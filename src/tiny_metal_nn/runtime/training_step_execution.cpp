/**
 * @file training_step_execution.cpp
 * @brief Internal tmnn-owned helpers for dispatch-plan binding and
 *        enqueue/finalize step orchestration.
 */

#include "tiny_metal_nn/runtime/training_step_execution.h"

#include "tiny_metal_nn/runtime/command_batch.h"
#include "tiny_metal_nn/runtime/metal_context_internal.h"
#include "tiny_metal_nn/runtime/metal_device.h"
#include "tiny_metal_nn/runtime/numerics_guard.h"
#include "tiny_metal_nn/runtime/parameter_store.h"
#include "tiny_metal_nn/runtime/step_lane_coordinator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace tmnn::detail {
namespace {

std::runtime_error make_runtime_error(const char *runtime_label,
                                      const std::string &message) {
  return std::runtime_error(std::string(runtime_label) + ": " + message);
}

std::runtime_error make_runtime_step_error(const char *runtime_label,
                                           const char *step_label,
                                           const std::string &message) {
  return std::runtime_error(std::string(runtime_label) + "::" + step_label +
                            ": " + message);
}

uint64_t elapsed_ns(std::chrono::steady_clock::time_point begin,
                    std::chrono::steady_clock::time_point end) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}

metal::DispatchDesc::BufferBind bind_view(const BufferView &view,
                                          uint32_t slot) {
  return {view.gpu_buffer, static_cast<uint32_t>(view.offset), slot};
}

BufferView resolve_binding_view(BindingRole role, const ParameterStore &ps,
                                const StepBufferSet *lane) {
  switch (role) {
  case BindingRole::StepPositions:
    if (!lane) {
      throw std::runtime_error("Missing step lane for StepPositions binding");
    }
    return lane->positions;
  case BindingRole::StepTargets:
    if (!lane) {
      throw std::runtime_error("Missing step lane for StepTargets binding");
    }
    return lane->targets;
  case BindingRole::StepLossReduction:
    if (!lane) {
      throw std::runtime_error(
          "Missing step lane for StepLossReduction binding");
    }
    return lane->loss_reduction;
  case BindingRole::ExternalGradient:
    if (!lane) {
      throw std::runtime_error(
          "Missing step lane for ExternalGradient binding");
    }
    return lane->external_gradient;
  case BindingRole::ForwardOutput:
    if (!lane) {
      throw std::runtime_error(
          "Missing step lane for ForwardOutput binding");
    }
    return lane->forward_output;
  case BindingRole::ProbeBuffer:
    if (!lane) {
      throw std::runtime_error(
          "Missing step lane for ProbeBuffer binding");
    }
    return lane->probe_buffer;
  case BindingRole::ConfigWeights:
    return ps.config_weights();
  case BindingRole::HashWeights:
    return ps.hash_weights();
  case BindingRole::GradHash:
    return ps.grad_hash();
  case BindingRole::GradMlp:
    return ps.grad_mlp();
  case BindingRole::ActiveHashMask:
    return ps.active_hash_mask();
  case BindingRole::ActiveHashSummaryMask:
    return ps.active_hash_summary_mask();
  case BindingRole::ActiveHashIndices:
    return ps.active_hash_indices();
  case BindingRole::TrainParams:
    return ps.train_params();
  case BindingRole::AdamParams:
    return ps.adam_params();
  case BindingRole::MlpWeights:
    return ps.mlp_weights();
  case BindingRole::AdamMHash:
    return ps.adam_m_hash();
  case BindingRole::AdamVHash:
    return ps.adam_v_hash();
  case BindingRole::AdamMMlp:
    return ps.adam_m_mlp();
  case BindingRole::AdamVMlp:
    return ps.adam_v_mlp();
  case BindingRole::FusedWeights:
    return ps.fused_weights();
  case BindingRole::FusedM:
    return ps.fused_m();
  case BindingRole::FusedV:
    return ps.fused_v();
  // ExternalGradient and ForwardOutput handled above as step-lane-varying.
  }
  throw std::runtime_error("Unhandled binding role");
}

void add_binding_entry(DispatchBindingTemplate &binding_template, uint32_t index,
                       BindingRole role, BindingResolutionClass resolution,
                       uint32_t slot) {
  binding_template.entries[index] = {role, resolution, slot};
  binding_template.count = std::max(binding_template.count, index + 1);
}

void resolve_runtime_stable_bindings(DispatchPlan &plan,
                                     const ParameterStore &ps) {
  plan.resolved.count = plan.binding_template.count;
  for (uint32_t i = 0; i < plan.binding_template.count; ++i) {
    const auto &entry = plan.binding_template.entries[i];
    if (entry.resolution != BindingResolutionClass::RuntimeStable) {
      continue;
    }
    auto view = resolve_binding_view(entry.role, ps, nullptr);
    plan.resolved.binds[i] = bind_view(view, entry.slot);
  }
}

std::vector<uint8_t> copy_view_bytes(MetalContext &ctx, const BufferView &view,
                                     const char *runtime_label,
                                     const char *label) {
  if (view.bytes == 0) {
    return {};
  }
  if (view.data) {
    auto *p = static_cast<const uint8_t *>(view.data);
    return std::vector<uint8_t>(p, p + view.bytes);
  }
  if (!view.gpu_buffer) {
    throw make_runtime_error(runtime_label,
                             std::string(label) +
                                 " is not backed by a GPU buffer");
  }
  std::vector<uint8_t> out(view.bytes);
  context_blit_download(ctx, view, out.data(), out.size());
  return out;
}

template <typename T>
std::vector<T> copy_view_pod(MetalContext &ctx, const BufferView &view,
                             const char *runtime_label, const char *label) {
  const auto bytes = copy_view_bytes(ctx, view, runtime_label, label);
  if (bytes.size() % sizeof(T) != 0) {
    throw make_runtime_error(runtime_label,
                             std::string(label) +
                                 " byte size is not POD-aligned");
  }

  std::vector<T> out(bytes.size() / sizeof(T));
  if (!bytes.empty()) {
    std::memcpy(out.data(), bytes.data(), bytes.size());
  }
  return out;
}

struct BackwardProbe {
  bool finite = true;
  float max_abs_gradient = 0.0f;
  double grad_l2_sum = 0.0;
};

struct UpdateProbe {
  bool finite = true;
  float max_abs_update = 0.0f;
};

struct AdamProbeParams {
  float lr_hash = 0.0f;
  float lr_mlp = 0.0f;
  float beta1 = 0.0f;
  float beta2 = 0.0f;
  float epsilon = 0.0f;
  float bias_correction_1 = 1.0f;
  float bias_correction_2 = 1.0f;
  float l1_hash = 0.0f;
  float l2_mlp = 0.0f;
  float grad_clip = 0.0f;
  float weight_decay = 0.0f;
};

struct DerivedStepNumerics {
  NumericsReport report;
  float grad_norm = 0.0f;
};

constexpr float kIntAtomicGradInvScale = 1.0f / 65536.0f;

float signum(float value) {
  if (value > 0.0f) {
    return 1.0f;
  }
  if (value < 0.0f) {
    return -1.0f;
  }
  return 0.0f;
}

bool numerics_report_has_anomaly(const NumericsReport &report) {
  return !report.finite_forward || !report.finite_backward ||
         !report.finite_update;
}

std::string describe_numerics_report(const NumericsReport &report) {
  return std::string("forward=") + (report.finite_forward ? "finite" : "nonfinite") +
         ", backward=" + (report.finite_backward ? "finite" : "nonfinite") +
         ", update=" + (report.finite_update ? "finite" : "nonfinite") +
         ", max_abs_gradient=" + std::to_string(report.max_abs_gradient) +
         ", max_abs_update=" + std::to_string(report.max_abs_update);
}

BackwardProbe probe_backward_numerics(const std::vector<int32_t> &grad_hash,
                                      const std::vector<float> &grad_mlp) {
  BackwardProbe out;
  auto note_gradient = [&](float g) {
    if (!std::isfinite(g)) {
      out.finite = false;
      return;
    }
    out.max_abs_gradient = std::max(out.max_abs_gradient, std::abs(g));
    out.grad_l2_sum += static_cast<double>(g) * static_cast<double>(g);
  };

  for (int32_t raw : grad_hash) {
    note_gradient(static_cast<float>(raw) * kIntAtomicGradInvScale);
  }
  for (float g : grad_mlp) {
    note_gradient(g);
  }

  return out;
}

AdamProbeParams parse_adam_probe_params(const std::vector<float> &params,
                                        const char *runtime_label) {
  if (params.size() < kFusedAdamParamFloats) {
    throw make_runtime_error(
        runtime_label,
        "adam_params buffer is smaller than expected");
  }

  AdamProbeParams out;
  out.lr_hash = params[0];
  out.lr_mlp = params[1];
  out.beta1 = params[2];
  out.beta2 = params[3];
  out.epsilon = params[4];
  out.bias_correction_1 = params[5];
  out.bias_correction_2 = params[6];
  out.l1_hash = params[7];
  out.l2_mlp = params[8];
  out.grad_clip = params[11];
  if (params.size() >= kUnifiedAdamParamFloats) {
    out.weight_decay = params[12];
  }
  return out;
}

void validate_float_count(const std::vector<float> &values, size_t expected,
                          const char *runtime_label, const char *label) {
  if (values.size() != expected) {
    throw make_runtime_error(runtime_label,
                             std::string(label) + " float count mismatch");
  }
}

void accumulate_update_probe_for_range(
    UpdateProbe &probe, const AdamProbeParams &params, bool is_mlp,
    const std::vector<float> &weights, const std::vector<float> &m_values,
    const std::vector<float> &v_values, size_t start, size_t count,
    const std::function<float(size_t)> &raw_gradient_at) {
  const float lr = is_mlp ? params.lr_mlp : params.lr_hash;
  for (size_t i = 0; i < count; ++i) {
    const size_t idx = start + i;
    const float raw_gradient = raw_gradient_at(i);
    if (!std::isfinite(raw_gradient)) {
      probe.finite = false;
      continue;
    }

    float g = raw_gradient;
    if (params.grad_clip > 0.0f) {
      g = std::clamp(g, -params.grad_clip, params.grad_clip);
    }

    const float weight = weights[idx];
    if (params.weight_decay > 0.0f) {
      // AdamW path: decay is applied directly to the weight before the Adam step.
    } else if (!is_mlp && params.l1_hash > 0.0f) {
      g += params.l1_hash * signum(weight);
    } else if (is_mlp && params.l2_mlp > 0.0f) {
      g += params.l2_mlp * weight;
    }

    const float m_new =
        params.beta1 * m_values[idx] + (1.0f - params.beta1) * g;
    const float v_new =
        params.beta2 * v_values[idx] + (1.0f - params.beta2) * g * g;
    const float m_hat = m_new / params.bias_correction_1;
    const float v_hat = v_new / params.bias_correction_2;
    const float adam_delta =
        lr * m_hat / (std::sqrt(v_hat) + params.epsilon);
    const float weight_decay_delta =
        (params.weight_decay > 0.0f) ? (lr * params.weight_decay * weight) : 0.0f;
    const float candidate_weight = weight - weight_decay_delta - adam_delta;
    const float update_delta = weight - candidate_weight;

    if (!std::isfinite(g) || !std::isfinite(m_new) || !std::isfinite(v_new) ||
        !std::isfinite(m_hat) || !std::isfinite(v_hat) ||
        !std::isfinite(adam_delta) || !std::isfinite(weight_decay_delta) ||
        !std::isfinite(candidate_weight) || !std::isfinite(update_delta)) {
      probe.finite = false;
      continue;
    }

    probe.max_abs_update =
        std::max(probe.max_abs_update, std::abs(update_delta));
  }
}

UpdateProbe probe_update_numerics(MetalContext &ctx, ParameterStore &ps,
                                  FinalizeTrainingStepTimings *timings,
                                  const char *runtime_label) {
  UpdateProbe out;
  const auto hash_count = ps.desc().hash_grid_size;
  const auto mlp_count = ps.desc().mlp_weight_count;

  const auto readback_begin = std::chrono::steady_clock::now();
  const auto hash_weights =
      copy_view_pod<float>(ctx, ps.hash_weights(), runtime_label, "hash_weights");
  const auto mlp_weights =
      copy_view_pod<float>(ctx, ps.mlp_weights(), runtime_label, "mlp_weights");
  const auto grad_hash =
      copy_view_pod<int32_t>(ctx, ps.grad_hash(), runtime_label, "grad_hash");
  const auto grad_mlp =
      copy_view_pod<float>(ctx, ps.grad_mlp(), runtime_label, "grad_mlp");
  const auto adam_m_hash =
      copy_view_pod<float>(ctx, ps.adam_m_hash(), runtime_label, "adam_m_hash");
  const auto adam_v_hash =
      copy_view_pod<float>(ctx, ps.adam_v_hash(), runtime_label, "adam_v_hash");
  const auto adam_m_mlp =
      copy_view_pod<float>(ctx, ps.adam_m_mlp(), runtime_label, "adam_m_mlp");
  const auto adam_v_mlp =
      copy_view_pod<float>(ctx, ps.adam_v_mlp(), runtime_label, "adam_v_mlp");
  const auto adam_params =
      copy_view_pod<float>(ctx, ps.adam_params(), runtime_label, "adam_params");
  const auto readback_end = std::chrono::steady_clock::now();
  if (timings) {
    timings->numerics_update_readback_ns +=
        elapsed_ns(readback_begin, readback_end);
  }

  const auto scan_begin = std::chrono::steady_clock::now();
  validate_float_count(hash_weights, hash_count, runtime_label, "hash_weights");
  validate_float_count(mlp_weights, mlp_count, runtime_label, "mlp_weights");
  validate_float_count(adam_m_hash, hash_count, runtime_label, "adam_m_hash");
  validate_float_count(adam_v_hash, hash_count, runtime_label, "adam_v_hash");
  validate_float_count(adam_m_mlp, mlp_count, runtime_label, "adam_m_mlp");
  validate_float_count(adam_v_mlp, mlp_count, runtime_label, "adam_v_mlp");
  if (grad_hash.size() != hash_count) {
    throw make_runtime_error(runtime_label,
                             "grad_hash element count mismatch");
  }
  validate_float_count(grad_mlp, mlp_count, runtime_label, "grad_mlp");

  const auto params = parse_adam_probe_params(adam_params, runtime_label);

  accumulate_update_probe_for_range(
      out, params, false, hash_weights, adam_m_hash, adam_v_hash, 0, hash_count,
      [&](size_t i) {
        return static_cast<float>(grad_hash[i]) * kIntAtomicGradInvScale;
      });
  accumulate_update_probe_for_range(
      out, params, true, mlp_weights, adam_m_mlp, adam_v_mlp, 0, mlp_count,
      [&](size_t i) { return grad_mlp[i]; });
  const auto scan_end = std::chrono::steady_clock::now();
  if (timings) {
    timings->numerics_update_scan_ns += elapsed_ns(scan_begin, scan_end);
  }
  return out;
}

DerivedStepNumerics make_numerics_report(
    MetalContext &ctx, ParameterStore &ps,
    const ParameterStore::AsyncStepResult &step_result, uint32_t step,
    bool safe_family_active, bool include_deep_probes,
    NumericsOverrideHook override_hook, FinalizeTrainingStepTimings *timings,
    const char *runtime_label) {
  DerivedStepNumerics derived;
  derived.report.finite_forward = std::isfinite(step_result.mean_loss);

  if (include_deep_probes) {
    const auto backward_readback_begin = std::chrono::steady_clock::now();
    const auto grad_hash =
        copy_view_pod<int32_t>(ctx, ps.grad_hash(), runtime_label, "grad_hash");
    const auto grad_mlp =
        copy_view_pod<float>(ctx, ps.grad_mlp(), runtime_label, "grad_mlp");
    const auto backward_readback_end = std::chrono::steady_clock::now();
    if (timings) {
      timings->numerics_backward_readback_ns +=
          elapsed_ns(backward_readback_begin, backward_readback_end);
    }
    const auto backward_scan_begin = std::chrono::steady_clock::now();
    const auto backward = probe_backward_numerics(grad_hash, grad_mlp);
    const auto backward_scan_end = std::chrono::steady_clock::now();
    if (timings) {
      timings->numerics_backward_scan_ns +=
          elapsed_ns(backward_scan_begin, backward_scan_end);
    }
    derived.report.finite_backward = backward.finite;
    derived.report.max_abs_gradient = backward.max_abs_gradient;
    derived.grad_norm =
        static_cast<float>(std::sqrt(std::max(0.0, backward.grad_l2_sum)));

    if (derived.report.finite_backward) {
      const auto update = probe_update_numerics(ctx, ps, timings, runtime_label);
      derived.report.finite_update = update.finite;
      derived.report.max_abs_update = update.max_abs_update;
    } else {
      derived.report.finite_update = false;
    }
  }

  if (override_hook) {
    if (auto override = override_hook(step, safe_family_active)) {
      derived.report = *override;
    }
  }
  return derived;
}

BatchFence submit_forward_backward_batch(CommandBatchPool &pool,
                                         PipelineRegistry &reg,
                                         ParameterStore &ps,
                                         DispatchPlan &fwd_bwd_plan,
                                         StepBufferSet &lane_buf,
                                         uint32_t total_threads,
                                         bool clear_grad_buffers,
                                         SubmitMode mode,
                                         const char *runtime_label) {
  auto batch = pool.begin_batch();
  if (batch.generation == 0) {
    throw make_runtime_error(
        runtime_label,
        "command batch pool exhausted — cannot submit forward/backward step");
  }

  auto *cmd = pool.current_command_buffer(batch);

  auto zero_or_fill = [&](const BufferView &view) {
    if (view.bytes == 0) {
      return;
    }
    if (view.data) {
      std::memset(view.data, 0, view.bytes);
      return;
    }
    metal::encode_blit_fill(cmd, view.gpu_buffer, view.offset, view.bytes, 0);
  };

  if (clear_grad_buffers) {
    zero_or_fill(ps.grad_hash());
    zero_or_fill(ps.grad_mlp());
  }
  zero_or_fill(lane_buf.loss_reduction);

  metal::DispatchDesc dd{};
  dd.cmd_buf = cmd;
  dd.pipeline = reg.raw_pipeline(fwd_bwd_plan.pipeline);
  dd.bindings = fwd_bwd_plan.resolved.binds.data();
  dd.binding_count = fwd_bwd_plan.resolved.count;
  dd.grid_x = total_threads;
  dd.grid_y = 1;
  dd.grid_z = 1;
  dd.tg_x = fwd_bwd_plan.tg_x;
  dd.tg_y = fwd_bwd_plan.tg_y;
  dd.tg_z = fwd_bwd_plan.tg_z;
  dd.threadgroup_memory_bytes = fwd_bwd_plan.threadgroup_memory_bytes;
  metal::encode_dispatch(dd);

  return pool.submit(batch, mode);
}

BatchFence submit_forward_backward_fill_batch(CommandBatchPool &pool,
                                              ParameterStore &ps,
                                              StepBufferSet &lane_buf,
                                              SubmitMode mode,
                                              const char *runtime_label) {
  CommandBatchHandle batch{};
  void *cmd = nullptr;
  auto ensure_batch = [&]() {
    if (batch.generation != 0) {
      return;
    }
    batch = pool.begin_batch();
    if (batch.generation == 0) {
      throw make_runtime_error(
          runtime_label,
          "command batch pool exhausted — cannot submit forward/backward clears");
    }
    cmd = pool.current_command_buffer(batch);
  };

  auto zero_or_fill = [&](const BufferView &view) {
    if (view.bytes == 0) {
      return;
    }
    if (view.data) {
      std::memset(view.data, 0, view.bytes);
      return;
    }
    ensure_batch();
    metal::encode_blit_fill(cmd, view.gpu_buffer, view.offset, view.bytes, 0);
  };

  zero_or_fill(ps.grad_hash());
  zero_or_fill(ps.grad_mlp());
  zero_or_fill(lane_buf.loss_reduction);

  if (batch.generation == 0) {
    return {};
  }
  return pool.submit(batch, mode);
}

BatchFence submit_forward_backward_dispatch_batch(CommandBatchPool &pool,
                                                  PipelineRegistry &reg,
                                                  DispatchPlan &fwd_bwd_plan,
                                                  uint32_t total_threads,
                                                  SubmitMode mode,
                                                  const char *runtime_label) {
  auto batch = pool.begin_batch();
  if (batch.generation == 0) {
    throw make_runtime_error(
        runtime_label,
        "command batch pool exhausted — cannot submit forward/backward dispatch");
  }

  auto *cmd = pool.current_command_buffer(batch);
  metal::DispatchDesc dd{};
  dd.cmd_buf = cmd;
  dd.pipeline = reg.raw_pipeline(fwd_bwd_plan.pipeline);
  dd.bindings = fwd_bwd_plan.resolved.binds.data();
  dd.binding_count = fwd_bwd_plan.resolved.count;
  dd.grid_x = total_threads;
  dd.grid_y = 1;
  dd.grid_z = 1;
  dd.tg_x = fwd_bwd_plan.tg_x;
  dd.tg_y = fwd_bwd_plan.tg_y;
  dd.tg_z = fwd_bwd_plan.tg_z;
  dd.threadgroup_memory_bytes = fwd_bwd_plan.threadgroup_memory_bytes;
  metal::encode_dispatch(dd);

  return pool.submit(batch, mode);
}

BatchFence submit_adam_batch(CommandBatchPool &pool, PipelineRegistry &reg,
                             ParameterStore &ps, DispatchPlan &adam_plan,
                             uint32_t weight_count,
                             SubmitMode mode, const char *runtime_label) {
  auto batch = pool.begin_batch();
  if (batch.generation == 0) {
    throw make_runtime_error(runtime_label,
                             "command batch pool exhausted — cannot submit Adam "
                             "update");
  }

  auto *cmd = pool.current_command_buffer(batch);
  metal::DispatchDesc adam_dd{};
  adam_dd.cmd_buf = cmd;
  adam_dd.pipeline = reg.raw_pipeline(adam_plan.pipeline);
  adam_dd.bindings = adam_plan.resolved.binds.data();
  adam_dd.binding_count = adam_plan.resolved.count;
  adam_dd.grid_x = weight_count;
  adam_dd.grid_y = 1;
  adam_dd.grid_z = 1;
  adam_dd.tg_x = adam_plan.tg_x;
  adam_dd.tg_y = adam_plan.tg_y;
  adam_dd.tg_z = adam_plan.tg_z;
  metal::encode_dispatch(adam_dd);

  return pool.submit(batch, mode);
}

BatchFence submit_split_adam_batch_impl(CommandBatchPool &pool,
                                        PipelineRegistry &reg,
                                        ParameterStore &ps,
                                        DispatchPlan &hash_adam_plan,
                                        uint32_t active_hash_count,
                                        DispatchPlan &mlp_adam_plan,
                                        uint32_t mlp_weight_count,
                                        SubmitMode mode,
                                        const char *runtime_label) {
  auto batch = pool.begin_batch();
  if (batch.generation == 0) {
    throw make_runtime_error(runtime_label,
                             "command batch pool exhausted — cannot submit split "
                             "Adam update");
  }

  auto *cmd = pool.current_command_buffer(batch);

  if (active_hash_count > 0) {
    metal::DispatchDesc hash_dd{};
    hash_dd.cmd_buf = cmd;
    hash_dd.pipeline = reg.raw_pipeline(hash_adam_plan.pipeline);
    hash_dd.bindings = hash_adam_plan.resolved.binds.data();
    hash_dd.binding_count = hash_adam_plan.resolved.count;
    hash_dd.grid_x = active_hash_count;
    hash_dd.grid_y = 1;
    hash_dd.grid_z = 1;
    hash_dd.tg_x = hash_adam_plan.tg_x;
    hash_dd.tg_y = hash_adam_plan.tg_y;
    hash_dd.tg_z = hash_adam_plan.tg_z;
    metal::encode_dispatch(hash_dd);
  }

  if (mlp_weight_count > 0) {
    metal::DispatchDesc mlp_dd{};
    mlp_dd.cmd_buf = cmd;
    mlp_dd.pipeline = reg.raw_pipeline(mlp_adam_plan.pipeline);
    mlp_dd.bindings = mlp_adam_plan.resolved.binds.data();
    mlp_dd.binding_count = mlp_adam_plan.resolved.count;
    mlp_dd.grid_x = mlp_weight_count;
    mlp_dd.grid_y = 1;
    mlp_dd.grid_z = 1;
    mlp_dd.tg_x = mlp_adam_plan.tg_x;
    mlp_dd.tg_y = mlp_adam_plan.tg_y;
    mlp_dd.tg_z = mlp_adam_plan.tg_z;
    metal::encode_dispatch(mlp_dd);
  }

  return pool.submit(batch, mode);
}

} // namespace

BatchFence submit_split_adam_batch(CommandBatchPool &pool, PipelineRegistry &reg,
                                   ParameterStore &ps,
                                   DispatchPlan &hash_adam_plan,
                                   uint32_t active_hash_count,
                                   DispatchPlan &mlp_adam_plan,
                                   uint32_t mlp_weight_count, SubmitMode mode,
                                   const char *runtime_label) {
  return submit_split_adam_batch_impl(pool, reg, ps, hash_adam_plan,
                                      active_hash_count, mlp_adam_plan,
                                      mlp_weight_count, mode, runtime_label);
}

DispatchPlan make_fwd_bwd_plan(const TrainingDispatchKernels &kernels,
                               const ParameterStore &ps, bool emit_probes) {
  DispatchPlan plan;
  plan.pipeline = kernels.fwd_bwd;
  plan.tg_x = kernels.tg_size;
  plan.threadgroup_memory_bytes = kernels.tg_memory_bytes;
  add_binding_entry(plan.binding_template, 0, BindingRole::StepPositions,
                    BindingResolutionClass::StepLaneVarying, 0);
  add_binding_entry(plan.binding_template, 1, BindingRole::StepTargets,
                    BindingResolutionClass::StepLaneVarying, 1);
  add_binding_entry(plan.binding_template, 2, BindingRole::ConfigWeights,
                    BindingResolutionClass::RuntimeStable, 2);
  add_binding_entry(plan.binding_template, 3, BindingRole::HashWeights,
                    BindingResolutionClass::RuntimeStable, 3);
  add_binding_entry(plan.binding_template, 4, BindingRole::GradHash,
                    BindingResolutionClass::RuntimeStable, 4);
  add_binding_entry(plan.binding_template, 5, BindingRole::GradMlp,
                    BindingResolutionClass::RuntimeStable, 5);
  add_binding_entry(plan.binding_template, 6, BindingRole::StepLossReduction,
                    BindingResolutionClass::StepLaneVarying, 6);
  add_binding_entry(plan.binding_template, 7, BindingRole::TrainParams,
                    BindingResolutionClass::RuntimeStable, 7);
  add_binding_entry(plan.binding_template, 8, BindingRole::ActiveHashMask,
                    BindingResolutionClass::RuntimeStable, 8);
  add_binding_entry(plan.binding_template, 9,
                    BindingRole::ActiveHashSummaryMask,
                    BindingResolutionClass::RuntimeStable, 9);
  if (emit_probes) {
    add_binding_entry(plan.binding_template, 10, BindingRole::ProbeBuffer,
                      BindingResolutionClass::StepLaneVarying, 10);
  }
  add_binding_entry(plan.binding_template, emit_probes ? 11u : 10u,
                    BindingRole::MlpWeights,
                    BindingResolutionClass::RuntimeStable, 11);
  resolve_runtime_stable_bindings(plan, ps);
  return plan;
}

DispatchPlan make_forward_training_plan(const TrainingDispatchKernels &kernels,
                                        const ParameterStore &ps,
                                        bool emit_probes) {
  DispatchPlan plan;
  plan.pipeline = kernels.forward_train;
  plan.tg_x = kernels.forward_train_tg_size;
  plan.tg_y = 1;
  plan.tg_z = 1;
  plan.threadgroup_memory_bytes = kernels.forward_train_tg_memory_bytes;
  // Same position/config/hash bindings as fwd_bwd. No grad buffers.
  add_binding_entry(plan.binding_template, 0, BindingRole::StepPositions,
                    BindingResolutionClass::StepLaneVarying, 0);
  add_binding_entry(plan.binding_template, 1, BindingRole::StepTargets,
                    BindingResolutionClass::StepLaneVarying, 1);
  add_binding_entry(plan.binding_template, 2, BindingRole::ConfigWeights,
                    BindingResolutionClass::RuntimeStable, 2);
  add_binding_entry(plan.binding_template, 3, BindingRole::HashWeights,
                    BindingResolutionClass::RuntimeStable, 3);
  // Slots 4-5 bound but unused by forward-only kernel (Metal requires all slots bound).
  add_binding_entry(plan.binding_template, 4, BindingRole::GradHash,
                    BindingResolutionClass::RuntimeStable, 4);
  add_binding_entry(plan.binding_template, 5, BindingRole::GradMlp,
                    BindingResolutionClass::RuntimeStable, 5);
  add_binding_entry(plan.binding_template, 6, BindingRole::StepLossReduction,
                    BindingResolutionClass::StepLaneVarying, 6);
  add_binding_entry(plan.binding_template, 7, BindingRole::TrainParams,
                    BindingResolutionClass::RuntimeStable, 7);
  add_binding_entry(plan.binding_template, 8, BindingRole::ForwardOutput,
                    BindingResolutionClass::StepLaneVarying, 8);
  if (emit_probes) {
    add_binding_entry(plan.binding_template, 9, BindingRole::ProbeBuffer,
                      BindingResolutionClass::StepLaneVarying, 9);
  }
  add_binding_entry(plan.binding_template, emit_probes ? 10u : 9u,
                    BindingRole::MlpWeights,
                    BindingResolutionClass::RuntimeStable, 11);
  resolve_runtime_stable_bindings(plan, ps);
  return plan;
}

DispatchPlan make_backward_ext_plan(const TrainingDispatchKernels &kernels,
                                    const ParameterStore &ps,
                                    bool emit_probes) {
  DispatchPlan plan;
  plan.pipeline = kernels.backward_ext;
  plan.tg_x = kernels.backward_ext_tg_size;
  plan.tg_y = 1;
  plan.tg_z = 1;
  plan.threadgroup_memory_bytes = kernels.backward_ext_tg_memory_bytes;
  // Same bindings as fwd_bwd (slots 0-7) + external gradient at slot 8 +
  // active-hash mask at slot 9.
  add_binding_entry(plan.binding_template, 0, BindingRole::StepPositions,
                    BindingResolutionClass::StepLaneVarying, 0);
  add_binding_entry(plan.binding_template, 1, BindingRole::StepTargets,
                    BindingResolutionClass::StepLaneVarying, 1);
  add_binding_entry(plan.binding_template, 2, BindingRole::ConfigWeights,
                    BindingResolutionClass::RuntimeStable, 2);
  add_binding_entry(plan.binding_template, 3, BindingRole::HashWeights,
                    BindingResolutionClass::RuntimeStable, 3);
  add_binding_entry(plan.binding_template, 4, BindingRole::GradHash,
                    BindingResolutionClass::RuntimeStable, 4);
  add_binding_entry(plan.binding_template, 5, BindingRole::GradMlp,
                    BindingResolutionClass::RuntimeStable, 5);
  add_binding_entry(plan.binding_template, 6, BindingRole::StepLossReduction,
                    BindingResolutionClass::StepLaneVarying, 6);
  add_binding_entry(plan.binding_template, 7, BindingRole::TrainParams,
                    BindingResolutionClass::RuntimeStable, 7);
  add_binding_entry(plan.binding_template, 8, BindingRole::ExternalGradient,
                    BindingResolutionClass::StepLaneVarying, 8);
  add_binding_entry(plan.binding_template, 9, BindingRole::ActiveHashMask,
                    BindingResolutionClass::RuntimeStable, 9);
  add_binding_entry(plan.binding_template, 10,
                    BindingRole::ActiveHashSummaryMask,
                    BindingResolutionClass::RuntimeStable, 10);
  add_binding_entry(plan.binding_template, 11u,
                    BindingRole::MlpWeights,
                    BindingResolutionClass::RuntimeStable, 11);
  if (emit_probes) {
    add_binding_entry(plan.binding_template, 12, BindingRole::ProbeBuffer,
                      BindingResolutionClass::StepLaneVarying, 12);
  }
  resolve_runtime_stable_bindings(plan, ps);
  return plan;
}

DispatchPlan make_adam_plan(const ParameterStore &ps, PipelineHandle pipeline) {
  DispatchPlan plan;
  plan.pipeline = pipeline;
  plan.tg_x = 256;
  add_binding_entry(plan.binding_template, 0, BindingRole::FusedWeights,
                    BindingResolutionClass::RuntimeStable, 0);
  add_binding_entry(plan.binding_template, 1, BindingRole::GradHash,
                    BindingResolutionClass::RuntimeStable, 1);
  add_binding_entry(plan.binding_template, 2, BindingRole::FusedM,
                    BindingResolutionClass::RuntimeStable, 2);
  add_binding_entry(plan.binding_template, 3, BindingRole::FusedV,
                    BindingResolutionClass::RuntimeStable, 3);
  add_binding_entry(plan.binding_template, 4, BindingRole::AdamParams,
                    BindingResolutionClass::RuntimeStable, 4);
  add_binding_entry(plan.binding_template, 5, BindingRole::GradMlp,
                    BindingResolutionClass::RuntimeStable, 5);
  resolve_runtime_stable_bindings(plan, ps);
  return plan;
}

DispatchPlan make_sparse_hash_adam_plan(const ParameterStore &ps,
                                        PipelineHandle pipeline) {
  DispatchPlan plan;
  plan.pipeline = pipeline;
  plan.tg_x = 256;
  add_binding_entry(plan.binding_template, 0, BindingRole::HashWeights,
                    BindingResolutionClass::RuntimeStable, 0);
  add_binding_entry(plan.binding_template, 1, BindingRole::GradHash,
                    BindingResolutionClass::RuntimeStable, 1);
  add_binding_entry(plan.binding_template, 2, BindingRole::AdamMHash,
                    BindingResolutionClass::RuntimeStable, 2);
  add_binding_entry(plan.binding_template, 3, BindingRole::AdamVHash,
                    BindingResolutionClass::RuntimeStable, 3);
  add_binding_entry(plan.binding_template, 4, BindingRole::AdamParams,
                    BindingResolutionClass::RuntimeStable, 4);
  add_binding_entry(plan.binding_template, 5, BindingRole::ActiveHashIndices,
                    BindingResolutionClass::RuntimeStable, 5);
  resolve_runtime_stable_bindings(plan, ps);
  return plan;
}

DispatchPlan make_mlp_dense_adam_plan(const ParameterStore &ps,
                                      PipelineHandle pipeline) {
  DispatchPlan plan;
  plan.pipeline = pipeline;
  plan.tg_x = 256;
  add_binding_entry(plan.binding_template, 0, BindingRole::MlpWeights,
                    BindingResolutionClass::RuntimeStable, 0);
  add_binding_entry(plan.binding_template, 1, BindingRole::GradMlp,
                    BindingResolutionClass::RuntimeStable, 1);
  add_binding_entry(plan.binding_template, 2, BindingRole::AdamMMlp,
                    BindingResolutionClass::RuntimeStable, 2);
  add_binding_entry(plan.binding_template, 3, BindingRole::AdamVMlp,
                    BindingResolutionClass::RuntimeStable, 3);
  add_binding_entry(plan.binding_template, 4, BindingRole::AdamParams,
                    BindingResolutionClass::RuntimeStable, 4);
  resolve_runtime_stable_bindings(plan, ps);
  return plan;
}

void resolve_step_lane_bindings(DispatchPlan &plan, const ParameterStore &ps,
                                const StepBufferSet &lane) {
  plan.resolved.count = plan.binding_template.count;
  for (uint32_t i = 0; i < plan.binding_template.count; ++i) {
    const auto &entry = plan.binding_template.entries[i];
    if (entry.resolution != BindingResolutionClass::StepLaneVarying) {
      continue;
    }
    auto view = resolve_binding_view(entry.role, ps, &lane);
    plan.resolved.binds[i] = bind_view(view, entry.slot);
  }
}

DispatchPlanDebugSnapshot snapshot_dispatch_plan(const DispatchPlan &plan) {
  DispatchPlanDebugSnapshot out;
  out.binding_template_storage = &plan.binding_template;
  out.resolved_binding_storage = plan.resolved.binds.data();
  out.count = plan.binding_template.count;
  for (uint32_t i = 0; i < plan.binding_template.count; ++i) {
    out.entries[i].role = plan.binding_template.entries[i].role;
    out.entries[i].slot = plan.binding_template.entries[i].slot;
    out.entries[i].resolution = plan.binding_template.entries[i].resolution;
    out.entries[i].buffer = plan.resolved.binds[i].buffer;
    out.entries[i].offset = plan.resolved.binds[i].offset;
  }
  return out;
}

PendingTrainingStep enqueue_training_step(
    const EnqueueTrainingStepRequest &request) {
  const auto total_begin = std::chrono::steady_clock::now();
  const uint32_t num_tgs =
      (request.batch_size + request.kernels.pts_per_tg - 1) /
      request.kernels.pts_per_tg;
  const uint32_t total_threads = num_tgs * request.kernels.tg_size;

  const auto drain_begin = std::chrono::steady_clock::now();
  drain_pending_training_step(request.pool, request.parameter_store,
                              request.lane_coordinator, request.step_lanes,
                              request.pending_to_drain);
  const auto drain_end = std::chrono::steady_clock::now();
  if (request.timings) {
    request.timings->drain_pending_ns += elapsed_ns(drain_begin, drain_end);
  }

  const auto lane = request.lane_coordinator.acquire_lane();
  if (lane == UINT32_MAX) {
    throw make_runtime_error(request.runtime_label,
                             "all step lanes busy — cannot enqueue training "
                             "step (internal invariant violation)");
  }
  auto &lane_buf = request.step_lanes[lane];

  try {
    if (request.prepare_step_lane) {
      const auto prepare_begin = std::chrono::steady_clock::now();
      request.prepare_step_lane(lane_buf);
      const auto prepare_end = std::chrono::steady_clock::now();
      if (request.timings) {
        request.timings->prepare_step_lane_ns +=
            elapsed_ns(prepare_begin, prepare_end);
      }
    }

    if (request.fill_train_params) {
      const auto fill_begin = std::chrono::steady_clock::now();
      request.fill_train_params(request.batch_size, request.logical_step);
      const auto fill_end = std::chrono::steady_clock::now();
      if (request.timings) {
        request.timings->fill_train_params_ns +=
            elapsed_ns(fill_begin, fill_end);
      }
    }

    const auto bind_begin = std::chrono::steady_clock::now();
    resolve_step_lane_bindings(request.forward_backward_plan,
                               request.parameter_store, lane_buf);
    const auto bind_end = std::chrono::steady_clock::now();
    if (request.timings) {
      request.timings->resolve_bindings_ns += elapsed_ns(bind_begin, bind_end);
    }

    const auto submit_begin = std::chrono::steady_clock::now();
    BatchFence fill_fence{};
    const BatchFence fence = submit_forward_backward_batch(
        request.pool, request.registry, request.parameter_store,
        request.forward_backward_plan, lane_buf, total_threads,
        request.clear_grad_buffers,
        SubmitMode::Async, request.runtime_label);
    const auto submit_end = std::chrono::steady_clock::now();
    if (request.timings) {
      request.timings->submit_forward_backward_ns +=
          elapsed_ns(submit_begin, submit_end);
    }
    request.lane_coordinator.bind_fence(lane, fence);

    PendingTrainingStep pending;
    pending.fence = fence;
    pending.fill_fence = fill_fence;
    pending.lane = lane;
    pending.num_tgs = num_tgs;
    pending.batch_N = request.batch_size;
    pending.logical_step = request.logical_step;
    pending.valid = true;
    if (request.timings) {
      request.timings->total_ns +=
          elapsed_ns(total_begin, std::chrono::steady_clock::now());
    }
    return pending;
  } catch (...) {
    request.lane_coordinator.release_lane(lane);
    throw;
  }
}

TrainingStepResult finalize_training_step(
    const FinalizeTrainingStepRequest &request) {
  const auto total_begin = std::chrono::steady_clock::now();
  auto &guard = context_numerics_guard(request.context);

  TrainingStepResult out;
  out.step = request.pending_step.logical_step + 1;

  if (!request.pending_step.valid) {
    throw make_runtime_step_error(request.runtime_label, "finalize_step",
                                  "invalid PendingStep ticket");
  }
  if (request.pending_step.lane >= request.step_lanes.size()) {
    throw make_runtime_step_error(request.runtime_label, "finalize_step",
                                  "pending lane is out of range");
  }

  auto &ps = request.parameter_store;
  auto &lane_buf = request.step_lanes[request.pending_step.lane];
  bool lane_released = false;
  bool stats_recorded = false;
  uint64_t numerics_reports = 0;
  uint64_t numerics_anomalies = 0;
  uint64_t bad_steps_skipped = 0;
  uint64_t bad_steps_rolled_back = 0;
  uint64_t safe_family_recoveries = 0;

  struct AttemptOutcome {
    ParameterStore::AsyncStepResult step_result;
    NumericsReport report;
    float grad_norm = 0.0f;
    bool anomaly = false;
  };

  auto safe_family_active = [&]() {
    return request.is_safe_family_active ? request.is_safe_family_active() : false;
  };

  auto record_attempt = [&](const ParameterStore::AsyncStepResult &step_result,
                            bool force_record) {
    AttemptOutcome outcome;
    outcome.step_result = step_result;
    const bool probe_required =
        force_record ||
        request.recovery_mode != BadStepRecoveryMode::SignalOnly ||
        guard.should_sample(out.step, !std::isfinite(step_result.mean_loss));
    const auto numerics_begin = std::chrono::steady_clock::now();
    const auto derived = make_numerics_report(
        request.context, ps, step_result, out.step, safe_family_active(),
        probe_required, request.numerics_override_hook, request.timings,
        request.runtime_label);
    const auto numerics_end = std::chrono::steady_clock::now();
    if (request.timings) {
      request.timings->numerics_report_ns +=
          elapsed_ns(numerics_begin, numerics_end);
    }
    outcome.report = derived.report;
    outcome.grad_norm = derived.grad_norm;
    outcome.anomaly = numerics_report_has_anomaly(outcome.report);
    const bool should_record =
        force_record || guard.should_sample(out.step, outcome.anomaly);
    if (should_record) {
      guard.record_step(out.step, outcome.report);
      ++numerics_reports;
      out.numerics_reported = true;
    }
    if (outcome.anomaly) {
      ++numerics_anomalies;
    }
    out.has_numerics_anomaly = out.has_numerics_anomaly || outcome.anomaly;
    return outcome;
  };

  auto finish_step = [&]() {
    if (!lane_released) {
      request.lane_coordinator.release_lane(request.pending_step.lane);
      lane_released = true;
    }
    if (!stats_recorded) {
      context_record_training_step(request.context, numerics_reports,
                                   numerics_anomalies, bad_steps_skipped,
                                   bad_steps_rolled_back,
                                   safe_family_recoveries);
      stats_recorded = true;
    }
  };

  auto rollback_committed_step = [&]() {
    if (request.rollback_committed_step) {
      request.rollback_committed_step();
    }
  };

  auto finalize_timings = [&]() {
    if (!request.timings) {
      return;
    }
    request.timings->total_ns =
        elapsed_ns(total_begin, std::chrono::steady_clock::now());
    const uint64_t bucketed_ns =
        request.timings->wait_pending_ns +
        request.timings->fill_adam_params_pre_finalize_ns +
        request.timings->finalize_step_readback_ns +
        request.timings->numerics_report_ns +
        request.timings->fill_adam_params_apply_ns +
        request.timings->prepare_sparse_hash_adam_ns +
        request.timings->submit_adam_ns +
        request.timings->sync_config_weights_ns +
        request.timings->append_extra_losses_ns;
    request.timings->uncategorized_ns =
        request.timings->total_ns > bucketed_ns
            ? request.timings->total_ns - bucketed_ns
            : 0;
  };

  try {
    uint64_t fill_wait_ns = 0;
    if (request.pending_step.fill_fence.value != 0) {
      const auto fill_wait_begin = std::chrono::steady_clock::now();
      request.pool.complete(request.pending_step.fill_fence);
      const auto fill_wait_end = std::chrono::steady_clock::now();
      fill_wait_ns = elapsed_ns(fill_wait_begin, fill_wait_end);
      if (request.timings) {
        request.timings->wait_fwd_bwd_fill_ns += fill_wait_ns;
      }
    }
    uint64_t dispatch_wait_ns = 0;
    const auto dispatch_wait_begin = std::chrono::steady_clock::now();
    request.pool.complete(request.pending_step.fence);
    const auto dispatch_wait_end = std::chrono::steady_clock::now();
    dispatch_wait_ns = elapsed_ns(dispatch_wait_begin, dispatch_wait_end);
    if (request.timings) {
      request.timings->wait_fwd_bwd_dispatch_ns += dispatch_wait_ns;
      request.timings->wait_pending_ns += fill_wait_ns + dispatch_wait_ns;
      request.timings->gpu_fwd_bwd_us =
          request.pool.gpu_time_us(request.pending_step.fence);
    }

    bool apply_update = true;

    if (request.fill_adam_params) {
      const auto fill_begin = std::chrono::steady_clock::now();
      request.fill_adam_params(request.pending_step.logical_step);
      const auto fill_end = std::chrono::steady_clock::now();
      if (request.timings) {
        request.timings->fill_adam_params_pre_finalize_ns +=
            elapsed_ns(fill_begin, fill_end);
      }
    }
    const auto finalize_begin = std::chrono::steady_clock::now();
    const auto step_result =
        ps.finalize_async_step(lane_buf, request.pending_step.num_tgs,
                               request.pending_step.batch_N,
                               request.pending_step.logical_step + 1,
                               /*sync_config_weights=*/false);
    const auto finalize_end = std::chrono::steady_clock::now();
    if (request.timings) {
      request.timings->finalize_step_readback_ns +=
          elapsed_ns(finalize_begin, finalize_end);
    }
    auto attempt = record_attempt(step_result, false);

    if (attempt.anomaly) {
      if (request.recovery_mode == BadStepRecoveryMode::Throw) {
        out.numerics = attempt.report;
        out.grad_norm = attempt.grad_norm;
        out.loss = attempt.step_result.mean_loss;
        rollback_committed_step();
        finish_step();
        throw make_runtime_step_error(
            request.runtime_label, "finalize_step",
            "numerics anomaly detected at step " + std::to_string(out.step) +
                " (" + describe_numerics_report(attempt.report) + ")");
      }

      if (request.recovery_mode == BadStepRecoveryMode::Skip) {
        out.numerics = attempt.report;
        out.grad_norm = attempt.grad_norm;
        out.loss = attempt.step_result.mean_loss;
        out.recovery_action = BadStepRecoveryAction::Skipped;
        ++bad_steps_skipped;
        apply_update = false;
      } else if (request.recovery_mode == BadStepRecoveryMode::Rollback) {
        out.numerics = attempt.report;
        out.grad_norm = attempt.grad_norm;
        out.loss = attempt.step_result.mean_loss;
        out.recovery_action = BadStepRecoveryAction::RolledBack;
        ++bad_steps_rolled_back;
        rollback_committed_step();
        apply_update = false;
      } else if (request.recovery_mode ==
                 BadStepRecoveryMode::FallbackAndRetryWithSafeFamily) {
        if (!request.activate_safe_family || !request.activate_safe_family()) {
          out.numerics = attempt.report;
          out.grad_norm = attempt.grad_norm;
          out.loss = attempt.step_result.mean_loss;
          rollback_committed_step();
          finish_step();
          throw make_runtime_step_error(
              request.runtime_label, "finalize_step",
              "numerics anomaly persisted while SafeDebugMetal is already active "
              "at step " + std::to_string(out.step) + " (" +
                  describe_numerics_report(attempt.report) + ")");
        }

        if (request.fill_train_params) {
          request.fill_train_params(request.pending_step.batch_N,
                                    request.pending_step.logical_step);
        }
        resolve_step_lane_bindings(request.forward_backward_plan, ps, lane_buf);
        const uint32_t retry_num_tgs =
            (request.pending_step.batch_N + request.kernels.pts_per_tg - 1) /
            request.kernels.pts_per_tg;
        (void)submit_forward_backward_batch(
            request.pool, request.registry, ps, request.forward_backward_plan,
            lane_buf, retry_num_tgs * request.kernels.tg_size,
            /*clear_grad_buffers=*/true, SubmitMode::Sync,
            request.runtime_label);

        if (request.fill_adam_params) {
          const auto retry_fill_begin = std::chrono::steady_clock::now();
          request.fill_adam_params(request.pending_step.logical_step);
          const auto retry_fill_end = std::chrono::steady_clock::now();
          if (request.timings) {
            request.timings->fill_adam_params_pre_finalize_ns +=
                elapsed_ns(retry_fill_begin, retry_fill_end);
          }
        }
        const auto retry_finalize_begin = std::chrono::steady_clock::now();
        const auto retry_step_result =
            ps.finalize_async_step(lane_buf, retry_num_tgs,
                                   request.pending_step.batch_N,
                                   request.pending_step.logical_step + 1,
                                   /*sync_config_weights=*/false);
        const auto retry_finalize_end = std::chrono::steady_clock::now();
        if (request.timings) {
          request.timings->finalize_step_readback_ns +=
              elapsed_ns(retry_finalize_begin, retry_finalize_end);
        }
        auto retry_attempt = record_attempt(retry_step_result, true);
        if (retry_attempt.anomaly) {
          out.numerics = retry_attempt.report;
          out.grad_norm = retry_attempt.grad_norm;
          out.loss = retry_attempt.step_result.mean_loss;
          rollback_committed_step();
          finish_step();
          throw make_runtime_step_error(
              request.runtime_label, "finalize_step",
              "SafeDebugMetal retry still produced a numerics anomaly at step " +
                  std::to_string(out.step) + " (" +
                  describe_numerics_report(retry_attempt.report) + ")");
        }

        attempt = std::move(retry_attempt);
        out.recovery_action = BadStepRecoveryAction::RetriedWithSafeFamily;
        ++safe_family_recoveries;
      }
    }

    out.loss = attempt.step_result.mean_loss;
    out.numerics = attempt.report;
    out.grad_norm = attempt.grad_norm;

    if (apply_update) {
      if (request.fill_adam_params) {
        const auto fill_begin = std::chrono::steady_clock::now();
        request.fill_adam_params(request.pending_step.logical_step);
        const auto fill_end = std::chrono::steady_clock::now();
        if (request.timings) {
          request.timings->fill_adam_params_apply_ns +=
            elapsed_ns(fill_begin, fill_end);
        }
      }
      uint32_t active_hash_count = 0;
      const bool use_split_adam =
          request.sparse_hash_adam_plan != nullptr &&
          request.dense_mlp_adam_plan != nullptr &&
          static_cast<bool>(request.prepare_sparse_hash_adam);
      if (use_split_adam) {
        const auto prepare_sparse_begin = std::chrono::steady_clock::now();
        active_hash_count = request.prepare_sparse_hash_adam();
        const auto prepare_sparse_end = std::chrono::steady_clock::now();
        if (request.timings) {
          request.timings->prepare_sparse_hash_adam_ns +=
              elapsed_ns(prepare_sparse_begin, prepare_sparse_end);
        }
      }
      const auto adam_begin = std::chrono::steady_clock::now();
      const auto adam_fence = use_split_adam
                                  ? submit_split_adam_batch(
                                        request.pool, request.registry, ps,
                                        *request.sparse_hash_adam_plan,
                                        active_hash_count,
                                        *request.dense_mlp_adam_plan,
                                        static_cast<uint32_t>(
                                            ps.desc().mlp_weight_count),
                                        SubmitMode::Sync, request.runtime_label)
                                  : submit_adam_batch(
                                        request.pool, request.registry, ps,
                                        request.adam_plan,
                                        static_cast<uint32_t>(
                                            ps.desc().hash_grid_size +
                                            ps.desc().mlp_weight_count),
                                        SubmitMode::Sync, request.runtime_label);
      const auto adam_end = std::chrono::steady_clock::now();
      if (request.timings) {
        request.timings->submit_adam_ns += elapsed_ns(adam_begin, adam_end);
        request.timings->gpu_adam_us =
            request.pool.gpu_time_us(adam_fence);
      }

      // Post-Adam: zero gradients so fused path leaves clean state.
      // (Adam kernel no longer clears in-place; explicit blit-fill needed.)
      auto post_adam_batch = request.pool.begin_batch();
      auto *post_cmd = request.pool.current_command_buffer(post_adam_batch);
      metal::encode_blit_fill(post_cmd, ps.grad_hash().gpu_buffer,
                              ps.grad_hash().offset, ps.grad_hash().bytes, 0);
      metal::encode_blit_fill(post_cmd, ps.grad_mlp().gpu_buffer,
                              ps.grad_mlp().offset, ps.grad_mlp().bytes, 0);
      [[maybe_unused]] const auto post_fence =
          request.pool.submit(post_adam_batch, SubmitMode::Sync);
    }

    uint32_t count = 0;
    for (uint32_t i = 0;
         i < attempt.step_result.extra_loss_count &&
         count < TrainingStepResult::kMaxExtraLosses;
         ++i) {
      out.extra_losses[count++] = attempt.step_result.extra_losses[i];
    }
    out.extra_loss_count = count;

    if (request.append_extra_losses) {
      const auto extra_begin = std::chrono::steady_clock::now();
      request.append_extra_losses(out);
      const auto extra_end = std::chrono::steady_clock::now();
      if (request.timings) {
        request.timings->append_extra_losses_ns +=
            elapsed_ns(extra_begin, extra_end);
      }
    }

    finish_step();
    finalize_timings();
    return out;
  } catch (...) {
    finish_step();
    finalize_timings();
    throw;
  }
}

ProbeResult aggregate_probe_partials(const BufferView &probe_buf,
                                     uint32_t num_tgs,
                                     uint32_t num_hidden_layers) {
  ProbeResult result;
  result.num_hidden_layers = num_hidden_layers;
  if (!probe_buf.data || probe_buf.bytes == 0 || num_tgs == 0)
    return result;

  const uint32_t stride = ProbeResult::stride_for_layers(num_hidden_layers);
  const auto *data = static_cast<const float *>(probe_buf.data);
  const uint32_t L = num_hidden_layers;

  // Layout per TG: [fwd_nan, bwd_nan, hash_grad_l2,
  //                  act_max[0..L-1], output_abs_max, output_neg_min,
  //                  grad_l2[0..L-1]]
  float fwd_nan = 0.0f, bwd_nan = 0.0f, hash_l2 = 0.0f;
  float out_max = 0.0f, out_neg_min = 0.0f;
  float act_max[ProbeResult::kMaxLayers] = {};
  float grad_l2[ProbeResult::kMaxLayers] = {};

  for (uint32_t tg = 0; tg < num_tgs; ++tg) {
    const float *row = data + tg * stride;
    fwd_nan = std::max(fwd_nan, row[0]);
    bwd_nan = std::max(bwd_nan, row[1]);
    hash_l2 += row[2];
    for (uint32_t l = 0; l < L; ++l)
      act_max[l] = std::max(act_max[l], row[3 + l]);
    out_max = std::max(out_max, row[3 + L]);
    out_neg_min = std::max(out_neg_min, row[3 + L + 1]);
    for (uint32_t l = 0; l < L; ++l)
      grad_l2[l] += row[3 + L + 2 + l];
  }

  result.has_nan_forward = fwd_nan > 0.0f;
  result.has_nan_backward = bwd_nan > 0.0f;
  result.hash_grad_l2 = hash_l2;
  result.output_abs_max = out_max;
  result.output_min = -out_neg_min; // kernel wrote -min(sdf)
  for (uint32_t l = 0; l < L && l < ProbeResult::kMaxLayers; ++l) {
    result.act_max[l] = act_max[l];
    result.grad_l2[l] = grad_l2[l];
  }
  return result;
}

} // namespace tmnn::detail

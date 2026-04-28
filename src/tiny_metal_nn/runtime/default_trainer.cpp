/**
 * @file default_trainer.cpp
 * @brief Default ITrainerRuntime using only tmnn types.
 */

#include "tiny-metal-nn/default_trainer.h"
#include "tiny-metal-nn/factory.h"

#include "tiny_metal_nn/runtime/buffer_arena.h"
#include "tiny_metal_nn/runtime/command_batch.h"
#include "tiny_metal_nn/runtime/adam_params.h"
#include "tiny_metal_nn/runtime/metal_context_internal.h"
#include "tiny_metal_nn/runtime/morton_sort.h"
#include "tiny_metal_nn/runtime/default_trainer_policy.h"
#include "tiny_metal_nn/runtime/kernel_dispatch_contract.h"
#include "tiny_metal_nn/runtime/parameter_store.h"
#include "tiny_metal_nn/runtime/pipeline_registry.h"
#include "tiny_metal_nn/runtime/runtime_policy.h"
#include "tiny_metal_nn/runtime/step_lane_coordinator.h"
#include "tiny_metal_nn/runtime/training_step_execution.h"
#include "tiny_metal_nn/runtime/training_step_lifecycle.h"

#include "tiny-metal-nn/kernels/adam_kernel_msl.h"
#include "tiny-metal-nn/kernels/kernel_compiler.h"
#include "tiny-metal-nn/kernels/kernel_spec.h"
#include "tiny-metal-nn/extension/schema.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace tmnn {

namespace {

constexpr uint32_t kStepLaneCount = 2;
constexpr int kHeadlineTrainerLog2HashmapSize = 14;
constexpr int kHeadlineTrainerHiddenDim = 32;
constexpr int kHeadlineTrainerBatchSize = 1024;
constexpr uint32_t kFNVPrimeY = 2654435761u;
constexpr uint32_t kFNVPrimeZ = 805459861u;
constexpr uint32_t kFNVPrimeW = 3674653429u;

[[nodiscard]] inline uint32_t hash_coords_3d(int x, int y, int z,
                                             uint32_t table_size) {
  const uint32_t h = static_cast<uint32_t>(x) * 1u ^
                     static_cast<uint32_t>(y) * kFNVPrimeY ^
                     static_cast<uint32_t>(z) * kFNVPrimeZ;
  return h % table_size;
}

[[nodiscard]] inline uint32_t hash_coords_4d(int x, int y, int z, int w,
                                             uint32_t table_size) {
  const uint32_t h = static_cast<uint32_t>(x) * 1u ^
                     static_cast<uint32_t>(y) * kFNVPrimeY ^
                     static_cast<uint32_t>(z) * kFNVPrimeZ ^
                     static_cast<uint32_t>(w) * kFNVPrimeW;
  return h % table_size;
}

std::vector<uint8_t> copy_view_bytes(MetalContext &ctx, const BufferView &view,
                                     const char *label) {
  if (view.bytes == 0)
    return {};
  if (view.data) {
    auto *p = static_cast<const uint8_t *>(view.data);
    return std::vector<uint8_t>(p, p + view.bytes);
  }
  if (!view.gpu_buffer) {
    throw std::runtime_error(std::string("DefaultRuntime: ") + label +
                             " is not backed by a GPU buffer");
  }
  std::vector<uint8_t> out(view.bytes);
  detail::context_blit_download(ctx, view, out.data(), out.size());
  return out;
}

void append_u32(std::vector<uint8_t> &out, uint32_t value) {
  out.push_back(static_cast<uint8_t>(value & 0xFFu));
  out.push_back(static_cast<uint8_t>((value >> 8) & 0xFFu));
  out.push_back(static_cast<uint8_t>((value >> 16) & 0xFFu));
  out.push_back(static_cast<uint8_t>((value >> 24) & 0xFFu));
}

uint32_t read_u32(const uint8_t *&cursor, size_t &remaining) {
  if (remaining < 4) {
    throw std::runtime_error("DefaultRuntime: optimizer blob truncated");
  }
  uint32_t value = static_cast<uint32_t>(cursor[0]) |
                   (static_cast<uint32_t>(cursor[1]) << 8) |
                   (static_cast<uint32_t>(cursor[2]) << 16) |
                   (static_cast<uint32_t>(cursor[3]) << 24);
  cursor += 4;
  remaining -= 4;
  return value;
}

class InspectableTrainerRuntime {
public:
  virtual ~InspectableTrainerRuntime() = default;
  [[nodiscard]] virtual TrainerRuntimeInspection inspect_runtime() const = 0;
};

class InspectableEvaluator {
public:
  virtual ~InspectableEvaluator() = default;
  virtual void
  merge_runtime_inspection(TrainerRuntimeInspection &inspection) const = 0;
};

uint64_t elapsed_ns(std::chrono::steady_clock::time_point begin,
                    std::chrono::steady_clock::time_point end) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}

[[nodiscard]] TrainerKernelInspection
to_public_inspection(const detail::KernelDispatchContract &contract) {
  TrainerKernelInspection inspection;
  if (!contract.available)
    return inspection;
  inspection.available = true;
  inspection.entry_point = contract.entry_point;
  inspection.requested_simd = contract.simd.requested;
  inspection.realized_simd = contract.simd.realized;
  inspection.requested_fp16 = contract.fp16.requested;
  inspection.realized_fp16 = contract.fp16.realized;
  inspection.requested_tg_weight_cache = contract.tg_weight_cache.requested;
  inspection.realized_tg_weight_cache = contract.tg_weight_cache.realized;
  inspection.threadgroup_size = contract.geometry.tg_size;
  inspection.points_per_threadgroup = contract.geometry.pts_per_tg;
  inspection.threadgroup_memory_bytes =
      contract.geometry.threadgroup_memory_bytes;
  return inspection;
}

[[nodiscard]] TrainingStepProfile
to_public_training_step_profile(uint32_t step, uint32_t batch_size,
                                uint64_t morton_sort_ns,
                                const detail::EnqueueTrainingStepTimings &enqueue,
                                const detail::FinalizeTrainingStepTimings &finalize,
                                uint64_t probe_aggregation_ns,
                                uint64_t total_ns) {
  TrainingStepProfile profile;
  profile.step = step;
  profile.batch_size = batch_size;
  profile.total_ns = total_ns;
  profile.morton_sort_ns = morton_sort_ns;
  profile.enqueue_total_ns = enqueue.total_ns;
  profile.drain_pending_ns = enqueue.drain_pending_ns;
  profile.prepare_step_lane_ns = enqueue.prepare_step_lane_ns;
  profile.fill_train_params_ns = enqueue.fill_train_params_ns;
  profile.resolve_bindings_ns = enqueue.resolve_bindings_ns;
  profile.submit_forward_backward_ns = enqueue.submit_forward_backward_ns;
  profile.finalize_total_ns = finalize.total_ns;
  profile.wait_pending_ns = finalize.wait_pending_ns;
  profile.wait_fwd_bwd_fill_ns = finalize.wait_fwd_bwd_fill_ns;
  profile.wait_fwd_bwd_dispatch_ns = finalize.wait_fwd_bwd_dispatch_ns;
  profile.fill_adam_params_pre_finalize_ns =
      finalize.fill_adam_params_pre_finalize_ns;
  profile.finalize_step_readback_ns = finalize.finalize_step_readback_ns;
  profile.numerics_report_ns = finalize.numerics_report_ns;
  profile.numerics_backward_readback_ns =
      finalize.numerics_backward_readback_ns;
  profile.numerics_backward_scan_ns = finalize.numerics_backward_scan_ns;
  profile.numerics_update_readback_ns =
      finalize.numerics_update_readback_ns;
  profile.numerics_update_scan_ns = finalize.numerics_update_scan_ns;
  profile.fill_adam_params_apply_ns = finalize.fill_adam_params_apply_ns;
  profile.prepare_sparse_hash_adam_ns = finalize.prepare_sparse_hash_adam_ns;
  profile.submit_adam_ns = finalize.submit_adam_ns;
  profile.sync_config_weights_ns = finalize.sync_config_weights_ns;
  profile.append_extra_losses_ns = finalize.append_extra_losses_ns;
  profile.probe_aggregation_ns = probe_aggregation_ns;
  profile.uncategorized_ns = finalize.uncategorized_ns;
  profile.gpu_fwd_bwd_us = finalize.gpu_fwd_bwd_us;
  profile.gpu_adam_us = finalize.gpu_adam_us;
  return profile;
}

void verify_plan_matches_contract(const detail::DispatchPlan &plan,
                                  const detail::KernelDispatchContract &contract,
                                  const char *label) {
  detail::validate_dispatch_contract(contract);
  if (plan.tg_x != contract.geometry.tg_size) {
    throw std::logic_error(std::string("DefaultRuntime: ") + label +
                           " tg_x does not match dispatch contract");
  }
  if (plan.threadgroup_memory_bytes !=
      contract.geometry.threadgroup_memory_bytes) {
    throw std::logic_error(std::string("DefaultRuntime: ") + label +
                           " threadgroup memory does not match dispatch contract");
  }
}

void append_section(std::vector<uint8_t> &payload,
                    const std::vector<uint8_t> &section) {
  append_u32(payload, static_cast<uint32_t>(section.size()));
  payload.insert(payload.end(), section.begin(), section.end());
}

std::vector<uint8_t> read_section(const uint8_t *&cursor, size_t &remaining) {
  const uint32_t size = read_u32(cursor, remaining);
  if (remaining < size) {
    throw std::runtime_error("DefaultRuntime: optimizer blob section truncated");
  }
  std::vector<uint8_t> out(cursor, cursor + size);
  cursor += size;
  remaining -= size;
  return out;
}

void zero_gpu_only_view(MetalContext &ctx, const BufferView &view) {
  if (view.data || !view.gpu_buffer || view.bytes == 0)
    return;
  auto mutable_view = view;
  detail::context_blit_fill(ctx, mutable_view, 0);
}

void zero_buffer_view(MetalContext &ctx, const BufferView &view) {
  if (view.bytes == 0)
    return;
  if (view.data) {
    std::memset(view.data, 0, view.bytes);
    return;
  }
  if (!view.gpu_buffer)
    return;
  auto mutable_view = view;
  detail::context_blit_fill(ctx, mutable_view, 0);
}

// ── DefaultAuthority ────────────────────────────────────────────────────

class DefaultAuthority final : public RuntimeAuthority {
public:
  DefaultAuthority(std::shared_ptr<MetalContext> ctx,
                   std::shared_ptr<ParameterStore> ps,
                   const ParameterStoreDesc &desc)
      : ctx_(std::move(ctx)), ps_(std::move(ps)) {
    layout_.hash_grid_float_count = desc.hash_grid_size;
    layout_.mlp_weight_float_count = desc.mlp_weight_count;
    layout_.train_params_layout = desc.train_params_layout;
    layout_.adam_params_float_count = desc.adam_params_float_count;
    layout_.target_dims = desc.target_dims;
    layout_.reduction_terms = desc.reduction_terms;
    layout_.fused_adam = desc.use_fused_adam;
  }

  [[nodiscard]] const std::shared_ptr<MetalContext> &context() const override {
    return ctx_;
  }
  [[nodiscard]] ParameterLayout parameter_layout() const override {
    return layout_;
  }
  [[nodiscard]] RuntimeStoragePolicy storage_policy() const override {
    return {};
  }
  [[nodiscard]] RuntimeBufferView
  buffer(RuntimeBufferRole role) const override {
    auto to_view = [](BufferView bv) {
      RuntimeBufferView rv;
      rv.cpu_data = bv.data;
      rv.gpu_buffer = bv.gpu_buffer;
      rv.offset = bv.offset;
      rv.bytes = bv.bytes;
      return rv;
    };
    switch (role) {
    case RuntimeBufferRole::ConfigWeights:
      return to_view(ps_->config_weights());
    case RuntimeBufferRole::HashWeights:
      return to_view(ps_->hash_weights());
    case RuntimeBufferRole::MlpWeights:
      return to_view(ps_->mlp_weights());
    case RuntimeBufferRole::GradHash:
      return to_view(ps_->grad_hash());
    case RuntimeBufferRole::GradMlp:
      return to_view(ps_->grad_mlp());
    case RuntimeBufferRole::TrainParams:
      return to_view(ps_->train_params());
    case RuntimeBufferRole::AdamParams:
      return to_view(ps_->adam_params());
    case RuntimeBufferRole::FusedWeights:
      return to_view(ps_->fused_weights());
    case RuntimeBufferRole::FusedFirstMoment:
      return to_view(ps_->fused_m());
    case RuntimeBufferRole::FusedSecondMoment:
      return to_view(ps_->fused_v());
    }
    return {};
  }

private:
  std::shared_ptr<MetalContext> ctx_;
  std::shared_ptr<ParameterStore> ps_;
  ParameterLayout layout_;
};

// ── DefaultEvaluator ────────────────────────────────────────────────────

class DefaultEvaluator final : public FieldEvaluator, public InspectableEvaluator {
public:
  DefaultEvaluator(std::shared_ptr<const RuntimeAuthority> authority,
                   const KernelSpec &spec,
                   std::shared_ptr<MetalContext> ctx)
      : authority_(std::move(authority)), spec_(spec), ctx_(std::move(ctx)) {}

  [[nodiscard]] uint32_t n_input_dims() const override {
    return static_cast<uint32_t>(spec_.spatial_dims);
  }
  [[nodiscard]] uint32_t n_output_dims() const override {
    return static_cast<uint32_t>(spec_.num_outputs);
  }

  bool evaluate(const float *positions, float *output, int N) override {
    clear_diagnostic();
    if (N < 0) {
      set_diagnostic(DiagnosticCode::InvalidArgument,
                     "FieldEvaluator::evaluate",
                     "N must be non-negative");
      return false;
    }
    if (N == 0)
      return true;
    if (!positions || !output) {
      set_diagnostic(DiagnosticCode::InvalidArgument,
                     "FieldEvaluator::evaluate",
                     "positions and output must be non-null when N > 0");
      return false;
    }
    if (!authority_) {
      set_diagnostic(DiagnosticCode::MissingRuntimeAuthority,
                     "FieldEvaluator::evaluate",
                     "runtime authority is unavailable");
      return false;
    }
    if (!ctx_) {
      set_diagnostic(DiagnosticCode::MissingRuntimeContext,
                     "FieldEvaluator::evaluate",
                     "MetalContext is unavailable");
      return false;
    }
    if (!ctx_->is_gpu_available()) {
      set_diagnostic(DiagnosticCode::GpuUnavailable,
                     "FieldEvaluator::evaluate",
                     "Metal GPU is unavailable");
      return false;
    }

    // Compile eval kernel on first call.
    // Keep evaluator on the scalar non-FP16 path so it only needs the config
    // header plus the live MLP weight buffer.
    if (!eval_compiled_) {
      auto &reg = detail::context_pipeline_registry(*ctx_);
      extension::KernelCompileSpec compile_spec;
      compile_spec.allow_simd = false;
      compile_spec.allow_fp16 = false;
      auto result = KernelCompiler::compile(
          {KernelRole::Eval, spec_,
           KernelCompiler::makeDefaultSchema(spec_), compile_spec});
      if (result.source.empty()) {
        set_diagnostic(DiagnosticCode::KernelCompilationFailed,
                       "FieldEvaluator::evaluate",
                       "KernelCompiler returned empty source for Eval");
        return false;
      }
      PipelineKey key{result.key.hash(), result.entry_point.c_str(), false};
      eval_pipeline_ = reg.register_pipeline(
          key, result.source.c_str(), result.entry_point.c_str());
      eval_contract_ = detail::make_dispatch_contract(result);
      eval_tg_size_ = eval_contract_.geometry.tg_size;
      eval_compiled_ = true;
    }

    // Dispatch eval kernel.
    auto &pool = detail::context_batch_pool(*ctx_);
    auto &reg = detail::context_pipeline_registry(*ctx_);
    auto batch = pool.begin_batch();
    if (batch.generation == 0) {
      set_diagnostic(DiagnosticCode::BatchSubmissionUnavailable,
                     "FieldEvaluator::evaluate",
                     "command batch pool returned generation 0");
      return false;
    }

    auto *cmd = pool.current_command_buffer(batch);
    auto config_view = authority_->buffer(RuntimeBufferRole::ConfigWeights);
    auto hash_view = authority_->buffer(RuntimeBufferRole::HashWeights);
    auto mlp_view = authority_->buffer(RuntimeBufferRole::MlpWeights);
    if (!config_view.gpu_buffer) {
      set_diagnostic(DiagnosticCode::MissingRuntimeBuffer,
                     "FieldEvaluator::evaluate",
                     "ConfigWeights GPU buffer is unavailable");
      return false;
    }
    if (!hash_view.gpu_buffer) {
      set_diagnostic(DiagnosticCode::MissingRuntimeBuffer,
                     "FieldEvaluator::evaluate",
                     "HashWeights GPU buffer is unavailable");
      return false;
    }
    if (!mlp_view.gpu_buffer) {
      set_diagnostic(DiagnosticCode::MissingRuntimeBuffer,
                     "FieldEvaluator::evaluate",
                     "MlpWeights GPU buffer is unavailable");
      return false;
    }
    if (config_view.cpu_data && config_view.bytes >= kConfigPackedFloats * sizeof(float)) {
      auto *cfg = static_cast<float *>(config_view.cpu_data);
      cfg[kConfigPackedFloats - 1] = static_cast<float>(N);
    }

    // Upload positions to a temporary buffer.
    auto &arena = detail::context_arena(*ctx_);
    const size_t pos_bytes = static_cast<size_t>(N) * n_input_dims() * sizeof(float);
    const size_t out_bytes = static_cast<size_t>(N) * n_output_dims() * sizeof(float);
    BufferDesc pos_desc;
    pos_desc.bytes = pos_bytes;
    pos_desc.storage = BufferStorage::Shared;
    pos_desc.debug_name = "eval_positions";
    auto pos_handle = arena.allocate(pos_desc);
    auto pos_view = arena.view(pos_handle);
    BufferDesc out_desc;
    out_desc.bytes = out_bytes;
    out_desc.storage = BufferStorage::Shared;
    out_desc.debug_name = "eval_output";
    auto out_handle = arena.allocate(out_desc);
    auto out_view = arena.view(out_handle);
    std::memcpy(pos_view.data, positions, pos_bytes);

    metal::DispatchDesc dd{};
    dd.cmd_buf = cmd;
    dd.pipeline = reg.raw_pipeline(eval_pipeline_);
    // Eval kernel bindings: [0]=positions, [1]=output, [2]=config header,
    // [3]=hash grid, [5]=live mlp weights.
    metal::DispatchDesc::BufferBind binds[5] = {
        {pos_view.gpu_buffer, static_cast<uint32_t>(pos_view.offset), 0},
        {out_view.gpu_buffer, static_cast<uint32_t>(out_view.offset), 1},
        {config_view.gpu_buffer, static_cast<uint32_t>(config_view.offset), 2},
        {hash_view.gpu_buffer, static_cast<uint32_t>(hash_view.offset), 3},
        {mlp_view.gpu_buffer, static_cast<uint32_t>(mlp_view.offset), 5},
    };
    dd.bindings = binds;
    dd.binding_count = 5;
    dd.grid_x = static_cast<uint32_t>(N);
    dd.grid_y = 1;
    dd.grid_z = 1;
    dd.tg_x = eval_tg_size_;
    dd.tg_y = 1;
    dd.tg_z = 1;
    metal::encode_dispatch(dd);

    [[maybe_unused]] const auto eval_fence =
        pool.submit(batch, SubmitMode::Sync);
    std::memcpy(output, out_view.data, out_bytes);
    arena.release(pos_handle);
    arena.release(out_handle);
    return true;
  }

  bool evaluate_with_gradient(const float *positions, float *output,
                              float *gradients, int N) override {
    clear_diagnostic();
    if (N < 0) {
      set_diagnostic(DiagnosticCode::InvalidArgument,
                     "FieldEvaluator::evaluate_with_gradient",
                     "N must be non-negative");
      return false;
    }
    if (N == 0)
      return true;
    if (!positions || !output || !gradients) {
      set_diagnostic(DiagnosticCode::InvalidArgument,
                     "FieldEvaluator::evaluate_with_gradient",
                     "positions, output, and gradients must be non-null when N > 0");
      return false;
    }
    if (!authority_) {
      set_diagnostic(DiagnosticCode::MissingRuntimeAuthority,
                     "FieldEvaluator::evaluate_with_gradient",
                     "runtime authority is unavailable");
      return false;
    }
    if (!ctx_) {
      set_diagnostic(DiagnosticCode::MissingRuntimeContext,
                     "FieldEvaluator::evaluate_with_gradient",
                     "MetalContext is unavailable");
      return false;
    }
    if (!ctx_->is_gpu_available()) {
      set_diagnostic(DiagnosticCode::GpuUnavailable,
                     "FieldEvaluator::evaluate_with_gradient",
                     "Metal GPU is unavailable");
      return false;
    }
    // Gradient evaluation requires the EvalGradient kernel which has a
    // different buffer binding layout. Compile lazily on first call.
    if (!grad_compiled_) {
      auto &reg = detail::context_pipeline_registry(*ctx_);
      extension::KernelCompileSpec compile_spec;
      compile_spec.allow_simd = false;
      compile_spec.allow_fp16 = false;
      auto result = KernelCompiler::compile(
          {KernelRole::EvalGradient, spec_,
           KernelCompiler::makeDefaultSchema(spec_), compile_spec});
      if (result.source.empty()) {
        set_diagnostic(DiagnosticCode::KernelCompilationFailed,
                       "FieldEvaluator::evaluate_with_gradient",
                       "KernelCompiler returned empty source for EvalGradient");
        return false;
      }
      PipelineKey key{result.key.hash(), result.entry_point.c_str(), false};
      grad_pipeline_ = reg.register_pipeline(
          key, result.source.c_str(), result.entry_point.c_str());
      grad_contract_ = detail::make_dispatch_contract(result);
      grad_tg_size_ = grad_contract_.geometry.tg_size;
      grad_compiled_ = true;
    }

    auto &arena = detail::context_arena(*ctx_);
    auto &pool = detail::context_batch_pool(*ctx_);
    auto &reg = detail::context_pipeline_registry(*ctx_);

    const uint32_t in_dims = n_input_dims();
    const uint32_t out_dims = n_output_dims();
    const size_t pos_bytes = static_cast<size_t>(N) * in_dims * sizeof(float);
    const size_t out_bytes = static_cast<size_t>(N) * out_dims * sizeof(float);
    const size_t grad_bytes =
        static_cast<size_t>(N) * out_dims * in_dims * sizeof(float);

    auto config_view = authority_->buffer(RuntimeBufferRole::ConfigWeights);
    auto hash_view = authority_->buffer(RuntimeBufferRole::HashWeights);
    auto mlp_view = authority_->buffer(RuntimeBufferRole::MlpWeights);
    if (!config_view.gpu_buffer) {
      set_diagnostic(DiagnosticCode::MissingRuntimeBuffer,
                     "FieldEvaluator::evaluate_with_gradient",
                     "ConfigWeights GPU buffer is unavailable");
      return false;
    }
    if (!hash_view.gpu_buffer) {
      set_diagnostic(DiagnosticCode::MissingRuntimeBuffer,
                     "FieldEvaluator::evaluate_with_gradient",
                     "HashWeights GPU buffer is unavailable");
      return false;
    }
    if (!mlp_view.gpu_buffer) {
      set_diagnostic(DiagnosticCode::MissingRuntimeBuffer,
                     "FieldEvaluator::evaluate_with_gradient",
                     "MlpWeights GPU buffer is unavailable");
      return false;
    }
    if (config_view.cpu_data &&
        config_view.bytes >= kConfigPackedFloats * sizeof(float)) {
      auto *cfg = static_cast<float *>(config_view.cpu_data);
      cfg[kConfigPackedFloats - 1] = static_cast<float>(N);
    }

    auto batch = pool.begin_batch();
    if (batch.generation == 0) {
      set_diagnostic(DiagnosticCode::BatchSubmissionUnavailable,
                     "FieldEvaluator::evaluate_with_gradient",
                     "command batch pool returned generation 0");
      return false;
    }

    BufferDesc pos_desc;
    pos_desc.bytes = pos_bytes;
    pos_desc.storage = BufferStorage::Shared;
    auto pos_handle = arena.allocate(pos_desc);
    auto pos_view = arena.view(pos_handle);

    BufferDesc out_desc;
    out_desc.bytes = out_bytes + grad_bytes; // interleaved [sdf, gx, gy, gz]
    out_desc.storage = BufferStorage::Shared;
    auto out_handle = arena.allocate(out_desc);
    auto out_view = arena.view(out_handle);

    std::memcpy(pos_view.data, positions, pos_bytes);

    auto *cmd = pool.current_command_buffer(batch);
    metal::DispatchDesc dd{};
    dd.cmd_buf = cmd;
    dd.pipeline = reg.raw_pipeline(grad_pipeline_);
    metal::DispatchDesc::BufferBind binds[5] = {
        {pos_view.gpu_buffer, static_cast<uint32_t>(pos_view.offset), 0},
        {out_view.gpu_buffer, static_cast<uint32_t>(out_view.offset), 1},
        {config_view.gpu_buffer, static_cast<uint32_t>(config_view.offset), 2},
        {hash_view.gpu_buffer, static_cast<uint32_t>(hash_view.offset), 3},
        {mlp_view.gpu_buffer, static_cast<uint32_t>(mlp_view.offset), 5},
    };
    dd.bindings = binds;
    dd.binding_count = 5;
    dd.grid_x = static_cast<uint32_t>(N);
    dd.grid_y = 1;
    dd.grid_z = 1;
    dd.tg_x = grad_tg_size_;
    dd.tg_y = 1;
    dd.tg_z = 1;
    metal::encode_dispatch(dd);

    [[maybe_unused]] const auto grad_fence =
        pool.submit(batch, SubmitMode::Sync);

    // Deinterleave [sdf, gx, gy, gz] → separate output + gradients.
    const auto *interleaved = static_cast<const float *>(out_view.data);
    const uint32_t stride = out_dims + out_dims * in_dims; // 1 + 3 for SDF
    for (int i = 0; i < N; ++i) {
      output[i] = interleaved[i * stride];
      for (uint32_t d = 0; d < out_dims * in_dims; ++d)
        gradients[i * out_dims * in_dims + d] = interleaved[i * stride + 1 + d];
    }

    arena.release(pos_handle);
    arena.release(out_handle);
    return true;
  }

  void merge_runtime_inspection(
      TrainerRuntimeInspection &inspection) const override {
    inspection.evaluate = to_public_inspection(eval_contract_);
    inspection.evaluate_with_gradient = to_public_inspection(grad_contract_);
  }

  [[nodiscard]] std::optional<DiagnosticInfo> last_diagnostic() const override {
    return last_diagnostic_;
  }

  void clear_diagnostic() override {
    last_diagnostic_.reset();
  }

private:
  void set_diagnostic(DiagnosticCode code, std::string operation,
                      std::string message) {
    last_diagnostic_ = DiagnosticInfo{
        .code = code,
        .operation = std::move(operation),
        .message = std::move(message),
    };
    emit_diagnostic(*last_diagnostic_);
  }

  std::shared_ptr<const RuntimeAuthority> authority_;
  KernelSpec spec_;
  std::shared_ptr<MetalContext> ctx_;
  PipelineHandle eval_pipeline_{};
  PipelineHandle grad_pipeline_{};
  detail::KernelDispatchContract eval_contract_{};
  detail::KernelDispatchContract grad_contract_{};
  uint32_t eval_tg_size_ = 128;
  uint32_t grad_tg_size_ = 128;
  bool eval_compiled_ = false;
  bool grad_compiled_ = false;
  std::optional<DiagnosticInfo> last_diagnostic_;
};

KernelSpec make_spec(const HashGridEncoding::Config &enc,
                     const FullyFusedMLP::Config &net) {
  KernelSpec s;
  s.input_dim = static_cast<int>(enc.num_levels * enc.features_per_level);
  s.hidden_dim = static_cast<int>(net.hidden_dim);
  s.num_hidden_layers = static_cast<int>(net.num_hidden_layers);
  s.num_outputs = static_cast<int>(net.n_output);
  s.num_levels = static_cast<int>(enc.num_levels);
  s.features_per_level = static_cast<int>(enc.features_per_level);
  s.log2_hashmap_size = static_cast<int>(enc.log2_hashmap_size);
  s.base_resolution = enc.base_resolution;
  s.per_level_scale = enc.per_level_scale;
  s.spatial_dims = static_cast<int>(enc.input_dims);
  s.encoding = (enc.input_dims == 4) ? KernelSpec::FourD : KernelSpec::Standard;
  s.validate();
  return s;
}

FullyFusedMLP::Config
resolve_network_config(const HashGridEncoding::Config &enc_cfg,
                       const FullyFusedMLP::Config &net_cfg) {
  auto resolved = net_cfg;
  resolved.n_input = enc_cfg.num_levels * enc_cfg.features_per_level;
  return resolved;
}

template <typename T>
[[nodiscard]] Result<T> make_construction_error(
    DiagnosticCode code, std::string_view operation, std::string message,
    std::vector<DiagnosticDetail> details = {}) {
  return make_error_result<T>(DiagnosticInfo{
      .code = code,
      .operation = std::string(operation),
      .message = std::move(message),
      .details = std::move(details),
  });
}

template <typename T>
[[nodiscard]] Result<T> result_from_exception(std::string_view operation,
                                              const std::exception &e) {
  const auto *invalid_arg = dynamic_cast<const std::invalid_argument *>(&e);
  return make_construction_error<T>(
      invalid_arg ? DiagnosticCode::InvalidArgument
                  : DiagnosticCode::OperationFailed,
      operation, e.what());
}

[[nodiscard]] Result<std::unique_ptr<FieldEvaluator>>
try_make_bound_evaluator(ITrainerRuntime &runtime, std::string_view operation) {
  auto authority = runtime.runtime_authority();
  if (!authority) {
    return make_construction_error<std::unique_ptr<FieldEvaluator>>(
        DiagnosticCode::MissingRuntimeAuthority, operation,
        "runtime does not expose a RuntimeAuthority");
  }

  auto ctx = authority->context();
  if (!ctx) {
    return make_construction_error<std::unique_ptr<FieldEvaluator>>(
        DiagnosticCode::MissingRuntimeContext, operation,
        "RuntimeAuthority does not expose a MetalContext");
  }
  if (!ctx->is_gpu_available()) {
    return make_construction_error<std::unique_ptr<FieldEvaluator>>(
        DiagnosticCode::GpuUnavailable, operation, "Metal GPU is unavailable");
  }

  auto layout = authority->parameter_layout();
  auto config_view = authority->buffer(RuntimeBufferRole::ConfigWeights);
  if (!config_view.cpu_data) {
    return make_construction_error<std::unique_ptr<FieldEvaluator>>(
        DiagnosticCode::MissingHostVisibleConfigWeights, operation,
        "runtime authority does not expose host-visible config weights");
  }

  try {
    const uint32_t spatial_dims = runtime.batch_plan().input_dims;
    auto spec = KernelSpec::fromConfigHeader(
        static_cast<const float *>(config_view.cpu_data), layout.target_dims,
        spatial_dims);
    return std::make_unique<DefaultEvaluator>(std::move(authority),
                                              std::move(spec), std::move(ctx));
  } catch (const std::exception &e) {
    return result_from_exception<std::unique_ptr<FieldEvaluator>>(operation, e);
  }
}

[[nodiscard]] Result<std::pair<HashGridEncoding::Config, FullyFusedMLP::Config>>
try_extract_default_runtime_configs(const NetworkWithInputEncoding &model) {
  auto hash_grid = std::dynamic_pointer_cast<HashGridEncoding>(model.encoding());
  if (!hash_grid) {
    return make_construction_error<
        std::pair<HashGridEncoding::Config, FullyFusedMLP::Config>>(
        DiagnosticCode::UnsupportedModelType, "create_trainer(model)",
        "default tmnn runtime currently requires HashGridEncoding");
  }

  auto mlp = std::dynamic_pointer_cast<FullyFusedMLP>(model.network());
  if (!mlp) {
    return make_construction_error<
        std::pair<HashGridEncoding::Config, FullyFusedMLP::Config>>(
        DiagnosticCode::UnsupportedModelType, "create_trainer(model)",
        "default tmnn runtime currently requires FullyFusedMLP");
  }

  return std::pair<HashGridEncoding::Config, FullyFusedMLP::Config>{
      hash_grid->config(), mlp->config()};
}

[[nodiscard]] Result<TrainerConfig>
try_resolve_loss_config(const TrainerConfig &train_cfg, const Loss &loss) {
  auto resolved = train_cfg;
  extension::LossConfig explicit_loss;
  if (dynamic_cast<const L2Loss *>(&loss)) {
    explicit_loss = {};
  } else if (dynamic_cast<const L1Loss *>(&loss)) {
    explicit_loss = {.kind = extension::LossKind::L1, .huber_delta = 1.0f};
  } else if (const auto *huber = dynamic_cast<const HuberLoss *>(&loss)) {
    explicit_loss = {.kind = extension::LossKind::Huber,
                     .huber_delta = huber->delta()};
  } else if (dynamic_cast<const CosineLoss *>(&loss)) {
    explicit_loss = {.kind = extension::LossKind::Cosine, .huber_delta = 1.0f};
  } else {
    return make_construction_error<TrainerConfig>(
        DiagnosticCode::UnsupportedLossType, "create_trainer(model)",
        "default tmnn runtime currently requires built-in L2Loss, L1Loss, "
        "HuberLoss, or CosineLoss");
  }

  const bool train_cfg_declares_loss =
      train_cfg.loss_kind != extension::LossKind::L2 ||
      (train_cfg.loss_kind == extension::LossKind::Huber &&
       std::fabs(train_cfg.huber_delta - 1.0f) > 1e-12f);
  if (train_cfg_declares_loss && train_cfg.loss_kind != explicit_loss.kind) {
    return make_construction_error<TrainerConfig>(
        DiagnosticCode::ConfigurationConflict, "create_trainer(model)",
        "explicit loss object must match TrainerConfig.loss_kind");
  }
  if (train_cfg_declares_loss &&
      explicit_loss.kind == extension::LossKind::Huber &&
      std::fabs(train_cfg.huber_delta - explicit_loss.huber_delta) > 1e-12f) {
    return make_construction_error<TrainerConfig>(
        DiagnosticCode::ConfigurationConflict, "create_trainer(model)",
        "explicit HuberLoss delta must match TrainerConfig.huber_delta");
  }
  resolved.loss_kind = explicit_loss.kind;
  resolved.huber_delta = explicit_loss.huber_delta;
  return resolved;
}

[[nodiscard]] Result<std::shared_ptr<Loss>>
try_create_loss_from_config(const extension::LossConfig &config) {
  switch (config.kind) {
  case extension::LossKind::L2:
    return create_loss_l2();
  case extension::LossKind::L1:
    return create_loss_l1();
  case extension::LossKind::Huber:
    try {
      return create_loss_huber(config.huber_delta);
    } catch (const std::exception &e) {
      return result_from_exception<std::shared_ptr<Loss>>(
          "create_trainer(adapter)", e);
    }
  case extension::LossKind::Cosine:
    return create_loss_cosine();
  }
  return make_construction_error<std::shared_ptr<Loss>>(
      DiagnosticCode::UnsupportedLossType, "create_trainer(adapter)",
      "unsupported loss kind");
}

struct DefaultRuntime final : ITrainerRuntime, InspectableTrainerRuntime {
  DefaultRuntime(const HashGridEncoding::Config &enc_cfg,
                 const FullyFusedMLP::Config &net_cfg,
                 const TrainerConfig &train_cfg,
                 std::shared_ptr<MetalContext> ctx)
      : ctx_(std::move(ctx)), train_cfg_(train_cfg),
        spec_(make_spec(enc_cfg, net_cfg)),
        schema_(KernelCompiler::makeDefaultSchema(spec_)) {
    if (!ctx_ || !ctx_->is_gpu_available())
      throw std::runtime_error("DefaultTrainerRuntime: Metal GPU not available");

    // Phase 3.0 cold-startup attribution: env-gated stage timing dump,
    // off by default. Set TMNN_PROFILE_COLDSTART=1 to print to stderr.
    const bool profile = std::getenv("TMNN_PROFILE_COLDSTART") != nullptr;
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    init_parameter_store();
    const auto t1 = clk::now();
    rebuild_training_kernels();
    const auto t2 = clk::now();
    init_step_lanes();
    const auto t3 = clk::now();
    if (profile) {
      auto ms = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a)
                   .count() / 1.0e6;
      };
      std::fprintf(stderr,
          "[tmnn cold-startup] DefaultRuntime ctor: "
          "init_parameter_store=%.3f ms rebuild_training_kernels=%.3f ms "
          "init_step_lanes=%.3f ms total=%.3f ms\n",
          ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t0, t3));
    }
  }

  ~DefaultRuntime() override {
    drain_pending_if_needed();
  }

  TrainingStepResult training_step(const float *input, const float *target,
                                   int N) override {
    const auto step_begin = std::chrono::steady_clock::now();
    const uint32_t batch_N = std::min<uint32_t>(
        static_cast<uint32_t>(N), batch_plan_.max_batch_size);
    auto &pool = detail::context_batch_pool(*ctx_);
    auto &reg = detail::context_pipeline_registry(*ctx_);
    detail::EnqueueTrainingStepTimings enqueue_timings{};
    detail::FinalizeTrainingStepTimings finalize_timings{};
    const bool collect_profile = training_step_profiling_enabled_;
    const uint32_t logical_step = step_;
    uint64_t morton_sort_ns = 0;

    // Morton sort for spatial coherence.
    const auto morton_begin = std::chrono::steady_clock::now();
    sorted_pos_.resize(static_cast<size_t>(batch_N) * batch_plan_.input_dims);
    sorted_tgt_.resize(static_cast<size_t>(batch_N) * batch_plan_.target_dims);
    detail::morton_sort_batch(
        input, target, static_cast<int>(batch_N),
        batch_plan_.input_dims, batch_plan_.target_dims,
        morton_indices_, morton_codes_, sorted_pos_, sorted_tgt_);
    const auto morton_end = std::chrono::steady_clock::now();
    if (collect_profile) {
      morton_sort_ns = elapsed_ns(morton_begin, morton_end);
      enqueue_timings.morton_sort_ns = morton_sort_ns;
    }

    auto fill_train = [&](uint32_t bN, uint32_t /*logical_step*/) {
      auto tp_view = param_store_->train_params();
      auto *tp = static_cast<float *>(tp_view.data);
      auto cfg_view = param_store_->config_weights();
      if (cfg_view.data && cfg_view.bytes >= kConfigPackedFloats * sizeof(float)) {
        auto *cfg = static_cast<float *>(cfg_view.data);
        cfg[kConfigPackedFloats - 1] = static_cast<float>(bN);
      }
      fill_train_params(
          tp, schema_.train_params_layout, bN, false, training_loss_scale_,
          static_cast<uint32_t>(spec_.num_levels));
    };

    auto fill_adam = [&](uint32_t logical_step) {
      auto ap_view = param_store_->adam_params();
      auto *p = static_cast<float *>(ap_view.data);
      detail::fill_unified_adam_params(p, train_cfg_, param_store_->desc(),
                                       logical_step);
    };

    try {
      if (use_sparse_hash_adam_path() && step_state_clear_required_) {
        auto mask_view = param_store_->active_hash_mask();
        if (!mask_view.data) {
          throw std::runtime_error(
              "DefaultRuntime: active-hash mask buffer must be CPU-visible");
        }
        auto summary_view = param_store_->active_hash_summary_mask();
        if (!summary_view.data) {
          throw std::runtime_error(
              "DefaultRuntime: active-hash summary buffer must be CPU-visible");
        }
        zero_buffer_view(*ctx_, mask_view);
        zero_buffer_view(*ctx_, summary_view);
      }

      // Enqueue forward/backward.
      detail::EnqueueTrainingStepRequest enqueue_req{
          "DefaultRuntime",
          pool,
          reg,
          *param_store_,
          lane_coord_,
          step_lanes_,
          pending_,
          fwd_bwd_plan_,
          kernels_,
          batch_N,
          logical_step,
          [&](StepBufferSet &lane) {
            std::memcpy(lane.positions.data, sorted_pos_.data(),
                        static_cast<size_t>(batch_N) * batch_plan_.input_dims *
                            sizeof(float));
            std::memcpy(lane.targets.data, sorted_tgt_.data(),
                        static_cast<size_t>(batch_N) * batch_plan_.target_dims *
                            sizeof(float));
          },
          fill_train,
          step_state_clear_required_,
          collect_profile ? &enqueue_timings : nullptr};
      auto new_pending = detail::enqueue_training_step(enqueue_req);

      detail::DispatchPlan *sparse_hash_plan = nullptr;
      detail::DispatchPlan *dense_mlp_plan = nullptr;
      detail::PrepareSparseHashAdamFn prepare_sparse_hash_adam;
      if (use_sparse_hash_adam_path()) {
        sparse_hash_plan = &sparse_hash_adam_plan_;
        dense_mlp_plan = &dense_mlp_adam_plan_;
        prepare_sparse_hash_adam = [&, batch_N]() {
          return compact_active_hash_indices(batch_N);
        };
      }

      // Finalize (wait, read loss, submit Adam).
      pending_ = {};
      detail::FinalizeTrainingStepRequest finalize_req{
          "DefaultRuntime",
          *ctx_,
          pool,
          reg,
          *param_store_,
          lane_coord_,
          step_lanes_,
          new_pending,
          fwd_bwd_plan_,
          adam_plan_,
          kernels_,
          resolved_recovery_mode(),
          fill_train,
          fill_adam,
          [&]() { return activate_safe_family(); },
          [&]() { return safe_family_active_; },
          nullptr,
          nullptr,
          nullptr,
          collect_profile ? &finalize_timings : nullptr,
          sparse_hash_plan,
          dense_mlp_plan,
          std::move(prepare_sparse_hash_adam)};
      auto result = detail::finalize_training_step(finalize_req);

      // Probe aggregation (host-side reduce of per-TG partials).
      uint64_t probe_aggregation_ns = 0;
      if (train_cfg_.enable_probes && new_pending.valid) {
        const auto probe_begin = std::chrono::steady_clock::now();
        const auto &lane_buf = step_lanes_[new_pending.lane];
        const uint32_t num_tgs =
            detail::threadgroup_count_for_points(
                new_pending.batch_N, training_contract_.geometry);
        result.probe = detail::aggregate_probe_partials(
            lane_buf.probe_buffer, num_tgs,
            static_cast<uint32_t>(spec_.num_hidden_layers));
        const auto probe_end = std::chrono::steady_clock::now();
        if (collect_profile) {
          probe_aggregation_ns = elapsed_ns(probe_begin, probe_end);
        }
      }

      step_state_clear_required_ =
          result.recovery_action == BadStepRecoveryAction::Skipped ||
          result.recovery_action == BadStepRecoveryAction::RolledBack;
      pending_ = {};
      ++step_;
      if (collect_profile) {
        last_training_step_profile_ = to_public_training_step_profile(
            step_, batch_N, morton_sort_ns, enqueue_timings, finalize_timings,
            probe_aggregation_ns,
            elapsed_ns(step_begin, std::chrono::steady_clock::now()));
      }
      return result;
    } catch (...) {
      step_state_clear_required_ = true;
      pending_ = {};
      throw;
    }
  }

  uint32_t forward_for_training(const float *input, float *output,
                                int N) override {
    drain_pending_if_needed();
    ensure_backward_ext_kernel(); // also compiles forward kernel
    const uint32_t batch_N =
        std::min<uint32_t>(static_cast<uint32_t>(N), batch_plan_.max_batch_size);
    auto &pool = detail::context_batch_pool(*ctx_);
    auto &reg = detail::context_pipeline_registry(*ctx_);

    // Sort positions by Morton code for spatial coherence.
    // No targets needed for forward-only, but morton_sort requires a target array.
    sorted_pos_.resize(static_cast<size_t>(batch_N) * batch_plan_.input_dims);
    std::vector<float> dummy_targets(batch_N, 0.0f);
    sorted_tgt_.resize(batch_N);
    detail::morton_sort_batch(
        input, dummy_targets.data(), static_cast<int>(batch_N),
        static_cast<int>(batch_plan_.input_dims), 1,
        morton_indices_, morton_codes_, sorted_pos_, sorted_tgt_);

    // Acquire step lane, pack positions + output buffer.
    const auto lane = lane_coord_.acquire_lane();
    auto &lane_buf = step_lanes_[lane];
    std::memcpy(lane_buf.positions.data, sorted_pos_.data(),
                static_cast<size_t>(batch_N) * batch_plan_.input_dims *
                    sizeof(float));
    std::memset(lane_buf.targets.data, 0, lane_buf.targets.bytes);
    if (train_cfg_.enable_probes && lane_buf.probe_buffer.data)
      std::memset(lane_buf.probe_buffer.data, 0, lane_buf.probe_buffer.bytes);

    // Fill train params.
    {
      auto tp_view = param_store_->train_params();
      auto *tp = static_cast<float *>(tp_view.data);
      fill_train_params(tp, schema_.train_params_layout, batch_N, false, 1.0f,
                        static_cast<uint32_t>(spec_.num_levels));
    }

    // Resolve forward plan bindings.
    detail::resolve_step_lane_bindings(forward_train_plan_, *param_store_,
                                       lane_buf);

    // Dispatch forward kernel using its own realized specialization metadata.
    const uint32_t total_threads =
        detail::total_threads_for_points(batch_N, forward_train_contract_.geometry);
    auto fwd_batch = pool.begin_batch();
    auto *cmd = pool.current_command_buffer(fwd_batch);
    metal::DispatchDesc dd{};
    dd.cmd_buf = cmd;
    dd.pipeline = reg.raw_pipeline(forward_train_plan_.pipeline);
    dd.bindings = forward_train_plan_.resolved.binds.data();
    dd.binding_count = forward_train_plan_.resolved.count;
    dd.grid_x = total_threads;
    dd.grid_y = 1;
    dd.grid_z = 1;
    dd.tg_x = forward_train_contract_.geometry.tg_size;
    dd.tg_y = 1;
    dd.tg_z = 1;
    dd.threadgroup_memory_bytes =
        forward_train_contract_.geometry.threadgroup_memory_bytes;
    metal::encode_dispatch(dd);
    auto fwd_fence = pool.submit(fwd_batch, SubmitMode::Sync);
    lane_coord_.bind_fence(lane, fwd_fence);

    // Read output from forward_output lane buffer.
    // Output is in Morton-sorted order. Unsort back to original order.
    const uint32_t out_dims = batch_plan_.target_dims;
    const uint32_t out_count = batch_N * out_dims;
    const auto *sorted_output =
        static_cast<const float *>(lane_buf.forward_output.data);

    // morton_indices_[sorted_i] = original_i.
    for (uint32_t sorted_i = 0; sorted_i < batch_N; ++sorted_i) {
      const uint32_t orig_i = morton_indices_[sorted_i];
      for (uint32_t d = 0; d < out_dims; ++d) {
        output[orig_i * out_dims + d] = sorted_output[sorted_i * out_dims + d];
      }
    }

    lane_coord_.release_lane(lane);
    return out_count;
  }

  void backward_and_update(const float *input, const float *d_output,
                           int N) override {
    drain_pending_if_needed();
    ensure_backward_ext_kernel();
    const uint32_t batch_N =
        std::min<uint32_t>(static_cast<uint32_t>(N), batch_plan_.max_batch_size);
    auto &pool = detail::context_batch_pool(*ctx_);
    auto &reg = detail::context_pipeline_registry(*ctx_);

    // Morton sort (positions + d_output together).
    sorted_pos_.resize(static_cast<size_t>(batch_N) * batch_plan_.input_dims);
    sorted_tgt_.resize(static_cast<size_t>(batch_N) * batch_plan_.target_dims);
    detail::morton_sort_batch(
        input, d_output, static_cast<int>(batch_N),
        static_cast<int>(batch_plan_.input_dims),
        static_cast<int>(batch_plan_.target_dims),
        morton_indices_, morton_codes_, sorted_pos_, sorted_tgt_);

    // Acquire step lane and pack buffers.
    const auto lane = lane_coord_.acquire_lane();
    auto &lane_buf = step_lanes_[lane];
    std::memcpy(lane_buf.positions.data, sorted_pos_.data(),
                static_cast<size_t>(batch_N) * batch_plan_.input_dims *
                    sizeof(float));
    // Zero targets (backward kernel doesn't read them but slot is bound).
    std::memset(lane_buf.targets.data, 0, lane_buf.targets.bytes);
    // Upload external gradient to step lane buffer.
    std::memcpy(lane_buf.external_gradient.data, sorted_tgt_.data(),
                static_cast<size_t>(batch_N) * batch_plan_.target_dims *
                    sizeof(float));

    // Fill train params.
    {
      auto tp_view = param_store_->train_params();
      auto *tp = static_cast<float *>(tp_view.data);
      fill_train_params(tp, schema_.train_params_layout, batch_N, false, 1.0f,
                        static_cast<uint32_t>(spec_.num_levels));
    }

    // Blit-fill gradients + probe buffer.
    {
      if (use_sparse_hash_adam_path()) {
        auto mask_view = param_store_->active_hash_mask();
        if (!mask_view.data) {
          throw std::runtime_error(
              "DefaultRuntime: active-hash mask buffer must be CPU-visible");
        }
        auto summary_view = param_store_->active_hash_summary_mask();
        if (!summary_view.data) {
          throw std::runtime_error(
              "DefaultRuntime: active-hash summary buffer must be CPU-visible");
        }
        std::memset(mask_view.data, 0, mask_view.bytes);
        std::memset(summary_view.data, 0, summary_view.bytes);
      }
      auto fill_batch = pool.begin_batch();
      auto *fill_cmd = pool.current_command_buffer(fill_batch);
      metal::encode_blit_fill(fill_cmd, param_store_->grad_hash().gpu_buffer,
                              param_store_->grad_hash().offset,
                              param_store_->grad_hash().bytes, 0);
      metal::encode_blit_fill(fill_cmd, param_store_->grad_mlp().gpu_buffer,
                              param_store_->grad_mlp().offset,
                              param_store_->grad_mlp().bytes, 0);
      if (train_cfg_.enable_probes && lane_buf.probe_buffer.bytes > 0)
        metal::encode_blit_fill(fill_cmd, lane_buf.probe_buffer.gpu_buffer,
                                lane_buf.probe_buffer.offset,
                                lane_buf.probe_buffer.bytes, 0);
      [[maybe_unused]] const auto fill_fence =
          pool.submit(fill_batch, SubmitMode::Sync);
    }

    // Resolve backward plan bindings.
    detail::resolve_step_lane_bindings(backward_ext_plan_, *param_store_,
                                       lane_buf);

    // Dispatch backward kernel (scalar — uses its own TG params, not SIMD fwd_bwd).
    {
      const uint32_t total_threads =
          detail::total_threads_for_points(batch_N,
                                           backward_ext_contract_.geometry);
      auto bwd_batch = pool.begin_batch();
      auto *cmd = pool.current_command_buffer(bwd_batch);
      metal::DispatchDesc dd{};
      dd.cmd_buf = cmd;
      dd.pipeline = reg.raw_pipeline(backward_ext_plan_.pipeline);
      dd.bindings = backward_ext_plan_.resolved.binds.data();
      dd.binding_count = backward_ext_plan_.resolved.count;
      dd.grid_x = total_threads;
      dd.grid_y = 1;
      dd.grid_z = 1;
      dd.tg_x = backward_ext_contract_.geometry.tg_size;
      dd.tg_y = 1;
      dd.tg_z = 1;
      dd.threadgroup_memory_bytes =
          backward_ext_contract_.geometry.threadgroup_memory_bytes;
      metal::encode_dispatch(dd);
      auto bwd_fence = pool.submit(bwd_batch, SubmitMode::Sync);
      lane_coord_.bind_fence(lane, bwd_fence);
    }

    // Fill Adam params and dispatch.
    {
      auto ap_view = param_store_->adam_params();
      auto *p = static_cast<float *>(ap_view.data);
      detail::fill_unified_adam_params(p, train_cfg_, param_store_->desc(),
                                       step_);
    }
    {
      if (use_sparse_hash_adam_path()) {
        const uint32_t active_hash_count = compact_active_hash_indices(batch_N);
        [[maybe_unused]] const auto adam_fence = detail::submit_split_adam_batch(
            pool, reg, *param_store_, sparse_hash_adam_plan_, active_hash_count,
            dense_mlp_adam_plan_,
            static_cast<uint32_t>(param_store_->desc().mlp_weight_count),
            SubmitMode::Sync, "DefaultRuntime");
      } else {
        const uint32_t wc = detail::adam_total_param_count(param_store_->desc());
        auto adam_batch = pool.begin_batch();
        auto *adam_cmd = pool.current_command_buffer(adam_batch);
        metal::DispatchDesc adam_dd{};
        adam_dd.cmd_buf = adam_cmd;
        adam_dd.pipeline = reg.raw_pipeline(adam_plan_.pipeline);
        adam_dd.bindings = adam_plan_.resolved.binds.data();
        adam_dd.binding_count = adam_plan_.resolved.count;
        adam_dd.grid_x = wc;
        adam_dd.grid_y = 1;
        adam_dd.grid_z = 1;
        adam_dd.tg_x = adam_plan_.tg_x;
        adam_dd.tg_y = 1;
        adam_dd.tg_z = 1;
        metal::encode_dispatch(adam_dd);
        [[maybe_unused]] const auto adam_fence =
            pool.submit(adam_batch, SubmitMode::Sync);
      }
    }

    // Post-Adam: zero gradients for the legacy fused path (single-shot
    // backward_and_update is self-contained — callers expect clean state).
    // The accumulation API uses zero_gradients() explicitly instead.
    {
      auto post_batch = pool.begin_batch();
      auto *post_cmd = pool.current_command_buffer(post_batch);
      metal::encode_blit_fill(post_cmd, param_store_->grad_hash().gpu_buffer,
                              param_store_->grad_hash().offset,
                              param_store_->grad_hash().bytes, 0);
      metal::encode_blit_fill(post_cmd, param_store_->grad_mlp().gpu_buffer,
                              param_store_->grad_mlp().offset,
                              param_store_->grad_mlp().bytes, 0);
      [[maybe_unused]] const auto post_fence =
          pool.submit(post_batch, SubmitMode::Sync);
    }

    // Aggregate probes from backward kernel (same buffer, same layout).
    if (train_cfg_.enable_probes && lane_buf.probe_buffer.data) {
      const uint32_t num_tgs =
          detail::threadgroup_count_for_points(
              batch_N, backward_ext_contract_.geometry);
      last_split_probe_ = detail::aggregate_probe_partials(
          lane_buf.probe_buffer, num_tgs,
          static_cast<uint32_t>(spec_.num_hidden_layers));
    } else {
      last_split_probe_.reset();
    }

    lane_coord_.release_lane(lane);
    step_state_clear_required_ = false;
    ++step_;
  }

  // ── Multi-frame accumulation API ─────────────────────────────────

  void zero_gradients() override {
    drain_pending_if_needed();
    ensure_backward_ext_kernel();
    auto &pool = detail::context_batch_pool(*ctx_);

    // Clear sparse hash mask (if applicable).
    if (use_sparse_hash_adam_path()) {
      auto mask_view = param_store_->active_hash_mask();
      if (mask_view.data) std::memset(mask_view.data, 0, mask_view.bytes);
      auto summary_view = param_store_->active_hash_summary_mask();
      if (summary_view.data) std::memset(summary_view.data, 0, summary_view.bytes);
    }

    // GPU blit-fill gradient buffers to zero.
    auto fill_batch = pool.begin_batch();
    auto *fill_cmd = pool.current_command_buffer(fill_batch);
    metal::encode_blit_fill(fill_cmd, param_store_->grad_hash().gpu_buffer,
                            param_store_->grad_hash().offset,
                            param_store_->grad_hash().bytes, 0);
    metal::encode_blit_fill(fill_cmd, param_store_->grad_mlp().gpu_buffer,
                            param_store_->grad_mlp().offset,
                            param_store_->grad_mlp().bytes, 0);
    [[maybe_unused]] const auto fill_fence =
        pool.submit(fill_batch, SubmitMode::Sync);
  }

  void backward_accumulate(const float *input, const float *d_output,
                           int N) override {
    drain_pending_if_needed();
    ensure_backward_ext_kernel();
    const uint32_t batch_N =
        std::min<uint32_t>(static_cast<uint32_t>(N), batch_plan_.max_batch_size);
    auto &pool = detail::context_batch_pool(*ctx_);
    auto &reg = detail::context_pipeline_registry(*ctx_);

    // Morton sort (positions + d_output together).
    sorted_pos_.resize(static_cast<size_t>(batch_N) * batch_plan_.input_dims);
    sorted_tgt_.resize(static_cast<size_t>(batch_N) * batch_plan_.target_dims);
    detail::morton_sort_batch(
        input, d_output, static_cast<int>(batch_N),
        static_cast<int>(batch_plan_.input_dims),
        static_cast<int>(batch_plan_.target_dims),
        morton_indices_, morton_codes_, sorted_pos_, sorted_tgt_);

    // Acquire step lane and pack buffers.
    const auto lane = lane_coord_.acquire_lane();
    auto &lane_buf = step_lanes_[lane];
    std::memcpy(lane_buf.positions.data, sorted_pos_.data(),
                static_cast<size_t>(batch_N) * batch_plan_.input_dims *
                    sizeof(float));
    std::memset(lane_buf.targets.data, 0, lane_buf.targets.bytes);
    std::memcpy(lane_buf.external_gradient.data, sorted_tgt_.data(),
                static_cast<size_t>(batch_N) * batch_plan_.target_dims *
                    sizeof(float));

    // Fill train params.
    {
      auto tp_view = param_store_->train_params();
      auto *tp = static_cast<float *>(tp_view.data);
      fill_train_params(tp, schema_.train_params_layout, batch_N, false, 1.0f,
                        static_cast<uint32_t>(spec_.num_levels));
    }

    // NOTE: Do NOT zero gradients here — they accumulate across calls.
    // Resolve backward plan bindings.
    detail::resolve_step_lane_bindings(backward_ext_plan_, *param_store_,
                                       lane_buf);

    // Dispatch backward kernel (gradients accumulate via atomic_add).
    {
      const uint32_t total_threads =
          detail::total_threads_for_points(batch_N,
                                           backward_ext_contract_.geometry);
      auto bwd_batch = pool.begin_batch();
      auto *cmd = pool.current_command_buffer(bwd_batch);
      metal::DispatchDesc dd{};
      dd.cmd_buf = cmd;
      dd.pipeline = reg.raw_pipeline(backward_ext_plan_.pipeline);
      dd.bindings = backward_ext_plan_.resolved.binds.data();
      dd.binding_count = backward_ext_plan_.resolved.count;
      dd.grid_x = total_threads;
      dd.grid_y = 1;
      dd.grid_z = 1;
      dd.tg_x = backward_ext_contract_.geometry.tg_size;
      dd.tg_y = 1;
      dd.tg_z = 1;
      dd.threadgroup_memory_bytes =
          backward_ext_contract_.geometry.threadgroup_memory_bytes;
      metal::encode_dispatch(dd);
      auto bwd_fence = pool.submit(bwd_batch, SubmitMode::Sync);
      lane_coord_.bind_fence(lane, bwd_fence);
    }

    lane_coord_.release_lane(lane);
  }

  void adam_step() override {
    drain_pending_if_needed();
    auto &pool = detail::context_batch_pool(*ctx_);
    auto &reg = detail::context_pipeline_registry(*ctx_);

    // Fill Adam params.
    {
      auto ap_view = param_store_->adam_params();
      auto *p = static_cast<float *>(ap_view.data);
      detail::fill_unified_adam_params(p, train_cfg_, param_store_->desc(),
                                       step_);
    }

    // Dispatch Adam optimizer.
    // For multi-frame accumulation, the active hash mask may exceed sparse
    // capacity. try_compact returns nullopt on overflow → fall back to unified.
    auto dispatch_unified_adam = [&] {
      const uint32_t wc = detail::adam_total_param_count(param_store_->desc());
      auto adam_batch = pool.begin_batch();
      auto *adam_cmd = pool.current_command_buffer(adam_batch);
      metal::DispatchDesc adam_dd{};
      adam_dd.cmd_buf = adam_cmd;
      adam_dd.pipeline = reg.raw_pipeline(adam_plan_.pipeline);
      adam_dd.bindings = adam_plan_.resolved.binds.data();
      adam_dd.binding_count = adam_plan_.resolved.count;
      adam_dd.grid_x = wc;
      adam_dd.grid_y = 1;
      adam_dd.grid_z = 1;
      adam_dd.tg_x = adam_plan_.tg_x;
      adam_dd.tg_y = 1;
      adam_dd.tg_z = 1;
      metal::encode_dispatch(adam_dd);
      [[maybe_unused]] const auto adam_fence =
          pool.submit(adam_batch, SubmitMode::Sync);
    };

    if (auto active = try_compact_active_hash_indices(); active.has_value()) {
      detail::submit_split_adam_batch(
          pool, reg, *param_store_, sparse_hash_adam_plan_, *active,
          dense_mlp_adam_plan_,
          static_cast<uint32_t>(param_store_->desc().mlp_weight_count),
          SubmitMode::Sync, "DefaultRuntime");
    } else {
      dispatch_unified_adam();
    }

    ++step_;
  }

  void ensure_backward_ext_kernel() {
    if (backward_ext_compiled_)
      return;
    auto &reg = detail::context_pipeline_registry(*ctx_);

    extension::KernelCompileSpec backward_compile_spec;
    backward_compile_spec.allow_simd = false;
    backward_compile_spec.allow_fp16 = false;
    backward_compile_spec.allow_tg_weight_cache = true;
    backward_compile_spec.enable_probes = train_cfg_.enable_probes;

    auto result = KernelCompiler::compile({
        KernelRole::BackwardFromExternalGrad, spec_, schema_,
        backward_compile_spec});
    PipelineKey key{result.key.hash(), result.entry_point.c_str(), false};
    kernels_.backward_ext = reg.register_pipeline(
        key, result.source.c_str(), result.entry_point.c_str());

    backward_ext_contract_ = detail::make_dispatch_contract(result);
    const auto backward_geometry = backward_ext_contract_.geometry;
    kernels_.backward_ext_tg_size = backward_geometry.tg_size;
    kernels_.backward_ext_pts_per_tg = backward_geometry.pts_per_tg;
    kernels_.backward_ext_tg_memory_bytes =
        backward_geometry.threadgroup_memory_bytes;

    // Build dispatch plan with slot 8 for external gradient and slot 9 for
    // active-hash mask.
    backward_ext_plan_ = detail::make_backward_ext_plan(
        kernels_, *param_store_, train_cfg_.enable_probes);
    verify_plan_matches_contract(backward_ext_plan_, backward_ext_contract_,
                                 "BackwardFromExternalGrad");

    auto forward_compile_spec = backward_compile_spec;
    auto fwd_result = KernelCompiler::compile({
        KernelRole::ForwardForTraining, spec_, schema_, forward_compile_spec});
    PipelineKey fwd_key{fwd_result.key.hash(), fwd_result.entry_point.c_str(),
                        false};
    kernels_.forward_train = reg.register_pipeline(
        fwd_key, fwd_result.source.c_str(), fwd_result.entry_point.c_str());
    forward_train_contract_ = detail::make_dispatch_contract(fwd_result);
    const auto forward_geometry = forward_train_contract_.geometry;
    kernels_.forward_train_tg_size = forward_geometry.tg_size;
    kernels_.forward_train_pts_per_tg = forward_geometry.pts_per_tg;
    kernels_.forward_train_tg_memory_bytes =
        forward_geometry.threadgroup_memory_bytes;
    forward_train_plan_ =
        detail::make_forward_training_plan(kernels_, *param_store_,
                                           fwd_result.resolved_spec.emit_probes);
    verify_plan_matches_contract(forward_train_plan_, forward_train_contract_,
                                 "ForwardForTraining");

    // Ensure step lanes have external_gradient + forward_output buffers.
    if (!step_lanes_.empty() && step_lanes_[0].external_gradient.bytes == 0) {
      auto &arena = detail::context_arena(*ctx_);
      const size_t buf_bytes = static_cast<size_t>(batch_plan_.max_batch_size) *
                               batch_plan_.target_dims * sizeof(float);
      for (auto &lane : step_lanes_) {
        BufferDesc eg_desc{buf_bytes, 256, BufferStorage::Shared,
                           BufferLifetime::Transient, "step_external_gradient"};
        lane.external_gradient = arena.view(arena.allocate(eg_desc));
        BufferDesc fo_desc{buf_bytes, 256, BufferStorage::Shared,
                           BufferLifetime::Transient, "step_forward_output"};
        lane.forward_output = arena.view(arena.allocate(fo_desc));
      }
    }

    backward_ext_compiled_ = true;
  }

  [[nodiscard]] std::optional<ProbeResult>
  read_last_split_probe() const override {
    return last_split_probe_;
  }

  void sync_weights() override {
    drain_pending_if_needed();
  }

  [[nodiscard]] uint32_t step() const override { return step_; }
  [[nodiscard]] bool is_gpu_available() const override {
    return ctx_ && ctx_->is_gpu_available();
  }
  [[nodiscard]] std::shared_ptr<const RuntimeAuthority>
  runtime_authority() const override {
    return authority_;
  }
  [[nodiscard]] TrainerBatchPlan batch_plan() const override {
    return batch_plan_;
  }
  [[nodiscard]] TrainerRuntimeInspection inspect_runtime() const override {
    TrainerRuntimeInspection inspection;
    inspection.training_step = to_public_inspection(training_contract_);
    inspection.forward_for_training = to_public_inspection(forward_train_contract_);
    inspection.backward_from_output = to_public_inspection(backward_ext_contract_);
    inspection.batch_size = batch_plan_.max_batch_size;
    inspection.safe_family_active = safe_family_active_;
    return inspection;
  }
  void
  set_training_step_profiling(const TrainingStepProfilingOptions &options)
      override {
    training_step_profiling_enabled_ = options.enabled;
    if (!training_step_profiling_enabled_) {
      last_training_step_profile_.reset();
    }
  }
  [[nodiscard]] std::optional<TrainingStepProfile>
  last_training_step_profile() const override {
    return last_training_step_profile_;
  }
  [[nodiscard]] OptimizerStateBlob export_optimizer_state() override {
    OptimizerStateBlob blob;
    blob.version = kOptimizerStateBlobVersion;
    blob.step = step_;

    drain_pending_if_needed();

    if (!param_store_)
      return blob;

    auto &ps = *param_store_;
    const std::pair<BufferView, const char *> sections[] = {
        {ps.adam_m_hash(), "adam_m_hash"},
        {ps.adam_v_hash(), "adam_v_hash"},
        {ps.adam_m_mlp(),  "adam_m_mlp"},
        {ps.adam_v_mlp(),  "adam_v_mlp"},
    };

    // Phase 4 followup: collect Private downloads into one batch so we
    // pay one commit_and_wait round-trip for the optimizer-state export
    // instead of one per Private section.
    //
    // INVARIANT: private_bufs.reserve(N) + at-most-N emplace_back keeps
    // the outer vector from reallocating, so the inner vectors stay at
    // fixed addresses and the data() pointers captured below remain
    // valid until context_blit_download_views completes. Each section
    // contributes at most one emplace_back, so std::size(sections) is
    // a tight bound and the invariant holds by construction.
    std::vector<std::vector<uint8_t>> private_bufs;
    std::vector<detail::BlitDownloadRequest> downloads;
    private_bufs.reserve(std::size(sections));
    downloads.reserve(std::size(sections));
    for (const auto &[view, label] : sections) {
      if (view.bytes > 0 && !view.data && view.gpu_buffer) {
        private_bufs.emplace_back(view.bytes);
        downloads.push_back({view, private_bufs.back().data(), view.bytes});
      } else if (view.bytes > 0 && !view.data && !view.gpu_buffer) {
        throw std::runtime_error(std::string("DefaultRuntime: ") + label +
                                 " is not backed by a GPU buffer");
      }
    }
    detail::context_blit_download_views(*ctx_, downloads);

    // Now serialize sections in order — Shared straight from cpu_data,
    // Private from the just-downloaded staging copies.
    size_t priv_idx = 0;
    for (const auto &[view, label] : sections) {
      std::vector<uint8_t> bytes;
      if (view.bytes == 0) {
        // empty section — append_section writes a zero-length record
      } else if (view.data) {
        auto *p = static_cast<const uint8_t *>(view.data);
        bytes.assign(p, p + view.bytes);
      } else {
        bytes = std::move(private_bufs[priv_idx++]);
      }
      append_section(blob.payload, bytes);
    }
    return blob;
  }
  void import_optimizer_state(const OptimizerStateBlob &state) override {
    if (state.version != kOptimizerStateBlobVersion) {
      throw std::runtime_error(
          "DefaultRuntime: unsupported optimizer blob version");
    }

    drain_pending_if_needed();

    if (!param_store_) {
      step_ = state.step;
      return;
    }

    auto &ps = *param_store_;
    const std::pair<BufferView, const char *> sections[] = {
        {ps.adam_m_hash(), "adam_m_hash"},
        {ps.adam_v_hash(), "adam_v_hash"},
        {ps.adam_m_mlp(),  "adam_m_mlp"},
        {ps.adam_v_mlp(),  "adam_v_mlp"},
    };

    // Read all sections first, then batch the Private uploads.
    //
    // INVARIANT: section_bytes.reserve(N) + exactly N emplace_back keeps
    // the outer vector from reallocating; each inner vector's data()
    // pointer remains valid until context_blit_upload_views completes.
    const uint8_t *cursor = state.payload.data();
    size_t remaining = state.payload.size();
    std::vector<std::vector<uint8_t>> section_bytes;
    section_bytes.reserve(std::size(sections));
    for (size_t i = 0; i < std::size(sections); ++i) {
      section_bytes.emplace_back(read_section(cursor, remaining));
    }
    if (remaining != 0) {
      throw std::runtime_error(
          "DefaultRuntime: optimizer blob has trailing bytes");
    }

    // Phase 4 followup: route Private sections through one batched
    // upload (one commit_and_wait), Shared sections through CPU memcpy.
    std::vector<detail::BlitUploadRequest> uploads;
    uploads.reserve(std::size(sections));
    for (size_t i = 0; i < std::size(sections); ++i) {
      const auto &[view, label] = sections[i];
      const auto &bytes = section_bytes[i];
      if (view.bytes == 0) continue;
      if (bytes.size() != view.bytes) {
        throw std::runtime_error(std::string("DefaultRuntime: ") + label +
                                 " size mismatch");
      }
      if (view.data) {
        std::memcpy(view.data, bytes.data(), bytes.size());
      } else if (view.gpu_buffer) {
        uploads.push_back({view, bytes.data(), bytes.size()});
      } else {
        throw std::runtime_error(std::string("DefaultRuntime: ") + label +
                                 " is not backed by a GPU buffer");
      }
    }
    detail::context_blit_upload_views(*ctx_, uploads);

    clear_training_step_state();
    step_ = state.step;
  }
  void reset_optimizer() override {
    drain_pending_if_needed();

    param_store_->reset_adam_state();
    // Phase 4: batched blit-fill for the Private Adam buffers (Shared
    // are already zeroed by reset_adam_state's CPU memset).
    {
      const BufferView adam_zero_views[] = {
          param_store_->fused_m(),     param_store_->fused_v(),
          param_store_->adam_m_hash(), param_store_->adam_v_hash(),
          param_store_->adam_m_mlp(),  param_store_->adam_v_mlp(),
      };
      BufferView gpu_only[6];
      size_t n = 0;
      for (const auto &v : adam_zero_views) {
        if (!v.data && v.gpu_buffer && v.bytes > 0) gpu_only[n++] = v;
      }
      detail::context_blit_fill_views(*ctx_,
                                      std::span<const BufferView>(gpu_only, n),
                                      0);
    }
    clear_training_step_state();
    step_ = 0;
  }
  void apply_optimizer_config(const Optimizer &opt) override {
    train_cfg_.lr_encoding = opt.learning_rate();
    train_cfg_.lr_network = opt.learning_rate();
  }

private:
  [[nodiscard]] bool has_pending_work() const {
    return pending_.valid;
  }

  void drain_pending_if_needed() {
    if (!param_store_ || !has_pending_work()) {
      return;
    }
    auto &pool = detail::context_batch_pool(*ctx_);
    detail::drain_pending_training_step(pool, *param_store_, lane_coord_,
                                        step_lanes_, pending_);
  }

  void clear_training_step_state() {
    if (!ctx_ || !param_store_) {
      step_state_clear_required_ = false;
      return;
    }
    // Phase 4 followup: Shared views go through CPU memset; Private
    // views go into ONE batched blit-fill (was up to four sequential
    // commit_and_wait round-trips, one per Private grad buffer).
    const BufferView views[] = {
        param_store_->grad_hash(),
        param_store_->grad_mlp(),
        param_store_->active_hash_mask(),
        param_store_->active_hash_summary_mask(),
    };
    BufferView gpu_only[4];
    size_t n = 0;
    for (const auto &v : views) {
      if (v.bytes == 0) continue;
      if (v.data) {
        std::memset(v.data, 0, v.bytes);  // Shared: CPU memset
      } else if (v.gpu_buffer) {
        gpu_only[n++] = v;                // Private: defer to batched fill
      }
    }
    detail::context_blit_fill_views(*ctx_,
                                    std::span<const BufferView>(gpu_only, n),
                                    0);
    step_state_clear_required_ = false;
  }

  [[nodiscard]] extension::KernelCompileSpec
  make_training_compile_spec() const {
    extension::KernelCompileSpec compile_spec;
    const bool simd_requested = !safe_family_active_ && spec_.canUseSIMD();
    compile_spec.allow_simd = simd_requested;
    compile_spec.allow_fp16 =
        simd_requested && ctx_->capabilities().supports_fp16;
    compile_spec.allow_tg_weight_cache = !safe_family_active_;
    compile_spec.loss_kind = train_cfg_.loss_kind;
    compile_spec.huber_delta = train_cfg_.huber_delta;
    compile_spec.enable_probes = train_cfg_.enable_probes;
    return compile_spec;
  }

  [[nodiscard]] BadStepRecoveryMode resolved_recovery_mode() const {
    return detail::resolve_default_trainer_recovery_mode(ctx_->policy());
  }

  bool activate_safe_family() {
    if (safe_family_active_)
      return false;
    safe_family_active_ = true;
    rebuild_training_kernels();
    return true;
  }

  [[nodiscard]] bool use_sparse_hash_adam_path() const {
    return param_store_ != nullptr &&
           train_cfg_.l1_reg == 0.0f &&
           param_store_->desc().active_hash_mask_words != 0 &&
           param_store_->desc().active_hash_summary_words != 0 &&
           param_store_->active_hash_mask().data != nullptr &&
           param_store_->active_hash_summary_mask().data != nullptr &&
           param_store_->active_hash_indices().bytes != 0;
  }

  [[nodiscard]] uint32_t sparse_hash_entry_width() const {
    return spec_.features_per_level == 2 ? 2u : 1u;
  }

  uint32_t compact_active_hash_indices(uint32_t batch_N) {
    if (!use_sparse_hash_adam_path()) {
      return 0;
    }

    (void)batch_N;

    const auto index_view = param_store_->active_hash_indices();
    auto *indices = static_cast<uint32_t *>(index_view.data);
    if (!indices) {
      throw std::runtime_error(
          "DefaultRuntime: active-hash index buffer must be CPU-visible");
    }

    const auto mask_view = param_store_->active_hash_mask();
    auto *mask_words = static_cast<uint32_t *>(mask_view.data);
    if (!mask_words) {
      throw std::runtime_error(
          "DefaultRuntime: active-hash mask buffer must be CPU-visible");
    }

    const auto summary_view = param_store_->active_hash_summary_mask();
    auto *summary_words = static_cast<uint32_t *>(summary_view.data);
    if (!summary_words) {
      throw std::runtime_error(
          "DefaultRuntime: active-hash summary buffer must be CPU-visible");
    }

    const uint32_t mask_word_count = param_store_->desc().active_hash_mask_words;
    const uint32_t summary_word_count =
        param_store_->desc().active_hash_summary_words;
    const uint32_t capacity = param_store_->desc().active_hash_index_capacity;
    const uint32_t hash_grid_size = param_store_->desc().hash_grid_size;
    const uint32_t entry_width = sparse_hash_entry_width();
    const uint32_t tracked_slot_count = hash_grid_size / entry_width;
    const uint32_t child_word_count = (mask_word_count + 1u) / 2u;
    uint32_t count = 0;
    for (uint32_t summary_word_idx = 0; summary_word_idx < summary_word_count;
         ++summary_word_idx) {
      uint32_t child_bits = summary_words[summary_word_idx];
      summary_words[summary_word_idx] = 0u;
      while (child_bits != 0u) {
        const uint32_t child_bit =
            static_cast<uint32_t>(std::countr_zero(child_bits));
        const uint32_t child_word_idx = summary_word_idx * 32u + child_bit;
        child_bits &= (child_bits - 1u);
        if (child_word_idx >= child_word_count) {
          continue;
        }

        const uint32_t lo_word_idx = child_word_idx * 2u;
        const uint32_t hi_word_idx = lo_word_idx + 1u;
        const uint32_t lo_bits = mask_words[lo_word_idx];
        mask_words[lo_word_idx] = 0u;
        uint32_t hi_bits = 0u;
        if (hi_word_idx < mask_word_count) {
          hi_bits = mask_words[hi_word_idx];
          mask_words[hi_word_idx] = 0u;
        }

        uint64_t bits = static_cast<uint64_t>(lo_bits) |
                        (static_cast<uint64_t>(hi_bits) << 32u);
        const uint32_t slot_base = child_word_idx * 64u;
        while (bits != 0u) {
          const uint32_t bit = static_cast<uint32_t>(std::countr_zero(bits));
          const uint32_t slot = slot_base + bit;
          if (slot < tracked_slot_count) {
            if (count >= capacity) {
              throw std::runtime_error(
                  "DefaultRuntime: active-hash index capacity exhausted");
            }
            indices[count++] = slot * entry_width;
          }
          bits &= (bits - 1u);
        }
      }
    }
    return count;
  }

  /// Non-throwing variant: returns nullopt on capacity overflow.
  std::optional<uint32_t> try_compact_active_hash_indices() {
    if (!use_sparse_hash_adam_path()) return std::nullopt;
    try {
      return compact_active_hash_indices(batch_plan_.max_batch_size);
    } catch (const std::runtime_error&) {
      return std::nullopt;
    }
  }

  // Phase 5: dispatch GPU init kernels for the hash-grid table and the
  // MLP weight tensor. Caller-controllable via TrainerConfig.weight_init.
  // Both dispatches share ONE command buffer + ONE commit_and_wait via
  // context_dispatch_init_uniform_views; the two dispatches use distinct
  // counter_base offsets so they draw from non-overlapping streams of
  // the same Philox seed.
  void init_weights_with_config(const WeightInitConfig &wcfg) {
    const auto hash_view = param_store_->hash_weights();
    const auto mlp_view  = param_store_->mlp_weights();
    const std::uint32_t hash_count =
        static_cast<std::uint32_t>(hash_view.bytes / sizeof(float));
    const std::uint32_t mlp_count =
        static_cast<std::uint32_t>(mlp_view.bytes / sizeof(float));

    // Build one batch of init dispatches + a list of buffers needing
    // zero-fill (Zero mode goes through context_blit_fill_views).
    std::vector<detail::InitUniformRequest> init_reqs;
    std::vector<BufferView> zero_views;

    // Hash grid.
    if (hash_count > 0 && hash_view.gpu_buffer) {
      switch (wcfg.hash_grid_mode) {
        case HashGridInit::Uniform: {
          const float r = wcfg.hash_grid_range;
          init_reqs.push_back({hash_view, hash_count, -r, r,
                                wcfg.seed, /*counter_base=*/0u});
          break;
        }
        case HashGridInit::Zero:
          zero_views.push_back(hash_view);
          break;
      }
    }

    // MLP weight init. counter_base offsets so hash + mlp draw from
    // disjoint streams of the same Philox seed (each thread emits 4
    // outputs, so advance by ceil(hash_count/4) slots).
    const std::uint32_t mlp_counter_base =
        static_cast<std::uint32_t>((hash_count + 3u) / 4u);
    if (mlp_count > 0 && mlp_view.gpu_buffer) {
      // For Kaiming/Xavier we approximate fan_in / fan_out as hidden_dim
      // — most layers in a FullyFusedMLP are hidden_dim × hidden_dim, so
      // the slight under-init at the first/last boundary stays inside
      // the typical scaling tolerance. Per-layer correct init is a
      // future refinement.
      const float fan_in  = static_cast<float>(spec_.hidden_dim);
      const float fan_out = static_cast<float>(spec_.hidden_dim);
      const float a       = wcfg.mlp_kaiming_a;

      const auto add_uniform = [&](float low, float high) {
        init_reqs.push_back({mlp_view, mlp_count, low, high,
                              wcfg.seed, mlp_counter_base});
      };
      switch (wcfg.mlp_mode) {
        case MlpInit::KaimingUniform: {
          const float b = std::sqrt(6.0f / ((1.0f + a * a) * fan_in));
          add_uniform(-b, b); break;
        }
        case MlpInit::XavierUniform: {
          const float b = std::sqrt(6.0f / (fan_in + fan_out));
          add_uniform(-b, b); break;
        }
        case MlpInit::Uniform:
          add_uniform(-wcfg.mlp_uniform_range, wcfg.mlp_uniform_range);
          break;
        case MlpInit::Zero:
          zero_views.push_back(mlp_view); break;
        // P5.3: KaimingNormal / XavierNormal / Normal ride a future
        // Box-Muller variant of the Philox kernel. For now fall back
        // to KaimingUniform-equivalent so configs that selected those
        // modes still receive a sensible (if not exact) initializer.
        case MlpInit::KaimingNormal:
        case MlpInit::XavierNormal:
        case MlpInit::Normal: {
          const float b = std::sqrt(6.0f / ((1.0f + a * a) * fan_in));
          add_uniform(-b, b); break;
        }
      }
    }

    if (!init_reqs.empty()) {
      // wait_for_completion = false: the downstream Adam-zero
      // context_blit_fill_views in init_parameter_store and the first
      // training_step's GPU work naturally sequence after these
      // writes via Metal's in-order command queue, so blocking the
      // host here only adds queue-wait latency (especially under
      // shared-GPU load).
      detail::context_dispatch_init_uniform_views(
          *ctx_,
          std::span<const detail::InitUniformRequest>(
              init_reqs.data(), init_reqs.size()),
          /*wait_for_completion=*/false);
    }
    if (!zero_views.empty()) {
      detail::context_blit_fill_views(
          *ctx_, std::span<const BufferView>(zero_views.data(),
                                              zero_views.size()),
          0);
    }
  }

  void init_parameter_store() {
    auto &arena = detail::context_arena(*ctx_);

    ParameterStoreDesc ps_desc;
    ps_desc.hash_grid_size =
        static_cast<uint32_t>(spec_.num_levels) *
        static_cast<uint32_t>(1u << spec_.log2_hashmap_size) *
        static_cast<uint32_t>(spec_.features_per_level);
    ps_desc.mlp_weight_count = static_cast<uint32_t>(spec_.mlpWeightCount());
    ps_desc.use_private_buffers =
        train_cfg_.use_private_buffers && ctx_ && ctx_->is_gpu_available();
    ps_desc.use_fused_adam = true;
    ps_desc.target_dims = static_cast<uint32_t>(spec_.num_outputs);
    ps_desc.reduction_terms = schema_.reduction_terms;
    ps_desc.train_params_layout = schema_.train_params_layout;
    ps_desc.adam_params_float_count = 13;
    const uint32_t sparse_entry_width = sparse_hash_entry_width();
    const uint32_t tracked_slot_count =
        ps_desc.hash_grid_size / sparse_entry_width;
    ps_desc.active_hash_mask_words = (tracked_slot_count + 31u) / 32u;
    ps_desc.active_hash_summary_words = (tracked_slot_count + 2047u) / 2048u;
    const uint32_t batch_size = std::min<uint32_t>(
        static_cast<uint32_t>(train_cfg_.batch_size), 65536u);
    const uint32_t hash_corner_count =
        1u << std::min<uint32_t>(static_cast<uint32_t>(spec_.spatial_dims), 4u);
    const uint64_t active_touch_upper_bound =
        static_cast<uint64_t>(batch_size) *
        static_cast<uint64_t>(spec_.num_levels) *
        static_cast<uint64_t>(hash_corner_count) *
        static_cast<uint64_t>(spec_.features_per_level / sparse_entry_width);
    ps_desc.active_hash_index_capacity = static_cast<uint32_t>(std::min<uint64_t>(
        static_cast<uint64_t>(tracked_slot_count),
        active_touch_upper_bound));
    param_store_ = std::make_shared<ParameterStore>(ps_desc, arena);

    // Phase 5: GPU weight init via Philox-4x32-10. Replaces a single-thread
    // CPU std::mt19937 fill of `hash_grid_size + mlp_weight_count` floats
    // (~240 ms for default HashGridEncoding log2_hashmap=19) with one
    // batched GPU compute dispatch (~5-15 ms cold, sub-ms warm; GPU-
    // memory-bandwidth bound). On smaller hash configs the speedup is
    // noise; on default-encoding it's a clear order-of-magnitude win.
    init_weights_with_config(train_cfg_.weight_init);

    // Config header (8 floats describing hash grid + MLP geometry) is
    // tiny CPU-side data; route through hydrate_weights with null hash
    // and mlp pointers so it only writes the 8-float header.
    const float config_header[8] = {
        static_cast<float>(spec_.num_levels),
        static_cast<float>(spec_.features_per_level),
        static_cast<float>(spec_.log2_hashmap_size),
        spec_.base_resolution,
        spec_.per_level_scale,
        static_cast<float>(spec_.hidden_dim),
        static_cast<float>(spec_.num_hidden_layers),
        0.0f};
    param_store_->hydrate_weights(nullptr, 0, nullptr, 0, config_header);

    // Zero Adam optimizer state (m/v buffers may contain garbage from
    // arena reuse). Phase 4: replaces six sequential commit_and_wait
    // GPU sync round-trips with one batched blit-fill — Shared views
    // are skipped at the helper boundary (their reset_adam_state CPU
    // memset above already zeroed them), so the batch only does GPU
    // work for Private buffers.
    param_store_->reset_adam_state();
    {
      const BufferView adam_zero_views[] = {
          param_store_->fused_m(),     param_store_->fused_v(),
          param_store_->adam_m_hash(), param_store_->adam_v_hash(),
          param_store_->adam_m_mlp(),  param_store_->adam_v_mlp(),
      };
      // zero_gpu_only_view's "skip Shared" semantic is preserved by
      // pre-filtering; context_blit_fill_views also skips empty entries.
      BufferView gpu_only[6];
      size_t n = 0;
      for (const auto &v : adam_zero_views) {
        if (!v.data && v.gpu_buffer && v.bytes > 0) gpu_only[n++] = v;
      }
      detail::context_blit_fill_views(*ctx_,
                                      std::span<const BufferView>(gpu_only, n),
                                      0);
    }
    clear_training_step_state();

    // Create authority for evaluator access.
    authority_ = std::make_shared<DefaultAuthority>(ctx_, param_store_, ps_desc);
  }

  void rebuild_training_kernels() {
    auto &reg = detail::context_pipeline_registry(*ctx_);

    const auto compile_spec = make_training_compile_spec();

    // Forward/backward kernel.
    auto fwd_result = KernelCompiler::compile({
        KernelRole::TrainForwardBackward, spec_, schema_, compile_spec});

    PipelineKey fwd_key{fwd_result.key.hash(),
                        fwd_result.entry_point.c_str(), false};
    kernels_.fwd_bwd = reg.register_pipeline(
        fwd_key, fwd_result.source.c_str(), fwd_result.entry_point.c_str());
    training_contract_ = detail::make_dispatch_contract(fwd_result);
    const auto fused_geometry = training_contract_.geometry;
    kernels_.tg_size = fused_geometry.tg_size;
    kernels_.pts_per_tg = fused_geometry.pts_per_tg;
    kernels_.tg_memory_bytes = fused_geometry.threadgroup_memory_bytes;
    training_loss_scale_ = fwd_result.resolved_spec.use_fp16 ? 128.0f : 1.0f;
    kernels_.valid = true;

    // Adam optimizer kernel (static MSL, not emitter-generated).
    // Prepend USE_INT_ATOMICS_HASH to match the fwd/bwd kernel's gradient format.
    const std::string adam_source =
        std::string("#define USE_INT_ATOMICS_HASH\n") + kNeuralSDFAdamUnifiedMSL;
    constexpr const char *adam_entry = "neural_sdf_adam_unified";
    PipelineKey adam_key{fwd_result.key.hash() ^ 0xADA0ULL,
                         adam_entry, false};
    kernels_.adam_fused = reg.register_pipeline(
        adam_key, adam_source.c_str(), adam_entry);

    std::string split_adam_source = "#define USE_INT_ATOMICS_HASH\n";
    if (sparse_hash_entry_width() > 1u) {
      split_adam_source += "#define SPARSE_HASH_ENTRY_WIDTH 2u\n";
    }
    split_adam_source += kNeuralSDFAdamSplitMSL;
    constexpr const char *hash_adam_entry = "neural_sdf_adam_hash_sparse";
    PipelineKey hash_adam_key{fwd_result.key.hash() ^ 0xADA1ULL,
                              hash_adam_entry, false};
    kernels_.adam_hash_sparse = reg.register_pipeline(
        hash_adam_key, split_adam_source.c_str(), hash_adam_entry);

    constexpr const char *mlp_adam_entry = "neural_sdf_adam_mlp_dense";
    PipelineKey mlp_adam_key{fwd_result.key.hash() ^ 0xADA2ULL,
                             mlp_adam_entry, false};
    kernels_.adam_mlp_dense = reg.register_pipeline(
        mlp_adam_key, split_adam_source.c_str(), mlp_adam_entry);

    fwd_bwd_plan_ = detail::make_fwd_bwd_plan(kernels_, *param_store_,
                                                train_cfg_.enable_probes);
    verify_plan_matches_contract(fwd_bwd_plan_, training_contract_,
                                 "TrainForwardBackward");
    adam_plan_ = detail::make_adam_plan(*param_store_, kernels_.adam_fused);
    sparse_hash_adam_plan_ = detail::make_sparse_hash_adam_plan(
        *param_store_, kernels_.adam_hash_sparse);
    dense_mlp_adam_plan_ = detail::make_mlp_dense_adam_plan(
        *param_store_, kernels_.adam_mlp_dense);
    batch_plan_.threadgroup_size = training_contract_.geometry.tg_size;
    batch_plan_.points_per_threadgroup = training_contract_.geometry.pts_per_tg;
  }

  // Second copy removed — first stub at line ~619 is canonical.

  void init_step_lanes() {
    auto &arena = detail::context_arena(*ctx_);
    if (train_cfg_.batch_size <= 0) {
      throw std::invalid_argument(
          "DefaultRuntime: batch_size must be > 0 (got " +
          std::to_string(train_cfg_.batch_size) +
          "). batch_size=0 causes zero-sized GPU buffers and all-zero outputs.");
    }
    const uint32_t batch_size = std::min<uint32_t>(
        static_cast<uint32_t>(train_cfg_.batch_size), 65536u);
    batch_plan_.max_batch_size = batch_size;
    batch_plan_.lane_count = kStepLaneCount;
    batch_plan_.input_dims = static_cast<uint32_t>(spec_.spatial_dims);
    batch_plan_.target_dims = static_cast<uint32_t>(spec_.num_outputs);
    batch_plan_.reduction_terms = schema_.reduction_terms;
    batch_plan_.positions_bytes_per_lane =
        static_cast<size_t>(batch_size) * batch_plan_.input_dims *
        sizeof(float);
    batch_plan_.targets_bytes_per_lane =
        static_cast<size_t>(batch_size) * batch_plan_.target_dims *
        sizeof(float);
    const uint32_t reduction_threadgroups =
        detail::threadgroup_count_for_points(batch_size,
                                             training_contract_.geometry);
    batch_plan_.reduction_bytes_per_lane =
        static_cast<size_t>(reduction_threadgroups) *
        batch_plan_.reduction_terms * sizeof(float);

    step_lanes_ = arena.allocate_step_set(
        batch_plan_.positions_bytes_per_lane,
        batch_plan_.targets_bytes_per_lane,
        batch_plan_.reduction_bytes_per_lane,
        kStepLaneCount);

    // Allocate probe buffers when probes are enabled.
    if (train_cfg_.enable_probes) {
      const uint32_t probe_stride =
          ProbeResult::stride_for_layers(
              static_cast<uint32_t>(spec_.num_hidden_layers));
      const size_t probe_bytes = static_cast<size_t>(reduction_threadgroups) *
                                 probe_stride * sizeof(float);
      for (auto &lane : step_lanes_) {
        BufferDesc pb_desc{probe_bytes, 256, BufferStorage::Shared,
                           BufferLifetime::Transient, "step_probe"};
        lane.probe_buffer = arena.view(arena.allocate(pb_desc));
      }
    }
  }

  std::shared_ptr<MetalContext> ctx_;
  TrainerConfig train_cfg_;
  KernelSpec spec_;
  extension::ExtensionSchema schema_;
  std::shared_ptr<ParameterStore> param_store_;
  std::shared_ptr<const RuntimeAuthority> authority_;
  detail::TrainingDispatchKernels kernels_{};
  detail::KernelDispatchContract training_contract_{};
  detail::KernelDispatchContract backward_ext_contract_{};
  detail::KernelDispatchContract forward_train_contract_{};
  detail::DispatchPlan fwd_bwd_plan_;
  detail::DispatchPlan backward_ext_plan_;
  detail::DispatchPlan forward_train_plan_;
  detail::DispatchPlan adam_plan_;
  BufferHandle d_output_buf_{};
  bool backward_ext_compiled_ = false;
  detail::DispatchPlan sparse_hash_adam_plan_;
  detail::DispatchPlan dense_mlp_adam_plan_;
  std::vector<StepBufferSet> step_lanes_;
  StepLaneCoordinator lane_coord_{kStepLaneCount};
  detail::PendingTrainingStep pending_{};
  bool step_state_clear_required_ = false;
  TrainerBatchPlan batch_plan_{};
  uint32_t step_ = 0;
  bool safe_family_active_ = false;
  float training_loss_scale_ = 1.0f;
  std::vector<float> sorted_pos_;
  std::vector<float> sorted_tgt_;
  std::vector<uint32_t> morton_codes_;
  std::vector<uint32_t> morton_indices_;
  std::optional<ProbeResult> last_split_probe_;
  bool training_step_profiling_enabled_ = false;
  std::optional<TrainingStepProfile> last_training_step_profile_;
};

} // namespace

HashGridEncoding::Config default_trainer_encoding_config() {
  HashGridEncoding::Config cfg;
  cfg.log2_hashmap_size = kHeadlineTrainerLog2HashmapSize;
  return cfg;
}

FullyFusedMLP::Config
default_trainer_network_config(const HashGridEncoding::Config &enc_cfg) {
  FullyFusedMLP::Config cfg;
  cfg.hidden_dim = kHeadlineTrainerHiddenDim;
  cfg.n_input = enc_cfg.num_levels * enc_cfg.features_per_level;
  return cfg;
}

TrainerConfig default_trainer_config() {
  TrainerConfig cfg;
  cfg.batch_size = kHeadlineTrainerBatchSize;
  return cfg;
}

// ── Trainer methods ──────────────────────────────────────────────────────

void Trainer::clear_diagnostic_state() {
  last_diagnostic_.reset();
  if (runtime_)
    runtime_->clear_diagnostic();
  if (evaluator_)
    evaluator_->clear_diagnostic();
}

void Trainer::set_diagnostic(DiagnosticCode code, std::string operation,
                             std::string message) {
  last_diagnostic_ = DiagnosticInfo{
      .code = code,
      .operation = std::move(operation),
      .message = std::move(message),
  };
  emit_diagnostic(*last_diagnostic_);
}

void Trainer::capture_delegate_diagnostic(DiagnosticCode fallback_code,
                                          const char *operation,
                                          const char *fallback_message) {
  if (last_diagnostic_)
    return;
  if (evaluator_) {
    if (auto diag = evaluator_->last_diagnostic()) {
      last_diagnostic_ = std::move(*diag);
      return;
    }
  }
  if (runtime_) {
    if (auto diag = runtime_->last_diagnostic()) {
      last_diagnostic_ = std::move(*diag);
      return;
    }
  }
  set_diagnostic(fallback_code, operation, fallback_message);
}

std::optional<DiagnosticInfo> Trainer::last_diagnostic() const {
  if (last_diagnostic_)
    return last_diagnostic_;
  if (evaluator_) {
    if (auto diag = evaluator_->last_diagnostic())
      return diag;
  }
  if (runtime_)
    return runtime_->last_diagnostic();
  return std::nullopt;
}

void Trainer::ensure_evaluator() {
  if (!runtime_) {
    set_diagnostic(DiagnosticCode::MissingRuntime, "Trainer::ensure_evaluator",
                   "trainer runtime is unavailable");
    return;
  }
  if (evaluator_)
    return;
  auto evaluator_result =
      try_make_bound_evaluator(*runtime_, "Trainer::ensure_evaluator");
  if (!evaluator_result) {
    last_diagnostic_ = evaluator_result.error();
    return;
  }
  evaluator_ = std::move(*evaluator_result);
}

bool Trainer::evaluate(const float *positions, float *output, int N) {
  clear_diagnostic_state();
  if (runtime_) {
    runtime_->sync_weights();
  }
  ensure_evaluator();
  if (!evaluator_) {
    capture_delegate_diagnostic(DiagnosticCode::OperationFailed,
                                "Trainer::evaluate",
                                "failed to initialize evaluator");
    return false;
  }
  const bool ok = evaluator_->evaluate(positions, output, N);
  if (!ok) {
    capture_delegate_diagnostic(DiagnosticCode::OperationFailed,
                                "Trainer::evaluate",
                                "evaluator returned false");
  }
  return ok;
}

bool Trainer::evaluate_with_gradient(const float *positions, float *output,
                                     float *gradients, int N) {
  clear_diagnostic_state();
  if (runtime_) {
    runtime_->sync_weights();
  }
  ensure_evaluator();
  if (!evaluator_) {
    capture_delegate_diagnostic(DiagnosticCode::OperationFailed,
                                "Trainer::evaluate_with_gradient",
                                "failed to initialize evaluator");
    return false;
  }
  const bool ok = evaluator_->evaluate_with_gradient(positions, output, gradients, N);
  if (!ok) {
    capture_delegate_diagnostic(DiagnosticCode::OperationFailed,
                                "Trainer::evaluate_with_gradient",
                                "evaluator returned false");
  }
  return ok;
}

Result<std::unique_ptr<FieldEvaluator>> Trainer::try_create_evaluator() {
  clear_diagnostic_state();
  if (!runtime_) {
    return make_construction_error<std::unique_ptr<FieldEvaluator>>(
        DiagnosticCode::MissingRuntime, "Trainer::try_create_evaluator",
        "trainer runtime is unavailable");
  }

  runtime_->sync_weights();
  return try_make_bound_evaluator(*runtime_, "Trainer::try_create_evaluator");
}

std::unique_ptr<FieldEvaluator> Trainer::create_evaluator() {
  return unwrap_or_throw(try_create_evaluator(), "Trainer::create_evaluator");
}

// ── Split path: forward / backward / optimizer_step ─────────────────────

Trainer::ForwardPass Trainer::forward_for_training(const float *input,
                                                   const float *target,
                                                   int N) {
  clear_diagnostic_state();
  if (runtime_) {
    runtime_->sync_weights();
  }
  ForwardPass pass;
  if (!runtime_) {
    set_diagnostic(DiagnosticCode::MissingRuntime,
                   "Trainer::forward_for_training",
                   "trainer runtime is unavailable");
    return pass;
  }
  if (N < 0) {
    set_diagnostic(DiagnosticCode::InvalidArgument,
                   "Trainer::forward_for_training",
                   "N must be non-negative");
    return pass;
  }
  if (!runtime_->is_gpu_available()) {
    set_diagnostic(DiagnosticCode::GpuUnavailable,
                   "Trainer::forward_for_training",
                   "Metal GPU is unavailable");
    return pass;
  }
  if (N == 0) {
    set_diagnostic(DiagnosticCode::InvalidArgument,
                   "Trainer::forward_for_training",
                   "N must be greater than zero");
    return pass;
  }
  if (!input) {
    set_diagnostic(DiagnosticCode::InvalidArgument,
                   "Trainer::forward_for_training",
                   "input must be non-null");
    return pass;
  }
  // target may be nullptr for split path (no internal loss needed).

  const auto plan = runtime_->batch_plan();
  const uint32_t batch_N =
      std::min<uint32_t>(static_cast<uint32_t>(N), plan.max_batch_size);
  const uint32_t out_dims = plan.target_dims;

  // Use ForwardForTraining kernel if available, otherwise fall back to evaluate.
  pass.output_storage_ = std::make_shared<std::vector<float>>(
      static_cast<size_t>(batch_N) * out_dims);
  uint32_t written = runtime_->forward_for_training(
      input, pass.output_storage_->data(), static_cast<int>(batch_N));
  if (written == 0) {
    // Fallback to evaluate (different kernel path but numerically equivalent
    // for scalar non-FP16).
    bool ok = evaluate(input, pass.output_storage_->data(),
                       static_cast<int>(batch_N));
    if (!ok)
      return pass;
  }

  // Store positions for backward dispatch.
  split_positions_.assign(input, input +
      static_cast<size_t>(batch_N) * plan.input_dims);
  split_batch_N_ = batch_N;

  pass.valid_ = true;
  pass.batch_N_ = batch_N;
  pass.output_dims_ = out_dims;
  pass.output_data_ = pass.output_storage_->data();
  return pass;
}

void Trainer::backward_from_output(const ForwardPass &pass,
                                   const float *d_output) {
  if (!pass.valid() || !runtime_)
    throw std::runtime_error("backward_from_output: invalid ForwardPass");
  // d_output size = N * output_dims.
  split_d_output_.assign(d_output, d_output + pass.output_count());
  split_has_external_grad_ = true;
}

void Trainer::optimizer_step() {
  if (!split_has_external_grad_ || !runtime_)
    throw std::runtime_error("optimizer_step: no pending backward");

  // Real dispatch: backward_and_update runs BackwardFromExternalGrad kernel
  // + Adam + weight sync.
  runtime_->sync_weights();
  runtime_->backward_and_update(split_positions_.data(),
                                split_d_output_.data(),
                                static_cast<int>(split_batch_N_));
  split_has_external_grad_ = false;
}

// ── Multi-frame accumulation API ────────────────────────────────────

Trainer::ForwardPass Trainer::forward_for_training(const float *input, int N) {
  return forward_for_training(input, nullptr, N);
}

void Trainer::zero_gradients() {
  if (!runtime_)
    throw std::runtime_error("zero_gradients: no runtime");
  runtime_->sync_weights();
  runtime_->zero_gradients();
  accum_zeroed_ = true;
  accum_count_ = 0;
}

void Trainer::backward_accumulate(const ForwardPass &pass,
                                  const float *d_output) {
  if (!pass.valid() || !runtime_)
    throw std::runtime_error("backward_accumulate: invalid ForwardPass");
  if (!accum_zeroed_)
    throw std::runtime_error(
        "backward_accumulate: must call zero_gradients() first");

  // Use the positions stored by forward_for_training.
  runtime_->backward_accumulate(split_positions_.data(),
                                d_output,
                                static_cast<int>(pass.batch_size()));
  ++accum_count_;
}

void Trainer::adam_step() {
  if (!runtime_)
    throw std::runtime_error("adam_step: no runtime");
  if (accum_count_ == 0)
    throw std::runtime_error(
        "adam_step: no backward calls since last zero_gradients()");

  runtime_->adam_step();
  accum_zeroed_ = false;
  accum_count_ = 0;
}

std::optional<TrainerRuntimeInspection> Trainer::inspect_runtime() const {
  const auto *inspectable_runtime =
      dynamic_cast<const InspectableTrainerRuntime *>(runtime_.get());
  if (!inspectable_runtime)
    return std::nullopt;

  auto inspection = inspectable_runtime->inspect_runtime();
  if (evaluator_) {
    if (const auto *inspectable_evaluator =
            dynamic_cast<const InspectableEvaluator *>(evaluator_.get())) {
      inspectable_evaluator->merge_runtime_inspection(inspection);
    }
  }
  return inspection;
}

// ── Factory function ────────────────────────────────────────────────────

Result<Trainer> try_create_trainer(const HashGridEncoding::Config &enc_cfg,
                                   const FullyFusedMLP::Config &net_cfg,
                                   const TrainerConfig &train_cfg,
                                   std::shared_ptr<MetalContext> ctx) {
  try {
    const auto resolved_net_cfg = resolve_network_config(enc_cfg, net_cfg);
    auto enc = std::make_shared<HashGridEncoding>(enc_cfg);
    auto net = std::make_shared<FullyFusedMLP>(resolved_net_cfg);
    auto model = std::make_shared<NetworkWithInputEncoding>(enc, net);
    if (!ctx)
      ctx = MetalContext::create();
    if (!ctx || !ctx->is_gpu_available()) {
      return make_construction_error<Trainer>(
          DiagnosticCode::GpuUnavailable, "create_trainer",
          "Metal GPU not available");
    }
    auto runtime = std::make_unique<DefaultRuntime>(
        enc_cfg, resolved_net_cfg, train_cfg, ctx);
    return Trainer(std::move(model), create_loss_l2(), create_optimizer_adam(),
                   std::move(runtime));
  } catch (const std::exception &e) {
    return result_from_exception<Trainer>("create_trainer", e);
  }
}

Trainer create_trainer(const HashGridEncoding::Config &enc_cfg,
                       const FullyFusedMLP::Config &net_cfg,
                       const TrainerConfig &train_cfg,
                       std::shared_ptr<MetalContext> ctx) {
  return unwrap_or_throw(
      try_create_trainer(enc_cfg, net_cfg, train_cfg, std::move(ctx)),
      "create_trainer");
}

Result<Trainer> try_create_trainer(std::shared_ptr<NetworkWithInputEncoding> model,
                                   const TrainerConfig &train_cfg,
                                   std::shared_ptr<MetalContext> ctx) {
  return try_create_trainer(std::move(model), create_loss_l2(),
                            create_optimizer_adam(), train_cfg,
                            std::move(ctx));
}

Trainer create_trainer(std::shared_ptr<NetworkWithInputEncoding> model,
                       const TrainerConfig &train_cfg,
                       std::shared_ptr<MetalContext> ctx) {
  return unwrap_or_throw(
      try_create_trainer(std::move(model), train_cfg, std::move(ctx)),
      "create_trainer(model)");
}

Result<Trainer> try_create_trainer(std::shared_ptr<NetworkWithInputEncoding> model,
                                   std::shared_ptr<Loss> loss,
                                   std::shared_ptr<Optimizer> optimizer,
                                   const TrainerConfig &train_cfg,
                                   std::shared_ptr<MetalContext> ctx) {
  if (!model) {
    return make_construction_error<Trainer>(
        DiagnosticCode::NullObject, "create_trainer(model)",
        "model must not be null");
  }
  if (!loss) {
    return make_construction_error<Trainer>(
        DiagnosticCode::NullObject, "create_trainer(model)",
        "loss must not be null");
  }
  if (!optimizer) {
    return make_construction_error<Trainer>(
        DiagnosticCode::NullObject, "create_trainer(model)",
        "optimizer must not be null");
  }

  auto resolved_cfg = try_resolve_loss_config(train_cfg, *loss);
  if (!resolved_cfg) {
    return std::unexpected(std::move(resolved_cfg.error()));
  }

  auto configs = try_extract_default_runtime_configs(*model);
  if (!configs) {
    return std::unexpected(std::move(configs.error()));
  }

  try {
    if (!ctx)
      ctx = MetalContext::create();
    if (!ctx || !ctx->is_gpu_available()) {
      return make_construction_error<Trainer>(
          DiagnosticCode::GpuUnavailable, "create_trainer(model)",
          "Metal GPU not available");
    }

    auto runtime = std::make_unique<DefaultRuntime>(
        configs->first, configs->second, *resolved_cfg, ctx);
    return Trainer(std::move(model), std::move(loss), std::move(optimizer),
                   std::move(runtime));
  } catch (const std::exception &e) {
    return result_from_exception<Trainer>("create_trainer(model)", e);
  }
}

Trainer create_trainer(std::shared_ptr<NetworkWithInputEncoding> model,
                       std::shared_ptr<Loss> loss,
                       std::shared_ptr<Optimizer> optimizer,
                       const TrainerConfig &train_cfg,
                       std::shared_ptr<MetalContext> ctx) {
  return unwrap_or_throw(try_create_trainer(std::move(model), std::move(loss),
                                            std::move(optimizer), train_cfg,
                                            std::move(ctx)),
                         "create_trainer(model)");
}

Result<Trainer>
try_create_trainer_with_adapter(const extension::TrainingAdapter &adapter,
                                std::shared_ptr<NetworkWithInputEncoding> model,
                                const TrainerConfig &train_cfg,
                                std::shared_ptr<MetalContext> ctx) {
  if (!model) {
    return make_construction_error<Trainer>(
        DiagnosticCode::NullObject, "create_trainer(adapter)",
        "model must not be null");
  }

  const auto schema = adapter.schema();
  try {
    schema.validate();
  } catch (const std::exception &e) {
    return result_from_exception<Trainer>("create_trainer(adapter)", e);
  }
  if (schema.input_dims != static_cast<uint32_t>(model->n_input_dims())) {
    return make_construction_error<Trainer>(
        DiagnosticCode::SchemaMismatch, "create_trainer(adapter)",
        "adapter schema input_dims must match model input dims");
  }
  if (schema.target_dims != static_cast<uint32_t>(model->n_output_dims())) {
    return make_construction_error<Trainer>(
        DiagnosticCode::SchemaMismatch, "create_trainer(adapter)",
        "adapter schema target_dims must match model output dims");
  }

  extension::KernelCompileSpec adapter_spec;
  adapter.configure_compile_spec(adapter_spec);

  const auto loss_cfg = adapter.loss_config();
  adapter_spec.loss_kind = loss_cfg.kind;
  adapter_spec.huber_delta = loss_cfg.huber_delta;
  try {
    adapter_spec.validate(schema);
  } catch (const std::exception &e) {
    return result_from_exception<Trainer>("create_trainer(adapter)", e);
  }

  auto loss = try_create_loss_from_config(loss_cfg);
  if (!loss) {
    return std::unexpected(std::move(loss.error()));
  }

  auto resolved_cfg = train_cfg;
  resolved_cfg.loss_kind = loss_cfg.kind;
  resolved_cfg.huber_delta = loss_cfg.huber_delta;

  return try_create_trainer(std::move(model), std::move(*loss),
                            create_optimizer_adam(), resolved_cfg,
                            std::move(ctx));
}

Trainer create_trainer_with_adapter(const extension::TrainingAdapter &adapter,
                                    std::shared_ptr<NetworkWithInputEncoding> model,
                                    const TrainerConfig &train_cfg,
                                    std::shared_ptr<MetalContext> ctx) {
  return unwrap_or_throw(try_create_trainer_with_adapter(
                             adapter, std::move(model), train_cfg,
                             std::move(ctx)),
                         "create_trainer(adapter)");
}

} // namespace tmnn

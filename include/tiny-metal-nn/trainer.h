#pragma once

/**
 * @file trainer.h
 * @brief Trainer — the single object for training and inference.
 *
 * Usage:
 *   auto trainer = tmnn::create_trainer();
 *   trainer.training_step(input, target, N);
 *   trainer.evaluate(input, output, N);
 */

#include "tiny-metal-nn/common.h"
#include "tiny-metal-nn/evaluator.h"
#include "tiny-metal-nn/extension/kernel_compile_spec.h"
#include "tiny-metal-nn/loss.h"
#include "tiny-metal-nn/detail/module.h"
#include "tiny-metal-nn/optimizer.h"
#include "tiny-metal-nn/detail/runtime_authority.h"
#include "tiny-metal-nn/weight_init.h"

#include <memory>
#include <optional>
#include <string>

namespace tmnn {

class MetalContext;
class NetworkWithInputEncoding;
class HashGridEncoding;
class FullyFusedMLP;

/// Configuration for training behavior.
struct TrainerConfig {
  int batch_size = 1024;
  float lr_encoding = 1e-2f;
  float lr_network = 1e-3f;
  float beta1 = 0.9f;
  float beta2 = 0.99f;
  float epsilon = 1e-15f;
  float l1_reg = 0.0f;
  float l2_reg = 0.0f;
  bool use_private_buffers = true;

  /// Loss function (lowered to KernelSpec at compile time).
  /// L1/Huber only supported for signed SDF (unsigned_mode not supported).
  /// Cosine is supported on generic multi-output paths with target_dims >= 2.
  extension::LossKind loss_kind = extension::LossKind::L2;
  float huber_delta = 1.0f; ///< Only used when loss_kind == Huber.

  /// Opt-in per-step probe telemetry. When true, training kernels emit
  /// activation/gradient stats; results available in TrainingStepResult.probe.
  /// Disables SIMD training (scalar-only for probe correctness).
  bool enable_probes = false;

  /// Weight initialization for the hash grid + MLP at trainer construction.
  /// Defaults follow literature (hash grid uniform [-1e-4, 1e-4],
  /// MLP Kaiming uniform). Override the seed for reproducibility across
  /// multi-seed sweeps; see weight_init.h.
  WeightInitConfig weight_init;
};

/// Snapshot of one compiled runtime kernel's requested vs realized dispatch.
struct TrainerKernelInspection {
  bool available = false;
  std::string entry_point;
  bool requested_simd = false;
  bool realized_simd = false;
  bool requested_fp16 = false;
  bool realized_fp16 = false;
  bool requested_tg_weight_cache = false;
  bool realized_tg_weight_cache = false;
  uint32_t threadgroup_size = 0;
  uint32_t points_per_threadgroup = 0;
  uint32_t threadgroup_memory_bytes = 0;
};

/// Lightweight runtime inspection snapshot for debugging specialization issues.
struct TrainerRuntimeInspection {
  TrainerKernelInspection training_step;
  TrainerKernelInspection forward_for_training;
  TrainerKernelInspection backward_from_output;
  TrainerKernelInspection evaluate;
  TrainerKernelInspection evaluate_with_gradient;
  uint32_t batch_size = 0;
  bool safe_family_active = false;
};

/// Opt-in per-step profiling for fused Trainer::training_step().
///
/// Profiling is disabled by default so normal training does not pay for the
/// extra host-side timing and profile materialization work.
struct TrainingStepProfilingOptions {
  bool enabled = false;
};

/// Last fused training-step timing profile recorded by the runtime.
///
/// All duration fields are host-observed wall-clock nanoseconds unless the
/// field name ends in `_us`, in which case the value comes from Metal command
/// buffer GPU execution timestamps.
struct TrainingStepProfile {
  uint32_t step = 0;
  uint32_t batch_size = 0;
  uint64_t total_ns = 0;
  uint64_t morton_sort_ns = 0;
  uint64_t enqueue_total_ns = 0;
  uint64_t drain_pending_ns = 0;
  uint64_t prepare_step_lane_ns = 0;
  uint64_t fill_train_params_ns = 0;
  uint64_t resolve_bindings_ns = 0;
  uint64_t submit_forward_backward_ns = 0;
  uint64_t finalize_total_ns = 0;
  uint64_t wait_pending_ns = 0;
  uint64_t wait_fwd_bwd_fill_ns = 0;
  uint64_t wait_fwd_bwd_dispatch_ns = 0;
  uint64_t fill_adam_params_pre_finalize_ns = 0;
  uint64_t finalize_step_readback_ns = 0;
  uint64_t numerics_report_ns = 0;
  uint64_t numerics_backward_readback_ns = 0;
  uint64_t numerics_backward_scan_ns = 0;
  uint64_t numerics_update_readback_ns = 0;
  uint64_t numerics_update_scan_ns = 0;
  uint64_t fill_adam_params_apply_ns = 0;
  uint64_t prepare_sparse_hash_adam_ns = 0;
  uint64_t submit_adam_ns = 0;
  uint64_t sync_config_weights_ns = 0;
  uint64_t append_extra_losses_ns = 0;
  uint64_t probe_aggregation_ns = 0;
  uint64_t uncategorized_ns = 0;
  double gpu_fwd_bwd_us = 0.0;
  double gpu_adam_us = 0.0;
};

/// Runtime interface — the actual training implementation.
class ITrainerRuntime {
public:
  virtual ~ITrainerRuntime() = default;

  virtual TrainingStepResult training_step(const float *input,
                                           const float *target, int N) = 0;

  /// Run forward-only for split training. Writes output to out_buf.
  /// Returns number of outputs written (N * output_dims), or 0 on failure.
  virtual uint32_t forward_for_training(const float *input, float *output,
                                        int N) {
    (void)input; (void)output; (void)N;
    return 0; // Default: not supported.
  }

  /// Run backward pass with externally-provided output gradient + Adam update.
  /// (Legacy: use zero_gradients + backward_accumulate + adam_step instead.)
  virtual void backward_and_update(const float *input, const float *d_output,
                                   int N) {
    (void)input; (void)d_output; (void)N;
    throw std::runtime_error("backward_and_update: not supported by this runtime");
  }

  /// Zero gradient buffers. Call once at the start of each accumulation window.
  virtual void zero_gradients() {
    throw std::runtime_error("zero_gradients: not supported by this runtime");
  }

  /// Dispatch backward kernel, accumulating gradients via atomic_add.
  /// May be called multiple times between zero_gradients() and adam_step().
  virtual void backward_accumulate(const float *input, const float *d_output,
                                   int N) {
    (void)input; (void)d_output; (void)N;
    throw std::runtime_error("backward_accumulate: not supported by this runtime");
  }

  /// Apply Adam optimizer update using accumulated gradients. Advances step counter.
  virtual void adam_step() {
    throw std::runtime_error("adam_step: not supported by this runtime");
  }

  /// Probe result from the last backward_and_update() call (split path).
  /// Returns nullopt when probes are disabled.
  [[nodiscard]] virtual std::optional<ProbeResult>
  read_last_split_probe() const {
    return std::nullopt;
  }
  virtual void sync_weights() = 0;
  [[nodiscard]] virtual uint32_t step() const = 0;
  [[nodiscard]] virtual bool is_gpu_available() const = 0;
  [[nodiscard]] virtual std::shared_ptr<const RuntimeAuthority>
  runtime_authority() const = 0;
  [[nodiscard]] virtual TrainerBatchPlan batch_plan() const = 0;
  [[nodiscard]] virtual OptimizerStateBlob export_optimizer_state() = 0;
  virtual void import_optimizer_state(const OptimizerStateBlob &state) = 0;
  virtual void reset_optimizer() = 0;
  virtual void apply_optimizer_config(const Optimizer &opt) = 0;
  [[nodiscard]] virtual std::optional<DiagnosticInfo>
  last_diagnostic() const {
    return std::nullopt;
  }
  virtual void clear_diagnostic() {}
  virtual void
  set_training_step_profiling(const TrainingStepProfilingOptions &) {}
  [[nodiscard]] virtual std::optional<TrainingStepProfile>
  last_training_step_profile() const {
    return std::nullopt;
  }
};

/// Trainer — the single object for training and inference on hash-grid + MLP.
class Trainer {
public:
  // ── Compositional construction ──────────────────────────────
  Trainer(std::shared_ptr<Module> model, std::shared_ptr<Loss> loss,
          std::shared_ptr<Optimizer> optimizer,
          std::unique_ptr<ITrainerRuntime> runtime)
      : model_(std::move(model)), loss_(std::move(loss)),
        optimizer_(std::move(optimizer)), runtime_(std::move(runtime)) {}


  // ── Training (fused path) ───────────────────────────────────

  TrainingStepResult training_step(const float *input, const float *target,
                                   int N) {
    clear_diagnostic_state();
    return runtime_->training_step(input, target, N);
  }

  // ── Training (split path — external gradient) ─────────────

  /// Opaque handle for an in-flight forward pass.
  struct ForwardPass {
    ForwardPass() = default;
    ForwardPass(const ForwardPass &other)
        : valid_(other.valid_), batch_N_(other.batch_N_),
          output_dims_(other.output_dims_), output_data_(other.output_data_),
          output_storage_(other.output_storage_) {
      refresh_output_pointer();
    }
    ForwardPass(ForwardPass &&other) noexcept
        : valid_(other.valid_), batch_N_(other.batch_N_),
          output_dims_(other.output_dims_), output_data_(other.output_data_),
          output_storage_(std::move(other.output_storage_)) {
      refresh_output_pointer();
    }
    ForwardPass &operator=(const ForwardPass &other) {
      if (this == &other)
        return *this;
      valid_ = other.valid_;
      batch_N_ = other.batch_N_;
      output_dims_ = other.output_dims_;
      output_data_ = other.output_data_;
      output_storage_ = other.output_storage_;
      refresh_output_pointer();
      return *this;
    }
    ForwardPass &operator=(ForwardPass &&other) noexcept {
      if (this == &other)
        return *this;
      valid_ = other.valid_;
      batch_N_ = other.batch_N_;
      output_dims_ = other.output_dims_;
      output_data_ = other.output_data_;
      output_storage_ = std::move(other.output_storage_);
      refresh_output_pointer();
      return *this;
    }

    [[nodiscard]] bool valid() const { return valid_; }
    [[nodiscard]] uint32_t batch_size() const { return batch_N_; }
    [[nodiscard]] uint32_t output_dims() const { return output_dims_; }
    /// Total float count = batch_size * output_dims.
    [[nodiscard]] uint32_t output_count() const {
      return batch_N_ * output_dims_;
    }
    /// Flat index into [batch_size * output_dims] array.
    [[nodiscard]] float output(uint32_t i) const {
      const float *data =
          (output_storage_ && !output_storage_->empty()) ? output_storage_->data()
                                                         : output_data_;
      return (i < output_count() && data) ? data[i] : 0.0f;
    }
    [[nodiscard]] const float *output_data_ptr() const {
      return (output_storage_ && !output_storage_->empty()) ? output_storage_->data()
                                                            : output_data_;
    }
  private:
    friend class Trainer;
    void refresh_output_pointer() {
      if (output_storage_ && !output_storage_->empty()) {
        output_data_ = output_storage_->data();
      }
    }
    bool valid_ = false;
    uint32_t batch_N_ = 0;
    uint32_t output_dims_ = 1;
    const float *output_data_ = nullptr;
    std::shared_ptr<std::vector<float>> output_storage_;
  };

  /// Forward pass only — returns output for external loss computation.
  ForwardPass forward_for_training(const float *input, const float *target,
                                   int N);

  /// Forward pass without target (split path: target not needed).
  ForwardPass forward_for_training(const float *input, int N);

  /// Apply external gradient and run backward pass.
  /// (Legacy single-shot: zero + backward + adam in one call.)
  void backward_from_output(const ForwardPass &pass, const float *d_output);

  /// Apply optimizer update (Adam) and advance step counter.
  /// (Legacy single-shot: requires prior backward_from_output.)
  void optimizer_step();

  // ── Multi-frame accumulation API ───────────────────────────

  /// Zero gradient buffers. Call once at the start of each training step.
  void zero_gradients();

  /// Accumulate gradients from one frame. May be called multiple times
  /// between zero_gradients() and adam_step().
  /// @param pass  ForwardPass from forward_for_training (same frame).
  /// @param d_output  External gradient [N × output_dims].
  void backward_accumulate(const ForwardPass &pass, const float *d_output);

  /// Apply Adam update using accumulated gradients. Advances step counter.
  void adam_step();

  /// Probe result from the last split-path optimizer_step() call.
  /// Returns nullopt when probes are disabled or no split step was run.
  [[nodiscard]] std::optional<ProbeResult> read_last_split_probe() const {
    return runtime_->read_last_split_probe();
  }

  // ── Inference (same object — tcnn pattern) ──────────────────

  /// Evaluate the trained model at N positions.
  bool evaluate(const float *positions, float *output, int N);

  /// tcnn-style alias for evaluate().
  bool inference(const float *positions, float *output, int N) {
    return evaluate(positions, output, N);
  }

  /// Evaluate with analytical gradients.
  bool evaluate_with_gradient(const float *positions, float *output,
                              float *gradients, int N);

  /// Convenience alias for gradient-capable inference.
  bool inference_with_gradient(const float *positions, float *output,
                               float *gradients, int N) {
    return evaluate_with_gradient(positions, output, gradients, N);
  }

  /// Export a separate evaluator bound to this trainer's runtime-owned weights.
  ///
  /// This is the honest evaluator-only surface for tmnn's default runtime: it
  /// derives from an already-constructed `Trainer` rather than pretending that a
  /// descriptor-only `create_evaluator(model, ...)` can hydrate arbitrary
  /// external weights on its own.
  ///
  /// The trainer synchronizes pending weight updates before exporting the
  /// evaluator. The returned evaluator is backed by the trainer's
  /// `RuntimeAuthority`, so advanced consumers can keep an evaluator-only handle
  /// without carrying the training API everywhere.
  [[nodiscard]] Result<std::unique_ptr<FieldEvaluator>> try_create_evaluator();

  /// Throwing wrapper over try_create_evaluator().
  [[nodiscard]] std::unique_ptr<FieldEvaluator> create_evaluator();

  /// Structured diagnostic for the most recent non-throwing trainer failure.
  /// Successful calls clear this state.
  [[nodiscard]] std::optional<DiagnosticInfo> last_diagnostic() const;

  // ── Accessors ───────────────────────────────────────────────

  void sync_weights() { runtime_->sync_weights(); }
  [[nodiscard]] uint32_t step() const { return runtime_->step(); }
  [[nodiscard]] bool is_gpu_available() const {
    return runtime_->is_gpu_available();
  }
  [[nodiscard]] TrainerBatchPlan batch_plan() const {
    return runtime_->batch_plan();
  }
  /// Returns compiled-kernel specialization/dispatch metadata when the runtime
  /// exposes it. This is a lightweight inspection API, not per-step telemetry.
  [[nodiscard]] std::optional<TrainerRuntimeInspection>
  inspect_runtime() const;
  /// Enables or disables fused training-step profiling on the bound runtime.
  void
  set_training_step_profiling(const TrainingStepProfilingOptions &options) {
    runtime_->set_training_step_profiling(options);
  }
  /// Returns the most recent successful profiled fused training step.
  [[nodiscard]] std::optional<TrainingStepProfile>
  last_training_step_profile() const {
    return runtime_->last_training_step_profile();
  }

  void set_learning_rate(float lr) {
    if (optimizer_) {
      optimizer_->set_learning_rate(lr);
      runtime_->apply_optimizer_config(*optimizer_);
    }
  }

  void reset_optimizer() { runtime_->reset_optimizer(); }
  [[nodiscard]] OptimizerStateBlob export_optimizer_state() {
    return runtime_->export_optimizer_state();
  }
  void import_optimizer_state(const OptimizerStateBlob &state) {
    runtime_->import_optimizer_state(state);
  }

  Module &model() { return *model_; }
  const Module &model() const { return *model_; }
  const Optimizer &optimizer() const { return *optimizer_; }
  ITrainerRuntime &runtime() { return *runtime_; }
  const ITrainerRuntime &runtime() const { return *runtime_; }
  /// Shared runtime authority for advanced consumers. Retaining the returned
  /// `shared_ptr` keeps the exported runtime-backed parameter storage alive.
  [[nodiscard]] std::shared_ptr<const RuntimeAuthority>
  runtime_authority() const {
    return runtime_->runtime_authority();
  }

private:
  std::shared_ptr<Module> model_;
  std::shared_ptr<Loss> loss_;
  std::shared_ptr<Optimizer> optimizer_;
  std::unique_ptr<ITrainerRuntime> runtime_;
  std::unique_ptr<FieldEvaluator> evaluator_;
  std::optional<DiagnosticInfo> last_diagnostic_;

  // Split path state (legacy single-shot).
  std::vector<float> split_positions_;
  std::vector<float> split_targets_;
  std::vector<float> split_d_output_;
  uint32_t split_batch_N_ = 0;
  bool split_has_external_grad_ = false;

  // Multi-frame accumulation state.
  bool accum_zeroed_ = false;
  uint32_t accum_count_ = 0;

  void clear_diagnostic_state();
  void set_diagnostic(DiagnosticCode code, std::string operation,
                      std::string message);
  void capture_delegate_diagnostic(DiagnosticCode fallback_code,
                                   const char *operation,
                                   const char *fallback_message);
  void ensure_evaluator();
};

} // namespace tmnn

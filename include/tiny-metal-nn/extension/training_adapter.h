#pragma once

/**
 * @file extension/training_adapter.h
 * @brief TrainingAdapter + AdamConfig + ResultMetrics — SDK surface for
 *        extension authors declaring training data/compute semantics.
 *
 * A TrainingAdapter is a *declarative contract*: the runtime (Trainer) owns
 * the training loop; the adapter only declares computation and data semantics.
 * This mirrors tiny-cuda-nn's "Trainer owns the loop, components declare
 * schemas" philosophy.
 *
 * Dependencies: extension/schema.h, extension/kernel_compile_spec.h,
 *               <array>, <cstdint>.
 */

#include "tiny-metal-nn/extension/schema.h"
#include "tiny-metal-nn/extension/kernel_compile_spec.h"

#include <array>
#include <cstdint>

namespace tmnn {
namespace extension {

/// Loss function configuration declared by an adapter.
/// Runtime lowers this to KernelCompileSpec::loss_kind at pipeline construction.
struct LossConfig {
  LossKind kind = LossKind::L2;
  float huber_delta = 1.0f; ///< Only used when kind == Huber.
};

/// Adam optimizer configuration with split encoding/network learning rates.
struct AdamConfig {
  float lr_encoding = 1e-2f;   ///< Learning rate for hash grid encoding.
  float lr_network = 1e-3f;    ///< Learning rate for MLP network.
  float beta1 = 0.9f;          ///< First moment decay.
  float beta2 = 0.99f;         ///< Second moment decay.
  float epsilon = 1e-8f;       ///< Numerical stability constant.
  float l1_reg = 0.0f;         ///< L1 regularization weight.
  float l2_reg = 1e-6f;        ///< L2 regularization weight.
  float grad_clip = 0.0f;      ///< Gradient clipping magnitude (0 = disabled).
  float weight_decay = 0.0f;   ///< Decoupled weight decay (AdamW).
};

/// Post-step result metrics reported by a training adapter.
struct ResultMetrics {
  std::array<float, 4> extra_losses = {}; ///< Up to 4 auxiliary loss values.
  uint32_t extra_loss_count = 0;          ///< Number of valid entries in extra_losses.
};

/// Abstract declarative adapter for custom training losses.
///
/// The runtime (Trainer) drives the loop. The adapter declares:
/// - I/O schema and parameter layout
/// - Kernel compilation preferences (as a patch on baseline policy)
/// - Adam optimizer configuration (per-step, enabling LR schedules)
/// - Batch packing for non-standard input/target layouts
/// - Config tail packing for extension-specific kernel parameters
/// - Per-step train_params filling
/// - Post-step metric extraction
class TrainingAdapter {
public:
  virtual ~TrainingAdapter() = default;

  /// I/O schema describing dimensions and parameter layout.
  [[nodiscard]] virtual ExtensionSchema schema() const = 0;

  /// Loss function configuration. Default is L2.
  /// Called once at pipeline construction; runtime lowers to compile spec.
  [[nodiscard]] virtual LossConfig loss_config() const { return {}; }

  /// Patch baseline compile spec with adapter preferences.
  /// Called once at pipeline construction; mutate `spec` in place.
  virtual void configure_compile_spec(KernelCompileSpec &spec) const = 0;

  /// Pack extension-specific config tail floats into `dst`.
  /// Buffer size is schema().config_tail_floats floats.
  /// Called once at pipeline construction and on config reload.
  virtual void pack_config_tail(float *dst) const = 0;

  /// Pack a training batch from raw input/target arrays.
  /// Default layout is identity copy; override for DNL (N * target_dims),
  /// RMHE rotation augmentation, etc.
  /// @param input       Raw input positions (N * schema().input_dims floats).
  /// @param target      Raw target values (N * schema().target_dims floats).
  /// @param N           Number of samples.
  /// @param positions_out  Output positions (N * schema().input_dims floats).
  /// @param targets_out    Output targets (N * schema().target_dims floats).
  virtual void pack_batch(const float *input, const float *target, int N,
                          float *positions_out, float *targets_out) const = 0;

  /// Fill the train_params buffer for the given step.
  /// The layout specifies where each field lives in the buffer.
  virtual void fill_train_params(float *dst, const TrainParamsLayout &layout,
                                 uint32_t N, uint32_t step) const = 0;

  /// Adam optimizer configuration for the next step.
  /// Called per-step to allow learning rate schedules.
  [[nodiscard]] virtual AdamConfig adam_config(uint32_t next_step) const = 0;

  /// Augment post-step metrics.  Core reduction outputs (from kernel
  /// loss_partials, e.g. bc/piezo for DNL) are delivered first; the runtime
  /// appends any extra_losses returned here after the core outputs.
  [[nodiscard]] virtual ResultMetrics result_metrics(float mean_loss,
                                                     uint32_t step) const = 0;
};

} // namespace extension
} // namespace tmnn

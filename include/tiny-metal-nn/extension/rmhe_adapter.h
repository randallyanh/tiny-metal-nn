#pragma once

/**
 * @file extension/rmhe_adapter.h
 * @brief RMHEAdapter — TrainingAdapter for R-MHE (Rotated Multi-resolution
 *        Hash Encoding) training.
 *
 * SDK-level code: depends ONLY on extension headers + standard library.
 * Rotation data is injected via constructor, not coupled to any external header.
 */

#include "tiny-metal-nn/extension/training_adapter.h"

#include <array>
#include <cstdint>

namespace tmnn {
namespace extension {

class RMHEAdapter : public TrainingAdapter {
public:
  struct Config {
    float lr_encoding = 1e-2f;
    float lr_network = 1e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.99f;
    float epsilon = 1e-15f;
    float l1_reg = 0.0f;           ///< L1 hash grid regularization.
    float l2_reg = 0.0f;           ///< L2 MLP regularization.
    bool use_fp16 = true;
    bool allow_tg_weight_cache = true;
    float loss_scale = 128.0f;
    bool unsigned_mode = false;
    float grad_clip = 0.0f;
  };

  /// Rotation data injected via constructor.
  /// @param rotation_data_144 Packed rotation matrices (16 levels × 9 floats).
  RMHEAdapter(const Config &cfg, const float *rotation_data_144);

  [[nodiscard]] ExtensionSchema schema() const override;
  void configure_compile_spec(KernelCompileSpec &spec) const override;
  void pack_config_tail(float *dst) const override;
  void pack_batch(const float *input, const float *target, int N,
                  float *positions_out, float *targets_out) const override;
  void fill_train_params(float *dst, const TrainParamsLayout &layout,
                         uint32_t N, uint32_t step) const override;
  [[nodiscard]] AdamConfig adam_config(uint32_t next_step) const override;
  [[nodiscard]] ResultMetrics result_metrics(float mean_loss,
                                             uint32_t step) const override;

private:
  Config cfg_;
  std::array<float, 144> rotations_;
};

} // namespace extension
} // namespace tmnn

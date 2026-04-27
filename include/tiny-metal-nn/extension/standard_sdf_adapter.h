#pragma once

/**
 * @file extension/standard_sdf_adapter.h
 * @brief StandardSDFAdapter — TrainingAdapter for the standard 3D scalar SDF
 *        path.
 */

#include "tiny-metal-nn/extension/training_adapter.h"

#include <cstdint>

namespace tmnn::extension {

class StandardSDFAdapter : public TrainingAdapter {
public:
  struct Config {
    float lr_encoding = 1e-2f;
    float lr_network = 1e-3f;
    float lr_decay = 1.0f;
    int lr_decay_step = 0;
    float beta1 = 0.9f;
    float beta2 = 0.99f;
    float epsilon = 1e-15f;
    float l1_reg = 0.0f;
    float l2_reg = 0.0f;
    float loss_scale = 128.0f;
    bool unsigned_mode = false;
    int initial_active_levels = 16;
    int level_activation_interval = 0;
    float grad_clip = 0.0f;
    bool allow_simd = true;
    bool allow_fp16 = true;
    bool allow_tg_weight_cache = true;
  };

  explicit StandardSDFAdapter(const Config &cfg);

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
};

} // namespace tmnn::extension

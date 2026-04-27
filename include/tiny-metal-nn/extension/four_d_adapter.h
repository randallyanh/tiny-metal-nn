#pragma once

/**
 * @file extension/four_d_adapter.h
 * @brief FourDAdapter — TrainingAdapter for 4D spatiotemporal training.
 *
 * SDK-level code: depends ONLY on extension headers + standard library.
 */

#include "tiny-metal-nn/extension/training_adapter.h"

#include <algorithm>
#include <cstdint>

namespace tmnn {
namespace extension {

class FourDAdapter : public TrainingAdapter {
public:
  struct Config {
    float lr_encoding = 1e-2f;
    float lr_network = 1e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.99f;
    float epsilon = 1e-8f;
    float l1_reg = 0.0f;
    float l2_reg = 1e-6f;
    float loss_scale = 128.0f;
    bool use_fp16 = false;
    bool allow_tg_weight_cache = true;
    bool unsigned_mode = false;
    int initial_active_levels = 16;
    int level_activation_interval = 500;
    float grad_clip = 0.0f;
    int num_outputs = 1;   ///< 1 for scalar SDF, 3 for direct displacement (dx,dy,dz)
  };

  explicit FourDAdapter(const Config &cfg);

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

} // namespace extension
} // namespace tmnn

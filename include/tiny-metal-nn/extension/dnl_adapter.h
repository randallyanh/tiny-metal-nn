#pragma once

/**
 * @file extension/dnl_adapter.h
 * @brief DNLAdapter — TrainingAdapter for multi-output DNL training.
 *
 * SDK-level code: depends ONLY on extension headers + standard library.
 * Zero private runtime header deps.
 */

#include "tiny-metal-nn/extension/training_adapter.h"

#include <cstdint>

namespace tmnn {
namespace extension {

class DNLAdapter : public TrainingAdapter {
public:
  struct Config {
    int num_outputs = 4;              ///< MLP output dimensionality.
    int bc_dims = 1;                  ///< Boundary condition output dims (for loss decomposition).
    float weight_decay = 0.0f;        ///< Decoupled weight decay (AdamW).
    float loss_scale = 128.0f;        ///< FP16 loss scaling.
    bool use_fp16 = true;             ///< Allow FP16 mixed-precision.
    float lr_encoding = 1e-2f;        ///< Hash grid learning rate.
    float lr_network = 1e-3f;         ///< MLP learning rate.
    float beta1 = 0.9f;
    float beta2 = 0.99f;
    float epsilon = 1e-15f;
    float l1_reg = 0.0f;
    float l2_reg = 0.0f;
    int initial_active_levels = 16;   ///< Active hash levels at step 0.
    int level_activation_interval = 0; ///< Steps between level activation.
    float grad_clip = 0.0f;
    bool allow_tg_weight_cache = true;
  };

  explicit DNLAdapter(const Config &cfg);

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

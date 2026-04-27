#pragma once

/**
 * @file extension/multi_output_mlp_adapter.h
 * @brief MultiOutputMLPAdapter — generic 3D multi-output TrainingAdapter.
 */

#include "tiny-metal-nn/extension/training_adapter.h"

#include <cstdint>

namespace tmnn::extension {

class MultiOutputMLPAdapter : public TrainingAdapter {
public:
  struct Config {
    int num_outputs = 2;
    LossKind loss_kind = LossKind::L2;
    float huber_delta = 1.0f;
    float lr_encoding = 1e-2f;
    float lr_network = 1e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.99f;
    float epsilon = 1e-15f;
    float l1_reg = 0.0f;
    float l2_reg = 0.0f;
    float loss_scale = 128.0f;
    bool use_fp16 = true;
    int initial_active_levels = 16;
    int level_activation_interval = 0;
    float grad_clip = 0.0f;
    bool allow_tg_weight_cache = true;
  };

  explicit MultiOutputMLPAdapter(const Config &cfg);

  [[nodiscard]] ExtensionSchema schema() const override;
  [[nodiscard]] LossConfig loss_config() const override;
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

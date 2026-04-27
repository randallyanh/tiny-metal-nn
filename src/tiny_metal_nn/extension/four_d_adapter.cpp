/**
 * @file four_d_adapter.cpp
 * @brief FourDAdapter — 4D spatiotemporal training adapter.
 */

#include "tiny-metal-nn/extension/four_d_adapter.h"

#include <algorithm>
#include <cstring>

namespace tmnn {
namespace extension {

FourDAdapter::FourDAdapter(const Config &cfg) : cfg_(cfg) {}

ExtensionSchema FourDAdapter::schema() const {
  ExtensionSchema s;
  s.input_dims = 4;
  s.target_dims = static_cast<uint32_t>(cfg_.num_outputs);
  s.reduction_terms = static_cast<uint32_t>(cfg_.num_outputs);
  s.config_tail_floats = 0;
  s.train_params_layout.float_count = 5;
  return s;
}

void FourDAdapter::configure_compile_spec(KernelCompileSpec &spec) const {
  spec.allow_simd = false;   // 4D → scalar path.
  spec.allow_fp16 = cfg_.use_fp16;
  spec.allow_tg_weight_cache = cfg_.allow_tg_weight_cache;
}

void FourDAdapter::pack_config_tail(float * /*dst*/) const {
  // No config tail for 4D.
}

void FourDAdapter::pack_batch(const float *input, const float *target, int N,
                              float *positions_out,
                              float *targets_out) const {
  std::memcpy(positions_out, input,
              static_cast<size_t>(N) * 4 * sizeof(float));
  std::memcpy(targets_out, target,
              static_cast<size_t>(N) * cfg_.num_outputs * sizeof(float));
}

void FourDAdapter::fill_train_params(float *dst,
                                     const TrainParamsLayout &layout,
                                     uint32_t N, uint32_t step) const {
  int active = cfg_.initial_active_levels;
  if (cfg_.level_activation_interval > 0) {
    active = std::min(
        cfg_.initial_active_levels +
            static_cast<int>(step + 1) / cfg_.level_activation_interval,
        16);
  }
  tmnn::fill_train_params(dst, layout, N, cfg_.unsigned_mode,
                           cfg_.use_fp16 ? cfg_.loss_scale : 1.0f,
                           static_cast<uint32_t>(active));
}

AdamConfig FourDAdapter::adam_config(uint32_t /*next_step*/) const {
  AdamConfig cfg;
  cfg.lr_encoding = cfg_.lr_encoding;
  cfg.lr_network = cfg_.lr_network;
  cfg.beta1 = cfg_.beta1;
  cfg.beta2 = cfg_.beta2;
  cfg.epsilon = cfg_.epsilon;
  cfg.l1_reg = cfg_.l1_reg;
  cfg.l2_reg = cfg_.l2_reg;
  cfg.grad_clip = cfg_.grad_clip;
  cfg.weight_decay = 0.0f;
  return cfg;
}

ResultMetrics FourDAdapter::result_metrics(float /*mean_loss*/,
                                           uint32_t /*step*/) const {
  return {};
}

} // namespace extension
} // namespace tmnn

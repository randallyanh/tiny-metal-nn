/**
 * @file rmhe_adapter.cpp
 * @brief RMHEAdapter — R-MHE training adapter.
 */

#include "tiny-metal-nn/extension/rmhe_adapter.h"

#include <cstring>

namespace tmnn {
namespace extension {

RMHEAdapter::RMHEAdapter(const Config &cfg, const float *rotation_data_144)
    : cfg_(cfg) {
  std::memcpy(rotations_.data(), rotation_data_144, 144 * sizeof(float));
}

ExtensionSchema RMHEAdapter::schema() const {
  ExtensionSchema s;
  s.input_dims = 3;
  s.target_dims = 1;
  s.reduction_terms = 1;
  s.config_tail_floats = 144; // 16 levels × 9 floats (rotation matrices).
  s.train_params_layout.float_count = 4;
  return s;
}

void RMHEAdapter::configure_compile_spec(KernelCompileSpec &spec) const {
  spec.encoding = KernelEncoding::RMHE;
  spec.allow_simd = false;   // RMHE → scalar path.
  spec.allow_fp16 = cfg_.use_fp16;
  spec.allow_tg_weight_cache = cfg_.allow_tg_weight_cache;
}

void RMHEAdapter::pack_config_tail(float *dst) const {
  std::memcpy(dst, rotations_.data(), 144 * sizeof(float));
}

void RMHEAdapter::pack_batch(const float *input, const float *target, int N,
                             float *positions_out,
                             float *targets_out) const {
  std::memcpy(positions_out, input,
              static_cast<size_t>(N) * 3 * sizeof(float));
  std::memcpy(targets_out, target,
              static_cast<size_t>(N) * 1 * sizeof(float));
}

void RMHEAdapter::fill_train_params(float *dst, const TrainParamsLayout &layout,
                                    uint32_t N, uint32_t /*step*/) const {
  tmnn::fill_train_params(dst, layout, N, cfg_.unsigned_mode,
                           cfg_.use_fp16 ? cfg_.loss_scale : 1.0f,
                           /*num_active_levels=*/16);
}

AdamConfig RMHEAdapter::adam_config(uint32_t /*next_step*/) const {
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

ResultMetrics RMHEAdapter::result_metrics(float /*mean_loss*/,
                                          uint32_t /*step*/) const {
  return {};
}

} // namespace extension
} // namespace tmnn

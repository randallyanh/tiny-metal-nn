/**
 * @file standard_sdf_adapter.cpp
 * @brief StandardSDFAdapter — standard 3D scalar SDF training adapter.
 */

#include "tiny-metal-nn/extension/standard_sdf_adapter.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace tmnn::extension {
namespace {

float decayed_learning_rate(float base, float decay, int decay_step,
                            uint32_t next_step) {
  if (decay_step <= 0 || next_step == 0)
    return base;

  const uint32_t completed_steps = next_step - 1;
  const uint32_t decays =
      completed_steps / static_cast<uint32_t>(decay_step);
  return base * std::pow(decay, static_cast<float>(decays));
}

} // namespace

StandardSDFAdapter::StandardSDFAdapter(const Config &cfg) : cfg_(cfg) {}

ExtensionSchema StandardSDFAdapter::schema() const {
  return {};
}

void StandardSDFAdapter::configure_compile_spec(KernelCompileSpec &spec) const {
  spec.encoding = KernelEncoding::Standard;
  spec.allow_simd = cfg_.allow_simd;
  spec.allow_fp16 = cfg_.allow_fp16;
  spec.allow_tg_weight_cache = cfg_.allow_tg_weight_cache;
}

void StandardSDFAdapter::pack_config_tail(float * /*dst*/) const {
  // No config tail for the standard SDF path.
}

void StandardSDFAdapter::pack_batch(const float *input, const float *target,
                                    int N, float *positions_out,
                                    float *targets_out) const {
  std::memcpy(positions_out, input,
              static_cast<size_t>(N) * 3 * sizeof(float));
  std::memcpy(targets_out, target,
              static_cast<size_t>(N) * 1 * sizeof(float));
}

void StandardSDFAdapter::fill_train_params(float *dst,
                                           const TrainParamsLayout &layout,
                                           uint32_t N, uint32_t step) const {
  int active = cfg_.initial_active_levels;
  if (cfg_.level_activation_interval > 0) {
    active = std::min(
        16, cfg_.initial_active_levels +
                static_cast<int>(step + 1) / cfg_.level_activation_interval);
  }
  tmnn::fill_train_params(dst, layout, N, cfg_.unsigned_mode,
                          cfg_.allow_fp16 ? cfg_.loss_scale : 1.0f,
                          static_cast<uint32_t>(active));
}

AdamConfig StandardSDFAdapter::adam_config(uint32_t next_step) const {
  AdamConfig cfg;
  cfg.lr_encoding = decayed_learning_rate(cfg_.lr_encoding, cfg_.lr_decay,
                                          cfg_.lr_decay_step, next_step);
  cfg.lr_network = decayed_learning_rate(cfg_.lr_network, cfg_.lr_decay,
                                         cfg_.lr_decay_step, next_step);
  cfg.beta1 = cfg_.beta1;
  cfg.beta2 = cfg_.beta2;
  cfg.epsilon = cfg_.epsilon;
  cfg.l1_reg = cfg_.l1_reg;
  cfg.l2_reg = cfg_.l2_reg;
  cfg.grad_clip = cfg_.grad_clip;
  cfg.weight_decay = 0.0f;
  return cfg;
}

ResultMetrics StandardSDFAdapter::result_metrics(float /*mean_loss*/,
                                                 uint32_t /*step*/) const {
  return {};
}

} // namespace tmnn::extension

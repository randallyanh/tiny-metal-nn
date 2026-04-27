/**
 * @file multi_output_mlp_adapter.cpp
 * @brief MultiOutputMLPAdapter — generic 3D multi-output training adapter.
 */

#include "tiny-metal-nn/extension/multi_output_mlp_adapter.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace tmnn::extension {

MultiOutputMLPAdapter::MultiOutputMLPAdapter(const Config &cfg) : cfg_(cfg) {
  if (cfg_.num_outputs < 2) {
    throw std::invalid_argument(
        "MultiOutputMLPAdapter: num_outputs must be >= 2 (got " +
        std::to_string(cfg_.num_outputs) + ")");
  }
}

ExtensionSchema MultiOutputMLPAdapter::schema() const {
  ExtensionSchema s;
  s.input_dims = 3;
  s.target_dims = static_cast<uint32_t>(cfg_.num_outputs);
  s.reduction_terms = 1;
  s.config_tail_floats = 0;
  s.train_params_layout.float_count = 5;
  return s;
}

LossConfig MultiOutputMLPAdapter::loss_config() const {
  return {.kind = cfg_.loss_kind, .huber_delta = cfg_.huber_delta};
}

void MultiOutputMLPAdapter::configure_compile_spec(KernelCompileSpec &spec) const {
  spec.encoding = KernelEncoding::Standard;
  spec.allow_simd = false;
  spec.allow_fp16 = cfg_.use_fp16;
  spec.allow_tg_weight_cache = cfg_.allow_tg_weight_cache;
  spec.output_semantics = KernelOutputSemantics::Generic;
}

void MultiOutputMLPAdapter::pack_config_tail(float * /*dst*/) const {
  // No config tail for the generic multi-output path.
}

void MultiOutputMLPAdapter::pack_batch(const float *input, const float *target,
                                       int N, float *positions_out,
                                       float *targets_out) const {
  std::memcpy(positions_out, input,
              static_cast<size_t>(N) * 3 * sizeof(float));
  std::memcpy(targets_out, target,
              static_cast<size_t>(N) * static_cast<size_t>(cfg_.num_outputs) *
                  sizeof(float));
}

void MultiOutputMLPAdapter::fill_train_params(float *dst,
                                              const TrainParamsLayout &layout,
                                              uint32_t N, uint32_t step) const {
  int active = cfg_.initial_active_levels;
  if (cfg_.level_activation_interval > 0) {
    active = std::min(
        16, cfg_.initial_active_levels +
                static_cast<int>(step + 1) / cfg_.level_activation_interval);
  }
  tmnn::fill_train_params(dst, layout, N, /*unsigned_mode=*/false,
                          cfg_.use_fp16 ? cfg_.loss_scale : 1.0f,
                          static_cast<uint32_t>(active));
}

AdamConfig MultiOutputMLPAdapter::adam_config(uint32_t /*next_step*/) const {
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

ResultMetrics MultiOutputMLPAdapter::result_metrics(float /*mean_loss*/,
                                                    uint32_t /*step*/) const {
  return {};
}

} // namespace tmnn::extension

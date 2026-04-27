/**
 * @file dnl_adapter.cpp
 * @brief DNLAdapter — multi-output DNL training adapter.
 */

#include "tiny-metal-nn/extension/dnl_adapter.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace tmnn {
namespace extension {

DNLAdapter::DNLAdapter(const Config &cfg) : cfg_(cfg) {
  if (cfg_.num_outputs < 2)
    throw std::invalid_argument(
        "DNLAdapter: num_outputs must be >= 2 (got "
        + std::to_string(cfg_.num_outputs) + ")");
  if (cfg_.bc_dims < 1)
    throw std::invalid_argument(
        "DNLAdapter: bc_dims must be >= 1 (got "
        + std::to_string(cfg_.bc_dims) + ")");
  if (cfg_.bc_dims >= cfg_.num_outputs)
    throw std::invalid_argument(
        "DNLAdapter: bc_dims must be < num_outputs (bc_dims="
        + std::to_string(cfg_.bc_dims) + ", num_outputs="
        + std::to_string(cfg_.num_outputs) + ")");
}

ExtensionSchema DNLAdapter::schema() const {
  ExtensionSchema s;
  s.input_dims = 3;
  s.target_dims = static_cast<uint32_t>(cfg_.num_outputs);
  // 3 reduction terms: total loss, bc loss, piezo loss.
  s.reduction_terms = 3;
  s.config_tail_floats = 0;
  s.train_params_layout.float_count = 5;
  return s;
}

void DNLAdapter::configure_compile_spec(KernelCompileSpec &spec) const {
  spec.allow_simd = false;   // Multi-output → scalar path.
  spec.allow_fp16 = cfg_.use_fp16;
  spec.allow_tg_weight_cache = cfg_.allow_tg_weight_cache;
  spec.output_semantics = cfg_.num_outputs == 4
                              ? KernelOutputSemantics::DNL
                              : KernelOutputSemantics::Generic;
  spec.bc_dim_count = static_cast<uint32_t>(cfg_.bc_dims);
}

void DNLAdapter::pack_config_tail(float * /*dst*/) const {
  // No config tail for DNL.
}

void DNLAdapter::pack_batch(const float *input, const float *target, int N,
                            float *positions_out, float *targets_out) const {
  // Identity copy.
  std::memcpy(positions_out, input,
              static_cast<size_t>(N) * 3 * sizeof(float));
  std::memcpy(targets_out, target,
              static_cast<size_t>(N) * static_cast<size_t>(cfg_.num_outputs) *
                  sizeof(float));
}

void DNLAdapter::fill_train_params(float *dst, const TrainParamsLayout &layout,
                                   uint32_t N, uint32_t step) const {
  int active = cfg_.initial_active_levels;
  if (cfg_.level_activation_interval > 0) {
    active = std::min(
        cfg_.initial_active_levels +
            static_cast<int>(step + 1) / cfg_.level_activation_interval,
        16);
  }
  tmnn::fill_train_params(dst, layout, N, /*unsigned_mode=*/false,
                           cfg_.use_fp16 ? cfg_.loss_scale : 1.0f,
                           static_cast<uint32_t>(active));
}

AdamConfig DNLAdapter::adam_config(uint32_t /*next_step*/) const {
  AdamConfig cfg;
  cfg.lr_encoding = cfg_.lr_encoding;
  cfg.lr_network = cfg_.lr_network;
  cfg.beta1 = cfg_.beta1;
  cfg.beta2 = cfg_.beta2;
  cfg.epsilon = cfg_.epsilon;
  cfg.weight_decay = cfg_.weight_decay;
  cfg.l1_reg = cfg_.l1_reg;
  cfg.l2_reg = cfg_.l2_reg;
  cfg.grad_clip = cfg_.grad_clip;
  return cfg;
}

ResultMetrics DNLAdapter::result_metrics(float /*mean_loss*/,
                                         uint32_t /*step*/) const {
  // Diagnostic losses are delivered via schema-driven reduction_terms (3 terms).
  // The compat shim maps extra_losses[0/1] → TrainingResult.loss_bc/loss_piezo.
  // result_metrics is a pass-through for DNL.
  return {};
}

} // namespace extension
} // namespace tmnn

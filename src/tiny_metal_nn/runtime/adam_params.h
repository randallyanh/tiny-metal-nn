#pragma once

/**
 * @file adam_params.h
 * @brief Internal helpers for filling the fused Adam parameter buffer.
 */

#include "tiny-metal-nn/trainer.h"

#include "tiny_metal_nn/runtime/parameter_store.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace tmnn::detail {

inline uint32_t adam_total_param_count(const ParameterStoreDesc &desc) {
  const uint64_t total = static_cast<uint64_t>(desc.hash_grid_size) +
                         static_cast<uint64_t>(desc.mlp_weight_count);
  if (total > std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error("Adam param count exceeds uint32_t range");
  }
  return static_cast<uint32_t>(total);
}

inline uint32_t decode_split_u16_u32(float low, float high) {
  return static_cast<uint32_t>(low) |
         (static_cast<uint32_t>(high) << 16u);
}

inline void fill_unified_adam_params(float *params, const TrainerConfig &cfg,
                                     const ParameterStoreDesc &desc,
                                     uint32_t logical_step,
                                     float grad_clip = 0.0f,
                                     float weight_decay = 0.0f) {
  const float s = static_cast<float>(logical_step + 1);
  params[0] = cfg.lr_encoding;
  params[1] = cfg.lr_network;
  params[2] = cfg.beta1;
  params[3] = cfg.beta2;
  params[4] = cfg.epsilon;
  params[5] = 1.0f - std::pow(cfg.beta1, s);
  params[6] = 1.0f - std::pow(cfg.beta2, s);
  params[7] = cfg.l1_reg;
  params[8] = cfg.l2_reg;
  params[9] = static_cast<float>(desc.hash_grid_size & 0xFFFFu);
  params[10] = static_cast<float>(desc.hash_grid_size >> 16u);
  params[11] = grad_clip;
  params[12] = weight_decay;
}

} // namespace tmnn::detail

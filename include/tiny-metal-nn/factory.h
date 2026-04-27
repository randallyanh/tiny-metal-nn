#pragma once

/**
 * @file factory.h
 * @brief Typed factory functions for tmnn core types.
 */

#include "tiny-metal-nn/detail/adam.h"
#include "tiny-metal-nn/detail/cosine_loss.h"
#include "tiny-metal-nn/detail/huber_loss.h"
#include "tiny-metal-nn/detail/l1_loss.h"
#include "tiny-metal-nn/fully_fused_mlp.h"
#include "tiny-metal-nn/hash_grid.h"
#include "tiny-metal-nn/detail/l2_loss.h"
#include "tiny-metal-nn/network_with_input_encoding.h"
#include "tiny-metal-nn/rotated_hash_grid.h"

#include <memory>
#include <utility>

namespace tmnn {

inline std::shared_ptr<HashGridEncoding>
create_encoding(const HashGridEncoding::Config &cfg) {
  return std::make_shared<HashGridEncoding>(cfg);
}

inline std::shared_ptr<HashGridEncoding> create_encoding() {
  return std::make_shared<HashGridEncoding>();
}

inline std::shared_ptr<RotatedHashGridEncoding>
create_rotated_encoding(const RotatedHashGridEncoding::Config &cfg) {
  return std::make_shared<RotatedHashGridEncoding>(cfg);
}

inline std::shared_ptr<RotatedHashGridEncoding> create_rotated_encoding() {
  return std::make_shared<RotatedHashGridEncoding>();
}

inline std::shared_ptr<FullyFusedMLP>
create_network(const FullyFusedMLP::Config &cfg) {
  return std::make_shared<FullyFusedMLP>(cfg);
}

inline std::shared_ptr<FullyFusedMLP> create_network() {
  return std::make_shared<FullyFusedMLP>();
}

inline std::shared_ptr<NetworkWithInputEncoding>
create_network_with_input_encoding(std::shared_ptr<Encoding> encoding,
                                   std::shared_ptr<Network> network) {
  return std::make_shared<NetworkWithInputEncoding>(std::move(encoding),
                                                    std::move(network));
}

inline std::shared_ptr<L2Loss> create_loss_l2() {
  return std::make_shared<L2Loss>();
}

inline std::shared_ptr<L1Loss> create_loss_l1() {
  return std::make_shared<L1Loss>();
}

inline std::shared_ptr<HuberLoss> create_loss_huber(float delta = 1.0f) {
  return std::make_shared<HuberLoss>(delta);
}

inline std::shared_ptr<CosineLoss> create_loss_cosine(uint32_t output_dims = 0,
                                                      float epsilon = 1e-8f) {
  return std::make_shared<CosineLoss>(output_dims, epsilon);
}

inline std::shared_ptr<Adam>
create_optimizer_adam(const Adam::Config &cfg) {
  return std::make_shared<Adam>(cfg);
}

inline std::shared_ptr<Adam> create_optimizer_adam() {
  return std::make_shared<Adam>();
}

} // namespace tmnn

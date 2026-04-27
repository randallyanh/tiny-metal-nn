#pragma once

/**
 * @file fully_fused_mlp.h
 * @brief FullyFusedMLP — width-aligned fully-fused MLP network.
 *
 * n_params() returns the total MLP weight count for a given config.
 */

#include "tiny-metal-nn/network.h"

#include <string>

namespace tmnn {

class FullyFusedMLP final : public Network {
public:
  struct Config {
    int hidden_dim = 64;
    int num_hidden_layers = 2;
    int n_input = 32;   // num_levels * features_per_level
    int n_output = 1;   // 1 for SDF, 4 for DNL
  };

  FullyFusedMLP() = default;
  explicit FullyFusedMLP(const Config &cfg) : config_(cfg) {}

  [[nodiscard]] int n_input_dims() const override { return config_.n_input; }
  [[nodiscard]] int n_output_dims() const override { return config_.n_output; }

  /// Total MLP parameter count: matches KernelSpec::mlpWeightCount().
  [[nodiscard]] int n_params() const override {
    int hd = config_.hidden_dim;
    // Layer 0: input -> hidden (W0 + b0)
    int count = config_.n_input * hd + hd;
    // Hidden layers 1..N-1: hidden -> hidden (W + b)
    for (int i = 1; i < config_.num_hidden_layers; ++i)
      count += hd * hd + hd;
    // Output layer: hidden -> output (W_out + b_out)
    count += hd * config_.n_output + config_.n_output;
    return count;
  }

  [[nodiscard]] std::string name() const override { return "FullyFusedMLP"; }
  [[nodiscard]] const Config &config() const { return config_; }

private:
  Config config_;
};

} // namespace tmnn

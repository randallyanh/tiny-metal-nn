#pragma once

/**
 * @file hash_grid.h
 * @brief HashGridEncoding — multi-resolution hash grid encoding (tcnn-style).
 *
 * Pure core header with no external dependencies.
 */

#include "tiny-metal-nn/encoding.h"

#include <cstddef>
#include <string>

namespace tmnn {

class HashGridEncoding final : public Encoding {
public:
  struct Config {
    int num_levels = 16;
    int features_per_level = 2;
    int log2_hashmap_size = 19;
    float base_resolution = 16.0f;
    float per_level_scale = 1.447f;
    int input_dims = 3;
  };

  HashGridEncoding() = default;
  explicit HashGridEncoding(const Config &cfg) : config_(cfg) {}

  [[nodiscard]] int n_input_dims() const override {
    return config_.input_dims;
  }

  [[nodiscard]] int n_output_dims() const override {
    return config_.num_levels * config_.features_per_level;
  }

  [[nodiscard]] int n_params() const override {
    size_t table_size = static_cast<size_t>(1) << config_.log2_hashmap_size;
    return static_cast<int>(static_cast<size_t>(config_.num_levels) *
                            table_size *
                            static_cast<size_t>(config_.features_per_level));
  }

  [[nodiscard]] std::string name() const override {
    return "HashGridEncoding";
  }

  [[nodiscard]] const Config &config() const { return config_; }

private:
  Config config_;
};

} // namespace tmnn

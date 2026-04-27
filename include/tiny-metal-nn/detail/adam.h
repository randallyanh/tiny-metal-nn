#pragma once

/**
 * @file adam.h
 * @brief Adam optimizer descriptor — pure config holder, no step state.
 *
 * Step count is runtime-owned (ITrainerRuntime::step()).
 */

#include "tiny-metal-nn/optimizer.h"

#include <string>

namespace tmnn {

class Adam final : public Optimizer {
public:
  struct Config {
    float lr = 1e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.99f;
    float epsilon = 1e-15f;
  };

  Adam() = default;
  explicit Adam(const Config &cfg) : config_(cfg) {}

  [[nodiscard]] std::string name() const override { return "Adam"; }
  [[nodiscard]] float learning_rate() const override { return config_.lr; }

  void set_learning_rate(float lr) override { config_.lr = lr; }

  [[nodiscard]] const Config &config() const { return config_; }

private:
  Config config_;
};

} // namespace tmnn

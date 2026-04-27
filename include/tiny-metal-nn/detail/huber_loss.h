#pragma once

/**
 * @file huber_loss.h
 * @brief HuberLoss — smooth L1 loss with CPU evaluation.
 */

#include "tiny-metal-nn/loss.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace tmnn {

class HuberLoss final : public Loss {
public:
  explicit HuberLoss(float delta = 1.0f) : delta_(delta) {
    if (delta_ <= 0.0f) {
      throw std::invalid_argument("HuberLoss: delta must be positive");
    }
  }

  [[nodiscard]] std::string name() const override { return "Huber"; }

  [[nodiscard]] float evaluate_cpu(const float *predicted, const float *target,
                                   int N) const override {
    if (N <= 0)
      return 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
      const float residual = predicted[i] - target[i];
      const float abs_residual = std::fabs(residual);
      if (abs_residual <= delta_) {
        sum += 0.5f * residual * residual;
      } else {
        sum += delta_ * (abs_residual - 0.5f * delta_);
      }
    }
    return sum / static_cast<float>(N);
  }

  [[nodiscard]] float delta() const { return delta_; }

private:
  float delta_ = 1.0f;
};

} // namespace tmnn

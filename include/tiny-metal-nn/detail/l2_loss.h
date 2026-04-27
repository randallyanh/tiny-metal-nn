#pragma once

/**
 * @file l2_loss.h
 * @brief L2Loss — mean squared error loss with CPU evaluation.
 */

#include "tiny-metal-nn/loss.h"

#include <string>

namespace tmnn {

class L2Loss final : public Loss {
public:
  [[nodiscard]] std::string name() const override { return "L2"; }

  [[nodiscard]] float evaluate_cpu(const float *predicted, const float *target,
                                   int N) const override {
    if (N <= 0)
      return 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
      float diff = predicted[i] - target[i];
      sum += diff * diff;
    }
    return sum / static_cast<float>(N);
  }
};

} // namespace tmnn

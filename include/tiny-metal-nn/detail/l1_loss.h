#pragma once

/**
 * @file l1_loss.h
 * @brief L1Loss — mean absolute error loss with CPU evaluation.
 */

#include "tiny-metal-nn/loss.h"

#include <cmath>
#include <string>

namespace tmnn {

class L1Loss final : public Loss {
public:
  [[nodiscard]] std::string name() const override { return "L1"; }

  [[nodiscard]] float evaluate_cpu(const float *predicted, const float *target,
                                   int N) const override {
    if (N <= 0)
      return 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
      sum += std::fabs(predicted[i] - target[i]);
    }
    return sum / static_cast<float>(N);
  }
};

} // namespace tmnn

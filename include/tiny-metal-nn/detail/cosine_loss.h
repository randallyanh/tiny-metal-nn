#pragma once

/**
 * @file cosine_loss.h
 * @brief CosineLoss — cosine-distance loss with CPU evaluation.
 */

#include "tiny-metal-nn/loss.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace tmnn {

class CosineLoss final : public Loss {
public:
  explicit CosineLoss(uint32_t output_dims = 0, float epsilon = 1e-8f)
      : output_dims_(output_dims), epsilon_(epsilon) {
    if (output_dims_ == 1u) {
      throw std::invalid_argument(
          "CosineLoss: output_dims must be 0 or >= 2");
    }
    if (epsilon_ <= 0.0f) {
      throw std::invalid_argument("CosineLoss: epsilon must be positive");
    }
  }

  [[nodiscard]] std::string name() const override { return "Cosine"; }

  [[nodiscard]] float evaluate_cpu(const float *predicted, const float *target,
                                   int N) const override {
    if (N <= 0)
      return 0.0f;
    if (output_dims_ == 0u) {
      return segment_loss(predicted, target, N);
    }
    if ((N % static_cast<int>(output_dims_)) != 0) {
      throw std::invalid_argument(
          "CosineLoss: N must be divisible by output_dims");
    }

    const int sample_count = N / static_cast<int>(output_dims_);
    float sum = 0.0f;
    for (int sample = 0; sample < sample_count; ++sample) {
      const int offset = sample * static_cast<int>(output_dims_);
      sum +=
          segment_loss(predicted + offset, target + offset,
                       static_cast<int>(output_dims_));
    }
    return sum / static_cast<float>(sample_count);
  }

  [[nodiscard]] uint32_t output_dims() const { return output_dims_; }
  [[nodiscard]] float epsilon() const { return epsilon_; }

private:
  [[nodiscard]] float segment_loss(const float *predicted, const float *target,
                                   int dims) const {
    float dot = 0.0f;
    float pred_norm_sq = 0.0f;
    float target_norm_sq = 0.0f;
    for (int i = 0; i < dims; ++i) {
      const float pred = predicted[i];
      const float tgt = target[i];
      dot += pred * tgt;
      pred_norm_sq += pred * pred;
      target_norm_sq += tgt * tgt;
    }
    const float pred_norm = std::sqrt(pred_norm_sq + epsilon_);
    const float target_norm = std::sqrt(target_norm_sq + epsilon_);
    return 1.0f - dot / (pred_norm * target_norm);
  }

  uint32_t output_dims_ = 0;
  float epsilon_ = 1e-8f;
};

} // namespace tmnn

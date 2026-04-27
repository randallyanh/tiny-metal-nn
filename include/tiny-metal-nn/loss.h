#pragma once

/**
 * @file loss.h
 * @brief Loss base class — named loss function with CPU evaluation.
 */

#include <string>

namespace tmnn {

class Loss {
public:
  virtual ~Loss() = default;

  [[nodiscard]] virtual std::string name() const = 0;

  /// Evaluate loss on CPU (for validation/testing).
  /// @param predicted  Predicted values (N floats).
  /// @param target     Ground truth values (N floats).
  /// @param N          Number of samples.
  /// @return Mean loss value.
  [[nodiscard]] virtual float evaluate_cpu(const float *predicted,
                                           const float *target, int N) const = 0;
};

} // namespace tmnn

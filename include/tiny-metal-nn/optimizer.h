#pragma once

/**
 * @file optimizer.h
 * @brief Optimizer base class — named optimizer with learning rate control.
 *
 * Step count is runtime-owned; optimizer is a pure config holder.
 */

#include <string>

namespace tmnn {

class Optimizer {
public:
  virtual ~Optimizer() = default;

  [[nodiscard]] virtual std::string name() const = 0;
  [[nodiscard]] virtual float learning_rate() const = 0;
  virtual void set_learning_rate(float lr) = 0;
};

} // namespace tmnn

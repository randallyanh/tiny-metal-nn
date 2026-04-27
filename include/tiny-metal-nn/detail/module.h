#pragma once

/**
 * @file module.h
 * @brief Module base class — describes a parameterized compute unit.
 *
 * Descriptive only (n_params, shape). No forward/backward dispatch.
 */

#include <string>

namespace tmnn {

class Module {
public:
  virtual ~Module() = default;

  [[nodiscard]] virtual int n_input_dims() const = 0;
  [[nodiscard]] virtual int n_output_dims() const = 0;
  [[nodiscard]] virtual int n_params() const = 0;
  [[nodiscard]] virtual std::string name() const = 0;

  /// Initialize parameters (Phase C will use this for GPU buffer init).
  virtual void initialize_params() {}
};

} // namespace tmnn

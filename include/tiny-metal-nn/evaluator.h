#pragma once

/**
 * @file evaluator.h
 * @brief FieldEvaluator — core public API for consuming trained neural fields.
 *
 * Pure abstract contract for meshing, metrics, and deformation consumers.
 * Depends only on <cstdint>; no Metal, no runtime internals.
 *
 * Gradient buffer layout is row-major Jacobian:
 *   gradients[(sample * n_output_dims() + output_dim) * n_input_dims() + input_dim]
 */

#include "tiny-metal-nn/common.h"

#include <cstdint>
#include <optional>

namespace tmnn {

/// Abstract evaluator for trained neural fields.
class FieldEvaluator {
public:
  virtual ~FieldEvaluator() = default;

  /// Number of input dimensions (e.g. 3 for spatial, 4 for spatiotemporal).
  [[nodiscard]] virtual uint32_t n_input_dims() const = 0;

  /// Number of output dimensions (e.g. 1 for SDF, 3 for RGB).
  [[nodiscard]] virtual uint32_t n_output_dims() const = 0;

  /// Evaluate the field at N positions.
  /// @param positions  Input buffer of N * n_input_dims() floats.
  /// @param output     Output buffer of N * n_output_dims() floats.
  /// @param N          Number of samples.
  /// @return true on success. On failure, consult last_diagnostic().
  virtual bool evaluate(const float *positions, float *output, int N) = 0;

  /// Evaluate with analytical gradients (row-major Jacobian).
  /// @param positions  Input buffer of N * n_input_dims() floats.
  /// @param output     Output buffer of N * n_output_dims() floats.
  /// @param gradients  Output buffer of N * n_output_dims() * n_input_dims() floats.
  /// @param N          Number of samples.
  /// @return true on success. On failure, consult last_diagnostic().
  virtual bool evaluate_with_gradient(const float *positions, float *output,
                                      float *gradients, int N) = 0;

  /// Structured diagnostic for the most recent non-throwing failure.
  [[nodiscard]] virtual std::optional<DiagnosticInfo>
  last_diagnostic() const {
    return std::nullopt;
  }

  /// Clear any stored diagnostic state. Successful calls should normally leave
  /// this empty; explicit clearing is mainly for wrappers/adapters.
  virtual void clear_diagnostic() {}
};

} // namespace tmnn

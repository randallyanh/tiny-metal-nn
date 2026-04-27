#pragma once

/**
 * @file common.h
 * @brief Core types for the tmnn (Tiny Metal Neural Network) namespace.
 */

#include "tiny-metal-nn/diagnostics.h"
#include "tiny-metal-nn/detail/runtime_stats.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace tmnn {

/// Precision for compute kernels.
enum class Precision { F16, F32 };

/// Traits mapping C++ types to Precision enum values.
template <typename T> struct PrecisionTraits;
template <> struct PrecisionTraits<float> {
  static constexpr Precision value = Precision::F32;
};
template <> struct PrecisionTraits<uint16_t> {
  static constexpr Precision value = Precision::F16;
};

/// N-dimensional tensor shape (up to 4D).
struct TensorShape {
  int dims[4] = {0, 0, 0, 0};
  int rank = 0;

  static TensorShape make_1d(int n) { return {{n, 0, 0, 0}, 1}; }
  static TensorShape make_2d(int r, int c) { return {{r, c, 0, 0}, 2}; }

  [[nodiscard]] size_t numel() const {
    size_t n = 1;
    for (int i = 0; i < rank; ++i)
      n *= static_cast<size_t>(dims[i]);
    return n;
  }
};

/// Unowned view of a contiguous buffer with shape metadata.
struct TensorRef {
  void *data = nullptr;
  TensorShape shape;
  Precision precision = Precision::F32;

  template <typename T> T *typed_data() {
    assert(precision == PrecisionTraits<T>::value &&
           "TensorRef precision mismatch");
    return static_cast<T *>(data);
  }

  template <typename T> const T *typed_data() const {
    assert(precision == PrecisionTraits<T>::value &&
           "TensorRef precision mismatch");
    return static_cast<const T *>(data);
  }
};

/// Opaque handle for execution context (Phase C placeholder).
struct ContextHandle {
  uint64_t generation = 0;
  explicit operator bool() const { return generation != 0; }
};

enum class BadStepRecoveryAction : uint32_t {
  None = 0,
  Skipped = 1,
  RolledBack = 2,
  RetriedWithSafeFamily = 3,
};

/// Per-step probe telemetry (opt-in, enable_probes=true).
struct ProbeResult {
  static constexpr uint32_t kMaxLayers = 8;
  uint32_t num_hidden_layers = 0;
  bool has_nan_forward = false;
  bool has_nan_backward = false;
  float hash_grad_l2 = 0.0f;
  float act_max[kMaxLayers] = {};      ///< Per hidden layer activation max.
  float grad_l2[kMaxLayers] = {};      ///< Per hidden layer gradient L2.
  float output_abs_max = 0.0f;
  float output_min = 0.0f;

  /// Probe stride per threadgroup = 2*num_hidden_layers + 5.
  [[nodiscard]] static uint32_t stride_for_layers(uint32_t nhl) {
    return 2 * nhl + 5;
  }
};

/// Result from a single training step.
struct TrainingStepResult {
  float loss = 0.0f;
  float loss_reg = 0.0f;
  float grad_norm = 0.0f;
  NumericsReport numerics;
  bool numerics_reported = false;
  bool has_numerics_anomaly = false;
  BadStepRecoveryAction recovery_action = BadStepRecoveryAction::None;
  uint32_t step = 0;
  static constexpr uint32_t kMaxExtraLosses = 4;
  float extra_losses[kMaxExtraLosses] = {};
  uint32_t extra_loss_count = 0;
  std::optional<ProbeResult> probe; ///< Non-null when enable_probes=true.
};

/// Result from inference.
struct InferenceResult {
  float elapsed_ms = 0.0f;
  int num_points = 0;
};

} // namespace tmnn

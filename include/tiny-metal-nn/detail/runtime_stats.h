#pragma once

/**
 * @file runtime_stats.h
 * @brief Runtime telemetry and numerics report structures.
 */

#include <cstddef>
#include <cstdint>

namespace tmnn {

/// Cumulative runtime telemetry counters.
struct RuntimeStats {
  uint64_t pipeline_cache_hits = 0;
  uint64_t pipeline_cache_misses = 0;
  uint64_t archive_hits = 0;
  uint64_t archive_misses = 0;
  uint64_t compile_count = 0;
  uint64_t async_batches_submitted = 0;
  uint64_t training_steps_completed = 0;
  uint64_t numerics_report_count = 0;
  uint64_t numerics_anomaly_count = 0;
  uint64_t bad_steps_skipped = 0;
  uint64_t bad_steps_rolled_back = 0;
  uint64_t safe_family_recoveries = 0;
  size_t persistent_bytes = 0;
  size_t transient_peak_bytes = 0;
};

/// Per-step numerics health report.
struct NumericsReport {
  bool finite_forward = true;
  bool finite_backward = true;
  bool finite_update = true;
  float max_abs_activation = 0.0f;
  float max_abs_gradient = 0.0f;
  float max_abs_update = 0.0f;
  uint32_t first_bad_layer = 0;
};

} // namespace tmnn

#pragma once

/**
 * @file numerics_guard.h
 * @brief NumericsGuard — finite/overflow detection + sampling policy (internal).
 *
 * Three-tier sampling strategy:
 * - **always-on**: host-side schema/config sanity, CPU counters
 * - **sampled GPU numerics**: first step, every 128 steps, on anomaly
 * - **full per-step**: debug/benchmark/explicit opt-in only
 */

#include "tiny-metal-nn/runtime_policy.h"
#include "tiny-metal-nn/detail/runtime_stats.h"

#include <cstdint>

namespace tmnn {

class NumericsGuard {
public:
  /// Default sampling interval (steps between GPU readback).
  static constexpr uint32_t kDefaultSampleInterval = 128;

  explicit NumericsGuard(
      NumericsSamplingMode mode = NumericsSamplingMode::Sampled);

  /// Should this step's numerics be sampled from GPU?
  [[nodiscard]] bool should_sample(uint32_t step, bool force = false) const;

  /// Record a numerics report for the given step.
  void record_step(uint32_t step, const NumericsReport &report);

  /// Latest recorded report.
  [[nodiscard]] const NumericsReport &latest_report() const {
    return latest_;
  }

  /// Total number of anomalies detected (non-finite forward/backward/update).
  [[nodiscard]] uint32_t anomaly_count() const { return anomaly_count_; }

  /// Total number of reports recorded.
  [[nodiscard]] uint32_t report_count() const { return report_count_; }

  /// The step of the most recent anomaly (0 if none).
  [[nodiscard]] uint32_t last_anomaly_step() const {
    return last_anomaly_step_;
  }

  /// Current sampling mode.
  [[nodiscard]] NumericsSamplingMode mode() const { return mode_; }

  /// Override sampling interval (default 128).
  void set_sample_interval(uint32_t interval) {
    sample_interval_ = interval > 0 ? interval : 1;
  }

private:
  NumericsSamplingMode mode_;
  uint32_t sample_interval_ = kDefaultSampleInterval;
  NumericsReport latest_;
  uint32_t anomaly_count_ = 0;
  uint32_t report_count_ = 0;
  uint32_t last_anomaly_step_ = 0;
};

} // namespace tmnn

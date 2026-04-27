/**
 * @file numerics_guard.cpp
 * @brief NumericsGuard implementation.
 */

#include "tiny_metal_nn/runtime/numerics_guard.h"

namespace tmnn {

NumericsGuard::NumericsGuard(NumericsSamplingMode mode) : mode_(mode) {}

bool NumericsGuard::should_sample(uint32_t step, bool force) const {
  if (force)
    return true;

  switch (mode_) {
  case NumericsSamplingMode::Disabled:
    return false;
  case NumericsSamplingMode::FullPerStep:
    return true;
  case NumericsSamplingMode::Periodic:
    if (anomaly_count_ > 0 && step == last_anomaly_step_ + 1)
      return true;
    return step != 0 && (step % sample_interval_) == 0;
  case NumericsSamplingMode::Sampled:
    // First two bootstrap steps are always sampled.
    if (step == 0 || step == 1)
      return true;
    // Sample after anomaly (next step after last anomaly).
    if (anomaly_count_ > 0 && step == last_anomaly_step_ + 1)
      return true;
    // Regular interval.
    return (step % sample_interval_) == 0;
  }
  return false;
}

void NumericsGuard::record_step(uint32_t step, const NumericsReport &report) {
  latest_ = report;
  ++report_count_;

  bool anomaly =
      !report.finite_forward || !report.finite_backward || !report.finite_update;
  if (anomaly) {
    ++anomaly_count_;
    last_anomaly_step_ = step;
  }
}

} // namespace tmnn

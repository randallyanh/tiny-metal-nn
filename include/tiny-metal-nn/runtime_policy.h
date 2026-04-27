#pragma once

/**
 * @file runtime_policy.h
 * @brief Resolved runtime policy snapshot and API override knobs.
 */

#include <cstdint>
#include <optional>

namespace tmnn {

enum class NumericsSamplingMode : uint32_t {
  Sampled = 0,
  Periodic = 1,
  FullPerStep = 2,
  Disabled = 3,
};

enum class BadStepRecoveryMode : uint32_t {
  SignalOnly = 0,
  Throw = 1,
  Skip = 2,
  Rollback = 3,
  FallbackAndRetryWithSafeFamily = 4,
};

/// Resolved low-level runtime policy frozen at MetalContext creation time.
struct RuntimePolicy {
  bool allow_cpu_fallback = false;
  bool force_precise_math = false;
  bool disable_binary_archive = false;
  bool disable_private_buffers = false;
  bool emit_runtime_stats = false;
  NumericsSamplingMode numerics_sampling_mode =
      NumericsSamplingMode::Sampled;
  uint32_t numerics_sample_interval = 128u;
  BadStepRecoveryMode bad_step_recovery = BadStepRecoveryMode::SignalOnly;
};

/// Optional API overrides applied on top of env/default policy resolution.
struct RuntimePolicyOverrides {
  std::optional<bool> allow_cpu_fallback;
  std::optional<bool> force_precise_math;
  std::optional<bool> disable_binary_archive;
  std::optional<bool> disable_private_buffers;
  std::optional<bool> emit_runtime_stats;
  std::optional<NumericsSamplingMode> numerics_sampling_mode;
  std::optional<uint32_t> numerics_sample_interval;
  std::optional<BadStepRecoveryMode> bad_step_recovery;
};

} // namespace tmnn

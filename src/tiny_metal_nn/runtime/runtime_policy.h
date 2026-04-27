#pragma once

/**
 * @file runtime_policy.h
 * @brief Structured runtime policy for tmnn NN runtime (internal).
 *
 * Priority: API parameters > environment variables > defaults.
 */

#include "tiny-metal-nn/runtime_policy.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace tmnn {

namespace detail {

inline bool icmp(const char *a, const char *b) {
  for (; *a && *b; ++a, ++b) {
    if (std::tolower(static_cast<unsigned char>(*a)) !=
        std::tolower(static_cast<unsigned char>(*b)))
      return false;
  }
  return *a == '\0' && *b == '\0';
}

inline const char *trimmed_env_value(const char *name) {
  const char *val = std::getenv(name);
  if (!val)
    return nullptr;
  while (*val && std::isspace(static_cast<unsigned char>(*val)))
    ++val;
  if (*val == '\0')
    return nullptr;
  return val;
}

/// Case-insensitive truthy env check (matches tmnn::gpu::isTruthyEnv).
inline bool is_truthy_env(const char *name) {
  const char *val = trimmed_env_value(name);
  if (!val)
    return false;
  if (std::strcmp(val, "0") == 0)
    return false;
  if (icmp(val, "false") || icmp(val, "off") || icmp(val, "no"))
    return false;
  return true;
}

inline void apply_override(bool &field, const std::optional<bool> &override) {
  if (override.has_value())
    field = *override;
}

inline void apply_override(BadStepRecoveryMode &field,
                           const std::optional<BadStepRecoveryMode> &override) {
  if (override.has_value())
    field = *override;
}

inline void apply_override(NumericsSamplingMode &field,
                           const std::optional<NumericsSamplingMode> &override) {
  if (override.has_value())
    field = *override;
}

inline void apply_override(uint32_t &field,
                           const std::optional<uint32_t> &override) {
  if (override.has_value())
    field = *override > 0 ? *override : 1u;
}

inline std::optional<BadStepRecoveryMode>
bad_step_mode_from_env(const char *name) {
  const char *val = trimmed_env_value(name);
  if (!val)
    return std::nullopt;

  if (icmp(val, "signal") || icmp(val, "signal-only"))
    return BadStepRecoveryMode::SignalOnly;
  if (icmp(val, "throw") || icmp(val, "strict"))
    return BadStepRecoveryMode::Throw;
  if (icmp(val, "skip"))
    return BadStepRecoveryMode::Skip;
  if (icmp(val, "rollback") || icmp(val, "revert"))
    return BadStepRecoveryMode::Rollback;
  if (icmp(val, "fallback") || icmp(val, "fallback-and-retry") ||
      icmp(val, "fallback-and-retry-with-safe-family") ||
      icmp(val, "safe-family-fallback")) {
    return BadStepRecoveryMode::FallbackAndRetryWithSafeFamily;
  }

  throw std::invalid_argument(std::string("TMNN_BAD_STEP_MODE has unsupported value '")
                              + val + "'");
}

inline std::optional<NumericsSamplingMode>
numerics_sampling_mode_from_env(const char *name) {
  const char *val = trimmed_env_value(name);
  if (!val)
    return std::nullopt;

  if (icmp(val, "sampled") || icmp(val, "default"))
    return NumericsSamplingMode::Sampled;
  if (icmp(val, "periodic") || icmp(val, "steady-state") ||
      icmp(val, "interval-only")) {
    return NumericsSamplingMode::Periodic;
  }
  if (icmp(val, "full") || icmp(val, "full-per-step"))
    return NumericsSamplingMode::FullPerStep;
  if (icmp(val, "disabled") || icmp(val, "off"))
    return NumericsSamplingMode::Disabled;

  throw std::invalid_argument(
      std::string("TMNN_NUMERICS_SAMPLING_MODE has unsupported value '") + val +
      "'");
}

inline std::optional<uint32_t> uint32_from_env(const char *name) {
  const char *val = trimmed_env_value(name);
  if (!val)
    return std::nullopt;

  char *end = nullptr;
  const unsigned long parsed = std::strtoul(val, &end, 10);
  if (end == val || *end != '\0' ||
      parsed > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument(std::string(name) +
                                " has unsupported value '" + val + "'");
  }
  return static_cast<uint32_t>(parsed > 0 ? parsed : 1ul);
}

} // namespace detail

/// Load RuntimePolicy from environment variables, then overlay API overrides.
inline RuntimePolicy load_runtime_policy(
    const RuntimePolicyOverrides &overrides = {}) {
  RuntimePolicy p;
  p.allow_cpu_fallback =
      detail::is_truthy_env("TMNN_ALLOW_CPU_FALLBACK");
  p.force_precise_math =
      detail::is_truthy_env("TMNN_FORCE_PRECISE_MATH");
  p.disable_binary_archive =
      detail::is_truthy_env("TMNN_DISABLE_BINARY_ARCHIVE");
  p.disable_private_buffers =
      detail::is_truthy_env("TMNN_DISABLE_PRIVATE_BUFFERS");
  p.emit_runtime_stats =
      detail::is_truthy_env("TMNN_EMIT_RUNTIME_STATS");
  p.numerics_sampling_mode =
      detail::numerics_sampling_mode_from_env("TMNN_NUMERICS_SAMPLING_MODE")
          .value_or(NumericsSamplingMode::Sampled);
  p.numerics_sample_interval =
      detail::uint32_from_env("TMNN_NUMERICS_SAMPLE_INTERVAL").value_or(128u);
  p.bad_step_recovery = detail::bad_step_mode_from_env("TMNN_BAD_STEP_MODE")
                            .value_or(BadStepRecoveryMode::SignalOnly);
  detail::apply_override(p.allow_cpu_fallback, overrides.allow_cpu_fallback);
  detail::apply_override(p.force_precise_math, overrides.force_precise_math);
  detail::apply_override(p.disable_binary_archive,
                         overrides.disable_binary_archive);
  detail::apply_override(p.disable_private_buffers,
                         overrides.disable_private_buffers);
  detail::apply_override(p.emit_runtime_stats, overrides.emit_runtime_stats);
  detail::apply_override(p.numerics_sampling_mode,
                         overrides.numerics_sampling_mode);
  detail::apply_override(p.numerics_sample_interval,
                         overrides.numerics_sample_interval);
  detail::apply_override(p.bad_step_recovery, overrides.bad_step_recovery);
  return p;
}

} // namespace tmnn

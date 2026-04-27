#pragma once

/**
 * @file autotune_manifest.h
 * @brief Persisted autotune/prewarm decision state for tmnn.
 */

#include "tiny-metal-nn/detail/network_planning.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace tmnn {

inline constexpr std::string_view kAutotuneManifestVersion =
    "tmnn-autotune-v1";

struct AutotuneCandidateMeasurement {
  NetworkFamily family = NetworkFamily::SafeDebugMetal;
  uint64_t warmup_step_ns = 0;
  uint64_t measured_step_ns = 0;
  bool succeeded = false;
  std::string failure;
  uint64_t build_ns = 0;
  uint64_t objective_ns = 0;
};

struct AutotuneManifestEntry {
  uint64_t planner_fingerprint = 0;
  NetworkFamily selected_family = NetworkFamily::SafeDebugMetal;
  std::vector<NetworkFamily> candidate_families;
  std::vector<PlannerFallbackReason> reasons;
  bool family_was_forced = false;
  bool selected_by_measurement = false;
  uint32_t measurement_batch_size = 0;
  uint32_t measurement_warmup_steps = 2;
  uint32_t measurement_steps = 0;
  uint64_t selected_family_measured_step_ns = 0;
  std::vector<AutotuneCandidateMeasurement> measurements;
  AutotuneSearchObjective measurement_objective =
      AutotuneSearchObjective::SteadyStateStep;
  uint64_t selected_family_objective_ns = 0;
};

struct AutotuneManifest {
  std::string version = std::string(kAutotuneManifestVersion);
  std::string device_name;
  uint32_t device_family = 0;
  std::vector<AutotuneManifestEntry> entries;
};

[[nodiscard]] AutotuneManifest load_autotune_manifest(const std::string &path);

void save_autotune_manifest(const std::string &path,
                            const AutotuneManifest &manifest);

} // namespace tmnn

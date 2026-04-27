#pragma once

/**
 * @file network_planning.h
 * @brief Planner vocabulary for tmnn network family selection.
 */

#include "tiny-metal-nn/device_capabilities.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace tmnn {

class MetalContext;
class Encoding;
class Network;
class NetworkWithInputEncoding;

enum class ExecutionBackendPreference : uint32_t {
  RequireMetal = 0,
  PreferMetal = 1,
  RequireCPUReference = 2,
};

enum class NetworkFamily : uint32_t {
  FullyFusedMetal = 0,
  TiledMetal = 1,
  SafeDebugMetal = 2,
};

enum class MultiOutputSemanticProfile : uint32_t {
  Generic = 0,
  DNL = 1,
};

enum class AutotuneSearchObjective : uint32_t {
  SteadyStateStep = 0,
  BuildPlusMeasureWindow = 1,
};

enum class PlannerFallbackReason : uint32_t {
  None = 0,
  NoGPUContext = 1,
  UnsupportedBackendPreference = 2,
  MissingSIMDGroupMatrix = 3,
  ThreadgroupMemoryExceeded = 4,
  UnsupportedEncoding = 5,
  UnsupportedInputDims = 6,
  UnsupportedOutputDims = 7,
  MultiOutputFusedUnsupported = 8,
  HiddenDimUnsupported = 9,
  ForcedTiled = 10,
  ForcedSafeDebug = 11,
  EncodingRequiresTiledFamily = 12,
};

inline constexpr std::string_view
to_string(ExecutionBackendPreference pref) noexcept {
  switch (pref) {
  case ExecutionBackendPreference::RequireMetal:
    return "RequireMetal";
  case ExecutionBackendPreference::PreferMetal:
    return "PreferMetal";
  case ExecutionBackendPreference::RequireCPUReference:
    return "RequireCPUReference";
  }
  return "UnknownExecutionBackendPreference";
}

inline constexpr std::string_view to_string(NetworkFamily family) noexcept {
  switch (family) {
  case NetworkFamily::FullyFusedMetal:
    return "FullyFusedMetal";
  case NetworkFamily::TiledMetal:
    return "TiledMetal";
  case NetworkFamily::SafeDebugMetal:
    return "SafeDebugMetal";
  }
  return "UnknownNetworkFamily";
}

inline constexpr std::string_view
to_string(MultiOutputSemanticProfile profile) noexcept {
  switch (profile) {
  case MultiOutputSemanticProfile::Generic:
    return "Generic";
  case MultiOutputSemanticProfile::DNL:
    return "DNL";
  }
  return "UnknownMultiOutputSemanticProfile";
}

inline constexpr std::string_view
to_string(AutotuneSearchObjective objective) noexcept {
  switch (objective) {
  case AutotuneSearchObjective::SteadyStateStep:
    return "SteadyStateStep";
  case AutotuneSearchObjective::BuildPlusMeasureWindow:
    return "BuildPlusMeasureWindow";
  }
  return "UnknownAutotuneSearchObjective";
}

inline constexpr std::string_view
to_string(PlannerFallbackReason reason) noexcept {
  switch (reason) {
  case PlannerFallbackReason::None:
    return "None";
  case PlannerFallbackReason::NoGPUContext:
    return "NoGPUContext";
  case PlannerFallbackReason::UnsupportedBackendPreference:
    return "UnsupportedBackendPreference";
  case PlannerFallbackReason::MissingSIMDGroupMatrix:
    return "MissingSIMDGroupMatrix";
  case PlannerFallbackReason::ThreadgroupMemoryExceeded:
    return "ThreadgroupMemoryExceeded";
  case PlannerFallbackReason::UnsupportedEncoding:
    return "UnsupportedEncoding";
  case PlannerFallbackReason::UnsupportedInputDims:
    return "UnsupportedInputDims";
  case PlannerFallbackReason::UnsupportedOutputDims:
    return "UnsupportedOutputDims";
  case PlannerFallbackReason::MultiOutputFusedUnsupported:
    return "MultiOutputFusedUnsupported";
  case PlannerFallbackReason::HiddenDimUnsupported:
    return "HiddenDimUnsupported";
  case PlannerFallbackReason::ForcedTiled:
    return "ForcedTiled";
  case PlannerFallbackReason::ForcedSafeDebug:
    return "ForcedSafeDebug";
  case PlannerFallbackReason::EncodingRequiresTiledFamily:
    return "EncodingRequiresTiledFamily";
  }
  return "UnknownPlannerFallbackReason";
}

struct MultiOutputFactoryOptions {
  MultiOutputSemanticProfile semantics = MultiOutputSemanticProfile::Generic;
  uint32_t bc_dim_count = 0;
};

struct NetworkFactoryOptions {
  ExecutionBackendPreference backend = ExecutionBackendPreference::PreferMetal;
  std::optional<NetworkFamily> forced_family;
  std::shared_ptr<MetalContext> metal_context;
  bool enable_safe_debug_family = true;
  bool enable_bounded_autotune_search = false;
  uint32_t autotune_search_batch_size = 1024;
  uint32_t autotune_search_warmup_steps = 2;
  uint32_t autotune_search_measure_steps = 2;
  AutotuneSearchObjective autotune_search_objective =
      AutotuneSearchObjective::SteadyStateStep;
  std::optional<MultiOutputFactoryOptions> multi_output;
};

struct NetworkPlan {
  NetworkFamily selected_family = NetworkFamily::SafeDebugMetal;
  bool fused_eval = false;
  bool fused_train = false;
  uint64_t planner_fingerprint = 0;
  bool from_autotune_manifest = false;
  bool selected_by_autotune_search = false;
  uint64_t autotune_measured_step_ns = 0;
  uint64_t autotune_measured_objective_ns = 0;
  AutotuneSearchObjective autotune_search_objective =
      AutotuneSearchObjective::SteadyStateStep;
  DeviceCapabilities capabilities;
  std::vector<NetworkFamily> candidate_families;
  std::vector<PlannerFallbackReason> reasons;
};

[[nodiscard]] NetworkPlan
plan_network(const Encoding &encoding, const Network &network,
             const DeviceCapabilities &capabilities,
             const NetworkFactoryOptions &options = {});

[[nodiscard]] NetworkPlan
plan_network(const NetworkWithInputEncoding &model,
             const DeviceCapabilities &capabilities,
             const NetworkFactoryOptions &options = {});

[[nodiscard]] NetworkPlan
plan_network(const Encoding &encoding, const Network &network,
             const NetworkFactoryOptions &options = {});

[[nodiscard]] NetworkPlan
plan_network(const NetworkWithInputEncoding &model,
             const NetworkFactoryOptions &options = {});

} // namespace tmnn

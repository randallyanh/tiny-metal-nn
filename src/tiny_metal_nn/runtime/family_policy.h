#pragma once

/**
 * @file family_policy.h
 * @brief Internal helpers for applying tmnn network-family policy consistently.
 */

#include "tiny-metal-nn/detail/network_planning.h"

namespace tmnn::detail {

inline bool family_is_safe_debug(NetworkFamily family) {
  return family == NetworkFamily::SafeDebugMetal;
}

inline bool family_is_fully_fused(NetworkFamily family) {
  return family == NetworkFamily::FullyFusedMetal;
}

inline bool plan_uses_fully_fused_training(const NetworkPlan &plan) {
  return family_is_fully_fused(plan.selected_family);
}

inline bool plan_allows_fp16_training(const NetworkPlan &plan) {
  return plan.capabilities.supports_fp16 &&
         !family_is_safe_debug(plan.selected_family);
}

inline bool plan_allows_threadgroup_weight_cache(const NetworkPlan &plan) {
  return !family_is_safe_debug(plan.selected_family);
}

inline NetworkPlan resolved_plan_with_selected_family(
    const NetworkPlan &baseline_plan, NetworkFamily selected_family,
    bool from_autotune_manifest, bool selected_by_autotune_search,
    uint64_t measured_step_ns,
    AutotuneSearchObjective objective =
        AutotuneSearchObjective::SteadyStateStep,
    uint64_t measured_objective_ns = 0) {
  NetworkPlan resolved = baseline_plan;
  resolved.selected_family = selected_family;
  resolved.from_autotune_manifest = from_autotune_manifest;
  resolved.selected_by_autotune_search = selected_by_autotune_search;
  resolved.autotune_measured_step_ns = measured_step_ns;
  resolved.autotune_search_objective = objective;
  resolved.autotune_measured_objective_ns =
      measured_objective_ns == 0 ? measured_step_ns : measured_objective_ns;
  resolved.fused_train =
      family_is_fully_fused(selected_family) && baseline_plan.fused_train;
  if (family_is_safe_debug(selected_family))
    resolved.fused_eval = false;
  return resolved;
}

} // namespace tmnn::detail

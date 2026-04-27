/**
 * @file test_family_policy.cpp
 * @brief Tests for tmnn-owned runtime family policy helpers.
 */

#include <gtest/gtest.h>

#include "tiny-metal-nn/detail/network_planning.h"

#include "tiny_metal_nn/runtime/family_policy.h"

using namespace tmnn;

TEST(FamilyPolicy, SafeDebugDisablesFusedEligibility) {
  NetworkPlan baseline;
  baseline.selected_family = NetworkFamily::FullyFusedMetal;
  baseline.fused_eval = true;
  baseline.fused_train = true;
  baseline.capabilities.supports_fp16 = true;

  const auto resolved = detail::resolved_plan_with_selected_family(
      baseline, NetworkFamily::SafeDebugMetal, true, true, 1234u);

  EXPECT_EQ(resolved.selected_family, NetworkFamily::SafeDebugMetal);
  EXPECT_TRUE(resolved.from_autotune_manifest);
  EXPECT_TRUE(resolved.selected_by_autotune_search);
  EXPECT_EQ(resolved.autotune_measured_step_ns, 1234u);
  EXPECT_EQ(resolved.autotune_measured_objective_ns, 1234u);
  EXPECT_EQ(resolved.autotune_search_objective,
            AutotuneSearchObjective::SteadyStateStep);
  EXPECT_FALSE(resolved.fused_eval);
  EXPECT_FALSE(resolved.fused_train);
  EXPECT_FALSE(detail::plan_allows_fp16_training(resolved));
  EXPECT_FALSE(detail::plan_allows_threadgroup_weight_cache(resolved));
}

TEST(FamilyPolicy, TiledKeepsEvalEligibilityButDisablesFusedTraining) {
  NetworkPlan baseline;
  baseline.selected_family = NetworkFamily::FullyFusedMetal;
  baseline.fused_eval = true;
  baseline.fused_train = true;
  baseline.capabilities.supports_fp16 = true;

  const auto resolved = detail::resolved_plan_with_selected_family(
      baseline, NetworkFamily::TiledMetal, false, true, 5678u);

  EXPECT_EQ(resolved.selected_family, NetworkFamily::TiledMetal);
  EXPECT_FALSE(resolved.from_autotune_manifest);
  EXPECT_TRUE(resolved.selected_by_autotune_search);
  EXPECT_EQ(resolved.autotune_measured_step_ns, 5678u);
  EXPECT_EQ(resolved.autotune_measured_objective_ns, 5678u);
  EXPECT_EQ(resolved.autotune_search_objective,
            AutotuneSearchObjective::SteadyStateStep);
  EXPECT_TRUE(resolved.fused_eval);
  EXPECT_FALSE(resolved.fused_train);
  EXPECT_TRUE(detail::plan_allows_fp16_training(resolved));
  EXPECT_TRUE(detail::plan_allows_threadgroup_weight_cache(resolved));
}

TEST(FamilyPolicy, FullyFusedPreservesTrainingPrivileges) {
  NetworkPlan baseline;
  baseline.selected_family = NetworkFamily::TiledMetal;
  baseline.fused_eval = true;
  baseline.fused_train = true;
  baseline.capabilities.supports_fp16 = true;

  const auto resolved = detail::resolved_plan_with_selected_family(
      baseline, NetworkFamily::FullyFusedMetal, false, false, 0u);

  EXPECT_EQ(resolved.selected_family, NetworkFamily::FullyFusedMetal);
  EXPECT_EQ(resolved.autotune_measured_objective_ns, 0u);
  EXPECT_TRUE(resolved.fused_eval);
  EXPECT_TRUE(resolved.fused_train);
  EXPECT_TRUE(detail::plan_uses_fully_fused_training(resolved));
  EXPECT_TRUE(detail::plan_allows_fp16_training(resolved));
  EXPECT_TRUE(detail::plan_allows_threadgroup_weight_cache(resolved));
}

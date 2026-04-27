/**
 * @file test_network_planning.cpp
 * @brief Tests for C4 planner vocabulary and family selection.
 */

#include <gtest/gtest.h>

#include "tiny-metal-nn/autotune_manifest.h"
#include "tiny-metal-nn/factory.h"
#include "tiny-metal-nn/metal_context.h"
#include "tiny-metal-nn/detail/network_planning.h"

#include <algorithm>

using namespace tmnn;

namespace {

DeviceCapabilities make_caps(bool simdgroup_matrix = true,
                             uint32_t tg_mem_bytes = 32768) {
  DeviceCapabilities caps;
  caps.device_name = "SyntheticMetal";
  caps.max_threads_per_tg = 1024;
  caps.max_threadgroup_memory_bytes = tg_mem_bytes;
  caps.supports_fp16 = true;
  caps.supports_simdgroup_matrix = simdgroup_matrix;
  caps.supports_nonuniform_threadgroups = true;
  return caps;
}

bool has_reason(const NetworkPlan &plan, PlannerFallbackReason reason) {
  return std::find(plan.reasons.begin(), plan.reasons.end(), reason) !=
         plan.reasons.end();
}

bool has_candidate(const NetworkPlan &plan, NetworkFamily family) {
  return std::find(plan.candidate_families.begin(), plan.candidate_families.end(),
                   family) != plan.candidate_families.end();
}

std::shared_ptr<NetworkWithInputEncoding>
make_model(const HashGridEncoding::Config &enc_cfg = {},
           const FullyFusedMLP::Config &net_cfg = {}) {
  auto enc = create_encoding(enc_cfg);
  auto net = create_network(net_cfg);
  return create_network_with_input_encoding(enc, net);
}

} // namespace

TEST(NetworkPlanning, FusedFamilyOnEligibleStandardModel) {
  auto model = make_model();
  auto plan = plan_network(*model, make_caps());

  EXPECT_EQ(plan.selected_family, NetworkFamily::FullyFusedMetal);
  EXPECT_TRUE(plan.fused_eval);
  EXPECT_TRUE(plan.fused_train);
  ASSERT_EQ(plan.candidate_families.size(), 2u);
  EXPECT_EQ(plan.candidate_families[0], NetworkFamily::FullyFusedMetal);
  EXPECT_EQ(plan.candidate_families[1], NetworkFamily::TiledMetal);
  EXPECT_TRUE(plan.reasons.empty());
}

TEST(NetworkPlanning, TiledFamilyWhenSimdgroupMatrixMissing) {
  auto model = make_model();
  auto plan = plan_network(*model, make_caps(false, 32768));

  EXPECT_EQ(plan.selected_family, NetworkFamily::TiledMetal);
  EXPECT_FALSE(plan.fused_eval);
  EXPECT_FALSE(plan.fused_train);
  EXPECT_TRUE(has_reason(plan,
                         PlannerFallbackReason::MissingSIMDGroupMatrix));
}

TEST(NetworkPlanning, EvalMayStayFusedWhenTrainThreadgroupMemoryExceedsLimit) {
  FullyFusedMLP::Config net_cfg;
  net_cfg.hidden_dim = 128;
  net_cfg.num_hidden_layers = 2;
  auto model = make_model(HashGridEncoding::Config{}, net_cfg);
  auto plan = plan_network(*model, make_caps(true, 32768));

  EXPECT_EQ(plan.selected_family, NetworkFamily::TiledMetal);
  EXPECT_TRUE(plan.fused_eval);
  EXPECT_FALSE(plan.fused_train);
  EXPECT_TRUE(
      has_reason(plan, PlannerFallbackReason::ThreadgroupMemoryExceeded));
}

TEST(NetworkPlanning, MultiOutputFallsBackToTiledMetal) {
  FullyFusedMLP::Config net_cfg;
  net_cfg.n_output = 4;
  auto model = make_model(HashGridEncoding::Config{}, net_cfg);
  auto plan = plan_network(*model, make_caps());

  EXPECT_EQ(plan.selected_family, NetworkFamily::TiledMetal);
  EXPECT_FALSE(plan.fused_eval);
  EXPECT_FALSE(plan.fused_train);
  EXPECT_TRUE(has_reason(plan,
                         PlannerFallbackReason::MultiOutputFusedUnsupported));
}

TEST(NetworkPlanning, FourDFallsBackToTiledMetal) {
  HashGridEncoding::Config enc_cfg;
  enc_cfg.input_dims = 4;
  FullyFusedMLP::Config net_cfg;
  net_cfg.n_input = 32;
  auto model = make_model(enc_cfg, net_cfg);
  auto plan = plan_network(*model, make_caps());

  EXPECT_EQ(plan.selected_family, NetworkFamily::TiledMetal);
  EXPECT_FALSE(plan.fused_eval);
  EXPECT_FALSE(plan.fused_train);
  EXPECT_TRUE(has_reason(plan, PlannerFallbackReason::UnsupportedInputDims));
}

TEST(NetworkPlanning, RotatedEncodingUsesTiledMetalFamily) {
  auto model = create_network_with_input_encoding(create_rotated_encoding(),
                                                  create_network());
  auto plan = plan_network(*model, make_caps());

  EXPECT_EQ(plan.selected_family, NetworkFamily::TiledMetal);
  EXPECT_FALSE(plan.fused_eval);
  EXPECT_FALSE(plan.fused_train);
  ASSERT_EQ(plan.candidate_families.size(), 1u);
  EXPECT_EQ(plan.candidate_families[0], NetworkFamily::TiledMetal);
  EXPECT_TRUE(has_reason(plan,
                         PlannerFallbackReason::EncodingRequiresTiledFamily));
}

TEST(NetworkPlanning, ForcedSafeDebugWinsEvenWhenFusedIsEligible) {
  auto model = make_model();
  NetworkFactoryOptions options;
  options.forced_family = NetworkFamily::SafeDebugMetal;

  auto plan = plan_network(*model, make_caps(), options);
  EXPECT_EQ(plan.selected_family, NetworkFamily::SafeDebugMetal);
  EXPECT_FALSE(plan.fused_eval);
  EXPECT_FALSE(plan.fused_train);
  ASSERT_EQ(plan.candidate_families.size(), 1u);
  EXPECT_EQ(plan.candidate_families[0], NetworkFamily::SafeDebugMetal);
  EXPECT_TRUE(has_reason(plan, PlannerFallbackReason::ForcedSafeDebug));
}

TEST(NetworkPlanning, ManifestPrewarmReusesRecordedDecision) {
  auto model = make_model();
  auto ctx = MetalContext::create();

  NetworkFactoryOptions options;
  options.metal_context = ctx;

  const auto first = plan_network(*model, make_caps(), options);
  EXPECT_FALSE(first.from_autotune_manifest);

  const auto manifest = ctx->snapshot_autotune_manifest();
  ASSERT_EQ(manifest.entries.size(), 1u);
  EXPECT_EQ(manifest.entries[0].planner_fingerprint, first.planner_fingerprint);

  auto ctx2 = MetalContext::create();
  ctx2->prewarm_autotune_manifest(manifest);
  options.metal_context = ctx2;

  const auto second = plan_network(*model, make_caps(), options);
  EXPECT_TRUE(second.from_autotune_manifest);
  EXPECT_EQ(second.selected_family, first.selected_family);
  EXPECT_EQ(second.planner_fingerprint, first.planner_fingerprint);
}

TEST(NetworkPlanning, MeasuredManifestRestoresAutotuneMetadata) {
  auto model = make_model();
  auto ctx = MetalContext::create();

  NetworkFactoryOptions options;
  options.metal_context = ctx;
  options.autotune_search_objective =
      AutotuneSearchObjective::BuildPlusMeasureWindow;
  const auto baseline = plan_network(*model, make_caps(), options);

  AutotuneManifest manifest;
  manifest.entries.push_back(AutotuneManifestEntry{
      baseline.planner_fingerprint,
      NetworkFamily::TiledMetal,
      baseline.candidate_families,
      baseline.reasons,
      false,
      true,
      256u,
      2u,
      2u,
      12345u,
      {
          AutotuneCandidateMeasurement{NetworkFamily::FullyFusedMetal, 200u,
                                       15000u, true, ""},
          AutotuneCandidateMeasurement{NetworkFamily::TiledMetal, 180u, 12345u,
                                       true, ""},
      },
      AutotuneSearchObjective::BuildPlusMeasureWindow,
      12705u,
  });

  auto ctx2 = MetalContext::create();
  ctx2->prewarm_autotune_manifest(manifest);
  options.metal_context = ctx2;

  const auto plan = plan_network(*model, make_caps(), options);
  EXPECT_TRUE(plan.from_autotune_manifest);
  EXPECT_TRUE(plan.selected_by_autotune_search);
  EXPECT_EQ(plan.autotune_measured_step_ns, 12345u);
  EXPECT_EQ(plan.autotune_measured_objective_ns, 12705u);
  EXPECT_EQ(plan.autotune_search_objective,
            AutotuneSearchObjective::BuildPlusMeasureWindow);
  EXPECT_EQ(plan.selected_family, NetworkFamily::TiledMetal);
  EXPECT_FALSE(plan.fused_train);
}

TEST(NetworkPlanning, MeasuredManifestIgnoredWhenObjectivePolicyMismatches) {
  auto model = make_model();
  auto ctx = MetalContext::create();

  NetworkFactoryOptions options;
  options.metal_context = ctx;
  const auto baseline = plan_network(*model, make_caps(), options);

  AutotuneManifest manifest;
  manifest.entries.push_back(AutotuneManifestEntry{
      baseline.planner_fingerprint,
      NetworkFamily::TiledMetal,
      baseline.candidate_families,
      baseline.reasons,
      false,
      true,
      256u,
      2u,
      2u,
      12345u,
      {},
      AutotuneSearchObjective::BuildPlusMeasureWindow,
      12705u,
  });

  auto ctx2 = MetalContext::create();
  ctx2->prewarm_autotune_manifest(manifest);
  options.metal_context = ctx2;
  options.autotune_search_objective =
      AutotuneSearchObjective::SteadyStateStep;

  const auto plan = plan_network(*model, make_caps(), options);
  EXPECT_FALSE(plan.from_autotune_manifest);
  EXPECT_EQ(plan.selected_family, NetworkFamily::FullyFusedMetal);
  EXPECT_EQ(plan.autotune_measured_step_ns, 0u);
  EXPECT_EQ(plan.autotune_measured_objective_ns, 0u);
  EXPECT_EQ(plan.autotune_search_objective,
            AutotuneSearchObjective::SteadyStateStep);
}

TEST(NetworkPlanning, ManifestEntryIgnoredWhenFamilyIsNoLongerEligible) {
  auto model = create_network_with_input_encoding(create_rotated_encoding(),
                                                  create_network());
  auto ctx = MetalContext::create();

  AutotuneManifest manifest;
  manifest.entries.push_back(AutotuneManifestEntry{
      0u, NetworkFamily::FullyFusedMetal, {NetworkFamily::FullyFusedMetal}, {},
      false, false, 0u, 2u, 0u, 0u, {}});

  NetworkFactoryOptions options;
  options.metal_context = ctx;
  const auto baseline = plan_network(*model, make_caps(), options);
  manifest.entries[0].planner_fingerprint = baseline.planner_fingerprint;

  auto ctx2 = MetalContext::create();
  ctx2->prewarm_autotune_manifest(manifest);
  options.metal_context = ctx2;

  const auto plan = plan_network(*model, make_caps(), options);
  EXPECT_FALSE(plan.from_autotune_manifest);
  EXPECT_EQ(plan.selected_family, NetworkFamily::TiledMetal);
  EXPECT_TRUE(has_candidate(plan, NetworkFamily::TiledMetal));
  EXPECT_FALSE(has_candidate(plan, NetworkFamily::FullyFusedMetal));
}

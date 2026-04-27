/**
 * @file test_autotune_manifest.cpp
 * @brief Tests for C6 manifest persistence and context prewarm hooks.
 */

#include <gtest/gtest.h>

#include "tiny-metal-nn/autotune_manifest.h"
#include "tiny-metal-nn/metal_context.h"

#include <filesystem>

using namespace tmnn;

TEST(AutotuneManifest, SaveLoadRoundTrip) {
  AutotuneManifest manifest;
  manifest.device_name = "SyntheticMetal";
  manifest.device_family = 7;
  manifest.entries = {
      AutotuneManifestEntry{
          11u,
          NetworkFamily::FullyFusedMetal,
          {NetworkFamily::FullyFusedMetal, NetworkFamily::TiledMetal},
          {PlannerFallbackReason::ThreadgroupMemoryExceeded},
          false,
          false,
          0u,
          2u,
          0u,
          0u,
          {},
          AutotuneSearchObjective::SteadyStateStep,
          0u,
      },
      AutotuneManifestEntry{
          22u,
          NetworkFamily::TiledMetal,
          {NetworkFamily::TiledMetal, NetworkFamily::SafeDebugMetal},
          {PlannerFallbackReason::EncodingRequiresTiledFamily},
          true,
          true,
          128u,
          2u,
          2u,
          456u,
          {
              AutotuneCandidateMeasurement{NetworkFamily::TiledMetal, 512u,
                                           456u, true, "", 2048u, 2960u},
              AutotuneCandidateMeasurement{NetworkFamily::SafeDebugMetal, 640u,
                                           612u, true, "", 3072u, 3684u},
          },
          AutotuneSearchObjective::BuildPlusMeasureWindow,
          2960u,
      },
  };

  const auto path = std::filesystem::temp_directory_path() /
                    "tmnn_autotune_manifest_roundtrip.json";
  save_autotune_manifest(path.string(), manifest);
  const auto loaded = load_autotune_manifest(path.string());
  std::filesystem::remove(path);

  EXPECT_EQ(loaded.version, std::string(kAutotuneManifestVersion));
  EXPECT_EQ(loaded.device_name, manifest.device_name);
  EXPECT_EQ(loaded.device_family, manifest.device_family);
  ASSERT_EQ(loaded.entries.size(), 2u);
  EXPECT_EQ(loaded.entries[0].planner_fingerprint, 11u);
  EXPECT_EQ(loaded.entries[0].selected_family,
            NetworkFamily::FullyFusedMetal);
  EXPECT_EQ(loaded.entries[1].family_was_forced, true);
  EXPECT_TRUE(loaded.entries[1].selected_by_measurement);
  EXPECT_EQ(loaded.entries[1].measurement_batch_size, 128u);
  EXPECT_EQ(loaded.entries[1].measurement_warmup_steps, 2u);
  EXPECT_EQ(loaded.entries[1].measurement_steps, 2u);
  EXPECT_EQ(loaded.entries[1].measurement_objective,
            AutotuneSearchObjective::BuildPlusMeasureWindow);
  EXPECT_EQ(loaded.entries[1].selected_family_measured_step_ns, 456u);
  EXPECT_EQ(loaded.entries[1].selected_family_objective_ns, 2960u);
  ASSERT_EQ(loaded.entries[1].candidate_families.size(), 2u);
  EXPECT_EQ(loaded.entries[1].candidate_families[1],
            NetworkFamily::SafeDebugMetal);
  ASSERT_EQ(loaded.entries[1].measurements.size(), 2u);
  EXPECT_EQ(loaded.entries[1].measurements[0].family,
            NetworkFamily::TiledMetal);
  EXPECT_EQ(loaded.entries[1].measurements[0].measured_step_ns, 456u);
  EXPECT_EQ(loaded.entries[1].measurements[0].build_ns, 2048u);
  EXPECT_EQ(loaded.entries[1].measurements[0].objective_ns, 2960u);
  EXPECT_TRUE(loaded.entries[1].measurements[0].succeeded);
}

TEST(AutotuneManifest, ContextRejectsMismatchedDeviceFamilyWhenKnown) {
  auto ctx = MetalContext::create();
  if (ctx->capabilities().gpu_family == 0) {
    GTEST_SKIP() << "Synthetic mismatch check requires a probed GPU family";
  }

  AutotuneManifest manifest;
  manifest.device_name = "ForeignDevice";
  manifest.device_family = ctx->capabilities().gpu_family + 1;
  manifest.entries.push_back(
      AutotuneManifestEntry{77u, NetworkFamily::TiledMetal,
                            {NetworkFamily::TiledMetal}, {}, false, false, 0u,
                            2u, 0u, 0u, {}});

  ctx->prewarm_autotune_manifest(manifest);
  EXPECT_TRUE(ctx->snapshot_autotune_manifest().entries.empty());
}

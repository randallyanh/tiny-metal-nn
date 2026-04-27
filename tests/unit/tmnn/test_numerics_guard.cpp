/**
 * @file test_numerics_guard.cpp
 * @brief Tests for NumericsGuard sampling policy and anomaly tracking.
 */

#include <gtest/gtest.h>

#include "tiny_metal_nn/runtime/metal_context_internal.h"
#include "tiny_metal_nn/runtime/numerics_guard.h"

using namespace tmnn;

// ---------------------------------------------------------------------------
// Standalone NumericsGuard tests
// ---------------------------------------------------------------------------


TEST(NumericsGuard, DisabledNeverSamples) {
  NumericsGuard guard(NumericsSamplingMode::Disabled);
  EXPECT_FALSE(guard.should_sample(0));
  EXPECT_FALSE(guard.should_sample(1));
  EXPECT_FALSE(guard.should_sample(128));
  EXPECT_FALSE(guard.should_sample(999));
}

TEST(NumericsGuard, DisabledForceSamples) {
  NumericsGuard guard(NumericsSamplingMode::Disabled);
  EXPECT_TRUE(guard.should_sample(42, /*force=*/true));
}

TEST(NumericsGuard, FullPerStepAlwaysSamples) {
  NumericsGuard guard(NumericsSamplingMode::FullPerStep);
  for (uint32_t i = 0; i < 300; ++i) {
    EXPECT_TRUE(guard.should_sample(i));
  }
}

TEST(NumericsGuard, PeriodicSkipsBootstrapSteps) {
  NumericsGuard guard(NumericsSamplingMode::Periodic);
  EXPECT_FALSE(guard.should_sample(0));
  EXPECT_FALSE(guard.should_sample(1));
  EXPECT_FALSE(guard.should_sample(127));
  EXPECT_TRUE(guard.should_sample(128));
}

TEST(NumericsGuard, SampledFirstSteps) {
  NumericsGuard guard(NumericsSamplingMode::Sampled);
  EXPECT_TRUE(guard.should_sample(0));
  EXPECT_TRUE(guard.should_sample(1));
}

TEST(NumericsGuard, SampledInterval) {
  NumericsGuard guard; // Sampled, interval=128
  // Step 2 through 127 should not sample.
  for (uint32_t i = 2; i < 128; ++i) {
    EXPECT_FALSE(guard.should_sample(i)) << "step=" << i;
  }
  // Step 128 should sample (128 % 128 == 0).
  EXPECT_TRUE(guard.should_sample(128));
  EXPECT_TRUE(guard.should_sample(256));
}

TEST(NumericsGuard, CustomInterval) {
  NumericsGuard guard;
  guard.set_sample_interval(10);
  EXPECT_FALSE(guard.should_sample(5));
  EXPECT_TRUE(guard.should_sample(10));
  EXPECT_TRUE(guard.should_sample(20));
  EXPECT_FALSE(guard.should_sample(15));
}

TEST(NumericsGuard, SampledAfterAnomaly) {
  NumericsGuard guard; // Sampled mode
  NumericsReport bad;
  bad.finite_forward = false; // anomaly

  guard.record_step(50, bad);
  EXPECT_EQ(guard.anomaly_count(), 1u);
  EXPECT_EQ(guard.last_anomaly_step(), 50u);

  // Step 51 (next after anomaly) should sample.
  EXPECT_TRUE(guard.should_sample(51));
  // Step 52 should not (not interval, not post-anomaly).
  EXPECT_FALSE(guard.should_sample(52));
}

TEST(NumericsGuard, RecordStepTracksReports) {
  NumericsGuard guard;
  NumericsReport ok;
  guard.record_step(0, ok);
  guard.record_step(1, ok);
  EXPECT_EQ(guard.report_count(), 2u);
  EXPECT_EQ(guard.anomaly_count(), 0u);
}

TEST(NumericsGuard, LatestReport) {
  NumericsGuard guard;
  NumericsReport r;
  r.max_abs_activation = 42.0f;
  guard.record_step(5, r);
  EXPECT_FLOAT_EQ(guard.latest_report().max_abs_activation, 42.0f);
}

TEST(NumericsGuard, MultipleAnomalies) {
  NumericsGuard guard;
  NumericsReport bad;
  bad.finite_backward = false;

  guard.record_step(10, bad);
  guard.record_step(20, bad);
  bad.finite_update = false;
  guard.record_step(30, bad);

  EXPECT_EQ(guard.anomaly_count(), 3u);
  EXPECT_EQ(guard.last_anomaly_step(), 30u);
}

TEST(NumericsGuard, ZeroIntervalClampedToOne) {
  NumericsGuard guard;
  guard.set_sample_interval(0); // should clamp to 1
  // Every step should sample (interval=1 means step%1==0 always).
  EXPECT_TRUE(guard.should_sample(5));
  EXPECT_TRUE(guard.should_sample(99));
}

// ---------------------------------------------------------------------------
// MetalContext integration
// ---------------------------------------------------------------------------

TEST(NumericsGuard, ComponentAccess) {
  auto ctx = MetalContext::create();
  auto &ng = detail::context_numerics_guard(*ctx);

  // Default mode is Sampled.
  EXPECT_EQ(ng.mode(), NumericsSamplingMode::Sampled);

  // Record a step and verify through the accessor.
  NumericsReport r;
  r.max_abs_gradient = 7.5f;
  ng.record_step(0, r);
  EXPECT_EQ(ng.report_count(), 1u);
  EXPECT_FLOAT_EQ(ng.latest_report().max_abs_gradient, 7.5f);
}

TEST(NumericsGuard, MetalContextAppliesPeriodicModeOverride) {
  MetalContextDesc desc;
  desc.policy_overrides.numerics_sampling_mode = NumericsSamplingMode::Periodic;
  desc.policy_overrides.numerics_sample_interval = 64u;
  auto ctx = MetalContext::create(desc);
  auto &ng = detail::context_numerics_guard(*ctx);

  EXPECT_EQ(ng.mode(), NumericsSamplingMode::Periodic);
  EXPECT_FALSE(ng.should_sample(1));
  EXPECT_TRUE(ng.should_sample(64));
}

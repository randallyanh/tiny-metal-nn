/**
 * @file test_runtime_skeleton.cpp
 * @brief C0 skeleton tests for tmnn MetalContext, DeviceCapabilities,
 *        RuntimeStats, RuntimePolicy. GPU-conditional dispatch tests.
 */

#include "tiny-metal-nn/device_capabilities.h"
#include "tiny-metal-nn/metal_context.h"
#include "tiny-metal-nn/detail/runtime_stats.h"

// Internal headers — test-accessible via src/ include path.
#include "tiny_metal_nn/runtime/metal_context_internal.h"
#include "tiny_metal_nn/runtime/metal_device.h"
#include "tiny_metal_nn/runtime/pipeline_registry.h"
#include "tiny_metal_nn/runtime/runtime_policy.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <span>
#include <string>

using namespace tmnn;

// --- MetalContext creation ---

TEST(RuntimeSkeleton, MetalContextCreation) {
  auto ctx = MetalContext::create();
  ASSERT_NE(ctx, nullptr);
}

TEST(RuntimeSkeleton, MetalContextDefaultDesc) {
  auto ctx = MetalContext::create();
  const auto &desc = ctx->desc();
  EXPECT_TRUE(desc.enable_binary_archive);
  EXPECT_TRUE(desc.enable_disk_cache);
  EXPECT_TRUE(desc.enable_precise_math_fallback);
  EXPECT_EQ(desc.max_inflight_batches, 2u);
}

TEST(RuntimeSkeleton, MetalContextCustomDesc) {
  MetalContextDesc desc;
  desc.enable_binary_archive = false;
  desc.max_inflight_batches = 4;
  auto ctx = MetalContext::create(desc);
  EXPECT_FALSE(ctx->desc().enable_binary_archive);
  EXPECT_EQ(ctx->desc().max_inflight_batches, 4u);
}

// Phase 3.2 (default ON since this checkout): BufferArena routes
// Persistent allocations through the Heap. Capacity derives from
// MTLDevice.recommendedMaxWorkingSetSize, clamped to the documented
// 1 GiB floor / 4 GiB ceiling.
TEST(RuntimeSkeleton, MetalContextHeapDefaultIsRouted) {
  auto ctx = MetalContext::create();
  auto *heap = detail::context_heap(*ctx);
  if (ctx->is_gpu_available()) {
    ASSERT_NE(heap, nullptr);
    const auto stats = heap->stats();
    // Default 30/70 split with 1 GiB floor.
    EXPECT_GE(stats.persistent_shared_capacity_bytes,
              300ull * 1024 * 1024);
    EXPECT_LE(stats.persistent_shared_capacity_bytes,
              4ull * 1024 * 1024 * 1024);
    EXPECT_GE(stats.persistent_private_capacity_bytes,
              700ull * 1024 * 1024);
    EXPECT_LE(stats.persistent_private_capacity_bytes,
              4ull * 1024 * 1024 * 1024);
    EXPECT_EQ(stats.live_transient_buffers, 0u);
    EXPECT_EQ(stats.live_staging_buffers, 0u);
    EXPECT_EQ(stats.live_external_buffers, 0u);
  } else {
    EXPECT_EQ(heap, nullptr);
  }
}

// Explicit opt-out: route_buffer_arena_through_heap = false keeps the
// Heap dormant (tiny reservation) so wired_memory matches the pre-3.2
// baseline on memory-constrained systems.
TEST(RuntimeSkeleton, MetalContextHeapOptOutIsDormant) {
  MetalContextDesc desc;
  desc.heap_config.route_buffer_arena_through_heap = false;
  auto ctx = MetalContext::create(desc);
  if (!ctx->is_gpu_available()) GTEST_SKIP() << "No GPU";
  auto *heap = detail::context_heap(*ctx);
  ASSERT_NE(heap, nullptr);
  const auto stats = heap->stats();
  EXPECT_LE(stats.persistent_shared_capacity_bytes,
            16ull * 1024 * 1024);
  EXPECT_LE(stats.persistent_private_capacity_bytes,
            16ull * 1024 * 1024);
}

// Staging cap override: explicit value lands verbatim.
TEST(RuntimeSkeleton, MetalContextStagingCapacityOverride) {
  MetalContextDesc desc;
  desc.heap_config.staging_capacity_bytes = 8ull * 1024 * 1024;
  auto ctx = MetalContext::create(desc);
  if (!ctx->is_gpu_available()) GTEST_SKIP() << "No GPU";
  auto *heap = detail::context_heap(*ctx);
  ASSERT_NE(heap, nullptr);
  EXPECT_EQ(heap->stats().staging_capacity_bytes, 8ull * 1024 * 1024);
}

// Heap capacity overrides land verbatim — for memory-constrained
// environments or workloads with known peak usage.
TEST(RuntimeSkeleton, MetalContextHeapConfigOverrideIsVerbatim) {
  MetalContextDesc desc;
  desc.heap_config.route_buffer_arena_through_heap   = true;
  desc.heap_config.persistent_shared_capacity_bytes  = 256 * 1024 * 1024;
  desc.heap_config.persistent_private_capacity_bytes = 128 * 1024 * 1024;
  auto ctx = MetalContext::create(desc);
  if (!ctx->is_gpu_available()) GTEST_SKIP() << "No GPU";
  auto *heap = detail::context_heap(*ctx);
  ASSERT_NE(heap, nullptr);
  const auto stats = heap->stats();
  EXPECT_EQ(stats.persistent_shared_capacity_bytes,
            256ull * 1024 * 1024);
  EXPECT_EQ(stats.persistent_private_capacity_bytes,
            128ull * 1024 * 1024);
}

// --- DeviceCapabilities ---

TEST(RuntimeSkeleton, DeviceCapabilitiesDefaults) {
  auto ctx = MetalContext::create();
  const auto &caps = ctx->capabilities();
#if __APPLE__
  // Real device probing on macOS.
  EXPECT_FALSE(caps.device_name.empty());
  EXPECT_GT(caps.max_threads_per_tg, 0u);
  EXPECT_TRUE(ctx->is_gpu_available());
#else
  EXPECT_TRUE(caps.device_name.empty());
  EXPECT_EQ(caps.gpu_family, 0u);
  EXPECT_EQ(caps.max_threads_per_tg, 0u);
  EXPECT_FALSE(ctx->is_gpu_available());
#endif
}

// --- RuntimeStats ---

// --- RuntimePolicy ---

TEST(RuntimeSkeleton, RuntimePolicyDefaults) {
  // Ensure env vars are unset for this test.
  ::unsetenv("TMNN_ALLOW_CPU_FALLBACK");
  ::unsetenv("TMNN_FORCE_PRECISE_MATH");
  ::unsetenv("TMNN_DISABLE_BINARY_ARCHIVE");
  ::unsetenv("TMNN_DISABLE_PRIVATE_BUFFERS");
  ::unsetenv("TMNN_EMIT_RUNTIME_STATS");
  ::unsetenv("TMNN_NUMERICS_SAMPLING_MODE");
  ::unsetenv("TMNN_NUMERICS_SAMPLE_INTERVAL");
  ::unsetenv("TMNN_BAD_STEP_MODE");

  auto policy = load_runtime_policy();
  EXPECT_FALSE(policy.allow_cpu_fallback);
  EXPECT_FALSE(policy.force_precise_math);
  EXPECT_FALSE(policy.disable_binary_archive);
  EXPECT_FALSE(policy.disable_private_buffers);
  EXPECT_FALSE(policy.emit_runtime_stats);
  EXPECT_EQ(policy.numerics_sampling_mode, NumericsSamplingMode::Sampled);
  EXPECT_EQ(policy.numerics_sample_interval, 128u);
  EXPECT_EQ(policy.bad_step_recovery, BadStepRecoveryMode::SignalOnly);
}

TEST(RuntimeSkeleton, RuntimePolicyFromEnv) {
  ::setenv("TMNN_ALLOW_CPU_FALLBACK", "1", 1);
  ::setenv("TMNN_FORCE_PRECISE_MATH", "true", 1);
  ::setenv("TMNN_DISABLE_BINARY_ARCHIVE", "0", 1);   // falsy
  ::setenv("TMNN_DISABLE_PRIVATE_BUFFERS", "off", 1); // falsy
  ::setenv("TMNN_EMIT_RUNTIME_STATS", "yes", 1);
  ::setenv("TMNN_NUMERICS_SAMPLING_MODE", "periodic", 1);
  ::setenv("TMNN_NUMERICS_SAMPLE_INTERVAL", "64", 1);
  ::setenv("TMNN_BAD_STEP_MODE", "throw", 1);

  auto policy = load_runtime_policy();
  EXPECT_TRUE(policy.allow_cpu_fallback);
  EXPECT_TRUE(policy.force_precise_math);
  EXPECT_FALSE(policy.disable_binary_archive);
  EXPECT_FALSE(policy.disable_private_buffers);
  EXPECT_TRUE(policy.emit_runtime_stats);
  EXPECT_EQ(policy.numerics_sampling_mode, NumericsSamplingMode::Periodic);
  EXPECT_EQ(policy.numerics_sample_interval, 64u);
  EXPECT_EQ(policy.bad_step_recovery, BadStepRecoveryMode::Throw);

  // Cleanup.
  ::unsetenv("TMNN_ALLOW_CPU_FALLBACK");
  ::unsetenv("TMNN_FORCE_PRECISE_MATH");
  ::unsetenv("TMNN_DISABLE_BINARY_ARCHIVE");
  ::unsetenv("TMNN_DISABLE_PRIVATE_BUFFERS");
  ::unsetenv("TMNN_EMIT_RUNTIME_STATS");
  ::unsetenv("TMNN_NUMERICS_SAMPLING_MODE");
  ::unsetenv("TMNN_NUMERICS_SAMPLE_INTERVAL");
  ::unsetenv("TMNN_BAD_STEP_MODE");
}

TEST(RuntimeSkeleton, RuntimePolicyRollbackFromEnv) {
  ::setenv("TMNN_BAD_STEP_MODE", "rollback", 1);
  const auto policy = load_runtime_policy();
  EXPECT_EQ(policy.bad_step_recovery, BadStepRecoveryMode::Rollback);
  ::unsetenv("TMNN_BAD_STEP_MODE");
}

TEST(RuntimeSkeleton, RuntimePolicyApiOverridesTakePrecedence) {
  ::setenv("TMNN_ALLOW_CPU_FALLBACK", "1", 1);
  ::setenv("TMNN_FORCE_PRECISE_MATH", "1", 1);
  ::setenv("TMNN_DISABLE_BINARY_ARCHIVE", "0", 1);
  ::setenv("TMNN_DISABLE_PRIVATE_BUFFERS", "0", 1);
  ::setenv("TMNN_EMIT_RUNTIME_STATS", "0", 1);
  ::setenv("TMNN_NUMERICS_SAMPLING_MODE", "sampled", 1);
  ::setenv("TMNN_NUMERICS_SAMPLE_INTERVAL", "256", 1);
  ::setenv("TMNN_BAD_STEP_MODE", "throw", 1);

  MetalContextDesc desc;
  desc.policy_overrides.allow_cpu_fallback = false;
  desc.policy_overrides.force_precise_math = false;
  desc.policy_overrides.disable_binary_archive = true;
  desc.policy_overrides.disable_private_buffers = true;
  desc.policy_overrides.emit_runtime_stats = true;
  desc.policy_overrides.numerics_sampling_mode = NumericsSamplingMode::Disabled;
  desc.policy_overrides.numerics_sample_interval = 7u;
  desc.policy_overrides.bad_step_recovery = BadStepRecoveryMode::Skip;

  auto ctx = MetalContext::create(desc);
  const auto &policy = ctx->policy();
  EXPECT_FALSE(policy.allow_cpu_fallback);
  EXPECT_FALSE(policy.force_precise_math);
  EXPECT_TRUE(policy.disable_binary_archive);
  EXPECT_TRUE(policy.disable_private_buffers);
  EXPECT_TRUE(policy.emit_runtime_stats);
  EXPECT_EQ(policy.numerics_sampling_mode, NumericsSamplingMode::Disabled);
  EXPECT_EQ(policy.numerics_sample_interval, 7u);
  EXPECT_EQ(policy.bad_step_recovery, BadStepRecoveryMode::Skip);

  ::unsetenv("TMNN_ALLOW_CPU_FALLBACK");
  ::unsetenv("TMNN_FORCE_PRECISE_MATH");
  ::unsetenv("TMNN_DISABLE_BINARY_ARCHIVE");
  ::unsetenv("TMNN_DISABLE_PRIVATE_BUFFERS");
  ::unsetenv("TMNN_EMIT_RUNTIME_STATS");
  ::unsetenv("TMNN_NUMERICS_SAMPLING_MODE");
  ::unsetenv("TMNN_NUMERICS_SAMPLE_INTERVAL");
  ::unsetenv("TMNN_BAD_STEP_MODE");
}

TEST(RuntimeSkeleton, MetalContextPolicySnapshotMatchesResolvedPolicy) {
  MetalContextDesc desc;
  desc.policy_overrides.force_precise_math = true;
  desc.policy_overrides.emit_runtime_stats = true;
  desc.policy_overrides.numerics_sampling_mode = NumericsSamplingMode::Periodic;
  desc.policy_overrides.numerics_sample_interval = 32u;
  desc.policy_overrides.bad_step_recovery =
      BadStepRecoveryMode::FallbackAndRetryWithSafeFamily;

  auto ctx = MetalContext::create(desc);
  const auto &policy = ctx->policy();
  EXPECT_TRUE(policy.force_precise_math);
  EXPECT_TRUE(policy.emit_runtime_stats);
  EXPECT_EQ(policy.numerics_sampling_mode, NumericsSamplingMode::Periodic);
  EXPECT_EQ(policy.numerics_sample_interval, 32u);
  EXPECT_EQ(policy.bad_step_recovery,
            BadStepRecoveryMode::FallbackAndRetryWithSafeFamily);
}

TEST(RuntimeSkeleton, SnapshotStatsEmitsTelemetryWhenEnabled) {
  MetalContextDesc desc;
  desc.policy_overrides.emit_runtime_stats = true;
  auto ctx = MetalContext::create(desc);

  testing::internal::CaptureStderr();
  const auto stats = ctx->snapshot_stats();
  const std::string captured = testing::internal::GetCapturedStderr();

  EXPECT_EQ(stats.training_steps_completed, 0u);
  EXPECT_NE(captured.find("TMNN_STATS"), std::string::npos);
  EXPECT_NE(captured.find("training_steps_completed=0"), std::string::npos);
  EXPECT_NE(captured.find("numerics_anomaly_count=0"), std::string::npos);
  EXPECT_NE(captured.find("bad_steps_skipped=0"), std::string::npos);
  EXPECT_NE(captured.find("bad_steps_rolled_back=0"), std::string::npos);
  EXPECT_NE(captured.find("safe_family_recoveries=0"), std::string::npos);
}

// --- GPU availability ---

// --- GPU dispatch tests (GTEST_SKIP on non-Apple) ---

// Phase 4: batched blit-fill. Allocates three Shared buffers, writes
// known content, fires one batched fill at value=0, and verifies all
// three are zeroed. Helper does NOT skip Shared internally — that's the
// caller's pre-filter responsibility — so this test pins the general
// fill-N-buffers-with-one-commit_and_wait contract.
TEST(RuntimeSkeleton, BlitFillViewsBatchesMultipleBuffers) {
  auto ctx = MetalContext::create();
  if (!ctx->is_gpu_available())
    GTEST_SKIP() << "No GPU";
  auto &arena = detail::context_arena(*ctx);

  auto h0 = arena.allocate(
      {256, 256, BufferStorage::Shared, BufferLifetime::Persistent, "fv0"});
  auto h1 = arena.allocate(
      {512, 256, BufferStorage::Shared, BufferLifetime::Persistent, "fv1"});
  auto h2 = arena.allocate(
      {128, 256, BufferStorage::Shared, BufferLifetime::Persistent, "fv2"});

  auto v0 = arena.view(h0);
  auto v1 = arena.view(h1);
  auto v2 = arena.view(h2);

  // Stamp known non-zero content via CPU.
  std::memset(v0.data, 0xAB, v0.bytes);
  std::memset(v1.data, 0xCD, v1.bytes);
  std::memset(v2.data, 0xEF, v2.bytes);

  const BufferView views[] = {v0, v1, v2};
  detail::context_blit_fill_views(*ctx, std::span<const BufferView>(views),
                                  0);

  for (size_t i = 0; i < v0.bytes; ++i)
    ASSERT_EQ(static_cast<const uint8_t *>(v0.data)[i], 0u);
  for (size_t i = 0; i < v1.bytes; ++i)
    ASSERT_EQ(static_cast<const uint8_t *>(v1.data)[i], 0u);
  for (size_t i = 0; i < v2.bytes; ++i)
    ASSERT_EQ(static_cast<const uint8_t *>(v2.data)[i], 0u);
}

// Empty-batch fast-path: no work, no command buffer created, no commit.
// Verified indirectly by "doesn't hang and returns" — under contention
// an erroneous commit_and_wait on an empty cmdbuf could stall.
TEST(RuntimeSkeleton, BlitFillViewsEmptyBatchIsNoOp) {
  auto ctx = MetalContext::create();
  if (!ctx->is_gpu_available())
    GTEST_SKIP() << "No GPU";
  detail::context_blit_fill_views(*ctx, std::span<const BufferView>(), 0);
  // All views with no gpu_buffer are also a no-op.
  BufferView empties[2] = {};
  detail::context_blit_fill_views(*ctx,
                                  std::span<const BufferView>(empties), 0);
}

// Phase 5: Philox init kernel correctness.
//   * All samples land within [low, high].
//   * Same seed reproduces the exact same byte sequence.
//   * Different seeds produce different sequences.
//   * Sample mean/variance roughly match a uniform distribution
//     (loose bounds; we're not running statistical tests, just sanity).
TEST(RuntimeSkeleton, InitUniformPhiloxKernelProducesUniform) {
  auto ctx = MetalContext::create();
  if (!ctx->is_gpu_available())
    GTEST_SKIP() << "No GPU";
  auto &arena = detail::context_arena(*ctx);

  constexpr size_t kN = 65536;  // 256 KiB Shared buffer
  auto h = arena.allocate({kN * sizeof(float), 256, BufferStorage::Shared,
                           BufferLifetime::Persistent, "init_test"});
  auto v = arena.view(h);

  detail::context_dispatch_init_uniform(*ctx, v, kN, /*low=*/-0.5f,
                                        /*high=*/0.5f, /*seed=*/123u,
                                        /*counter_base=*/0u);
  const auto *out = static_cast<const float *>(v.data);
  double sum = 0.0, sq = 0.0;
  for (size_t i = 0; i < kN; ++i) {
    ASSERT_GE(out[i], -0.5f);
    ASSERT_LE(out[i], 0.5f);
    sum += out[i];
    sq += static_cast<double>(out[i]) * out[i];
  }
  const double mean = sum / kN;
  const double var = sq / kN - mean * mean;
  // Uniform[-0.5, 0.5]: theoretical mean=0, variance=1/12≈0.0833.
  EXPECT_NEAR(mean, 0.0, 0.01);
  EXPECT_NEAR(var, 1.0 / 12.0, 0.005);

  // Reproducibility: fresh dispatch with same seed produces identical bytes.
  std::vector<float> snapshot(out, out + kN);
  detail::context_dispatch_init_uniform(*ctx, v, kN, -0.5f, 0.5f, 123u, 0u);
  for (size_t i = 0; i < kN; ++i) {
    ASSERT_EQ(out[i], snapshot[i]) << "i=" << i;
  }

  // Different seed → different sequence (sample first 4 floats; vanishingly
  // small chance of collision under Philox).
  detail::context_dispatch_init_uniform(*ctx, v, kN, -0.5f, 0.5f,
                                        /*seed=*/124u, 0u);
  bool any_diff = false;
  for (size_t i = 0; i < 4; ++i) {
    if (out[i] != snapshot[i]) { any_diff = true; break; }
  }
  EXPECT_TRUE(any_diff);
}

// Phase 5.5: cross-build / cross-process reproducibility pin. The
// Philox kernel is deterministic at the algorithm level (counter-based,
// no shared state), and the uint32 → float conversion is bit-exact on
// IEEE-754 hardware, so the same seed must produce the same first N
// outputs across rebuilds. This test captures a frozen snapshot of the
// first 8 floats produced by Uniform(seed=42, low=-0.5, high=0.5,
// counter_base=0) on a 1024-float buffer and asserts byte-equality.
// Failure here means either the kernel changed semantically or the
// host's float-conversion semantics drifted — both are red flags worth
// surfacing immediately.
TEST(RuntimeSkeleton, InitUniformPhiloxSeedFortyTwoGoldenSnapshot) {
  auto ctx = MetalContext::create();
  if (!ctx->is_gpu_available())
    GTEST_SKIP() << "No GPU";
  auto &arena = detail::context_arena(*ctx);

  constexpr size_t kN = 1024;
  auto h = arena.allocate({kN * sizeof(float), 256, BufferStorage::Shared,
                           BufferLifetime::Persistent, "init_golden"});
  auto v = arena.view(h);

  detail::context_dispatch_init_uniform(*ctx, v, kN,
                                        /*low=*/-0.5f, /*high=*/0.5f,
                                        /*seed=*/42u,
                                        /*counter_base=*/0u);

  // Captured 2026-04-28 on Apple M1 Pro / macOS 24.6.0 with the P5.5
  // build of init_uniform_philox. Update this snapshot intentionally
  // when you change the Philox round constants, the uniform-conversion
  // formula, or the per-thread output layout — never silently.
  static const float kGolden[8] = {
       0.1129598618f,
      -0.0314134955f,
      -0.4267682731f,
      -0.1591384411f,
       0.4877186418f,
      -0.1729366183f,
       0.0139061213f,
      -0.0456843972f,
  };
  const auto *out = static_cast<const float *>(v.data);
  for (size_t i = 0; i < 8; ++i) {
    ASSERT_EQ(out[i], kGolden[i])
        << "Reproducibility regression at i=" << i
        << " — Philox / uint32→float pipeline drifted";
  }
}

// Phase 5.3: Box-Muller normal-distribution kernel correctness.
//   * Sample mean ≈ requested mean
//   * Sample stddev ≈ requested stddev
//   * No NaN / Inf (Box-Muller is well-behaved if u1 stays clamped > 0)
//   * Same-seed reproducibility
TEST(RuntimeSkeleton, InitNormalPhiloxKernelProducesNormal) {
  auto ctx = MetalContext::create();
  if (!ctx->is_gpu_available())
    GTEST_SKIP() << "No GPU";
  auto &arena = detail::context_arena(*ctx);

  constexpr size_t kN = 65536;
  auto h = arena.allocate({kN * sizeof(float), 256, BufferStorage::Shared,
                           BufferLifetime::Persistent, "init_normal_test"});
  auto v = arena.view(h);

  detail::InitNormalRequest req{v, kN, /*mean=*/0.5f, /*stddev=*/2.0f,
                                /*seed=*/42u, /*counter_base=*/0u};
  detail::context_dispatch_init_normal_views(
      *ctx, std::span<const detail::InitNormalRequest>(&req, 1));

  const auto *out = static_cast<const float *>(v.data);
  double sum = 0.0, sq = 0.0;
  for (size_t i = 0; i < kN; ++i) {
    ASSERT_TRUE(std::isfinite(out[i])) << "i=" << i;
    sum += out[i];
    sq += static_cast<double>(out[i]) * out[i];
  }
  const double mean = sum / kN;
  const double var  = sq / kN - mean * mean;
  // Sampling tolerance for kN = 65536: std error of mean ≈ 2 / sqrt(N)
  // ≈ 0.008. We allow 0.05 to keep the test robust.
  EXPECT_NEAR(mean, 0.5, 0.05);
  EXPECT_NEAR(std::sqrt(var), 2.0, 0.05);

  // Reproducibility (same seed → same bytes).
  std::vector<float> snapshot(out, out + kN);
  detail::context_dispatch_init_normal_views(
      *ctx, std::span<const detail::InitNormalRequest>(&req, 1));
  for (size_t i = 0; i < kN; ++i) {
    ASSERT_EQ(out[i], snapshot[i]) << "i=" << i;
  }
}

// counter_base offsets the per-thread Philox counter so two call sites
// (e.g. hash grid + MLP) can draw from non-overlapping streams of the
// same seed without correlation.
TEST(RuntimeSkeleton, InitUniformPhiloxCounterBaseShiftsStream) {
  auto ctx = MetalContext::create();
  if (!ctx->is_gpu_available())
    GTEST_SKIP() << "No GPU";
  auto &arena = detail::context_arena(*ctx);

  constexpr size_t kN = 1024;
  auto h = arena.allocate({kN * sizeof(float), 256, BufferStorage::Shared,
                           BufferLifetime::Persistent, "init_offset"});
  auto v = arena.view(h);

  detail::context_dispatch_init_uniform(*ctx, v, kN, -0.5f, 0.5f, 7u, 0u);
  std::vector<float> base(static_cast<const float *>(v.data),
                          static_cast<const float *>(v.data) + kN);

  detail::context_dispatch_init_uniform(*ctx, v, kN, -0.5f, 0.5f, 7u,
                                        /*counter_base=*/1024u);
  // Same seed, shifted counter → completely different stream.
  size_t equal_count = 0;
  for (size_t i = 0; i < kN; ++i) {
    if (static_cast<const float *>(v.data)[i] == base[i]) ++equal_count;
  }
  EXPECT_LT(equal_count, kN / 64u)
      << "counter_base offset should produce a near-uncorrelated stream; "
         "saw " << equal_count << " / " << kN << " collisions";
}

TEST(RuntimeSkeleton, BlitFillEndToEnd) {
  auto ctx = MetalContext::create();
  if (!ctx->is_gpu_available())
    GTEST_SKIP() << "No GPU";
  auto &arena = detail::context_arena(*ctx);

  // Allocate Shared buffer, write known pattern via CPU.
  auto h = arena.allocate(
      {256, 256, BufferStorage::Shared, BufferLifetime::Transient, "blit_test"});
  auto v = arena.view(h);
  auto *f = static_cast<float *>(v.data);
  f[0] = 99.0f;
  EXPECT_FLOAT_EQ(f[0], 99.0f);

  // Blit fill with zeros via GPU.
  detail::context_blit_fill(*ctx, v, 0);

  // Read back via CPU — should be zero (Shared buffer is coherent).
  EXPECT_FLOAT_EQ(f[0], 0.0f);
}

TEST(RuntimeSkeleton, ComputeDispatchEndToEnd) {
  auto ctx = MetalContext::create();
  if (!ctx->is_gpu_available())
    GTEST_SKIP() << "No GPU";
  auto &arena = detail::context_arena(*ctx);
  auto &reg = detail::context_pipeline_registry(*ctx);

  // Compile a simple fill kernel.
  const char *msl = R"(
    #include <metal_stdlib>
    using namespace metal;
    kernel void write_42(device float* buf [[buffer(0)]],
                         uint tid [[thread_position_in_grid]]) {
      buf[tid] = 42.0f;
    }
  )";
  PipelineKey key{0xBEEF, "write_42", false, 1, 0};
  auto pso = reg.register_pipeline(key, msl, "write_42");
  ASSERT_NE(reg.raw_pipeline(pso), nullptr);

  // Allocate 64 floats.
  auto h = arena.allocate({64 * sizeof(float), 256, BufferStorage::Shared,
                           BufferLifetime::Transient, "dispatch_test"});
  auto v = arena.view(h);

  // Dispatch via metal_device helpers.
  auto *cmd = metal::create_command_buffer(detail::context_raw_queue(*ctx));
  metal::DispatchDesc dd{};
  dd.cmd_buf = cmd;
  dd.pipeline = reg.raw_pipeline(pso);
  metal::DispatchDesc::BufferBind bind{v.gpu_buffer, 0, 0};
  dd.bindings = &bind;
  dd.binding_count = 1;
  dd.grid_x = 64;
  dd.grid_y = 1;
  dd.grid_z = 1;
  dd.tg_x = 64;
  dd.tg_y = 1;
  dd.tg_z = 1;
  metal::encode_dispatch(dd);
  metal::commit_and_wait(cmd);
  metal::release_command_buffer(cmd);

  // Read back.
  auto *f = static_cast<float *>(v.data);
  for (int i = 0; i < 64; ++i)
    EXPECT_FLOAT_EQ(f[i], 42.0f);
}

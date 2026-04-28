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

// Phase 3.2: by default the Heap stays dormant — small reservation,
// BufferArena unchanged. Opt-in routing via
// heap_config.route_buffer_arena_through_heap for high-frequency
// allocation workloads.
TEST(RuntimeSkeleton, MetalContextHeapDefaultIsDormant) {
  auto ctx = MetalContext::create();
  auto *heap = detail::context_heap(*ctx);
  if (ctx->is_gpu_available()) {
    ASSERT_NE(heap, nullptr);
    const auto stats = heap->stats();
    // Dormant: tiny reservation so wired_memory matches pre-3.2 baseline.
    // The Heap remains reachable for adopt_external etc.
    EXPECT_LE(stats.persistent_shared_capacity_bytes,
              16ull * 1024 * 1024);
    EXPECT_LE(stats.persistent_private_capacity_bytes,
              16ull * 1024 * 1024);
    EXPECT_EQ(stats.live_persistent_buffers, 0u);
    EXPECT_EQ(stats.live_transient_buffers, 0u);
    EXPECT_EQ(stats.live_staging_buffers, 0u);
    EXPECT_EQ(stats.live_external_buffers, 0u);
  } else {
    EXPECT_EQ(heap, nullptr);
  }
}

// Routed mode: heap derives capacity from MTLDevice.recommendedMaxWorkingSetSize
// and BufferArena allocates against it.
TEST(RuntimeSkeleton, MetalContextHeapRoutedSizesFromWorkingSet) {
  MetalContextDesc desc;
  desc.heap_config.route_buffer_arena_through_heap = true;
  auto ctx = MetalContext::create(desc);
  if (!ctx->is_gpu_available()) GTEST_SKIP() << "No GPU";
  auto *heap = detail::context_heap(*ctx);
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

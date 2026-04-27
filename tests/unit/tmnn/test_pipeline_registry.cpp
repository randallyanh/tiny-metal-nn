/**
 * @file test_pipeline_registry.cpp
 * @brief C4 tests for PipelineRegistry.
 */

#include "tiny_metal_nn/runtime/pipeline_registry.h"
#include "tiny_metal_nn/runtime/metal_context_internal.h"

#include <gtest/gtest.h>

#include <thread>
#include <vector>

using namespace tmnn;

namespace {

PipelineKey make_key(uint64_t spec_hash, const char *entry = "main",
                     bool precise = false) {
  return {spec_hash, entry, precise, 1, 0};
}

} // namespace

// --- Basic lookup miss/hit ---

TEST(PipelineRegistry, MissOnEmpty) {
  PipelineRegistry reg;
  auto h = reg.lookup(make_key(42));
  EXPECT_FALSE(static_cast<bool>(h));

  auto s = reg.stats();
  EXPECT_EQ(s.misses, 1u);
  EXPECT_EQ(s.hits, 0u);
}

TEST(PipelineRegistry, RegisterAndLookup) {
  PipelineRegistry reg;
  auto key = make_key(42, "eval_kernel");
  auto h1 = reg.register_pipeline(key);
  EXPECT_TRUE(static_cast<bool>(h1));

  auto h2 = reg.lookup(key);
  EXPECT_TRUE(static_cast<bool>(h2));
  EXPECT_EQ(h1.index, h2.index);
  EXPECT_EQ(h1.generation, h2.generation);

  auto s = reg.stats();
  EXPECT_EQ(s.hits, 1u);
  EXPECT_EQ(s.compile_count, 1u);
  EXPECT_EQ(s.entries, 1u);
}

TEST(PipelineRegistry, DuplicateRegister) {
  PipelineRegistry reg;
  auto key = make_key(42);
  auto h1 = reg.register_pipeline(key);
  auto h2 = reg.register_pipeline(key); // duplicate
  EXPECT_EQ(h1.index, h2.index);

  auto s = reg.stats();
  EXPECT_EQ(s.compile_count, 1u); // only counted once
}

// --- Different keys ---

TEST(PipelineRegistry, DifferentKeys) {
  PipelineRegistry reg;
  auto k1 = make_key(1, "eval");
  auto k2 = make_key(1, "train");
  auto k3 = make_key(2, "eval");
  auto k4 = make_key(1, "eval", true); // precise

  auto h1 = reg.register_pipeline(k1);
  auto h2 = reg.register_pipeline(k2);
  auto h3 = reg.register_pipeline(k3);
  auto h4 = reg.register_pipeline(k4);

  EXPECT_NE(h1.index, h2.index);
  EXPECT_NE(h1.index, h3.index);
  EXPECT_NE(h1.index, h4.index);

  EXPECT_EQ(reg.stats().entries, 4u);
}

TEST(PipelineRegistry, DeviceFamilyDifferentiatesEntries) {
  PipelineRegistry reg;
  PipelineKey family7{7, "eval", false, 1, 7};
  PipelineKey family8{7, "eval", false, 1, 8};

  auto h1 = reg.register_pipeline(family7);
  auto h2 = reg.register_pipeline(family8);

  EXPECT_NE(h1.index, h2.index);
  EXPECT_EQ(reg.stats().entries, 2u);
}

// --- Clear ---

TEST(PipelineRegistry, Clear) {
  PipelineRegistry reg;
  auto key = make_key(42);
  (void)reg.register_pipeline(key);
  EXPECT_EQ(reg.stats().entries, 1u);

  reg.clear();
  EXPECT_EQ(reg.stats().entries, 0u);
  EXPECT_EQ(reg.stats().hits, 0u);
  EXPECT_EQ(reg.stats().misses, 0u);

  // After clear, previous key is a miss.
  auto h = reg.lookup(key);
  EXPECT_FALSE(static_cast<bool>(h));
  EXPECT_EQ(reg.stats().misses, 1u);
}

TEST(PipelineRegistry, ClearInvalidatesPriorHandles) {
  PipelineRegistry reg;
  auto key = make_key(314, "invalidate");
  auto h1 = reg.register_pipeline(key);
  ASSERT_TRUE(static_cast<bool>(h1));

  reg.clear();
  EXPECT_EQ(reg.raw_pipeline(h1), nullptr);
  EXPECT_FALSE(reg.has_failure(h1));
  EXPECT_TRUE(reg.failure_diagnostic(h1).empty());

  auto h2 = reg.register_pipeline(key);
  EXPECT_TRUE(static_cast<bool>(h2));
  EXPECT_NE(h2.generation, h1.generation);
}

// --- Thread safety ---

TEST(PipelineRegistry, ConcurrentAccess) {
  PipelineRegistry reg;

  constexpr int N = 100;
  std::vector<std::thread> threads;
  threads.reserve(N);

  for (int i = 0; i < N; ++i) {
    threads.emplace_back([&reg, i] {
      auto key = make_key(static_cast<uint64_t>(i % 10));
      (void)reg.register_pipeline(key);
      (void)reg.lookup(key);
    });
  }
  for (auto &t : threads)
    t.join();

  auto s = reg.stats();
  EXPECT_EQ(s.entries, 10u); // 10 unique keys
  EXPECT_GE(s.hits + s.misses, static_cast<uint64_t>(N));
}

// --- MetalContext integration ---

TEST(PipelineRegistry, ContextRegistryAccess) {
  auto ctx = MetalContext::create();
  auto &reg = detail::context_pipeline_registry(*ctx);

  auto key = make_key(99, "fwd_bwd");
  (void)reg.register_pipeline(key);

  // Stats flow through to context.
  auto stats = ctx->snapshot_stats();
  EXPECT_EQ(stats.compile_count, 1u);
  EXPECT_EQ(stats.pipeline_cache_hits, 0u);

  (void)reg.lookup(key);
  stats = ctx->snapshot_stats();
  EXPECT_EQ(stats.pipeline_cache_hits, 1u);
}

TEST(PipelineRegistry, ContextClearCachesResetsRegistry) {
  auto ctx = MetalContext::create();
  auto &reg = detail::context_pipeline_registry(*ctx);
  (void)reg.register_pipeline(make_key(1));
  EXPECT_EQ(reg.stats().entries, 1u);

  ctx->clear_runtime_caches();
  EXPECT_EQ(reg.stats().entries, 0u);
}

// --- GPU-conditional: real MSL compilation ---

TEST(PipelineRegistry, CompileRealPipeline) {
  auto ctx = MetalContext::create();
  if (!ctx->is_gpu_available())
    GTEST_SKIP() << "No GPU";
  auto &reg = detail::context_pipeline_registry(*ctx);
  const char *msl = R"(
    #include <metal_stdlib>
    using namespace metal;
    kernel void fill_zero(device float* buf [[buffer(0)]],
                          uint tid [[thread_position_in_grid]]) {
      buf[tid] = 0.0f;
    }
  )";
  PipelineKey key{0x1234, "fill_zero", false, 1, 0};
  auto h = reg.register_pipeline(key, msl, "fill_zero");
  EXPECT_TRUE(static_cast<bool>(h));
  EXPECT_NE(reg.raw_pipeline(h), nullptr);
  // Second lookup is a cache hit.
  auto h2 = reg.register_pipeline(key, msl, "fill_zero");
  auto stats = reg.stats();
  EXPECT_GE(stats.hits, 1u);
}

TEST(PipelineRegistry, CompileFailureIsStickyAndDiagnosable) {
  auto ctx = MetalContext::create();
  if (!ctx->is_gpu_available())
    GTEST_SKIP() << "No GPU";
  auto &reg = detail::context_pipeline_registry(*ctx);
  const char *bad_msl = R"(
    #include <metal_stdlib>
    using namespace metal;
    kernel void broken(device float* buf [[buffer(0)]]) {
      this_is_not_valid_msl
    }
  )";
  PipelineKey key{0x4321, "broken", false, 1, 0};

  auto h1 = reg.register_pipeline(key, bad_msl, "broken");
  EXPECT_TRUE(static_cast<bool>(h1));
  EXPECT_EQ(reg.raw_pipeline(h1), nullptr);
  EXPECT_TRUE(reg.has_failure(h1));
  auto diag = reg.failure_diagnostic(h1);
  EXPECT_FALSE(diag.empty());

  auto stats_before_retry = reg.stats();
  auto h2 = reg.register_pipeline(key, bad_msl, "broken");
  auto stats_after_retry = reg.stats();
  EXPECT_EQ(h2.index, h1.index);
  EXPECT_EQ(h2.generation, h1.generation);
  EXPECT_EQ(stats_after_retry.compile_count, stats_before_retry.compile_count);
  EXPECT_EQ(reg.failure_diagnostic(h2), diag);
}

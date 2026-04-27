/**
 * @file test_command_batch.cpp
 * @brief C1 tests for CommandBatchPool and BatchFence.
 */

#include "tiny_metal_nn/runtime/command_batch.h"
#include "tiny_metal_nn/runtime/metal_context_internal.h"

#include <gtest/gtest.h>

using namespace tmnn;

// --- Basic lifecycle ---

TEST(CommandBatch, BeginAndSubmitSync) {
  CommandBatchPool pool(2);
  auto batch = pool.begin_batch();
  EXPECT_NE(batch.generation, 0u);
  EXPECT_EQ(pool.inflight_count(), 1u);

  auto fence = pool.submit(batch, SubmitMode::Sync);
  EXPECT_TRUE(static_cast<bool>(fence));
  EXPECT_TRUE(pool.is_complete(fence));
  EXPECT_EQ(pool.total_submitted(), 1u);
  EXPECT_EQ(pool.inflight_count(), 0u);
}

TEST(CommandBatch, BeginAndSubmitAsync) {
  CommandBatchPool pool(2);
  auto batch = pool.begin_batch();
  auto fence = pool.submit(batch, SubmitMode::Async);

  // C8: async stays Submitted until explicit complete().
  EXPECT_FALSE(pool.is_complete(fence));
  EXPECT_EQ(pool.submitted_count(), 1u);

  pool.complete(fence);
  EXPECT_TRUE(pool.is_complete(fence));
  EXPECT_EQ(pool.submitted_count(), 0u);
  EXPECT_EQ(pool.total_submitted(), 1u);
}

TEST(CommandBatch, MultipleBatches) {
  CommandBatchPool pool(3);
  auto b1 = pool.begin_batch();
  auto b2 = pool.begin_batch();
  EXPECT_EQ(pool.inflight_count(), 2u);

  auto f1 = pool.submit(b1, SubmitMode::Sync);
  EXPECT_EQ(pool.inflight_count(), 1u);

  auto f2 = pool.submit(b2, SubmitMode::Sync);
  EXPECT_EQ(pool.inflight_count(), 0u);

  EXPECT_TRUE(pool.is_complete(f1));
  EXPECT_TRUE(pool.is_complete(f2));
  EXPECT_NE(f1.value, f2.value);
  EXPECT_EQ(pool.total_submitted(), 2u);
}

// --- Pool exhaustion ---

TEST(CommandBatch, PoolExhaustion) {
  CommandBatchPool pool(1);
  auto b1 = pool.begin_batch();
  EXPECT_NE(b1.generation, 0u);

  // Pool is full — next begin returns invalid handle.
  auto b2 = pool.begin_batch();
  EXPECT_EQ(b2.generation, 0u); // invalid

  // Submit first, then a new batch is possible.
  (void)pool.submit(b1, SubmitMode::Sync);
  auto b3 = pool.begin_batch();
  EXPECT_NE(b3.generation, 0u);
  (void)pool.submit(b3, SubmitMode::Sync);
}

// --- Fence ordering ---

TEST(CommandBatch, FenceOrdering) {
  CommandBatchPool pool(4);
  BatchFence fences[4];
  for (int i = 0; i < 4; ++i) {
    auto b = pool.begin_batch();
    fences[i] = pool.submit(b, SubmitMode::Sync);
  }
  // Fence values are monotonically increasing.
  for (int i = 1; i < 4; ++i)
    EXPECT_GT(fences[i].value, fences[i - 1].value);
}

// --- Null fence ---

TEST(CommandBatch, NullFence) {
  CommandBatchPool pool(2);
  BatchFence null_fence{};
  EXPECT_FALSE(static_cast<bool>(null_fence));
  EXPECT_TRUE(pool.is_complete(null_fence));
}

// --- Generation prevents reuse ---

TEST(CommandBatch, GenerationPreventsReuse) {
  CommandBatchPool pool(1);
  auto b1 = pool.begin_batch();
  auto gen1 = b1.generation;
  (void)pool.submit(b1, SubmitMode::Sync);

  auto b2 = pool.begin_batch();
  // Same slot, different generation.
  EXPECT_EQ(b2.slot, b1.slot);
  EXPECT_NE(b2.generation, gen1);
  (void)pool.submit(b2, SubmitMode::Sync);
}

// --- C8: Real async semantics ---

TEST(CommandBatch, AsyncLifecycle) {
  CommandBatchPool pool(2);
  auto batch = pool.begin_batch();
  EXPECT_EQ(pool.inflight_count(), 1u);

  auto fence = pool.submit(batch, SubmitMode::Async);
  EXPECT_FALSE(pool.is_complete(fence));
  EXPECT_EQ(pool.inflight_count(), 1u);

  pool.complete(fence);
  EXPECT_TRUE(pool.is_complete(fence));
  EXPECT_EQ(pool.inflight_count(), 0u);
}

TEST(CommandBatch, MultiAsyncFences) {
  CommandBatchPool pool(2);
  auto b1 = pool.begin_batch();
  auto f1 = pool.submit(b1, SubmitMode::Async);
  auto b2 = pool.begin_batch();
  auto f2 = pool.submit(b2, SubmitMode::Async);

  EXPECT_FALSE(pool.is_complete(f1));
  EXPECT_FALSE(pool.is_complete(f2));
  EXPECT_EQ(pool.submitted_count(), 2u);

  // Complete in FIFO order.
  pool.complete(f1);
  EXPECT_TRUE(pool.is_complete(f1));
  EXPECT_FALSE(pool.is_complete(f2));
  EXPECT_EQ(pool.submitted_count(), 1u);

  pool.complete(f2);
  EXPECT_TRUE(pool.is_complete(f2));
  EXPECT_EQ(pool.submitted_count(), 0u);
  EXPECT_EQ(pool.total_submitted(), 2u);
}

TEST(CommandBatch, SyncStillImmediate) {
  CommandBatchPool pool(2);
  auto batch = pool.begin_batch();
  auto fence = pool.submit(batch, SubmitMode::Sync);
  EXPECT_TRUE(pool.is_complete(fence));
  EXPECT_EQ(pool.inflight_count(), 0u);
  EXPECT_EQ(pool.submitted_count(), 0u);
}

TEST(CommandBatch, PoolExhaustionWithAsync) {
  CommandBatchPool pool(1);
  auto b1 = pool.begin_batch();
  auto f1 = pool.submit(b1, SubmitMode::Async);
  EXPECT_EQ(pool.inflight_count(), 1u);

  // Pool is full — Submitted slot blocks new begins.
  auto b2 = pool.begin_batch();
  EXPECT_EQ(b2.generation, 0u); // invalid

  // Complete the async batch → slot freed.
  pool.complete(f1);
  auto b3 = pool.begin_batch();
  EXPECT_NE(b3.generation, 0u);
  (void)pool.submit(b3, SubmitMode::Sync);
}

TEST(CommandBatch, SubmittedCount) {
  CommandBatchPool pool(3);
  EXPECT_EQ(pool.submitted_count(), 0u);

  auto b1 = pool.begin_batch();
  auto f1 = pool.submit(b1, SubmitMode::Async);
  EXPECT_EQ(pool.submitted_count(), 1u);

  auto b2 = pool.begin_batch();
  (void)pool.submit(b2, SubmitMode::Sync);
  EXPECT_EQ(pool.submitted_count(), 1u); // sync didn't add to submitted

  auto b3 = pool.begin_batch();
  auto f3 = pool.submit(b3, SubmitMode::Async);
  EXPECT_EQ(pool.submitted_count(), 2u);

  pool.complete(f1);
  EXPECT_EQ(pool.submitted_count(), 1u);
  pool.complete(f3);
  EXPECT_EQ(pool.submitted_count(), 0u);
}

// --- GPU-conditional: real Metal command buffers ---

TEST(CommandBatch, RealCommandBufferLifecycle) {
  auto ctx = MetalContext::create();
  if (!ctx->is_gpu_available())
    GTEST_SKIP() << "No GPU";
  auto &pool = detail::context_batch_pool(*ctx);
  auto batch = pool.begin_batch();
  ASSERT_NE(batch.generation, 0u);
  // Submit sync — should commit and complete immediately.
  auto fence = pool.submit(batch, SubmitMode::Sync);
  EXPECT_TRUE(pool.is_complete(fence));
}

TEST(CommandBatch, AsyncCommandBufferWait) {
  auto ctx = MetalContext::create();
  if (!ctx->is_gpu_available())
    GTEST_SKIP() << "No GPU";
  auto &pool = detail::context_batch_pool(*ctx);
  auto batch = pool.begin_batch();
  auto fence = pool.submit(batch, SubmitMode::Async);
  EXPECT_FALSE(pool.is_complete(fence));
  pool.complete(fence);
  EXPECT_TRUE(pool.is_complete(fence));
}

TEST(CommandBatch, GpuTimeUsReturnsNonNegativeAfterSyncSubmit) {
  auto ctx = MetalContext::create();
  if (!ctx->is_gpu_available())
    GTEST_SKIP() << "No GPU";
  auto &pool = detail::context_batch_pool(*ctx);
  auto batch = pool.begin_batch();
  auto fence = pool.submit(batch, SubmitMode::Sync);
  // GPU time may be 0 for empty command buffers, but must not be negative.
  EXPECT_GE(pool.gpu_time_us(fence), 0.0);
}

TEST(CommandBatch, GpuTimeUsReturnsNonNegativeAfterAsyncComplete) {
  auto ctx = MetalContext::create();
  if (!ctx->is_gpu_available())
    GTEST_SKIP() << "No GPU";
  auto &pool = detail::context_batch_pool(*ctx);
  auto batch = pool.begin_batch();
  auto fence = pool.submit(batch, SubmitMode::Async);
  pool.complete(fence);
  EXPECT_GE(pool.gpu_time_us(fence), 0.0);
}

TEST(CommandBatch, GpuTimeUsReturnsZeroForUnknownFence) {
  CommandBatchPool pool(1u);
  BatchFence unknown{999u};
  EXPECT_EQ(pool.gpu_time_us(unknown), 0.0);
}

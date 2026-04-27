/**
 * @file test_buffer_arena.cpp
 * @brief C1 tests for BufferArena, BufferHandle, BufferView, StepBufferSet.
 */

#include "tiny_metal_nn/runtime/buffer_arena.h"
#include "tiny_metal_nn/runtime/metal_context_internal.h"

#include <gtest/gtest.h>

using namespace tmnn;

// --- Basic allocation ---

TEST(BufferArena, AllocateAndRelease) {
  BufferArena arena;
  BufferDesc desc{1024, 256, BufferStorage::Shared,
                  BufferLifetime::Persistent, "test_buffer"};
  auto h = arena.allocate(desc);
  EXPECT_TRUE(arena.is_valid(h));
  EXPECT_EQ(arena.slot_bytes(h), 1024u);
  EXPECT_STREQ(arena.slot_debug_name(h), "test_buffer");
  EXPECT_EQ(arena.slot_lifetime(h), BufferLifetime::Persistent);
  EXPECT_EQ(arena.slot_storage(h), BufferStorage::Shared);
  EXPECT_EQ(arena.bytes_allocated(), 1024u);
  EXPECT_EQ(arena.live_count(), 1u);

  arena.release(h);
  EXPECT_FALSE(arena.is_valid(h));
  EXPECT_EQ(arena.bytes_allocated(), 0u);
  EXPECT_EQ(arena.live_count(), 0u);
}

TEST(BufferArena, SlotReuse) {
  BufferArena arena;
  BufferDesc desc{512};
  auto h1 = arena.allocate(desc);
  uint32_t slot1 = h1.arena_slot;
  arena.release(h1);

  // Next allocation should reuse the slot.
  auto h2 = arena.allocate(desc);
  EXPECT_EQ(h2.arena_slot, slot1);
  // But generation is different.
  EXPECT_NE(h2.generation, h1.generation);
  EXPECT_TRUE(arena.is_valid(h2));
  EXPECT_FALSE(arena.is_valid(h1)); // stale
  arena.release(h2);
}

TEST(BufferArena, SharedBackingReuseZeroesCpuData) {
  BufferArena arena;
  BufferDesc desc{256, 256, BufferStorage::Shared,
                  BufferLifetime::Persistent, "shared_reuse"};
  auto h1 = arena.allocate(desc);
  auto v1 = arena.view(h1);
  ASSERT_NE(v1.data, nullptr);
  auto *ptr1 = static_cast<float *>(v1.data);
  ptr1[0] = 42.0f;
  ptr1[63] = -7.5f;
  void *base1 = v1.data;
  arena.release(h1);

  auto h2 = arena.allocate(desc);
  auto v2 = arena.view(h2);
  ASSERT_NE(v2.data, nullptr);
  EXPECT_EQ(v2.data, base1);
  auto *ptr2 = static_cast<float *>(v2.data);
  EXPECT_FLOAT_EQ(ptr2[0], 0.0f);
  EXPECT_FLOAT_EQ(ptr2[63], 0.0f);
  arena.release(h2);
}

TEST(BufferArena, LifetimeTierDoesNotCrossReuse) {
  BufferArena arena;
  BufferDesc persistent{512, 256, BufferStorage::Shared,
                        BufferLifetime::Persistent, "persistent"};
  BufferDesc transient{512, 256, BufferStorage::Shared,
                       BufferLifetime::Transient, "transient"};
  BufferDesc staging{512, 256, BufferStorage::Shared,
                     BufferLifetime::Staging, "staging"};

  auto hp = arena.allocate(persistent);
  uint32_t persistent_slot = hp.arena_slot;
  arena.release(hp);

  auto ht = arena.allocate(transient);
  uint32_t transient_slot = ht.arena_slot;
  EXPECT_NE(transient_slot, persistent_slot);
  arena.release(ht);

  auto hs = arena.allocate(staging);
  EXPECT_NE(hs.arena_slot, persistent_slot);
  EXPECT_NE(hs.arena_slot, transient_slot);
  arena.release(hs);
}

TEST(BufferArena, ReusesMatchingLifetimeTierWhenAvailable) {
  BufferArena arena;
  BufferDesc persistent{512, 256, BufferStorage::Shared,
                        BufferLifetime::Persistent, "persistent"};
  BufferDesc transient{512, 256, BufferStorage::Shared,
                       BufferLifetime::Transient, "transient"};

  auto hp = arena.allocate(persistent);
  arena.release(hp);

  auto ht1 = arena.allocate(transient);
  uint32_t transient_slot = ht1.arena_slot;
  arena.release(ht1);

  auto ht2 = arena.allocate(transient);
  EXPECT_EQ(ht2.arena_slot, transient_slot);
  EXPECT_NE(ht2.generation, ht1.generation);
  arena.release(ht2);
}

TEST(BufferArena, MultipleAllocations) {
  BufferArena arena;
  auto h1 = arena.allocate({100});
  auto h2 = arena.allocate({200});
  auto h3 = arena.allocate({300});
  EXPECT_EQ(arena.live_count(), 3u);
  EXPECT_EQ(arena.bytes_allocated(), 600u);
  EXPECT_EQ(arena.total_slots(), 3u);

  arena.release(h2);
  EXPECT_EQ(arena.live_count(), 2u);
  EXPECT_EQ(arena.bytes_allocated(), 400u);

  arena.release(h1);
  arena.release(h3);
  EXPECT_EQ(arena.live_count(), 0u);
  EXPECT_EQ(arena.bytes_allocated(), 0u);
}

// --- Stale handle detection ---

TEST(BufferArena, StaleHandleDetection) {
  BufferArena arena;
  auto h = arena.allocate({256});
  arena.release(h);

  // h is now stale.
  EXPECT_FALSE(arena.is_valid(h));

  // Out-of-range slot.
  BufferHandle bad{999, 0};
  EXPECT_FALSE(arena.is_valid(bad));

  // Default-constructed handle.
  BufferHandle empty{};
  EXPECT_FALSE(arena.is_valid(empty));
}

// --- Views ---

TEST(BufferArena, FullView) {
  BufferArena arena;
  auto h = arena.allocate({1024});
  auto v = arena.view(h);
  EXPECT_EQ(v.handle, h);
  EXPECT_EQ(v.offset, 0u);
  EXPECT_EQ(v.bytes, 1024u);
  arena.release(h);
}

TEST(BufferArena, SubView) {
  BufferArena arena;
  auto h = arena.allocate({1024});
  auto full = arena.view(h);

  auto sub = BufferArena::sub_view(full, 256, 512);
  EXPECT_EQ(sub.handle, h);
  EXPECT_EQ(sub.offset, 256u);
  EXPECT_EQ(sub.bytes, 512u);

  // Nested sub-view.
  auto nested = BufferArena::sub_view(sub, 64, 128);
  EXPECT_EQ(nested.handle, h);
  EXPECT_EQ(nested.offset, 320u); // 256 + 64
  EXPECT_EQ(nested.bytes, 128u);

  arena.release(h);
}

TEST(BufferArena, Binding) {
  BufferArena arena;
  auto h = arena.allocate({512});
  auto v = arena.view(h);
  auto binding = BufferArena::bind(v, 3);
  EXPECT_EQ(binding.view.handle, h);
  EXPECT_EQ(binding.binding_index, 3u);
  arena.release(h);
}

// --- Ring-buffered StepBufferSet ---

TEST(BufferArena, StepBufferSetDoubleBuf) {
  BufferArena arena;
  auto lanes = arena.allocate_step_set(
      /*positions_bytes=*/4096,
      /*targets_bytes=*/4096,
      /*reduction_bytes=*/256,
      /*num_lanes=*/2);

  EXPECT_EQ(lanes.size(), 2u);

  // Each lane has 3 buffers.
  EXPECT_EQ(arena.live_count(), 6u);

  // Lane 0 and lane 1 use different buffer handles.
  EXPECT_NE(lanes[0].positions.handle, lanes[1].positions.handle);
  EXPECT_NE(lanes[0].targets.handle, lanes[1].targets.handle);
  EXPECT_NE(lanes[0].loss_reduction.handle, lanes[1].loss_reduction.handle);

  // Sizes are correct.
  EXPECT_EQ(lanes[0].positions.bytes, 4096u);
  EXPECT_EQ(lanes[0].targets.bytes, 4096u);
  EXPECT_EQ(lanes[0].loss_reduction.bytes, 256u);
  EXPECT_EQ(lanes[1].positions.bytes, 4096u);

  // Step buffers are Transient.
  EXPECT_EQ(arena.slot_lifetime(lanes[0].positions.handle),
            BufferLifetime::Transient);

  // Total allocation: 2 * (4096 + 4096 + 256) = 16896
  EXPECT_EQ(arena.bytes_allocated(), 2 * (4096 + 4096 + 256));
}

TEST(BufferArena, ReleaseStepSetReturnsArenaCapacity) {
  BufferArena arena;
  auto lanes = arena.allocate_step_set(
      /*positions_bytes=*/4096,
      /*targets_bytes=*/4096,
      /*reduction_bytes=*/256,
      /*num_lanes=*/2);

  ASSERT_EQ(arena.live_count(), 6u);
  ASSERT_GT(arena.bytes_allocated(), 0u);

  arena.release_step_set(lanes);

  EXPECT_TRUE(lanes.empty());
  EXPECT_EQ(arena.live_count(), 0u);
  EXPECT_EQ(arena.bytes_allocated(), 0u);
}

// --- CPU-backed memory ---

TEST(BufferArena, SharedBufferHasCpuData) {
  BufferArena arena;
  BufferDesc desc{256, 256, BufferStorage::Shared,
                  BufferLifetime::Persistent, "shared_buf"};
  auto h = arena.allocate(desc);
  auto v = arena.view(h);

  ASSERT_NE(v.data, nullptr);

  // Write/read round-trip.
  auto *ptr = static_cast<float *>(v.data);
  ptr[0] = 42.0f;
  ptr[63] = -7.5f; // 256 bytes = 64 floats
  EXPECT_FLOAT_EQ(ptr[0], 42.0f);
  EXPECT_FLOAT_EQ(ptr[63], -7.5f);

  arena.release(h);
}

TEST(BufferArena, PrivateBufferHasNullData) {
  BufferArena arena;
  BufferDesc desc{512, 256, BufferStorage::Private,
                  BufferLifetime::Persistent, "private_buf"};
  auto h = arena.allocate(desc);
  auto v = arena.view(h);

  EXPECT_EQ(v.data, nullptr);

  arena.release(h);
}

TEST(BufferArena, SubViewDataPointer) {
  BufferArena arena;
  BufferDesc desc{1024, 256, BufferStorage::Shared,
                  BufferLifetime::Persistent, "parent_buf"};
  auto h = arena.allocate(desc);
  auto full = arena.view(h);
  ASSERT_NE(full.data, nullptr);

  // Write sentinel at offset 256 (float index 64).
  auto *base = static_cast<float *>(full.data);
  base[64] = 99.0f;

  auto sub = BufferArena::sub_view(full, 256, 512);
  ASSERT_NE(sub.data, nullptr);
  EXPECT_EQ(sub.data, static_cast<char *>(full.data) + 256);

  // Read the sentinel through the sub-view.
  auto *sub_ptr = static_cast<float *>(sub.data);
  EXPECT_FLOAT_EQ(sub_ptr[0], 99.0f);

  arena.release(h);
}

// --- GPU-backed buffers ---

TEST(BufferArena, SharedBufferHasGpuBuffer) {
  auto ctx = MetalContext::create();
  if (!ctx->is_gpu_available())
    GTEST_SKIP() << "No GPU";
  auto &arena = detail::context_arena(*ctx);
  auto h = arena.allocate(
      {1024, 256, BufferStorage::Shared, BufferLifetime::Persistent, "test"});
  auto v = arena.view(h);
  EXPECT_NE(v.data, nullptr);        // CPU accessible
  EXPECT_NE(v.gpu_buffer, nullptr);   // MTLBuffer exists
  // Write via CPU, verify round-trip.
  auto *f = static_cast<float *>(v.data);
  f[0] = 42.0f;
  EXPECT_FLOAT_EQ(f[0], 42.0f);
}

TEST(BufferArena, SharedGpuBackingReuseKeepsMetalBuffer) {
  auto ctx = MetalContext::create();
  if (!ctx->is_gpu_available())
    GTEST_SKIP() << "No GPU";
  auto &arena = detail::context_arena(*ctx);
  BufferDesc desc{1024, 256, BufferStorage::Shared,
                  BufferLifetime::Persistent, "reuse_gpu"};

  auto h1 = arena.allocate(desc);
  auto v1 = arena.view(h1);
  ASSERT_NE(v1.data, nullptr);
  ASSERT_NE(v1.gpu_buffer, nullptr);
  auto *f1 = static_cast<float *>(v1.data);
  f1[0] = 42.0f;
  void *gpu1 = v1.gpu_buffer;
  void *cpu1 = v1.data;
  arena.release(h1);

  auto h2 = arena.allocate(desc);
  auto v2 = arena.view(h2);
  ASSERT_NE(v2.data, nullptr);
  ASSERT_NE(v2.gpu_buffer, nullptr);
  EXPECT_EQ(v2.gpu_buffer, gpu1);
  EXPECT_EQ(v2.data, cpu1);
  auto *f2 = static_cast<float *>(v2.data);
  EXPECT_FLOAT_EQ(f2[0], 0.0f);
  arena.release(h2);
}

TEST(BufferArena, PrivateBufferHasGpuOnly) {
  auto ctx = MetalContext::create();
  if (!ctx->is_gpu_available())
    GTEST_SKIP() << "No GPU";
  auto &arena = detail::context_arena(*ctx);
  auto h = arena.allocate(
      {1024, 256, BufferStorage::Private, BufferLifetime::Persistent, "test"});
  auto v = arena.view(h);
  EXPECT_EQ(v.data, nullptr);         // No CPU access
  EXPECT_NE(v.gpu_buffer, nullptr);   // MTLBuffer exists
}

// --- MetalContext integration ---

TEST(BufferArena, ContextArenaAccess) {
  auto ctx = MetalContext::create();
  auto &arena = detail::context_arena(*ctx);

  auto h = arena.allocate({512, 256, BufferStorage::Private,
                           BufferLifetime::Persistent, "ctx_buf"});
  EXPECT_TRUE(arena.is_valid(h));
  EXPECT_EQ(arena.slot_bytes(h), 512u);

  // Stats flow through to context.
  auto stats = ctx->snapshot_stats();
  EXPECT_EQ(stats.persistent_bytes, 512u);

  arena.release(h);
}

TEST(BufferArena, ContextBatchPoolAccess) {
  auto ctx = MetalContext::create();
  auto &pool = detail::context_batch_pool(*ctx);

  auto batch = pool.begin_batch();
  EXPECT_NE(batch.generation, 0u);

  auto fence = pool.submit(batch, SubmitMode::Sync);
  EXPECT_TRUE(static_cast<bool>(fence));
  EXPECT_TRUE(pool.is_complete(fence));

  auto stats = ctx->snapshot_stats();
  EXPECT_EQ(stats.async_batches_submitted, 1u);
}

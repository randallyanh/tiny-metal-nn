/**
 * @file test_metal_heap.mm
 * @brief Persistent pool semantics for the standalone metal_heap module.
 */

#include "tiny_metal_nn/runtime/metal_heap/metal_heap.h"

#import <Metal/Metal.h>
#include <gtest/gtest.h>
#include <chrono>
#include <type_traits>
#include <utility>

namespace {

using metal_heap::AllocDesc;
using metal_heap::AllocError;
using metal_heap::HazardTracking;
using metal_heap::Heap;
using metal_heap::HeapConfig;
using metal_heap::Lifetime;
using metal_heap::MetalDevice;
using metal_heap::OwnedBuffer;
using metal_heap::Storage;

class MetalHeapTest : public ::testing::Test {
protected:
  void SetUp() override {
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) GTEST_SKIP() << "no Metal device";
  }
  void TearDown() override { [device_ release]; }

  std::unique_ptr<Heap> make_heap(std::size_t cap = 4 * 1024 * 1024) {
    HeapConfig cfg;
    cfg.persistent_shared_capacity_bytes  = cap;
    cfg.persistent_private_capacity_bytes = cap;
    cfg.transient_lane_bytes              = cap;
    cfg.transient_lane_count              = 4;
    return Heap::create((MetalDevice)device_, cfg);
  }

  AllocDesc desc(std::size_t bytes, const char *name = "test.persistent") {
    AllocDesc d;
    d.bytes = bytes;
    d.alignment = 256;
    d.lifetime = Lifetime::Persistent;
    d.storage = Storage::Shared;
    d.hazard_tracking = HazardTracking::Untracked;
    d.debug_name = name;
    return d;
  }

  AllocDesc transient_desc(std::size_t bytes,
                           const char *name = "test.transient") {
    AllocDesc d = desc(bytes, name);
    d.lifetime = Lifetime::Transient;
    return d;
  }

  id<MTLDevice> device_ = nil;
};

TEST_F(MetalHeapTest, CreateRejectsNullDevice) {
  HeapConfig cfg;
  EXPECT_EQ(Heap::create(nullptr, cfg).get(), nullptr);
}

TEST_F(MetalHeapTest, CreateRejectsZeroCapacity) {
  HeapConfig cfg;
  cfg.persistent_shared_capacity_bytes  = 0;
  cfg.persistent_private_capacity_bytes = 0;
  EXPECT_EQ(Heap::create((MetalDevice)device_, cfg).get(), nullptr);
}

TEST_F(MetalHeapTest, AllocateReturnsValidBufferAndStatsUpdate) {
  auto heap = make_heap();
  ASSERT_NE(heap, nullptr);

  auto result = heap->allocate(desc(4096));
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->valid());
  EXPECT_NE(result->mtl_buffer(), nullptr);
  EXPECT_NE(result->cpu_data(), nullptr);  // Shared storage
  EXPECT_EQ(result->bytes(), 4096u);

  const auto s = heap->stats();
  EXPECT_EQ(s.live_persistent_buffers, 1u);
  EXPECT_EQ(s.total_allocations, 1u);
  EXPECT_GE(s.persistent_shared_used_bytes, 4096u);
}

TEST_F(MetalHeapTest, DtorReleasesBufferToHeap) {
  auto heap = make_heap();
  ASSERT_NE(heap, nullptr);
  {
    auto buf = heap->allocate(desc(2048)).value();
    EXPECT_EQ(heap->stats().live_persistent_buffers, 1u);
  }
  EXPECT_EQ(heap->stats().live_persistent_buffers, 0u);
}

TEST_F(MetalHeapTest, MissingDebugNameIsInvalidConfig) {
  auto heap = make_heap();
  AllocDesc d = desc(1024);
  d.debug_name = nullptr;
  auto r = heap->allocate(d);
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error(), AllocError::InvalidConfig);
}

TEST_F(MetalHeapTest, NonPersistentLifetimeUnsupportedInPhase21) {
  auto heap = make_heap();
  AllocDesc d = desc(1024);
  // Lifetime::Transient / Staging not yet wired; current enum has only
  // Persistent so this path tests the safety check by force-casting.
  d.lifetime = static_cast<Lifetime>(99);
  auto r = heap->allocate(d);
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error(), AllocError::UnsupportedLifetime);
}

TEST_F(MetalHeapTest, ExhaustionReturnsPersistentExhausted) {
  auto heap = make_heap(64 * 1024);  // 64 KiB heap
  // Allocate beyond capacity — the heap should refuse.
  auto r = heap->allocate(desc(1024 * 1024));
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error(), AllocError::PersistentExhausted);
  EXPECT_EQ(heap->stats().alloc_failures, 1u);
}

TEST_F(MetalHeapTest, MoveCtorTransfersOwnershipNoLeak) {
  auto heap = make_heap();
  auto a = heap->allocate(desc(1024)).value();
  auto b = std::move(a);
  EXPECT_FALSE(a.valid());
  EXPECT_TRUE(b.valid());
  EXPECT_EQ(heap->stats().live_persistent_buffers, 1u);
}

TEST_F(MetalHeapTest, MoveAssignReleasesPreviousHoldsNew) {
  auto heap = make_heap();
  auto a = heap->allocate(desc(1024)).value();
  auto b = heap->allocate(desc(2048)).value();
  EXPECT_EQ(heap->stats().live_persistent_buffers, 2u);
  a = std::move(b);
  EXPECT_EQ(heap->stats().live_persistent_buffers, 1u);
  EXPECT_TRUE(a.valid());
  EXPECT_FALSE(b.valid());
}

TEST_F(MetalHeapTest, SelfMoveAssignIsSafe) {
  auto heap = make_heap();
  auto a = heap->allocate(desc(1024)).value();
  a = std::move(a);
  EXPECT_TRUE(a.valid());
  EXPECT_EQ(heap->stats().live_persistent_buffers, 1u);
}

TEST_F(MetalHeapTest, PrivateStorageHasNoCpuPointer) {
  auto heap = make_heap();
  AllocDesc d = desc(1024, "test.private");
  d.storage = Storage::Private;
  auto buf = heap->allocate(d).value();
  EXPECT_EQ(buf.cpu_data(), nullptr);
  EXPECT_NE(buf.mtl_buffer(), nullptr);
}

// G5 microbench gate — not a hard pass/fail in unit tests; printed for
// inspection. The release-build figure on the bench binary is the gate.
TEST_F(MetalHeapTest, AllocateMicrobench) {
  auto heap = make_heap(64 * 1024 * 1024);
  constexpr int kIter = 10000;
  std::vector<OwnedBuffer> bufs;
  bufs.reserve(kIter);
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kIter; ++i) {
    bufs.emplace_back(heap->allocate(desc(256, "test.micro")).value());
  }
  const auto t1 = std::chrono::steady_clock::now();
  const auto ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  std::printf("[metal_heap] allocate microbench (Persistent): "
              "%.0f ns/call (%d iter)\n",
              static_cast<double>(ns) / kIter, kIter);
}

// === Transient lifetime ===

TEST_F(MetalHeapTest, TransientAllocateBumpsLaneOffset) {
  auto heap = make_heap();
  heap->begin_transient_frame();
  auto a = heap->allocate(transient_desc(1024)).value();
  auto b = heap->allocate(transient_desc(1024)).value();
  EXPECT_NE(a.cpu_data(), nullptr);
  EXPECT_NE(b.cpu_data(), nullptr);
  EXPECT_EQ(a.mtl_buffer(), b.mtl_buffer());          // same lane buffer
  EXPECT_LT(a.offset(), b.offset());                  // bump forward
  EXPECT_EQ(b.offset(), 1024u);                       // 256-aligned: 1024 -> 1024
}

TEST_F(MetalHeapTest, TransientFrameRotationResetsOffset) {
  auto heap = make_heap();
  heap->begin_transient_frame();
  const auto lane0 = heap->stats().transient_current_lane;
  (void)heap->allocate(transient_desc(2048)).value();
  EXPECT_GT(heap->stats().transient_current_lane_used_bytes, 0u);
  heap->end_transient_frame();

  heap->begin_transient_frame();
  EXPECT_NE(heap->stats().transient_current_lane, lane0);
  EXPECT_EQ(heap->stats().transient_current_lane_used_bytes, 0u);
}

TEST_F(MetalHeapTest, TransientCapacityExceededReturnsError) {
  HeapConfig cfg;
  cfg.persistent_shared_capacity_bytes  = 1024 * 1024;
  cfg.persistent_private_capacity_bytes = 0;
  cfg.transient_lane_bytes              = 4096;
  cfg.transient_lane_count              = 2;
  auto heap = Heap::create((MetalDevice)device_, cfg);
  ASSERT_NE(heap, nullptr);
  heap->begin_transient_frame();
  auto r = heap->allocate(transient_desc(8192));
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error(), AllocError::TransientCapacityExceeded);
}

TEST_F(MetalHeapTest, TransientPrivateStorageRejected) {
  auto heap = make_heap();
  heap->begin_transient_frame();
  AllocDesc d = transient_desc(1024);
  d.storage = Storage::Private;
  auto r = heap->allocate(d);
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error(), AllocError::UnsupportedStorage);
}

TEST_F(MetalHeapTest, TransientDtorIsNoOpForLaneStorage) {
  auto heap = make_heap();
  heap->begin_transient_frame();
  std::size_t lane_used_before = 0;
  {
    auto buf = heap->allocate(transient_desc(1024)).value();
    lane_used_before = heap->stats().transient_current_lane_used_bytes;
    EXPECT_EQ(lane_used_before, 1024u);
  }
  // Lane offset is NOT decremented by buffer destruction; it resets only
  // on the next begin_transient_frame() landing on this lane.
  EXPECT_EQ(heap->stats().transient_current_lane_used_bytes, lane_used_before);
}

// G6: begin_transient_frame() rotation cost — sub-1µs target. Measured
// without a registered fence event (the with-fence path is dominated by GPU
// signal latency, not heap bookkeeping, and is exercised separately).
TEST_F(MetalHeapTest, BeginTransientFrameMicrobench) {
  HeapConfig cfg;
  cfg.persistent_shared_capacity_bytes  = 1024 * 1024;
  cfg.persistent_private_capacity_bytes = 0;
  cfg.transient_lane_bytes              = 1024 * 1024;
  cfg.transient_lane_count              = 4;
  auto heap = Heap::create((MetalDevice)device_, cfg);
  ASSERT_NE(heap, nullptr);
  constexpr int kIter = 10000;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kIter; ++i) {
    heap->begin_transient_frame();
    heap->end_transient_frame();
  }
  const auto t1 = std::chrono::steady_clock::now();
  const auto ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  std::printf("[metal_heap] begin/end transient frame: "
              "%.0f ns/pair (%d iter)\n",
              static_cast<double>(ns) / kIter, kIter);
}

// G5b: transient hot-path microbench — sub-100ns target.
TEST_F(MetalHeapTest, TransientAllocateMicrobench) {
  HeapConfig cfg;
  cfg.persistent_shared_capacity_bytes  = 1024 * 1024;
  cfg.persistent_private_capacity_bytes = 0;
  cfg.transient_lane_bytes              = 64 * 1024 * 1024;  // 64 MiB lane
  cfg.transient_lane_count              = 1;
  auto heap = Heap::create((MetalDevice)device_, cfg);
  heap->begin_transient_frame();
  constexpr int kIter = 10000;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kIter; ++i) {
    auto _ = heap->allocate(transient_desc(256, "test.t.micro")).value();
    (void)_;
  }
  const auto t1 = std::chrono::steady_clock::now();
  const auto ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  std::printf("[metal_heap] allocate microbench (Transient): "
              "%.0f ns/call (%d iter)\n",
              static_cast<double>(ns) / kIter, kIter);
}

// === Staging lifetime ===

TEST_F(MetalHeapTest, StagingAllocateRoundsUpToBucket) {
  auto heap = make_heap();
  AllocDesc d = desc(700, "test.staging");
  d.lifetime = Lifetime::Staging;
  auto buf = heap->allocate(d).value();
  EXPECT_NE(buf.cpu_data(), nullptr);
  EXPECT_EQ(buf.bytes(), 1024u);  // 700 → bucket 1024
  EXPECT_EQ(buf.lifetime(), Lifetime::Staging);
  EXPECT_EQ(heap->stats().live_staging_buffers, 1u);
  EXPECT_EQ(heap->stats().staging_buffers_resident, 1u);
}

TEST_F(MetalHeapTest, StagingReuseHitsFreeList) {
  auto heap = make_heap();
  AllocDesc d = desc(1024, "test.staging");
  d.lifetime = Lifetime::Staging;
  void *first_buf = nullptr;
  {
    auto a = heap->allocate(d).value();
    first_buf = a.mtl_buffer();
    EXPECT_EQ(heap->stats().staging_buffers_resident, 1u);
  }
  // Same bucket; pool returns the same MTLBuffer without re-allocating.
  auto b = heap->allocate(d).value();
  EXPECT_EQ(b.mtl_buffer(), first_buf);
  EXPECT_EQ(heap->stats().staging_buffers_resident, 1u);
  EXPECT_EQ(heap->stats().total_allocations, 2u);
}

TEST_F(MetalHeapTest, StagingExhaustionReturnsStagingExhausted) {
  HeapConfig cfg;
  cfg.persistent_shared_capacity_bytes  = 1024 * 1024;
  cfg.persistent_private_capacity_bytes = 0;
  cfg.transient_lane_count              = 0;
  cfg.staging_max_bytes                 = 4096;
  auto heap = Heap::create((MetalDevice)device_, cfg);
  ASSERT_NE(heap, nullptr);

  AllocDesc d = desc(4096, "test.staging.exhaust");
  d.lifetime = Lifetime::Staging;
  auto a = heap->allocate(d).value();
  // Cap fully consumed; second concurrent acquire must fail.
  auto r = heap->allocate(d);
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error(), AllocError::StagingExhausted);
}

TEST_F(MetalHeapTest, StagingPrivateStorageRejected) {
  auto heap = make_heap();
  AllocDesc d = desc(1024, "test.staging.priv");
  d.lifetime = Lifetime::Staging;
  d.storage = Storage::Private;
  auto r = heap->allocate(d);
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error(), AllocError::UnsupportedStorage);
}

TEST_F(MetalHeapTest, StagingDisabledReturnsUnsupportedLifetime) {
  HeapConfig cfg;
  cfg.persistent_shared_capacity_bytes  = 1024 * 1024;
  cfg.persistent_private_capacity_bytes = 0;
  cfg.transient_lane_count              = 0;
  cfg.staging_max_bytes                 = 0;  // disable staging
  auto heap = Heap::create((MetalDevice)device_, cfg);
  ASSERT_NE(heap, nullptr);
  AllocDesc d = desc(1024, "test.staging.disabled");
  d.lifetime = Lifetime::Staging;
  auto r = heap->allocate(d);
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error(), AllocError::UnsupportedLifetime);
}

// === adopt_external ===

TEST_F(MetalHeapTest, AdoptExternalDoesNotReleaseBuffer) {
  auto heap = make_heap();
  id<MTLBuffer> external =
      [device_ newBufferWithLength:2048
                           options:MTLResourceStorageModeShared];
  ASSERT_NE(external, nil);
  // Caller's +1 retain. adopt_external must NOT release it on dtor.
  {
    auto buf = heap->adopt_external((metal_heap::MetalBuffer)external,
                                    2048, Storage::Shared);
    EXPECT_TRUE(buf.valid());
    EXPECT_EQ(buf.lifetime(), Lifetime::External);
    EXPECT_EQ(buf.mtl_buffer(), (void *)external);
    EXPECT_NE(buf.cpu_data(), nullptr);
    EXPECT_EQ(heap->stats().live_external_buffers, 1u);
  }
  EXPECT_EQ(heap->stats().live_external_buffers, 0u);
  // External buffer still alive — touching it must not crash.
  EXPECT_EQ([external length], 2048u);
  [external release];
}

TEST_F(MetalHeapTest, AdoptExternalCarriesSyncEvent) {
  auto heap = make_heap();
  id<MTLBuffer> external =
      [device_ newBufferWithLength:1024
                           options:MTLResourceStorageModeShared];
  id<MTLSharedEvent> ev = [device_ newSharedEvent];
  ASSERT_NE(ev, nil);
  auto buf = heap->adopt_external((metal_heap::MetalBuffer)external, 1024,
                                  Storage::Shared, (void *)ev, /*signal*/ 7);
  const auto se = buf.sync_event();
  EXPECT_EQ(se.event, (void *)ev);
  EXPECT_EQ(se.signal_value, 7u);
  [ev release];
  [external release];
}

TEST_F(MetalHeapTest, ZeroBytesIsInvalidConfig) {
  auto heap = make_heap();
  AllocDesc d = desc(0);
  auto r = heap->allocate(d);
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error(), AllocError::InvalidConfig);
}

TEST_F(MetalHeapTest, AdoptExternalNullReturnsInvalid) {
  auto heap = make_heap();
  auto buf = heap->adopt_external(nullptr, 0, Storage::Shared);
  EXPECT_FALSE(buf.valid());
  EXPECT_EQ(heap->stats().live_external_buffers, 0u);
}

TEST_F(MetalHeapTest, AdoptExternalPrivateHasNoCpuPointer) {
  auto heap = make_heap();
  id<MTLBuffer> external =
      [device_ newBufferWithLength:1024
                           options:MTLResourceStorageModePrivate];
  ASSERT_NE(external, nil);
  {
    auto buf = heap->adopt_external((metal_heap::MetalBuffer)external,
                                    1024, Storage::Private);
    EXPECT_TRUE(buf.valid());
    EXPECT_EQ(buf.cpu_data(), nullptr);
  }
  [external release];
}

// === Transient frame-token invalidation (Phase 2.3) ===

TEST_F(MetalHeapTest, TransientFrameTokenInvalidatesAfterEndFrame) {
  auto heap = make_heap();
  heap->begin_transient_frame();
  auto buf = heap->allocate(transient_desc(1024)).value();
  EXPECT_TRUE(buf.valid());
  heap->end_transient_frame();
  // Frame counter has advanced past this allocation's token.
  EXPECT_FALSE(buf.valid());
}

// === Cross-queue fence wiring ===

TEST_F(MetalHeapTest, PendingLaneSignalValueWithoutEvent) {
  auto heap = make_heap();
  EXPECT_EQ(heap->pending_lane_signal_value(), 0u);
}

TEST_F(MetalHeapTest, OwnedBufferMovePreservesSyncEvent) {
  auto heap = make_heap();
  id<MTLBuffer> external =
      [device_ newBufferWithLength:1024
                           options:MTLResourceStorageModeShared];
  id<MTLSharedEvent> ev = [device_ newSharedEvent];
  ASSERT_NE(external, nil);
  ASSERT_NE(ev, nil);

  auto a = heap->adopt_external((metal_heap::MetalBuffer)external, 1024,
                                Storage::Shared, (void *)ev, /*signal*/ 42);
  // Move ctor copies sync_event; the source is reset to {nullptr, 0}.
  auto b = std::move(a);
  EXPECT_EQ(b.sync_event().event, (void *)ev);
  EXPECT_EQ(b.sync_event().signal_value, 42u);
  EXPECT_EQ(a.sync_event().event, nullptr);
  EXPECT_EQ(a.sync_event().signal_value, 0u);

  // Move assign overwrites the destination's sync_event.
  id<MTLBuffer> other =
      [device_ newBufferWithLength:512
                           options:MTLResourceStorageModeShared];
  auto c = heap->adopt_external((metal_heap::MetalBuffer)other, 512,
                                Storage::Shared);
  c = std::move(b);
  EXPECT_EQ(c.sync_event().event, (void *)ev);
  EXPECT_EQ(c.sync_event().signal_value, 42u);

  [ev release];
  [external release];
  [other release];
}

// Re-registering a different event without first detaching: the per-lane
// last_signal targets from the prior event would never be reached on the
// new event and would deadlock the next begin_transient_frame on wrap. The
// fix zeros lane.last_signal in register_fence; ctest's per-test TIMEOUT
// catches a regression by killing the test rather than letting it block.
TEST_F(MetalHeapTest, RegisterLaneFenceEventResetsLaneSignalsOnReplace) {
  HeapConfig cfg;
  cfg.persistent_shared_capacity_bytes  = 1024 * 1024;
  cfg.persistent_private_capacity_bytes = 0;
  cfg.transient_lane_bytes              = 1024 * 1024;
  cfg.transient_lane_count              = 1;  // every begin wraps to lane 0
  auto heap = Heap::create((MetalDevice)device_, cfg);
  ASSERT_NE(heap, nullptr);

  id<MTLSharedEvent> ev_a = [device_ newSharedEvent];
  ASSERT_NE(ev_a, nil);
  heap->register_lane_fence_event((void *)ev_a, 0);
  heap->begin_transient_frame();
  heap->end_transient_frame();  // lane[0].last_signal becomes 1 vs ev_a

  id<MTLSharedEvent> ev_b = [device_ newSharedEvent];
  ASSERT_NE(ev_b, nil);
  heap->register_lane_fence_event((void *)ev_b, 0);
  EXPECT_EQ(heap->pending_lane_signal_value(), 0u);
  heap->begin_transient_frame();  // would block-forever without the reset
  heap->end_transient_frame();
  EXPECT_EQ(heap->pending_lane_signal_value(), 1u);

  heap->register_lane_fence_event(nullptr, 0);
  [ev_a release];
  [ev_b release];
}

TEST_F(MetalHeapTest, RegisterLaneFenceEventAdvancesSignalOnEndFrame) {
  auto heap = make_heap();
  id<MTLSharedEvent> ev = [device_ newSharedEvent];
  ASSERT_NE(ev, nil);
  heap->register_lane_fence_event((void *)ev, /*baseline*/ 5);
  EXPECT_EQ(heap->pending_lane_signal_value(), 5u);

  heap->begin_transient_frame();  // first lane: last_signal == 0, no wait
  heap->end_transient_frame();
  EXPECT_EQ(heap->pending_lane_signal_value(), 6u);
  // Detach so dtor does not retain a stale event reference.
  heap->register_lane_fence_event(nullptr, 0);
  [ev release];
}

// === Split live counters ===

TEST_F(MetalHeapTest, LiveCountersSplitByLifetime) {
  auto heap = make_heap();
  // Persistent
  auto p = heap->allocate(desc(1024, "test.split.p")).value();
  // Transient
  heap->begin_transient_frame();
  auto t = heap->allocate(transient_desc(1024, "test.split.t")).value();
  // Staging
  AllocDesc sd = desc(1024, "test.split.s");
  sd.lifetime = Lifetime::Staging;
  auto s = heap->allocate(sd).value();
  // External
  id<MTLBuffer> ext =
      [device_ newBufferWithLength:512
                           options:MTLResourceStorageModeShared];
  auto e = heap->adopt_external((metal_heap::MetalBuffer)ext, 512,
                                Storage::Shared);

  const auto stats = heap->stats();
  EXPECT_EQ(stats.live_persistent_buffers, 1u);
  EXPECT_EQ(stats.live_transient_buffers,  1u);
  EXPECT_EQ(stats.live_staging_buffers,    1u);
  EXPECT_EQ(stats.live_external_buffers,   1u);
  EXPECT_EQ(stats.total_allocations, 4u);
  [ext release];
}

static_assert(!std::is_copy_constructible_v<OwnedBuffer>);
static_assert(!std::is_copy_assignable_v<OwnedBuffer>);
static_assert(std::is_nothrow_move_constructible_v<OwnedBuffer>);
static_assert(std::is_nothrow_move_assignable_v<OwnedBuffer>);

}  // namespace

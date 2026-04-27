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
  EXPECT_EQ(s.live_buffers, 1u);
  EXPECT_EQ(s.total_allocations, 1u);
  EXPECT_GE(s.persistent_shared_used_bytes, 4096u);
}

TEST_F(MetalHeapTest, DtorReleasesBufferToHeap) {
  auto heap = make_heap();
  ASSERT_NE(heap, nullptr);
  {
    auto buf = heap->allocate(desc(2048)).value();
    EXPECT_EQ(heap->stats().live_buffers, 1u);
  }
  EXPECT_EQ(heap->stats().live_buffers, 0u);
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
  EXPECT_EQ(heap->stats().live_buffers, 1u);
}

TEST_F(MetalHeapTest, MoveAssignReleasesPreviousHoldsNew) {
  auto heap = make_heap();
  auto a = heap->allocate(desc(1024)).value();
  auto b = heap->allocate(desc(2048)).value();
  EXPECT_EQ(heap->stats().live_buffers, 2u);
  a = std::move(b);
  EXPECT_EQ(heap->stats().live_buffers, 1u);
  EXPECT_TRUE(a.valid());
  EXPECT_FALSE(b.valid());
}

TEST_F(MetalHeapTest, SelfMoveAssignIsSafe) {
  auto heap = make_heap();
  auto a = heap->allocate(desc(1024)).value();
  a = std::move(a);
  EXPECT_TRUE(a.valid());
  EXPECT_EQ(heap->stats().live_buffers, 1u);
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
  std::printf("[metal_heap] allocate microbench: %.0f ns/call (%d iter)\n",
              static_cast<double>(ns) / kIter, kIter);
}

static_assert(!std::is_copy_constructible_v<OwnedBuffer>);
static_assert(!std::is_copy_assignable_v<OwnedBuffer>);
static_assert(std::is_nothrow_move_constructible_v<OwnedBuffer>);
static_assert(std::is_nothrow_move_assignable_v<OwnedBuffer>);

}  // namespace

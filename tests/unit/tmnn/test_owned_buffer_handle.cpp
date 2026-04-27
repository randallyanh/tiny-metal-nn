/**
 * @file test_owned_buffer_handle.cpp
 * @brief OwnedBufferHandle / BorrowedBufferView RAII semantics.
 */

#include "tiny_metal_nn/runtime/owned_buffer_handle.h"
#include "tiny_metal_nn/runtime/buffer_arena.h"

#include <gtest/gtest.h>
#include <type_traits>
#include <utility>

namespace {

using tmnn::BufferArena;
using tmnn::BufferDesc;
using tmnn::BufferLifetime;
using tmnn::BufferStorage;
using tmnn::detail::OwnedBufferHandle;
using tmnn::detail::BorrowedBufferView;

BufferDesc make_desc(std::size_t bytes) {
  BufferDesc d{};
  d.bytes = bytes;
  d.alignment = 256;
  d.storage = BufferStorage::Shared;
  d.lifetime = BufferLifetime::Persistent;
  d.debug_name = "test";
  return d;
}

class OwnedBufferHandleTest : public ::testing::Test {
protected:
  BufferArena arena_;
};

TEST_F(OwnedBufferHandleTest, NullCtor) {
  OwnedBufferHandle h;
  EXPECT_FALSE(h.valid());
  EXPECT_EQ(h.arena(), nullptr);
}

TEST_F(OwnedBufferHandleTest, AllocateOwnedHoldsSlotAndDtorReleases) {
  EXPECT_EQ(arena_.live_count(), 0u);
  {
    auto h = arena_.allocate_owned(make_desc(64));
    EXPECT_TRUE(h.valid());
    EXPECT_EQ(arena_.live_count(), 1u);
  }
  EXPECT_EQ(arena_.live_count(), 0u);
}

TEST_F(OwnedBufferHandleTest, MoveCtorTransfersOwnership) {
  auto a = arena_.allocate_owned(make_desc(64));
  const auto raw = a.raw();
  auto b = std::move(a);
  EXPECT_FALSE(a.valid());
  EXPECT_TRUE(b.valid());
  EXPECT_EQ(b.raw(), raw);
  EXPECT_EQ(arena_.live_count(), 1u);
}

TEST_F(OwnedBufferHandleTest, MoveAssignReleasesPreviousHoldsNew) {
  auto a = arena_.allocate_owned(make_desc(32));
  auto b = arena_.allocate_owned(make_desc(64));
  EXPECT_EQ(arena_.live_count(), 2u);
  a = std::move(b);
  EXPECT_EQ(arena_.live_count(), 1u);
  EXPECT_TRUE(a.valid());
  EXPECT_FALSE(b.valid());
}

TEST_F(OwnedBufferHandleTest, ReleaseHandsOffWithoutFree) {
  auto h = arena_.allocate_owned(make_desc(64));
  auto raw = h.release();
  EXPECT_FALSE(h.valid());
  EXPECT_EQ(arena_.live_count(), 1u);  // not yet released
  arena_.release(raw.handle);           // caller now responsible
  EXPECT_EQ(arena_.live_count(), 0u);
}

TEST_F(OwnedBufferHandleTest, SelfMoveAssignIsSafe) {
  auto h = arena_.allocate_owned(make_desc(64));
  h = std::move(h);
  EXPECT_TRUE(h.valid());
  EXPECT_EQ(arena_.live_count(), 1u);
}

TEST_F(OwnedBufferHandleTest, ViewRoundTripsBytesAndOffset) {
  auto h = arena_.allocate_owned(make_desc(128));
  auto v = h.view();
  EXPECT_TRUE(v.valid());
  EXPECT_EQ(v.bytes(), 128u);
  EXPECT_EQ(v.offset(), 0u);
}

TEST_F(OwnedBufferHandleTest, SubViewOffsetsAccumulate) {
  auto h = arena_.allocate_owned(make_desc(128));
  auto whole = h.view();
  auto half = whole.sub_view(32, 64);
  EXPECT_TRUE(half.valid());
  EXPECT_EQ(half.bytes(), 64u);
  EXPECT_EQ(half.offset(), 32u);
}

TEST_F(OwnedBufferHandleTest, BorrowedViewBecomesInvalidAfterHandleDtor) {
  BorrowedBufferView v;
  {
    auto h = arena_.allocate_owned(make_desc(64));
    v = h.view();
    EXPECT_TRUE(v.valid());
  }
  // Slot released; generation bumped; view's stored generation no longer matches.
  EXPECT_FALSE(v.valid());
}

TEST_F(OwnedBufferHandleTest, BindProducesIndexedBinding) {
  auto h = arena_.allocate_owned(make_desc(64));
  auto v = h.view();
  auto bind = v.bind(3);
  EXPECT_EQ(bind.binding_index, 3u);
  EXPECT_EQ(bind.view.handle, v.handle());
}

// Compile-time guarantees.
static_assert(!std::is_copy_constructible_v<OwnedBufferHandle>);
static_assert(!std::is_copy_assignable_v<OwnedBufferHandle>);
static_assert(std::is_nothrow_move_constructible_v<OwnedBufferHandle>);
static_assert(std::is_nothrow_move_assignable_v<OwnedBufferHandle>);

}  // namespace

/**
 * @file test_metal_obj_ptr.mm
 * @brief MetalObjPtr<T> RAII semantics under MRC.
 */

#include "tiny_metal_nn/runtime/metal_obj_ptr.h"

#import <Foundation/Foundation.h>
#include <gtest/gtest.h>
#include <type_traits>

namespace {

// dealloc-counting NSObject subclass; lets tests assert exact release timing
// without depending on -retainCount (which is flaky around autorelease pools).
static int g_dealloc_count = 0;

}  // namespace

@interface TmnnTestObj : NSObject
@end

@implementation TmnnTestObj
- (void)dealloc {
  ++g_dealloc_count;
  [super dealloc];
}
@end

namespace {

using tmnn::detail::MetalObjPtr;
using Ptr = MetalObjPtr<TmnnTestObj *>;

class MetalObjPtrTest : public ::testing::Test {
protected:
  void SetUp() override { g_dealloc_count = 0; }
};

TEST_F(MetalObjPtrTest, NullCtorAndDtorIsNoOp) {
  { Ptr p; }
  EXPECT_EQ(g_dealloc_count, 0);
}

TEST_F(MetalObjPtrTest, AdoptHoldsAndReleasesOnDtor) {
  {
    auto p = Ptr::adopt([[TmnnTestObj alloc] init]);
    ASSERT_TRUE(p);
    EXPECT_EQ(g_dealloc_count, 0);
  }
  EXPECT_EQ(g_dealloc_count, 1);
}

TEST_F(MetalObjPtrTest, RetainAddsRefcount) {
  TmnnTestObj *raw = [[TmnnTestObj alloc] init];
  {
    auto p = Ptr::retain(raw);
    EXPECT_EQ(g_dealloc_count, 0);
  }
  // p released its +1 retain; raw still has the original.
  EXPECT_EQ(g_dealloc_count, 0);
  [raw release];
  EXPECT_EQ(g_dealloc_count, 1);
}

TEST_F(MetalObjPtrTest, MoveCtorTransfersOwnership) {
  auto src = Ptr::adopt([[TmnnTestObj alloc] init]);
  auto *raw = src.get();
  auto dst = std::move(src);
  EXPECT_EQ(dst.get(), raw);
  EXPECT_FALSE(src);
  EXPECT_EQ(g_dealloc_count, 0);
}

TEST_F(MetalObjPtrTest, MoveAssignReleasesOldHolds_New) {
  auto a = Ptr::adopt([[TmnnTestObj alloc] init]);
  auto b = Ptr::adopt([[TmnnTestObj alloc] init]);
  ASSERT_EQ(g_dealloc_count, 0);
  a = std::move(b);
  // a's previous obj released; b now null; new obj still held by a.
  EXPECT_EQ(g_dealloc_count, 1);
  EXPECT_FALSE(b);
  EXPECT_TRUE(a);
}

TEST_F(MetalObjPtrTest, ReleaseHandsOwnershipToCaller) {
  auto p = Ptr::adopt([[TmnnTestObj alloc] init]);
  TmnnTestObj *raw = p.release();
  EXPECT_FALSE(p);
  EXPECT_EQ(g_dealloc_count, 0);
  // p's dtor runs at end-of-scope and is a no-op (already released).
  [raw release];
  EXPECT_EQ(g_dealloc_count, 1);
}

TEST_F(MetalObjPtrTest, ResetNilReleasesCurrent) {
  auto p = Ptr::adopt([[TmnnTestObj alloc] init]);
  p.reset();
  EXPECT_EQ(g_dealloc_count, 1);
  EXPECT_FALSE(p);
}

TEST_F(MetalObjPtrTest, ResetNewReleasesOld) {
  auto p = Ptr::adopt([[TmnnTestObj alloc] init]);
  p.reset([[TmnnTestObj alloc] init]);
  EXPECT_EQ(g_dealloc_count, 1);
  EXPECT_TRUE(p);
}

TEST_F(MetalObjPtrTest, SelfMoveAssignIsSafe) {
  auto p = Ptr::adopt([[TmnnTestObj alloc] init]);
  p = std::move(p);
  // No dealloc; ptr still alive.
  EXPECT_EQ(g_dealloc_count, 0);
  EXPECT_TRUE(p);
}

// Compile-time guarantees: copy is deleted, move-only.
static_assert(!std::is_copy_constructible_v<Ptr>, "must be move-only");
static_assert(!std::is_copy_assignable_v<Ptr>,    "must be move-only");
static_assert(std::is_nothrow_move_constructible_v<Ptr>);
static_assert(std::is_nothrow_move_assignable_v<Ptr>);

}  // namespace

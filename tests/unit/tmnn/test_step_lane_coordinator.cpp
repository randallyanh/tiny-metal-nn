/**
 * @file test_step_lane_coordinator.cpp
 * @brief C9 tests for StepLaneCoordinator.
 */

#include "tiny_metal_nn/runtime/step_lane_coordinator.h"

#include "tiny_metal_nn/runtime/buffer_arena.h"
#include "tiny_metal_nn/runtime/command_batch.h"

#include <gtest/gtest.h>

using namespace tmnn;

TEST(StepLaneCoordinator, AcquireAndRelease) {
  StepLaneCoordinator coord(3);
  EXPECT_EQ(coord.num_lanes(), 3u);
  EXPECT_EQ(coord.active_count(), 0u);

  auto l0 = coord.acquire_lane();
  EXPECT_NE(l0, UINT32_MAX);
  EXPECT_EQ(coord.active_count(), 1u);

  auto l1 = coord.acquire_lane();
  EXPECT_NE(l1, UINT32_MAX);
  EXPECT_NE(l0, l1);
  EXPECT_EQ(coord.active_count(), 2u);

  // Bind and release lane 0.
  coord.bind_fence(l0, {100});
  coord.release_lane(l0);
  EXPECT_EQ(coord.active_count(), 1u);

  // Bind and release lane 1.
  coord.bind_fence(l1, {101});
  coord.release_lane(l1);
  EXPECT_EQ(coord.active_count(), 0u);
}

TEST(StepLaneCoordinator, RingRotation) {
  StepLaneCoordinator coord(2);

  // Acquire lane 0.
  auto l0 = coord.acquire_lane();
  EXPECT_EQ(l0, 0u);
  coord.bind_fence(l0, {1});
  coord.release_lane(l0);

  // Next acquire should give lane 1 (round-robin), not 0.
  auto l1 = coord.acquire_lane();
  EXPECT_EQ(l1, 1u);
  coord.bind_fence(l1, {2});
  coord.release_lane(l1);

  // Next should wrap back to lane 0.
  auto l2 = coord.acquire_lane();
  EXPECT_EQ(l2, 0u);
}

TEST(StepLaneCoordinator, BindFenceAndLookup) {
  StepLaneCoordinator coord(2);
  auto l0 = coord.acquire_lane();
  BatchFence f{42};
  coord.bind_fence(l0, f);

  EXPECT_EQ(coord.lane_for_fence(f), l0);
  EXPECT_EQ(coord.lane_for_fence({99}), UINT32_MAX); // not found
}

TEST(StepLaneCoordinator, ExhaustionReturnsMax) {
  StepLaneCoordinator coord(2);
  auto l0 = coord.acquire_lane();
  auto l1 = coord.acquire_lane();
  EXPECT_NE(l0, UINT32_MAX);
  EXPECT_NE(l1, UINT32_MAX);

  // All lanes acquired — next acquire fails.
  EXPECT_EQ(coord.acquire_lane(), UINT32_MAX);

  // Release one → can acquire again.
  coord.bind_fence(l0, {1});
  coord.release_lane(l0);
  EXPECT_NE(coord.acquire_lane(), UINT32_MAX);
}

TEST(StepLaneCoordinator, FullCycle) {
  StepLaneCoordinator coord(2);
  // 4 steps cycling through 2 lanes.
  for (uint32_t step = 1; step <= 4; ++step) {
    auto lane = coord.acquire_lane();
    ASSERT_NE(lane, UINT32_MAX) << "step " << step;
    coord.bind_fence(lane, {step});
    // Verify lookup.
    EXPECT_EQ(coord.lane_for_fence({step}), lane);
    coord.release_lane(lane);
  }
  EXPECT_EQ(coord.active_count(), 0u);
}

TEST(StepLaneCoordinator, StepLaneReuseRequiresCompletionAndRelease) {
  BufferArena arena;
  auto lanes = arena.allocate_step_set(
      /*positions_bytes=*/1024,
      /*targets_bytes=*/512,
      /*reduction_bytes=*/256,
      /*num_lanes=*/2);
  ASSERT_EQ(lanes.size(), 2u);

  StepLaneCoordinator coord(2);
  CommandBatchPool pool(2);

  const auto lane0 = coord.acquire_lane();
  ASSERT_EQ(lane0, 0u);
  const auto lane0_positions = lanes[lane0].positions.handle;
  const auto lane0_targets = lanes[lane0].targets.handle;
  const auto lane0_reduction = lanes[lane0].loss_reduction.handle;
  const auto batch0 = pool.begin_batch();
  ASSERT_NE(batch0.generation, 0u);
  const auto fence0 = pool.submit(batch0, SubmitMode::Async);
  coord.bind_fence(lane0, fence0);

  const auto lane1 = coord.acquire_lane();
  ASSERT_EQ(lane1, 1u);
  const auto batch1 = pool.begin_batch();
  ASSERT_NE(batch1.generation, 0u);
  const auto fence1 = pool.submit(batch1, SubmitMode::Async);
  coord.bind_fence(lane1, fence1);

  EXPECT_EQ(coord.acquire_lane(), UINT32_MAX);
  EXPECT_EQ(coord.lane_for_fence(fence0), lane0);
  EXPECT_EQ(coord.lane_for_fence(fence1), lane1);

  pool.complete(fence0);
  EXPECT_TRUE(pool.is_complete(fence0));
  EXPECT_EQ(coord.acquire_lane(), UINT32_MAX)
      << "Completing the fence alone must not implicitly release the lane";

  coord.release_lane(lane0);
  const auto reacquired = coord.acquire_lane();
  EXPECT_EQ(reacquired, lane0);
  EXPECT_EQ(lanes[reacquired].positions.handle, lane0_positions);
  EXPECT_EQ(lanes[reacquired].targets.handle, lane0_targets);
  EXPECT_EQ(lanes[reacquired].loss_reduction.handle, lane0_reduction);

  pool.complete(fence1);
  coord.release_lane(lane1);
}

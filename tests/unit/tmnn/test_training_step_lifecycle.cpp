/**
 * @file test_training_step_lifecycle.cpp
 * @brief Tests for tmnn-owned pending-step lifecycle helpers.
 */

#include <gtest/gtest.h>

#include "tiny_metal_nn/runtime/buffer_arena.h"
#include "tiny_metal_nn/runtime/command_batch.h"
#include "tiny_metal_nn/runtime/parameter_store.h"
#include "tiny_metal_nn/runtime/step_lane_coordinator.h"
#include "tiny_metal_nn/runtime/training_step_lifecycle.h"

using namespace tmnn;

TEST(TrainingStepLifecycle, DrainPendingStepCompletesFenceAndReleasesLane) {
  BufferArena arena;

  ParameterStoreDesc desc;
  desc.hash_grid_size = 4u;
  desc.mlp_weight_count = 8u;
  ParameterStore store(desc, arena);

  auto step_lanes = arena.allocate_step_set(64u, 16u, 16u, 1u);

  CommandBatchPool pool(1u);
  auto batch = pool.begin_batch();
  ASSERT_NE(batch.generation, 0u);
  const auto fence = pool.submit(batch, SubmitMode::Async);
  ASSERT_TRUE(static_cast<bool>(fence));
  EXPECT_EQ(pool.submitted_count(), 1u);

  StepLaneCoordinator lane_coord(1u);
  const auto lane = lane_coord.acquire_lane();
  ASSERT_EQ(lane, 0u);
  lane_coord.bind_fence(lane, fence);
  EXPECT_EQ(lane_coord.active_count(), 1u);

  detail::PendingTrainingStep pending;
  pending.fence = fence;
  pending.lane = lane;
  pending.logical_step = 7u;
  pending.valid = true;

  detail::drain_pending_training_step(pool, store, lane_coord, step_lanes,
                                      pending);

  EXPECT_FALSE(pending.valid);
  EXPECT_EQ(pool.submitted_count(), 0u);
  EXPECT_EQ(lane_coord.active_count(), 0u);
}

TEST(TrainingStepLifecycle, DrainPendingStepWithoutValidStepIsNoop) {
  BufferArena arena;

  ParameterStoreDesc desc;
  desc.hash_grid_size = 4u;
  desc.mlp_weight_count = 8u;
  ParameterStore store(desc, arena);

  auto step_lanes = arena.allocate_step_set(64u, 16u, 16u, 1u);

  CommandBatchPool pool(1u);
  StepLaneCoordinator lane_coord(1u);

  detail::PendingTrainingStep pending;
  pending.valid = false;

  detail::drain_pending_training_step(pool, store, lane_coord, step_lanes,
                                      pending);

  EXPECT_FALSE(pending.valid);
  EXPECT_EQ(pool.submitted_count(), 0u);
  EXPECT_EQ(lane_coord.active_count(), 0u);
}

TEST(TrainingStepLifecycle, DrainPendingStepWithDispatchFenceOnlyCompletesNormally) {
  BufferArena arena;

  ParameterStoreDesc desc;
  desc.hash_grid_size = 4u;
  desc.mlp_weight_count = 8u;
  ParameterStore store(desc, arena);

  auto step_lanes = arena.allocate_step_set(64u, 16u, 16u, 1u);

  CommandBatchPool pool(1u);
  auto batch = pool.begin_batch();
  ASSERT_NE(batch.generation, 0u);
  const auto fence = pool.submit(batch, SubmitMode::Async);

  StepLaneCoordinator lane_coord(1u);
  const auto lane = lane_coord.acquire_lane();
  lane_coord.bind_fence(lane, fence);

  detail::PendingTrainingStep pending;
  pending.fence = fence;
  pending.lane = lane;
  pending.logical_step = 3u;
  pending.valid = true;

  detail::drain_pending_training_step(pool, store, lane_coord, step_lanes,
                                      pending);

  EXPECT_FALSE(pending.valid);
  EXPECT_EQ(pool.submitted_count(), 0u);
}

TEST(TrainingStepLifecycle, DrainPendingStepCompletesFillFenceBeforeDispatchFence) {
  BufferArena arena;

  ParameterStoreDesc desc;
  desc.hash_grid_size = 4u;
  desc.mlp_weight_count = 8u;
  ParameterStore store(desc, arena);

  auto step_lanes = arena.allocate_step_set(64u, 16u, 16u, 1u);

  CommandBatchPool pool(2u);
  auto fill_batch = pool.begin_batch();
  ASSERT_NE(fill_batch.generation, 0u);
  const auto fill_fence = pool.submit(fill_batch, SubmitMode::Async);
  ASSERT_TRUE(static_cast<bool>(fill_fence));

  auto dispatch_batch = pool.begin_batch();
  ASSERT_NE(dispatch_batch.generation, 0u);
  const auto dispatch_fence = pool.submit(dispatch_batch, SubmitMode::Async);
  ASSERT_TRUE(static_cast<bool>(dispatch_fence));
  EXPECT_EQ(pool.submitted_count(), 2u);

  StepLaneCoordinator lane_coord(1u);
  const auto lane = lane_coord.acquire_lane();
  ASSERT_EQ(lane, 0u);
  lane_coord.bind_fence(lane, dispatch_fence);
  EXPECT_EQ(lane_coord.active_count(), 1u);

  detail::PendingTrainingStep pending;
  pending.fence = dispatch_fence;
  pending.fill_fence = fill_fence;
  pending.lane = lane;
  pending.logical_step = 11u;
  pending.valid = true;

  detail::drain_pending_training_step(pool, store, lane_coord, step_lanes,
                                      pending);

  EXPECT_FALSE(pending.valid);
  EXPECT_EQ(pool.submitted_count(), 0u);
  EXPECT_EQ(lane_coord.active_count(), 0u);
}

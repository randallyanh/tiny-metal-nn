#pragma once

/**
 * @file training_step_lifecycle.h
 * @brief Internal helpers for tmnn pending-step lifecycle management.
 */

#include "tiny_metal_nn/runtime/buffer_handle.h"

#include <cstdint>
#include <vector>

namespace tmnn {

class CommandBatchPool;
class ParameterStore;
class StepLaneCoordinator;

namespace detail {

struct PendingTrainingStep {
  BatchFence fence{};
  BatchFence fill_fence{};
  uint32_t lane = UINT32_MAX;
  uint32_t num_tgs = 0;
  uint32_t batch_N = 0;
  uint32_t logical_step = 0;
  bool valid = false;
};

void drain_pending_training_step(CommandBatchPool &pool, ParameterStore &ps,
                                 StepLaneCoordinator &lane_coord,
                                 std::vector<StepBufferSet> &step_lanes,
                                 PendingTrainingStep &pending);

} // namespace detail
} // namespace tmnn

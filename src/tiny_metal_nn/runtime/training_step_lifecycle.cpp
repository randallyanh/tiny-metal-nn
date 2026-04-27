/**
 * @file training_step_lifecycle.cpp
 * @brief Internal helpers for tmnn pending-step lifecycle management.
 */

#include "tiny_metal_nn/runtime/training_step_lifecycle.h"

#include "tiny_metal_nn/runtime/command_batch.h"
#include "tiny_metal_nn/runtime/parameter_store.h"
#include "tiny_metal_nn/runtime/step_lane_coordinator.h"

#include <stdexcept>

namespace tmnn::detail {

void drain_pending_training_step(CommandBatchPool &pool, ParameterStore &ps,
                                 StepLaneCoordinator &lane_coord,
                                 std::vector<StepBufferSet> &step_lanes,
                                 PendingTrainingStep &pending) {
  if (!pending.valid)
    return;

  if (pending.valid) {
    if (pending.fill_fence)
      pool.complete(pending.fill_fence);
    if (pending.fence)
      pool.complete(pending.fence);
    if (pending.lane != UINT32_MAX) {
      if (pending.lane >= step_lanes.size()) {
        throw std::runtime_error(
            "drain_pending_training_step: pending lane is out of range");
      }
      (void)ps.finalize_async_step(step_lanes[pending.lane], pending.num_tgs,
                                   pending.batch_N, pending.logical_step + 1);
      lane_coord.release_lane(pending.lane);
    }
  }
  pending = {};
}

} // namespace tmnn::detail

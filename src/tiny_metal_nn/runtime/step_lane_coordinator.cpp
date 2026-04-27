/**
 * @file step_lane_coordinator.cpp
 * @brief StepLaneCoordinator implementation.
 */

#include "tiny_metal_nn/runtime/step_lane_coordinator.h"

#include <cassert>

namespace tmnn {

StepLaneCoordinator::StepLaneCoordinator(uint32_t num_lanes)
    : lanes_(num_lanes) {}

uint32_t StepLaneCoordinator::acquire_lane() {
  const uint32_t n = static_cast<uint32_t>(lanes_.size());
  if (n == 0)
    return UINT32_MAX;

  // Round-robin scan starting after current_lane_.
  uint32_t start = (current_lane_ == UINT32_MAX) ? 0 : (current_lane_ + 1) % n;
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t idx = (start + i) % n;
    if (lanes_[idx].state == LaneState::Free) {
      lanes_[idx].state = LaneState::Acquired;
      current_lane_ = idx;
      return idx;
    }
  }
  return UINT32_MAX; // All lanes busy.
}

void StepLaneCoordinator::bind_fence(uint32_t lane, BatchFence f) {
  assert(lane < lanes_.size() && "Lane index out of range");
  assert(lanes_[lane].state == LaneState::Acquired &&
         "Can only bind fence to an Acquired lane");
  lanes_[lane].state = LaneState::Submitted;
  lanes_[lane].fence = f;
}

void StepLaneCoordinator::release_lane(uint32_t lane) {
  assert(lane < lanes_.size() && "Lane index out of range");
  assert(lanes_[lane].state == LaneState::Submitted &&
         "Can only release a Submitted lane");
  lanes_[lane].state = LaneState::Free;
  lanes_[lane].fence = {};
}

uint32_t StepLaneCoordinator::lane_for_fence(BatchFence f) const {
  for (uint32_t i = 0; i < static_cast<uint32_t>(lanes_.size()); ++i) {
    if (lanes_[i].state == LaneState::Submitted &&
        lanes_[i].fence.value == f.value)
      return i;
  }
  return UINT32_MAX;
}

uint32_t StepLaneCoordinator::current_lane() const { return current_lane_; }

uint32_t StepLaneCoordinator::active_count() const {
  uint32_t count = 0;
  for (const auto &l : lanes_) {
    if (l.state != LaneState::Free)
      ++count;
  }
  return count;
}

uint32_t StepLaneCoordinator::num_lanes() const {
  return static_cast<uint32_t>(lanes_.size());
}

} // namespace tmnn

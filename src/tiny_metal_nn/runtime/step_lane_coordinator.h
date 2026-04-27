#pragma once

/**
 * @file step_lane_coordinator.h
 * @brief StepLaneCoordinator — ring-buffered lane→fence lifecycle manager.
 *
 * Maps StepBufferSet lanes to BatchFence values for async overlap.
 * Each lane cycles: Free → Acquired → Submitted (with fence) → Free.
 */

#include "tiny_metal_nn/runtime/buffer_handle.h"

#include <cstdint>
#include <vector>

namespace tmnn {

class StepLaneCoordinator {
public:
  explicit StepLaneCoordinator(uint32_t num_lanes);

  /// Acquire the next free lane (round-robin). Returns UINT32_MAX if exhausted.
  [[nodiscard]] uint32_t acquire_lane();

  /// Bind a fence to an acquired lane. Acquired → Submitted.
  void bind_fence(uint32_t lane, BatchFence f);

  /// Release a submitted lane. Submitted → Free.
  void release_lane(uint32_t lane);

  /// Find the lane holding a given fence, or UINT32_MAX if not found.
  [[nodiscard]] uint32_t lane_for_fence(BatchFence f) const;

  /// The most recently acquired lane (UINT32_MAX if none acquired yet).
  [[nodiscard]] uint32_t current_lane() const;

  /// Number of lanes in Acquired or Submitted state.
  [[nodiscard]] uint32_t active_count() const;

  /// Total number of lanes.
  [[nodiscard]] uint32_t num_lanes() const;

private:
  enum class LaneState { Free, Acquired, Submitted };

  struct LaneSlot {
    LaneState state = LaneState::Free;
    BatchFence fence{};
  };

  std::vector<LaneSlot> lanes_;
  uint32_t current_lane_ = UINT32_MAX;
};

} // namespace tmnn

#pragma once

/**
 * @file command_batch.h
 * @brief CommandBatchPool — batch lifecycle and fence tracking (internal).
 *
 * Sync batches complete immediately on submit. Async batches remain in
 * Submitted state until external complete() is called. When a Metal queue
 * is wired in via set_queue(), real MTLCommandBuffers are created and
 * committed. Without a queue, state tracking is metadata-only.
 */

#include "tiny_metal_nn/runtime/buffer_handle.h"

#include <cstdint>
#include <vector>

namespace tmnn {

class CommandBatchPool {
public:
  /// Create a pool with the given max inflight batch count.
  explicit CommandBatchPool(uint32_t max_inflight = 2);

  /// Set the Metal command queue for real command buffer creation.
  /// Called by MetalContext after device probing.
  void set_queue(void *queue);

  /// Begin a new command batch. Returns an invalid handle if all slots
  /// are in-flight (caller must wait or increase pool size).
  [[nodiscard]] CommandBatchHandle begin_batch();

  /// Submit a batch. Returns a fence for completion tracking.
  /// Sync: immediate completion. Async: stays Submitted until complete().
  [[nodiscard]] BatchFence submit(CommandBatchHandle handle, SubmitMode mode);

  /// Complete an async batch identified by its fence.
  /// Transitions Submitted → Completed.
  void complete(BatchFence fence);

  /// Check if a fence has completed.
  [[nodiscard]] bool is_complete(BatchFence fence) const;

  /// Number of batches currently in-flight (Recording + Submitted).
  [[nodiscard]] uint32_t inflight_count() const;

  /// Number of batches currently in Submitted (but not Completed) state.
  [[nodiscard]] uint32_t submitted_count() const;

  /// Total batch slots available for concurrent recording/submission.
  [[nodiscard]] uint32_t slot_capacity() const {
    return static_cast<uint32_t>(slots_.size());
  }

  /// Total number of batches submitted since creation.
  [[nodiscard]] uint64_t total_submitted() const;

  /// GPU execution time (us) of the command buffer identified by fence.
  /// Valid only after complete(fence). Returns 0.0 if unavailable.
  [[nodiscard]] double gpu_time_us(BatchFence fence) const;

  /// Get the raw MTLCommandBuffer for an active batch. nullptr if no queue.
  [[nodiscard]] void *current_command_buffer(CommandBatchHandle handle) const;

private:
  enum class SlotState { Idle, Recording, Submitted, Completed };

  struct BatchSlot {
    uint32_t generation = 0;
    SlotState state = SlotState::Idle;
    uint64_t fence_value = 0;
    void *cmd_buf = nullptr; ///< id<MTLCommandBuffer>
    double gpu_time_us = 0.0; ///< GPU execution time from last complete()
  };

  std::vector<BatchSlot> slots_;
  uint64_t next_fence_ = 1; // 0 is reserved for "no fence"
  uint64_t total_submitted_ = 0;
  uint32_t inflight_count_ = 0;
  void *queue_ = nullptr; ///< id<MTLCommandQueue>
};

} // namespace tmnn

/**
 * @file command_batch.cpp
 * @brief CommandBatchPool implementation — real MTLCommandBuffers when queue
 *        is set, metadata-only tracking otherwise.
 */

#include "tiny_metal_nn/runtime/command_batch.h"
#include "tiny_metal_nn/runtime/metal_device.h"
#include "tiny_metal_nn/runtime/command_batch_test_access.h"

#include <cassert>
#include <stdexcept>

namespace tmnn {

namespace {

detail::CommandBatchErrorHook g_command_batch_error_hook = nullptr;

std::string command_buffer_error_with_test_hook(void *cmd_buf) {
  if (g_command_batch_error_hook) {
    auto injected = g_command_batch_error_hook(cmd_buf);
    if (!injected.empty())
      return injected;
  }
  return metal::command_buffer_error(cmd_buf);
}

} // namespace

CommandBatchPool::CommandBatchPool(uint32_t max_inflight) {
  slots_.resize(max_inflight);
  // Generation starts at 1 so that default-constructed handles are invalid.
  for (auto &s : slots_)
    s.generation = 1;
}

void CommandBatchPool::set_queue(void *queue) { queue_ = queue; }

CommandBatchHandle CommandBatchPool::begin_batch() {
  // Find an idle or completed slot.
  for (uint32_t i = 0; i < static_cast<uint32_t>(slots_.size()); ++i) {
    auto &s = slots_[i];
    if (s.state == SlotState::Idle || s.state == SlotState::Completed) {
      s.state = SlotState::Recording;
      // Create a real command buffer if we have a queue.
      if (queue_) {
        s.cmd_buf = metal::create_command_buffer(queue_);
      }
      ++inflight_count_;
      return {i, s.generation};
    }
  }
  // No idle slot available — return invalid handle.
  return {0, 0};
}

BatchFence CommandBatchPool::submit(CommandBatchHandle handle, SubmitMode mode) {
  assert(handle.slot < slots_.size() && "CommandBatchHandle slot out of range");
  auto &s = slots_[handle.slot];
  assert(s.generation == handle.generation &&
         "Stale CommandBatchHandle (generation mismatch)");
  assert(s.state == SlotState::Recording &&
         "Cannot submit a batch that is not recording");

  uint64_t fence_val = next_fence_++;
  s.fence_value = fence_val;
  ++total_submitted_;

  // Bump generation so the handle can't be reused.
  ++s.generation;

  if (mode == SubmitMode::Sync) {
    // Sync: commit, wait, release, mark completed.
    if (s.cmd_buf) {
      metal::commit_and_wait(s.cmd_buf);
      auto error = command_buffer_error_with_test_hook(s.cmd_buf);
      if (!error.empty()) {
        metal::release_command_buffer(s.cmd_buf);
        s.cmd_buf = nullptr;
        s.state = SlotState::Completed;
        --inflight_count_;
        throw std::runtime_error("Metal batch dispatch failed: " + error);
      }
      s.gpu_time_us = metal::gpu_execution_time_us(s.cmd_buf);
      metal::release_command_buffer(s.cmd_buf);
      s.cmd_buf = nullptr;
    }
    s.state = SlotState::Completed;
    --inflight_count_;
  } else {
    // Async: commit without waiting.
    if (s.cmd_buf) {
      metal::commit_async(s.cmd_buf);
    }
    s.state = SlotState::Submitted;
    // inflight_count_ remains incremented — slot is in-flight.
  }

  return {fence_val};
}

void CommandBatchPool::complete(BatchFence fence) {
  assert(fence.value != 0 && "Cannot complete a null fence");
  for (auto &s : slots_) {
    if (s.state == SlotState::Submitted && s.fence_value == fence.value) {
      // Wait for the real command buffer if present.
      if (s.cmd_buf) {
        metal::wait_until_completed(s.cmd_buf);
        auto error = command_buffer_error_with_test_hook(s.cmd_buf);
        if (!error.empty()) {
          metal::release_command_buffer(s.cmd_buf);
          s.cmd_buf = nullptr;
          s.state = SlotState::Completed;
          --inflight_count_;
          throw std::runtime_error("Metal async batch failed: " + error);
        }
        s.gpu_time_us = metal::gpu_execution_time_us(s.cmd_buf);
        metal::release_command_buffer(s.cmd_buf);
        s.cmd_buf = nullptr;
      }
      s.state = SlotState::Completed;
      --inflight_count_;
      return;
    }
  }
  assert(false && "No Submitted slot found for fence");
}

bool CommandBatchPool::is_complete(BatchFence fence) const {
  if (fence.value == 0)
    return true; // null fence is always complete
  // Check if any slot holds this fence in Submitted (not yet completed) state.
  for (const auto &s : slots_) {
    if (s.state == SlotState::Submitted && s.fence_value == fence.value)
      return false;
  }
  // Either completed or fence predates all current slots.
  return fence.value < next_fence_;
}

void *CommandBatchPool::current_command_buffer(CommandBatchHandle handle) const {
  assert(handle.slot < slots_.size());
  return slots_[handle.slot].cmd_buf;
}

uint32_t CommandBatchPool::inflight_count() const { return inflight_count_; }

uint32_t CommandBatchPool::submitted_count() const {
  uint32_t count = 0;
  for (const auto &s : slots_) {
    if (s.state == SlotState::Submitted)
      ++count;
  }
  return count;
}

uint64_t CommandBatchPool::total_submitted() const {
  return total_submitted_;
}

double CommandBatchPool::gpu_time_us(BatchFence fence) const {
  for (const auto &s : slots_) {
    if (s.fence_value == fence.value &&
        (s.state == SlotState::Completed || s.state == SlotState::Idle))
      return s.gpu_time_us;
  }
  return 0.0;
}

} // namespace tmnn

namespace tmnn::detail {

void set_command_batch_error_hook_for_testing(CommandBatchErrorHook hook) {
  g_command_batch_error_hook = hook;
}

} // namespace tmnn

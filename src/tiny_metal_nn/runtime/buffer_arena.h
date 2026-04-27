#pragma once

/**
 * @file buffer_arena.h
 * @brief BufferArena — slot-based buffer lifecycle manager (internal).
 *
 * CPU-backed buffer lifecycle manager. Shared slots have host-accessible
 * memory; Private slots are GPU-only (backed when MetalContext has a device).
 * Manages allocation, release, generation tracking, sub-view creation,
 * and ring-buffered StepBufferSet lanes.
 */

#include "tiny_metal_nn/runtime/buffer_handle.h"

#include <cassert>
#include <cstddef>
#include <string>
#include <vector>

namespace tmnn {

class BufferArena {
public:
  /// Create an arena with the given initial slot capacity.
  explicit BufferArena(uint32_t initial_capacity = 64);

  /// Destructor — frees all CPU-backed slot memory and GPU buffers.
  ~BufferArena();

  /// Set the Metal device for GPU buffer allocation.
  /// Called by MetalContext after device probing.
  void set_device(void *device);

  // --- Allocation ---

  /// Allocate a buffer slot and return its handle.
  [[nodiscard]] BufferHandle allocate(const BufferDesc &desc);

  /// Release a buffer slot. Increments generation — existing handles
  /// become stale and will fail is_valid().
  void release(BufferHandle handle);

  // --- Validity ---

  /// Check if a handle is still valid (generation matches).
  [[nodiscard]] bool is_valid(BufferHandle handle) const;

  // --- Views ---

  /// Create a full-buffer view.
  [[nodiscard]] BufferView view(BufferHandle handle) const;

  /// Create a sub-range view from an existing view.
  [[nodiscard]] static BufferView sub_view(const BufferView &parent,
                                           size_t offset, size_t bytes);

  /// Create a binding from a view + pipeline slot index.
  [[nodiscard]] static BufferBinding bind(const BufferView &v,
                                          uint32_t binding_index);

  // --- Ring-buffered StepBufferSet ---

  /// Allocate a ring of StepBufferSet lanes for async overlap.
  /// Each lane gets its own positions/targets/loss_reduction buffers.
  [[nodiscard]] std::vector<StepBufferSet>
  allocate_step_set(size_t positions_bytes, size_t targets_bytes,
                    size_t reduction_bytes, uint32_t num_lanes = 2,
                    size_t external_grad_bytes = 0,
                    size_t forward_output_bytes = 0);

  /// Release all buffers owned by a ring of StepBufferSet lanes.
  void release_step_set(std::vector<StepBufferSet> &lanes);

  // --- Stats ---

  /// Total bytes tracked by the arena (sum of all live allocations).
  [[nodiscard]] size_t bytes_allocated() const;

  /// Number of live (non-released) slots.
  [[nodiscard]] uint32_t live_count() const;

  /// Total number of slots ever allocated (including released).
  [[nodiscard]] uint32_t total_slots() const;

  // --- Slot metadata queries ---

  /// Get the size of the buffer behind a handle.
  [[nodiscard]] size_t slot_bytes(BufferHandle handle) const;

  /// Get the debug name of a slot (may be nullptr).
  [[nodiscard]] const char *slot_debug_name(BufferHandle handle) const;

  /// Get the lifetime of a slot.
  [[nodiscard]] BufferLifetime slot_lifetime(BufferHandle handle) const;

  /// Get the storage mode of a slot.
  [[nodiscard]] BufferStorage slot_storage(BufferHandle handle) const;

private:
  struct SlotMeta {
    size_t bytes = 0;
    uint32_t generation = 0;
    BufferStorage storage = BufferStorage::Shared;
    BufferLifetime lifetime = BufferLifetime::Persistent;
    const char *debug_name = nullptr;
    void *cpu_data = nullptr;   ///< CPU backing for Shared buffers.
    void *gpu_buffer = nullptr; ///< id<MTLBuffer> when device is present.
    bool alive = false;
  };

  std::vector<SlotMeta> slots_;
  std::vector<uint32_t> free_list_;
  size_t total_bytes_ = 0;
  uint32_t live_count_ = 0;
  void *device_ = nullptr; ///< id<MTLDevice>, set by MetalContext.

  void assert_valid(BufferHandle handle) const;
};

} // namespace tmnn

#pragma once

/**
 * @file buffer_handle.h
 * @brief Buffer and command batch type definitions for tmnn runtime (internal).
 *
 * These types form the vocabulary for BufferArena, CommandBatchPool,
 * ParameterStore, and StepBufferSet. They are internal to the runtime
 * implementation and not part of the public tmnn API.
 */

#include <cstddef>
#include <cstdint>

namespace tmnn {

// ---------------------------------------------------------------------------
// Buffer types
// ---------------------------------------------------------------------------

/// Metal storage mode abstraction.
enum class BufferStorage { Shared, Private };

/// Buffer lifetime classification.
enum class BufferLifetime { Persistent, Transient, Staging };

/// Descriptor for buffer allocation.
struct BufferDesc {
  size_t bytes = 0;
  size_t alignment = 256;
  BufferStorage storage = BufferStorage::Shared;
  BufferLifetime lifetime = BufferLifetime::Persistent;
  const char *debug_name = nullptr;
};

/// Opaque handle to an arena-managed buffer.
/// Validity is checked via generation — a stale handle whose arena_slot
/// has been recycled will have a mismatched generation.
struct BufferHandle {
  uint32_t arena_slot = 0;
  uint32_t generation = 0;

  bool operator==(const BufferHandle &o) const {
    return arena_slot == o.arena_slot && generation == o.generation;
  }
  bool operator!=(const BufferHandle &o) const { return !(*this == o); }
};

/// A sub-range view into an arena buffer.
struct BufferView {
  BufferHandle handle;
  size_t offset = 0;
  size_t bytes = 0;
  void *data = nullptr;       ///< CPU-accessible pointer (non-null for Shared buffers).
  void *gpu_buffer = nullptr;  ///< id<MTLBuffer> (both Shared + Private when device present).
};

/// A BufferView bound to a specific pipeline slot index.
struct BufferBinding {
  BufferView view;
  uint32_t binding_index = 0;
};

/// Per-step IO buffer set (ring-buffered for async overlap).
struct StepBufferSet {
  BufferView positions;
  BufferView targets;
  BufferView loss_reduction;
  BufferView external_gradient;  ///< d_output for split training path.
  BufferView forward_output;     ///< Network output from forward pass.
  BufferView probe_buffer;       ///< Opt-in per-step probe telemetry.
};

// ---------------------------------------------------------------------------
// Command batch types
// ---------------------------------------------------------------------------

/// Opaque handle to an in-flight command batch.
struct CommandBatchHandle {
  uint32_t slot = 0;
  uint32_t generation = 0;

  bool operator==(const CommandBatchHandle &o) const {
    return slot == o.slot && generation == o.generation;
  }
  bool operator!=(const CommandBatchHandle &o) const {
    return !(*this == o);
  }
};

/// Fence value for tracking async batch completion.
struct BatchFence {
  uint64_t value = 0;
  explicit operator bool() const { return value != 0; }
};

/// Submission mode for command batches.
enum class SubmitMode { Sync, Async };

} // namespace tmnn

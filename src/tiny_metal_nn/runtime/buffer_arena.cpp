/**
 * @file buffer_arena.cpp
 * @brief BufferArena implementation — slot-based buffer lifecycle.
 *
 * When a Metal device is set via set_device(), Shared buffers are backed
 * by MTLBuffers (CPU pointer comes from [buffer contents]). Private buffers
 * get GPU-only MTLBuffers with no CPU access. Without a device, Shared
 * buffers fall back to calloc.
 */

#include "tiny_metal_nn/runtime/buffer_arena.h"
#include "tiny_metal_nn/runtime/metal_device.h"
#include "tiny_metal_nn/runtime/owned_buffer_handle.h"

#include <cassert>
#include <cstring>
#include <cstdlib>
#include <stdexcept>

namespace tmnn {

BufferArena::BufferArena(uint32_t initial_capacity) {
  slots_.reserve(initial_capacity);
}

BufferArena::~BufferArena() {
  for (auto &slot : slots_) {
    // Heap-backed slots: OwnedBuffer dtor returns the sub-buffer to the
    // Heap when the slot is destructed below (vector-of-SlotMeta dtor).
    if (slot.owned.valid())
      continue;
    if (slot.gpu_buffer) {
      metal::release_buffer(slot.gpu_buffer);
      slot.gpu_buffer = nullptr;
      slot.cpu_data = nullptr; // was pointing into MTLBuffer contents
    } else {
      std::free(slot.cpu_data);
      slot.cpu_data = nullptr;
    }
  }
}

void BufferArena::set_device(void *device) { device_ = device; }
void BufferArena::set_heap(metal_heap::Heap *heap) { heap_ = heap; }

// ---------------------------------------------------------------------------
// Allocation
// ---------------------------------------------------------------------------

BufferHandle BufferArena::allocate(const BufferDesc &desc) {
  uint32_t slot = UINT32_MAX;
  for (auto it = free_list_.begin(); it != free_list_.end(); ++it) {
    const auto candidate = *it;
    if (slots_[candidate].lifetime == desc.lifetime) {
      slot = candidate;
      free_list_.erase(it);
      break;
    }
  }

  if (slot == UINT32_MAX) {
    slot = static_cast<uint32_t>(slots_.size());
    slots_.push_back({});
    // First allocation in this slot gets generation 1.
    slots_[slot].generation = 1;
  } else if (slots_[slot].lifetime != desc.lifetime) {
    throw std::runtime_error(
        "BufferArena internal error: attempted cross-tier slot reuse");
  }

  auto &meta = slots_[slot];
  // Shared-backing reuse fast path: a previously-released slot still holds
  // an exact-size Shared buffer we can hand back after a memset. Heap-backed
  // slots qualify when the cached OwnedBuffer is still valid; legacy device
  // / calloc paths qualify on their own pointers.
  const bool can_reuse_shared_backing =
      !meta.alive && desc.storage == BufferStorage::Shared &&
      meta.storage == BufferStorage::Shared && meta.bytes == desc.bytes &&
      ((heap_ && meta.owned.valid()) ||
       (!heap_ && device_ && meta.gpu_buffer && meta.cpu_data) ||
       (!heap_ && !device_ && meta.cpu_data));

  if (!can_reuse_shared_backing) {
    // Drop whatever the slot was holding before re-binding it. For
    // heap-backed slots the OwnedBuffer dtor already releases the MTLBuffer
    // — DON'T also call metal::release_buffer or we double-release.
    if (meta.owned.valid()) {
      meta.owned = {};
      meta.gpu_buffer = nullptr;
      meta.cpu_data = nullptr;
    } else if (meta.gpu_buffer) {
      metal::release_buffer(meta.gpu_buffer);
      meta.gpu_buffer = nullptr;
      meta.cpu_data = nullptr;
    } else if (meta.cpu_data) {
      std::free(meta.cpu_data);
      meta.cpu_data = nullptr;
    }
  }

  meta.bytes = desc.bytes;
  meta.storage = desc.storage;
  meta.lifetime = desc.lifetime;
  meta.debug_name = desc.debug_name;
  meta.alive = true;

  if (desc.bytes > 0) {
    if (heap_) {
      if (can_reuse_shared_backing) {
        // MTLHeap sub-buffers are not zero-initialized on creation, but a
        // cached Shared backing may carry stale content from its prior use.
        // Match the legacy semantic: zero on reuse handover.
        std::memset(meta.cpu_data, 0, desc.bytes);
      } else {
        metal_heap::AllocDesc hd;
        hd.bytes = desc.bytes;
        hd.alignment = desc.alignment ? desc.alignment : 256;
        hd.lifetime = metal_heap::Lifetime::Persistent;
        hd.storage = (desc.storage == BufferStorage::Shared)
                         ? metal_heap::Storage::Shared
                         : metal_heap::Storage::Private;
        hd.hazard_tracking = metal_heap::HazardTracking::Untracked;
        hd.debug_name = desc.debug_name ? desc.debug_name : "tmnn.unnamed";
        auto r = heap_->allocate(hd);
        if (!r.has_value()) {
          const auto s = heap_->stats();
          throw std::runtime_error(
              std::string("BufferArena::allocate: metal_heap::Heap "
                          "exhausted (want=") +
              std::to_string(desc.bytes) + " bytes " +
              (desc.storage == BufferStorage::Shared ? "Shared" : "Private") +
              "; shared=" +
              std::to_string(s.persistent_shared_used_bytes) + "/" +
              std::to_string(s.persistent_shared_capacity_bytes) +
              " priv=" +
              std::to_string(s.persistent_private_used_bytes) + "/" +
              std::to_string(s.persistent_private_capacity_bytes) +
              "); raise MetalContextDesc.heap_config or free GPU memory");
        }
        meta.owned = std::move(*r);
        meta.gpu_buffer = meta.owned.mtl_buffer();
        meta.cpu_data = meta.owned.cpu_data();
      }
    } else if (desc.storage == BufferStorage::Shared) {
      if (can_reuse_shared_backing) {
        std::memset(meta.cpu_data, 0, desc.bytes);
      } else if (device_) {
        meta.gpu_buffer =
            metal::create_buffer(device_, desc.bytes, /*shared=*/true);
        meta.cpu_data = metal::buffer_contents(meta.gpu_buffer);
      } else {
        meta.cpu_data = std::calloc(1, desc.bytes);
      }
    } else if (desc.storage == BufferStorage::Private && device_) {
      meta.gpu_buffer =
          metal::create_buffer(device_, desc.bytes, /*shared=*/false);
      // cpu_data stays nullptr — Private is GPU-only.
    }
  }

  total_bytes_ += desc.bytes;
  ++live_count_;

  return {slot, meta.generation};
}

detail::OwnedBufferHandle BufferArena::allocate_owned(const BufferDesc &desc) {
  return detail::OwnedBufferHandle(this, allocate(desc));
}

void BufferArena::release(BufferHandle handle) {
  assert_valid(handle);

  auto &meta = slots_[handle.arena_slot];
  meta.alive = false;
  total_bytes_ -= meta.bytes;
  --live_count_;

  // Keep shared backing alive for reuse on a later matching allocation.
  // Reused shared buffers are zeroed on the next allocate() before they
  // are handed back out, so we avoid paying that cost twice during teardown.
  // Private slots release their backing immediately; heap-backed Private
  // slots drop the OwnedBuffer so the heap reclaims the sub-buffer.
  if (meta.storage != BufferStorage::Shared) {
    if (meta.owned.valid()) {
      meta.owned = {};
      meta.gpu_buffer = nullptr;
      meta.cpu_data = nullptr;
    } else if (meta.gpu_buffer) {
      metal::release_buffer(meta.gpu_buffer);
      meta.gpu_buffer = nullptr;
      meta.cpu_data = nullptr;
    } else if (meta.cpu_data) {
      std::free(meta.cpu_data);
      meta.cpu_data = nullptr;
    }
  }

  // Bump generation so stale handles are detected.
  ++meta.generation;

  free_list_.push_back(handle.arena_slot);
}

// ---------------------------------------------------------------------------
// Validity
// ---------------------------------------------------------------------------

bool BufferArena::is_valid(BufferHandle handle) const {
  if (handle.arena_slot >= slots_.size())
    return false;
  const auto &meta = slots_[handle.arena_slot];
  return meta.alive && meta.generation == handle.generation;
}

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

BufferView BufferArena::view(BufferHandle handle) const {
  assert_valid(handle);
  const auto &meta = slots_[handle.arena_slot];
  return {handle, 0, meta.bytes, meta.cpu_data, meta.gpu_buffer};
}

BufferView BufferArena::sub_view(const BufferView &parent, size_t offset,
                                 size_t bytes) {
  assert(offset + bytes <= parent.bytes && "sub_view out of range");
  void *data = nullptr;
  if (parent.data)
    data = static_cast<char *>(parent.data) + offset;
  // Sub-views share the parent's gpu_buffer — offset tracked in BufferView.
  return {parent.handle, parent.offset + offset, bytes, data,
          parent.gpu_buffer};
}

BufferBinding BufferArena::bind(const BufferView &v,
                                uint32_t binding_index) {
  return {v, binding_index};
}

// ---------------------------------------------------------------------------
// Ring-buffered StepBufferSet
// ---------------------------------------------------------------------------

std::vector<StepBufferSet>
BufferArena::allocate_step_set(size_t positions_bytes, size_t targets_bytes,
                               size_t reduction_bytes, uint32_t num_lanes,
                               size_t external_grad_bytes,
                               size_t forward_output_bytes) {
  std::vector<StepBufferSet> lanes;
  lanes.reserve(num_lanes);

  for (uint32_t i = 0; i < num_lanes; ++i) {
    BufferDesc pos_desc{positions_bytes, 256, BufferStorage::Shared,
                        BufferLifetime::Transient, "step_positions"};
    BufferDesc tgt_desc{targets_bytes, 256, BufferStorage::Shared,
                        BufferLifetime::Transient, "step_targets"};
    BufferDesc red_desc{reduction_bytes, 256, BufferStorage::Shared,
                        BufferLifetime::Transient, "step_loss_reduction"};

    StepBufferSet set;
    set.positions = view(allocate(pos_desc));
    set.targets = view(allocate(tgt_desc));
    set.loss_reduction = view(allocate(red_desc));

    if (external_grad_bytes > 0) {
      BufferDesc eg_desc{external_grad_bytes, 256, BufferStorage::Shared,
                         BufferLifetime::Transient, "step_external_gradient"};
      set.external_gradient = view(allocate(eg_desc));
    }
    if (forward_output_bytes > 0) {
      BufferDesc fo_desc{forward_output_bytes, 256, BufferStorage::Shared,
                         BufferLifetime::Transient, "step_forward_output"};
      set.forward_output = view(allocate(fo_desc));
    }

    lanes.push_back(set);
  }

  return lanes;
}

void BufferArena::release_step_set(std::vector<StepBufferSet> &lanes) {
  for (auto &lane : lanes) {
    if (is_valid(lane.positions.handle))
      release(lane.positions.handle);
    if (is_valid(lane.targets.handle))
      release(lane.targets.handle);
    if (is_valid(lane.loss_reduction.handle))
      release(lane.loss_reduction.handle);
    if (is_valid(lane.external_gradient.handle))
      release(lane.external_gradient.handle);
    if (is_valid(lane.forward_output.handle))
      release(lane.forward_output.handle);
    lane = {};
  }
  lanes.clear();
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

size_t BufferArena::bytes_allocated() const { return total_bytes_; }
uint32_t BufferArena::live_count() const { return live_count_; }
uint32_t BufferArena::total_slots() const {
  return static_cast<uint32_t>(slots_.size());
}

// ---------------------------------------------------------------------------
// Slot metadata queries
// ---------------------------------------------------------------------------

size_t BufferArena::slot_bytes(BufferHandle handle) const {
  assert_valid(handle);
  return slots_[handle.arena_slot].bytes;
}

const char *BufferArena::slot_debug_name(BufferHandle handle) const {
  assert_valid(handle);
  return slots_[handle.arena_slot].debug_name;
}

BufferLifetime BufferArena::slot_lifetime(BufferHandle handle) const {
  assert_valid(handle);
  return slots_[handle.arena_slot].lifetime;
}

BufferStorage BufferArena::slot_storage(BufferHandle handle) const {
  assert_valid(handle);
  return slots_[handle.arena_slot].storage;
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

void BufferArena::assert_valid(BufferHandle handle) const {
  assert(handle.arena_slot < slots_.size() && "BufferHandle slot out of range");
  assert(slots_[handle.arena_slot].alive && "BufferHandle slot not alive");
  assert(slots_[handle.arena_slot].generation == handle.generation &&
         "Stale BufferHandle (generation mismatch)");
}

} // namespace tmnn

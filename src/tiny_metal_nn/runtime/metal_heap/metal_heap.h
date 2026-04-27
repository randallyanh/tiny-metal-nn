#pragma once

// Standalone, reusable MTLHeap-backed allocator for Apple Metal C++ projects.
//
// The public API is pure C++; Metal types cross the boundary as void*. The
// implementation lives in metal_heap.mm. The module does not depend on any
// tmnn-specific symbols (a CI grep gate keeps it that way), so it can be
// lifted into another Metal project unchanged.
//
// Phase 2.1 ships only Lifetime::Persistent. TransientRing and StagingPool
// land in Phase 2.2 and 2.3 respectively.

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>

namespace metal_heap {

// Type-erased Metal handles. Cast back to id<MTLDevice> / id<MTLBuffer> in
// .mm code; runtime .cpp files never see Objective-C types.
using MetalDevice = void *;
using MetalBuffer = void *;

enum class Lifetime { Persistent, Transient };  // Staging — Phase 2.3

enum class Storage { Shared, Private };

// MTLResourceHazardTrackingMode mapping. Untracked is the SOTA default for
// hot-path buffers; Tracked is used for debug / safe-family / autotune
// caches where caller does not insert manual fences.
enum class HazardTracking { Tracked, Untracked };

struct HeapConfig {
  // MTLHeap fixes its storage mode at creation, so we maintain one heap per
  // storage class. Sized independently; either may be 0 to disable that
  // storage class.
  std::size_t persistent_shared_capacity_bytes  = 48 * 1024 * 1024;  // 48 MiB
  std::size_t persistent_private_capacity_bytes = 16 * 1024 * 1024;  // 16 MiB

  // Transient ring: N pre-allocated MTLBuffers, one per lane. allocate()
  // bumps the current lane's offset (no buffer creation in the hot path).
  // Set lane_count = 0 to disable Transient lifetime.
  std::size_t transient_lane_bytes  = 16 * 1024 * 1024;  // 16 MiB per lane
  std::uint32_t transient_lane_count = 4;
};

struct AllocDesc {
  std::size_t bytes;
  std::size_t alignment = 256;
  Lifetime lifetime = Lifetime::Persistent;
  Storage storage = Storage::Shared;
  HazardTracking hazard_tracking = HazardTracking::Untracked;
  // Mandatory; -setLabel: applied to every sub-buffer for Xcode capture.
  // Must point at a stable string (typically a string literal); the heap
  // caches the corresponding NSString keyed by raw pointer identity, so
  // distinct const char* with equal contents are treated as different
  // labels. Pass `nullptr` and the call returns AllocError::InvalidConfig.
  const char *debug_name = nullptr;
};

enum class AllocError {
  PersistentExhausted,
  TransientCapacityExceeded,
  InvalidConfig,
  DeviceUnavailable,
  UnsupportedLifetime,    // 2.3 (Staging) not yet implemented
  UnsupportedStorage,     // requested storage class disabled in HeapConfig
};

class Heap;
class OwnedBuffer;

class Heap {
public:
  // Construct a Heap that backs `mtl_device`. Returns nullptr if the device
  // is null or heap creation fails.
  [[nodiscard]] static std::unique_ptr<Heap>
  create(MetalDevice mtl_device, const HeapConfig &cfg) noexcept;

  ~Heap() noexcept;
  Heap(const Heap &) = delete;
  Heap &operator=(const Heap &) = delete;

  [[nodiscard]] std::expected<OwnedBuffer, AllocError>
  allocate(const AllocDesc &desc) noexcept;

  // Per-step transient ring control. begin_transient_frame() rotates to the
  // next lane and resets its bump offset; end_transient_frame() advances the
  // frame counter (Phase 2.2 MVP: no GPU fence — caller is responsible for
  // ensuring the next-lane reuse happens after prior GPU work completes).
  // Transient OwnedBuffers must NOT outlive the frame in which they were
  // allocated; their backing storage is recycled by the next
  // begin_transient_frame() that lands on the same lane.
  void begin_transient_frame() noexcept;
  void end_transient_frame() noexcept;

  struct Stats {
    std::size_t persistent_shared_capacity_bytes;
    std::size_t persistent_shared_used_bytes;
    std::size_t persistent_private_capacity_bytes;
    std::size_t persistent_private_used_bytes;
    std::size_t transient_lane_bytes;
    std::uint32_t transient_lane_count;
    std::uint32_t transient_current_lane;
    std::size_t transient_current_lane_used_bytes;
    std::uint64_t transient_frames_completed;
    std::uint64_t live_buffers;
    std::uint64_t total_allocations;
    std::uint64_t alloc_failures;
  };
  [[nodiscard]] Stats stats() const noexcept;

private:
  Heap() noexcept;
  // Called by OwnedBuffer's destructor to release a sub-allocation.
  // Transient buffers are no-op here; the lane reset on begin_transient_frame
  // handles their reuse.
  void deallocate(MetalBuffer buf, std::size_t bytes, Storage storage,
                  Lifetime lifetime) noexcept;
  friend class OwnedBuffer;

  class Impl;
  std::unique_ptr<Impl> impl_;
};

// Move-only owning view of a sub-buffer. dtor releases the underlying
// id<MTLBuffer>; the parent Heap keeps the backing MTLHeap alive.
class OwnedBuffer {
public:
  OwnedBuffer() noexcept = default;
  ~OwnedBuffer() noexcept;

  OwnedBuffer(OwnedBuffer &&) noexcept;
  OwnedBuffer &operator=(OwnedBuffer &&) noexcept;
  OwnedBuffer(const OwnedBuffer &) = delete;
  OwnedBuffer &operator=(const OwnedBuffer &) = delete;

  [[nodiscard]] bool valid() const noexcept { return buf_ != nullptr; }
  [[nodiscard]] MetalBuffer mtl_buffer() const noexcept { return buf_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
  // Offset within the underlying MTLBuffer. 0 for Persistent (each
  // allocation gets its own buffer); non-zero for Transient (multiple
  // sub-allocations share the lane buffer at different offsets).
  [[nodiscard]] std::size_t offset() const noexcept { return offset_; }
  // For Shared storage, the host-side pointer at this allocation's offset;
  // null on Private.
  [[nodiscard]] void *cpu_data() const noexcept { return cpu_; }

private:
  friend class Heap;
  OwnedBuffer(Heap *heap, MetalBuffer buf, void *cpu, std::size_t bytes,
              std::size_t offset, Storage storage, Lifetime lifetime) noexcept
      : heap_(heap), buf_(buf), cpu_(cpu), bytes_(bytes), offset_(offset),
        storage_(storage), lifetime_(lifetime) {}

  Heap *heap_ = nullptr;
  MetalBuffer buf_ = nullptr;
  void *cpu_ = nullptr;
  std::size_t bytes_ = 0;
  std::size_t offset_ = 0;
  Storage storage_ = Storage::Shared;
  Lifetime lifetime_ = Lifetime::Persistent;
};

}  // namespace metal_heap

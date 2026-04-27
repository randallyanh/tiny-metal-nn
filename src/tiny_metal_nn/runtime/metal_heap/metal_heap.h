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

enum class Lifetime {
  Persistent,  // long-lived; backed by an MTLHeap, released on OwnedBuffer dtor
  Transient,   // per-frame ring lane; sub-region of a lane buffer
  // Power-of-2 size-class pool for host↔device transfer. The OwnedBuffer's
  // bytes() reports the bucket size, not the requested size, since the
  // bucket is what gets recycled to the pool's free list on dtor.
  Staging,
  External,    // wraps a caller-owned MTLBuffer; dtor does not release
};

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

  // Staging pool: total cap shared across size-class buckets. Used for
  // host↔device transfer paths that allocate-and-free in tight bursts.
  // Set to 0 to disable Staging lifetime.
  std::size_t staging_max_bytes = 8 * 1024 * 1024;  // 8 MiB
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
  StagingExhausted,
  InvalidConfig,
  DeviceUnavailable,
  UnsupportedLifetime,
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
  // frame counter.
  //
  // GPU fence integration (Phase 2.3): if a non-null `sync_event`
  // (id<MTLSharedEvent>) was provided to register_lane_fence_event(), then
  // begin_transient_frame() blocks the host until prior GPU work signaled
  // the lane's fence value. Without a registered event, the caller is
  // responsible for ensuring the next-lane reuse happens after prior GPU
  // work completes.
  //
  // Transient OwnedBuffers MUST NOT outlive the frame they were allocated in;
  // their `valid()` checks the frame counter and returns false after the
  // owning frame has ended.
  void begin_transient_frame() noexcept;
  void end_transient_frame() noexcept;

  // Optional cross-queue fence used by begin_transient_frame() to wait for
  // prior GPU work on a lane to complete. The Heap takes a +1 retain on the
  // event; pass nullptr to detach.
  void register_lane_fence_event(void *mtl_shared_event,
                                 std::uint64_t initial_value) noexcept;
  // For caller's command buffer: signals the event with the next value at
  // end_transient_frame(); returns the value the caller should encode as
  // [commandBuffer encodeSignalEvent:event value:value]. Returns 0 if no
  // event is registered.
  [[nodiscard]] std::uint64_t pending_lane_signal_value() const noexcept;

  // Adopts an externally-owned MTLBuffer (e.g. from PyTorch MPS). Heap does
  // NOT take ownership — the returned OwnedBuffer's dtor leaves the buffer
  // alive. Optional `sync_event` lets the caller fence cross-queue access:
  // tmnn's command buffer should `encodeWaitForEvent:value:` before reading.
  [[nodiscard]] OwnedBuffer adopt_external(
      MetalBuffer external_buffer,
      std::size_t bytes,
      Storage storage,
      void *sync_event = nullptr,            // id<MTLSharedEvent>
      std::uint64_t signal_value = 0) noexcept;

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
    std::size_t staging_capacity_bytes;
    std::size_t staging_resident_bytes;
    std::uint32_t staging_buffers_resident;
    // Live-buffer counters split by lifetime so callers can spot leaks per
    // class without disambiguating in their own code.
    std::uint64_t live_persistent_buffers;
    std::uint64_t live_transient_buffers;
    std::uint64_t live_staging_buffers;
    std::uint64_t live_external_buffers;
    std::uint64_t total_allocations;
    std::uint64_t alloc_failures;
  };
  [[nodiscard]] Stats stats() const noexcept;

private:
  Heap() noexcept;
  // Called by OwnedBuffer's destructor; dispatches by lifetime.
  // - Persistent: releases the MTLBuffer + decrements counter.
  // - Transient:  no-op; lane reset on begin_transient_frame recycles.
  // - Staging:    returns the buffer to its size-class bucket.
  // - External:   no-op; caller owns the buffer.
  void deallocate(MetalBuffer buf, std::size_t bytes, Storage storage,
                  Lifetime lifetime,
                  std::uint64_t lifetime_token) noexcept;

  // Frame counter snapshot used by Transient OwnedBuffer.valid() to detect
  // dangling references after the owning frame has ended.
  [[nodiscard]] std::uint64_t current_transient_frame() const noexcept;

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

  // For Persistent / Staging / External: true iff the buffer is held.
  // For Transient: also requires the owning frame to still be active —
  // otherwise the lane has been recycled and the view is stale.
  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] MetalBuffer mtl_buffer() const noexcept { return buf_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
  // Offset within the underlying MTLBuffer. 0 for Persistent / Staging /
  // External (each allocation has its own buffer); non-zero for Transient
  // (multiple sub-allocations share the lane buffer at different offsets).
  [[nodiscard]] std::size_t offset() const noexcept { return offset_; }
  // For Shared storage, the host-side pointer at this allocation's offset;
  // null on Private.
  [[nodiscard]] void *cpu_data() const noexcept { return cpu_; }
  [[nodiscard]] Lifetime lifetime() const noexcept { return lifetime_; }

  // Cross-queue fence metadata for External buffers (and any other lifetime
  // the caller explicitly registers). For tmnn-managed lifetimes this is
  // {nullptr, 0}.
  struct SyncEvent {
    void *event = nullptr;          // id<MTLSharedEvent>
    std::uint64_t signal_value = 0;
  };
  [[nodiscard]] SyncEvent sync_event() const noexcept {
    return {sync_event_, sync_value_};
  }

private:
  friend class Heap;
  OwnedBuffer(Heap *heap, MetalBuffer buf, void *cpu, std::size_t bytes,
              std::size_t offset, Storage storage, Lifetime lifetime,
              std::uint64_t lifetime_token,
              void *sync_event = nullptr,
              std::uint64_t sync_value = 0) noexcept
      : heap_(heap), buf_(buf), cpu_(cpu), bytes_(bytes), offset_(offset),
        storage_(storage), lifetime_(lifetime),
        lifetime_token_(lifetime_token),
        sync_event_(sync_event), sync_value_(sync_value) {}

  Heap *heap_ = nullptr;
  MetalBuffer buf_ = nullptr;
  void *cpu_ = nullptr;
  std::size_t bytes_ = 0;
  std::size_t offset_ = 0;
  Storage storage_ = Storage::Shared;
  Lifetime lifetime_ = Lifetime::Persistent;
  // For Transient: the frame counter at which this allocation was made.
  // valid() compares it against the heap's current frame counter; mismatch
  // means the lane has been recycled.
  std::uint64_t lifetime_token_ = 0;
  void *sync_event_ = nullptr;
  std::uint64_t sync_value_ = 0;
};

}  // namespace metal_heap

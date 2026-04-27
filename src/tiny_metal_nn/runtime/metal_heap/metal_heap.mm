// metal_heap.mm — Apple Metal MTLHeap implementation. MRC; not ARC.

#include "tiny_metal_nn/runtime/metal_heap/metal_heap.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace metal_heap {

namespace {

MTLResourceOptions resource_options(Storage storage,
                                    HazardTracking hazard) {
  MTLResourceOptions opts =
      storage == Storage::Shared ? MTLResourceStorageModeShared
                                 : MTLResourceStorageModePrivate;
  if (hazard == HazardTracking::Untracked) {
    opts |= MTLResourceHazardTrackingModeUntracked;
  } else {
    opts |= MTLResourceHazardTrackingModeTracked;
  }
  return opts;
}

}  // namespace

// Pre-allocated N-lane ring of MTLBuffers. Hot-path allocate is just
// alignment + bump on the current lane — no buffer creation, no objc_msgSend
// for label (lane buffer is labeled once at construction).
//
// Phase 2.2 MVP: lane buffers are hardcoded Shared + Untracked. AllocDesc's
// storage / hazard_tracking are ignored for Transient and the call returns
// success only when matching that fixed config; otherwise UnsupportedStorage.
class TransientRing {
public:
  TransientRing(id<MTLDevice> device, std::size_t lane_bytes,
                std::uint32_t lane_count) {
    if (lane_count == 0 || lane_bytes == 0) return;
    lane_bytes_ = lane_bytes;
    lanes_.resize(lane_count);

    const MTLResourceOptions opts =
        MTLResourceStorageModeShared |
        MTLResourceHazardTrackingModeUntracked;

    for (std::uint32_t i = 0; i < lane_count; ++i) {
      id<MTLBuffer> buf =
          [device newBufferWithLength:lane_bytes options:opts];
      if (!buf) continue;
      NSString *label = [[NSString alloc] initWithFormat:
          @"metal_heap.transient.lane%u", i];
      [buf setLabel:label];
      [label release];
      lanes_[i].buffer = buf;            // +1 retained from -newBuffer
      lanes_[i].cpu_base = [buf contents];
    }
  }

  ~TransientRing() {
    for (auto &lane : lanes_) {
      if (lane.buffer) [lane.buffer release];
    }
  }

  TransientRing(const TransientRing &) = delete;
  TransientRing &operator=(const TransientRing &) = delete;

  bool enabled() const { return !lanes_.empty(); }
  std::size_t lane_bytes() const { return lane_bytes_; }
  std::uint32_t lane_count() const {
    return static_cast<std::uint32_t>(lanes_.size());
  }
  std::uint32_t current_lane() const { return current_; }
  std::size_t current_used() const {
    return enabled() ? lanes_[current_].offset : 0;
  }
  std::uint64_t frames_completed() const { return frames_completed_; }

  void begin_frame() {
    if (!enabled()) return;
    current_ = (current_ + 1) % static_cast<std::uint32_t>(lanes_.size());
    lanes_[current_].offset = 0;
  }

  void end_frame() {
    if (!enabled()) return;
    ++frames_completed_;
  }

  struct AllocResult {
    id<MTLBuffer> buffer = nil;
    void *cpu = nullptr;
    std::size_t offset = 0;
    bool capacity_exceeded = false;
  };
  // bytes + alignment must be > 0; aligned to alignment up from current
  // lane offset. Returns nil buffer + capacity_exceeded when the rounded-up
  // allocation does not fit.
  AllocResult allocate(std::size_t bytes, std::size_t alignment) {
    if (!enabled()) return {};
    auto &lane = lanes_[current_];
    if (alignment == 0) alignment = 1;
    std::size_t aligned =
        (lane.offset + alignment - 1) & ~(alignment - 1);
    if (aligned + bytes > lane_bytes_) {
      AllocResult r;
      r.capacity_exceeded = true;
      return r;
    }
    lane.offset = aligned + bytes;
    AllocResult r;
    r.buffer = lane.buffer;
    r.cpu = static_cast<std::uint8_t *>(lane.cpu_base) + aligned;
    r.offset = aligned;
    return r;
  }

private:
  struct Lane {
    id<MTLBuffer> buffer = nil;
    void *cpu_base = nullptr;
    std::size_t offset = 0;
  };
  std::vector<Lane> lanes_;
  std::size_t lane_bytes_ = 0;
  std::uint32_t current_ = 0;
  std::uint64_t frames_completed_ = 0;
};

class Heap::Impl {
public:
  Impl(id<MTLDevice> device, id<MTLHeap> shared, std::size_t shared_cap,
       id<MTLHeap> priv, std::size_t priv_cap,
       std::unique_ptr<TransientRing> transient)
      : device_(device), shared_(shared), shared_capacity_(shared_cap),
        private_(priv), private_capacity_(priv_cap),
        transient_(std::move(transient)) {
    [device_ retain];
    if (shared_)  [shared_  retain];
    if (private_) [private_ retain];
  }

  ~Impl() {
    for (auto &kv : label_cache_) [(NSString *)kv.second release];
    if (private_) [private_ release];
    if (shared_)  [shared_  release];
    [device_ release];
  }

  // Caller-stable `const char*` debug names (typically string literals) get
  // their NSString translation cached here, so the hot allocate path
  // pays one objc_msgSend instead of an NSString allocation per call.
  NSString *cached_label(const char *name) {
    auto it = label_cache_.find(name);
    if (it != label_cache_.end()) return (NSString *)it->second;
    NSString *s = [[NSString alloc] initWithUTF8String:name];  // +1 retained
    label_cache_.emplace(name, (void *)s);
    return s;
  }

  id<MTLHeap> shared_heap()  const { return shared_; }
  id<MTLHeap> private_heap() const { return private_; }
  std::size_t shared_capacity()  const { return shared_capacity_; }
  std::size_t private_capacity() const { return private_capacity_; }

  std::size_t shared_used()  const { return shared_used_; }
  std::size_t private_used() const { return private_used_; }
  std::uint64_t live_buffers() const { return live_buffers_; }
  std::uint64_t total_allocations() const { return total_allocations_; }
  std::uint64_t alloc_failures() const { return alloc_failures_; }

  void note_persistent_allocation(Storage s, std::size_t bytes) {
    if (s == Storage::Shared) shared_used_ += bytes; else private_used_ += bytes;
    ++live_buffers_;
    ++total_allocations_;
  }
  void note_persistent_deallocation(Storage s, std::size_t bytes) {
    auto &counter = (s == Storage::Shared) ? shared_used_ : private_used_;
    if (bytes <= counter) counter -= bytes;
    if (live_buffers_ > 0) --live_buffers_;
  }
  void note_transient_allocation() {
    ++live_buffers_;
    ++total_allocations_;
  }
  void note_transient_deallocation() {
    if (live_buffers_ > 0) --live_buffers_;
  }
  void note_failure() { ++alloc_failures_; }

  TransientRing *transient() { return transient_.get(); }
  const TransientRing *transient() const { return transient_.get(); }

private:
  id<MTLDevice> device_;
  id<MTLHeap> shared_;
  std::size_t shared_capacity_;
  id<MTLHeap> private_;
  std::size_t private_capacity_;
  std::unique_ptr<TransientRing> transient_;
  std::size_t shared_used_  = 0;
  std::size_t private_used_ = 0;
  std::uint64_t live_buffers_ = 0;
  std::uint64_t total_allocations_ = 0;
  std::uint64_t alloc_failures_ = 0;
  std::unordered_map<const char *, void *> label_cache_;
};

namespace {

id<MTLHeap> create_typed_heap(id<MTLDevice> device, std::size_t capacity,
                              MTLStorageMode storage_mode,
                              const char *label) {
  if (capacity == 0) return nil;
  MTLHeapDescriptor *desc = [[MTLHeapDescriptor alloc] init];
  desc.size = capacity;
  desc.storageMode = storage_mode;
  desc.cpuCacheMode = MTLCPUCacheModeDefaultCache;
  desc.hazardTrackingMode = MTLHazardTrackingModeUntracked;
  desc.type = MTLHeapTypeAutomatic;
  id<MTLHeap> heap = [device newHeapWithDescriptor:desc];
  [desc release];
  if (heap) [heap setLabel:[NSString stringWithUTF8String:label]];
  return heap;  // +1 retained, transferred to caller
}

}  // namespace

Heap::Heap() noexcept = default;

Heap::~Heap() noexcept = default;

std::unique_ptr<Heap> Heap::create(MetalDevice mtl_device,
                                   const HeapConfig &cfg) noexcept {
  if (!mtl_device) return nullptr;
  if (cfg.persistent_shared_capacity_bytes == 0 &&
      cfg.persistent_private_capacity_bytes == 0) {
    return nullptr;
  }
  id<MTLDevice> device = (id<MTLDevice>)mtl_device;

  id<MTLHeap> shared = create_typed_heap(
      device, cfg.persistent_shared_capacity_bytes,
      MTLStorageModeShared, "metal_heap.persistent.shared");
  id<MTLHeap> priv = create_typed_heap(
      device, cfg.persistent_private_capacity_bytes,
      MTLStorageModePrivate, "metal_heap.persistent.private");

  if (cfg.persistent_shared_capacity_bytes  > 0 && !shared) goto fail;
  if (cfg.persistent_private_capacity_bytes > 0 && !priv)   goto fail;

  {
    auto transient = std::make_unique<TransientRing>(
        device, cfg.transient_lane_bytes, cfg.transient_lane_count);
    std::unique_ptr<Heap> h{new Heap()};
    h->impl_ = std::make_unique<Impl>(
        device, shared, cfg.persistent_shared_capacity_bytes,
                priv,   cfg.persistent_private_capacity_bytes,
        std::move(transient));
    if (shared) [shared release];  // Impl took its own retain
    if (priv)   [priv   release];
    return h;
  }

fail:
  if (shared) [shared release];
  if (priv)   [priv   release];
  return nullptr;
}

std::expected<OwnedBuffer, AllocError>
Heap::allocate(const AllocDesc &desc) noexcept {
  if (!impl_ || !desc.debug_name) {
    if (impl_) impl_->note_failure();
    return std::unexpected(AllocError::InvalidConfig);
  }

  if (desc.lifetime == Lifetime::Persistent) {
    id<MTLHeap> selected = (desc.storage == Storage::Shared)
        ? impl_->shared_heap() : impl_->private_heap();
    if (!selected) {
      impl_->note_failure();
      return std::unexpected(AllocError::UnsupportedStorage);
    }
    const MTLResourceOptions opts =
        resource_options(desc.storage, desc.hazard_tracking);
    id<MTLBuffer> buf = [selected newBufferWithLength:desc.bytes options:opts];
    if (!buf) {
      impl_->note_failure();
      return std::unexpected(AllocError::PersistentExhausted);
    }
    [buf setLabel:impl_->cached_label(desc.debug_name)];
    void *cpu = (desc.storage == Storage::Shared) ? [buf contents] : nullptr;
    impl_->note_persistent_allocation(desc.storage, desc.bytes);
    return OwnedBuffer(this, (MetalBuffer)buf, cpu, desc.bytes, /*offset*/ 0,
                       desc.storage, Lifetime::Persistent);
  }

  if (desc.lifetime == Lifetime::Transient) {
    TransientRing *ring = impl_->transient();
    if (!ring || !ring->enabled()) {
      impl_->note_failure();
      return std::unexpected(AllocError::UnsupportedLifetime);
    }
    // Phase 2.2 MVP: lane buffers are fixed Shared+Untracked. Any other
    // request is rejected so the caller knows it is outside the supported
    // configuration rather than silently downgraded.
    if (desc.storage != Storage::Shared ||
        desc.hazard_tracking != HazardTracking::Untracked) {
      impl_->note_failure();
      return std::unexpected(AllocError::UnsupportedStorage);
    }
    auto r = ring->allocate(desc.bytes, desc.alignment);
    if (r.capacity_exceeded || !r.buffer) {
      impl_->note_failure();
      return std::unexpected(AllocError::TransientCapacityExceeded);
    }
    impl_->note_transient_allocation();
    return OwnedBuffer(this, (MetalBuffer)r.buffer, r.cpu, desc.bytes,
                       r.offset, Storage::Shared, Lifetime::Transient);
  }

  impl_->note_failure();
  return std::unexpected(AllocError::UnsupportedLifetime);
}

void Heap::begin_transient_frame() noexcept {
  if (impl_ && impl_->transient()) impl_->transient()->begin_frame();
}

void Heap::end_transient_frame() noexcept {
  if (impl_ && impl_->transient()) impl_->transient()->end_frame();
}

Heap::Stats Heap::stats() const noexcept {
  Stats s{};
  if (impl_) {
    s.persistent_shared_capacity_bytes  = impl_->shared_capacity();
    s.persistent_shared_used_bytes      = impl_->shared_used();
    s.persistent_private_capacity_bytes = impl_->private_capacity();
    s.persistent_private_used_bytes     = impl_->private_used();
    if (auto *ring = impl_->transient()) {
      s.transient_lane_bytes               = ring->lane_bytes();
      s.transient_lane_count               = ring->lane_count();
      s.transient_current_lane             = ring->current_lane();
      s.transient_current_lane_used_bytes  = ring->current_used();
      s.transient_frames_completed         = ring->frames_completed();
    }
    s.live_buffers      = impl_->live_buffers();
    s.total_allocations = impl_->total_allocations();
    s.alloc_failures    = impl_->alloc_failures();
  }
  return s;
}

void Heap::deallocate(MetalBuffer buf, std::size_t bytes, Storage storage,
                      Lifetime lifetime) noexcept {
  if (!buf || !impl_) return;
  if (lifetime == Lifetime::Persistent) {
    [(id<MTLBuffer>)buf release];
    impl_->note_persistent_deallocation(storage, bytes);
  } else {
    // Transient: backing storage is the lane buffer owned by TransientRing;
    // the lane reset on the next begin_transient_frame() recycles it.
    impl_->note_transient_deallocation();
  }
}

OwnedBuffer::~OwnedBuffer() noexcept {
  if (heap_ && buf_) {
    heap_->deallocate(buf_, bytes_, storage_, lifetime_);
  }
}

OwnedBuffer::OwnedBuffer(OwnedBuffer &&other) noexcept
    : heap_(other.heap_), buf_(other.buf_), cpu_(other.cpu_),
      bytes_(other.bytes_), offset_(other.offset_),
      storage_(other.storage_), lifetime_(other.lifetime_) {
  other.heap_ = nullptr;
  other.buf_ = nullptr;
  other.cpu_ = nullptr;
  other.bytes_ = 0;
  other.offset_ = 0;
}

OwnedBuffer &OwnedBuffer::operator=(OwnedBuffer &&other) noexcept {
  if (this != &other) {
    if (heap_ && buf_) {
      heap_->deallocate(buf_, bytes_, storage_, lifetime_);
    }
    heap_ = other.heap_;
    buf_ = other.buf_;
    cpu_ = other.cpu_;
    bytes_ = other.bytes_;
    offset_ = other.offset_;
    storage_ = other.storage_;
    lifetime_ = other.lifetime_;
    other.heap_ = nullptr;
    other.buf_ = nullptr;
    other.cpu_ = nullptr;
    other.bytes_ = 0;
    other.offset_ = 0;
  }
  return *this;
}

}  // namespace metal_heap

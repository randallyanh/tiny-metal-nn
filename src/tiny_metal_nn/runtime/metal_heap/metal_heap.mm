// metal_heap.mm — Apple Metal MTLHeap implementation. MRC; not ARC.

#include "tiny_metal_nn/runtime/metal_heap/metal_heap.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <unordered_map>

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

class Heap::Impl {
public:
  Impl(id<MTLDevice> device, id<MTLHeap> shared, std::size_t shared_cap,
       id<MTLHeap> priv, std::size_t priv_cap)
      : device_(device), shared_(shared), shared_capacity_(shared_cap),
        private_(priv), private_capacity_(priv_cap) {
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

  void note_allocation(Storage s, std::size_t bytes) {
    if (s == Storage::Shared) shared_used_ += bytes; else private_used_ += bytes;
    ++live_buffers_;
    ++total_allocations_;
  }
  void note_deallocation(Storage s, std::size_t bytes) {
    auto &counter = (s == Storage::Shared) ? shared_used_ : private_used_;
    if (bytes <= counter) counter -= bytes;
    if (live_buffers_ > 0) --live_buffers_;
  }
  void note_failure() { ++alloc_failures_; }

private:
  id<MTLDevice> device_;
  id<MTLHeap> shared_;
  std::size_t shared_capacity_;
  id<MTLHeap> private_;
  std::size_t private_capacity_;
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
    std::unique_ptr<Heap> h{new Heap()};
    h->impl_ = std::make_unique<Impl>(
        device, shared, cfg.persistent_shared_capacity_bytes,
                priv,   cfg.persistent_private_capacity_bytes);
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
  if (desc.lifetime != Lifetime::Persistent) {
    impl_->note_failure();
    return std::unexpected(AllocError::UnsupportedLifetime);
  }

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
  impl_->note_allocation(desc.storage, desc.bytes);
  return OwnedBuffer(this, (MetalBuffer)buf, cpu, desc.bytes, desc.storage);
}

Heap::Stats Heap::stats() const noexcept {
  Stats s{};
  if (impl_) {
    s.persistent_shared_capacity_bytes  = impl_->shared_capacity();
    s.persistent_shared_used_bytes      = impl_->shared_used();
    s.persistent_private_capacity_bytes = impl_->private_capacity();
    s.persistent_private_used_bytes     = impl_->private_used();
    s.live_buffers      = impl_->live_buffers();
    s.total_allocations = impl_->total_allocations();
    s.alloc_failures    = impl_->alloc_failures();
  }
  return s;
}

void Heap::deallocate(MetalBuffer buf, std::size_t bytes,
                      Storage storage) noexcept {
  if (!buf) return;
  [(id<MTLBuffer>)buf release];
  if (impl_) impl_->note_deallocation(storage, bytes);
}

OwnedBuffer::~OwnedBuffer() noexcept {
  if (heap_ && buf_) {
    heap_->deallocate(buf_, bytes_, storage_);
  }
}

OwnedBuffer::OwnedBuffer(OwnedBuffer &&other) noexcept
    : heap_(other.heap_), buf_(other.buf_), cpu_(other.cpu_),
      bytes_(other.bytes_), storage_(other.storage_) {
  other.heap_ = nullptr;
  other.buf_ = nullptr;
  other.cpu_ = nullptr;
  other.bytes_ = 0;
}

OwnedBuffer &OwnedBuffer::operator=(OwnedBuffer &&other) noexcept {
  if (this != &other) {
    if (heap_ && buf_) heap_->deallocate(buf_, bytes_, storage_);
    heap_ = other.heap_;
    buf_ = other.buf_;
    cpu_ = other.cpu_;
    bytes_ = other.bytes_;
    storage_ = other.storage_;
    other.heap_ = nullptr;
    other.buf_ = nullptr;
    other.cpu_ = nullptr;
    other.bytes_ = 0;
  }
  return *this;
}

}  // namespace metal_heap

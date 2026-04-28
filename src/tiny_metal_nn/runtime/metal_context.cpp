/**
 * @file metal_context.cpp
 * @brief MetalContext::Impl — owns BufferArena, CommandBatchPool,
 *        PipelineRegistry, NumericsGuard, and (on Apple) a real MTLDevice.
 *
 * On Apple platforms, probes the system default Metal device and populates
 * DeviceCapabilities from real hardware. Wires the device/queue into
 * BufferArena (for MTLBuffer creation), CommandBatchPool (for real command
 * buffers), and PipelineRegistry (for MSL compilation).
 *
 * On non-Apple, everything falls back to CPU-only stubs.
 */

#include "tiny-metal-nn/metal_context.h"
#include "tiny_metal_nn/runtime/buffer_arena.h"
#include "tiny_metal_nn/runtime/command_batch.h"
#include "tiny_metal_nn/runtime/metal_context_internal.h"
#include "tiny_metal_nn/runtime/metal_device.h"
#include "tiny_metal_nn/runtime/metal_heap/metal_heap.h"
#include "tiny_metal_nn/runtime/numerics_guard.h"
#include "tiny_metal_nn/runtime/pipeline_registry.h"
#include "tiny_metal_nn/runtime/runtime_policy.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace tmnn {

// ---------------------------------------------------------------------------
// MetalContext::Impl
// ---------------------------------------------------------------------------

class MetalContext::Impl {
public:
  explicit Impl(const MetalContextDesc &desc)
      : desc_(desc), policy_(load_runtime_policy(desc.policy_overrides)),
        arena_(64),
        batch_pool_(desc.max_inflight_batches),
        numerics_guard_(policy_.numerics_sampling_mode) {
    numerics_guard_.set_sample_interval(policy_.numerics_sample_interval);
    // Probe the system default Metal device.
    device_info_ = metal::probe_default_device();
    gpu_available_ = (device_info_.device != nullptr);

    if (gpu_available_) {
      // Populate capabilities from real hardware.
      caps_.device_name = device_info_.name;
      caps_.gpu_family = device_info_.gpu_family;
      caps_.max_threads_per_tg = device_info_.max_threads_per_tg;
      caps_.max_threadgroup_memory_bytes = device_info_.max_tg_memory_bytes;
      caps_.supports_fp16 = device_info_.supports_fp16;
      caps_.supports_simdgroup_matrix = device_info_.supports_simdgroup_matrix;
      caps_.supports_binary_archive = device_info_.supports_binary_archive;
      caps_.supports_nonuniform_threadgroups =
          device_info_.supports_nonuniform_tgs;

      // Wire device/queue into sub-components.
      arena_.set_device(device_info_.device);
      batch_pool_.set_queue(device_info_.queue);
      registry_.set_device(device_info_.device);

      // Phase 3.2: bring up the metal_heap::Heap. By default it stays
      // small (dormant) — BufferArena uses the legacy direct-device path
      // and the Heap is reserved for opt-in use cases (adopt_external,
      // future TransientRing). Set heap_config.route_buffer_arena_through_heap
      // to true to size the Heap from MTLDevice.recommendedMaxWorkingSetSize
      // and make BufferArena route Persistent allocations through it.
      heap_ = make_heap_or_throw(device_info_, desc_.heap_config);
      if (desc_.heap_config.route_buffer_arena_through_heap) {
        arena_.set_heap(heap_.get());
      }
    }
  }

  // Derive a metal_heap::HeapConfig from the user's MetalContextHeapConfig
  // and the device's recommendedMaxWorkingSetSize. Hard-fails if Heap::create
  // returns nullptr — silent degradation would just trade a clear startup
  // error for a cryptic runtime failure later.
  static std::unique_ptr<metal_heap::Heap>
  make_heap_or_throw(const metal::DeviceInfo &device_info,
                     const MetalContextHeapConfig &cfg) {
    metal_heap::HeapConfig hcfg;
    hcfg.transient_lane_count = 0;
    // Staging pool: cap is a CEILING, not eager reservation. Bucket
    // buffers are individual MTLBuffers (not heap sub-buffers), so they
    // don't trigger MTLHeap's first-bind eager-commit pattern; resident
    // bytes track only what gets acquired from the pool.
    //
    // 256 MiB sized to hold all four Adam-state sections concurrently
    // for a default HashGridEncoding (log2_hashmap=19, num_levels=16,
    // features_per_level=2 ⇒ 64 MiB hash-grid; m_hash + v_hash =
    // 128 MiB) plus headroom for the MLP buckets. Users on smaller
    // hash configs can drop this to 32 MiB without loss.
    constexpr std::size_t kDefaultStagingCap = 256 * 1024 * 1024;
    hcfg.staging_max_bytes = (cfg.staging_capacity_bytes != 0)
                                 ? cfg.staging_capacity_bytes
                                 : kDefaultStagingCap;

    const auto derive_total = [&]() -> std::size_t {
      const double ws = static_cast<double>(device_info.recommended_working_set_bytes);
      const auto raw = static_cast<std::size_t>(ws * cfg.working_set_fraction);
      return std::clamp(raw, cfg.min_total_bytes, cfg.max_total_bytes);
    };

    // Tiny dormant default when BufferArena is NOT routed through the
    // Heap — keeps wired memory at the Phase-2 baseline. The Heap stays
    // available for adopt_external / future TransientRing use cases.
    constexpr std::size_t kDormantDefault = 1 * 1024 * 1024;

    if (cfg.persistent_shared_capacity_bytes != 0) {
      hcfg.persistent_shared_capacity_bytes = cfg.persistent_shared_capacity_bytes;
    } else if (cfg.route_buffer_arena_through_heap) {
      const auto total = derive_total();
      hcfg.persistent_shared_capacity_bytes =
          static_cast<std::size_t>(total * cfg.shared_share);
    } else {
      hcfg.persistent_shared_capacity_bytes = kDormantDefault;
    }
    if (cfg.persistent_private_capacity_bytes != 0) {
      hcfg.persistent_private_capacity_bytes = cfg.persistent_private_capacity_bytes;
    } else if (cfg.route_buffer_arena_through_heap) {
      const auto total = derive_total();
      hcfg.persistent_private_capacity_bytes =
          total - static_cast<std::size_t>(total * cfg.shared_share);
    } else {
      hcfg.persistent_private_capacity_bytes = kDormantDefault;
    }

    auto heap = metal_heap::Heap::create(device_info.device, hcfg);
    if (!heap) {
      throw std::runtime_error(
          std::string("MetalContext: metal_heap::Heap::create failed "
                      "(shared=") +
          std::to_string(hcfg.persistent_shared_capacity_bytes) +
          " priv=" +
          std::to_string(hcfg.persistent_private_capacity_bytes) +
          " working_set=" +
          std::to_string(device_info.recommended_working_set_bytes) +
          " bytes); reduce MetalContextDesc.heap_config or free GPU memory");
    }
    return heap;
  }

  ~Impl() { metal::release_device(device_info_); }

  const DeviceCapabilities &capabilities() const { return caps_; }

  RuntimeStats snapshot_stats() const {
    RuntimeStats stats;
    {
      std::scoped_lock lock(stats_mu_);
      stats = stats_;
    }
    stats.persistent_bytes = arena_.bytes_allocated();
    stats.async_batches_submitted = batch_pool_.total_submitted();
    auto ps = registry_.stats();
    stats.pipeline_cache_hits = ps.hits;
    stats.pipeline_cache_misses = ps.misses;
    stats.compile_count = ps.compile_count;
    return stats;
  }

  void note_training_step(uint64_t numerics_reports, uint64_t numerics_anomalies,
                          uint64_t bad_steps_skipped,
                          uint64_t bad_steps_rolled_back,
                          uint64_t safe_family_recoveries) {
    std::scoped_lock lock(stats_mu_);
    ++stats_.training_steps_completed;
    stats_.numerics_report_count += numerics_reports;
    stats_.numerics_anomaly_count += numerics_anomalies;
    stats_.bad_steps_skipped += bad_steps_skipped;
    stats_.bad_steps_rolled_back += bad_steps_rolled_back;
    stats_.safe_family_recoveries += safe_family_recoveries;
  }

  void prewarm_autotune_manifest(const AutotuneManifest &manifest) {
    if (!manifest.version.empty() &&
        manifest.version != kAutotuneManifestVersion) {
      throw std::invalid_argument(
          "MetalContext::prewarm_autotune_manifest: unsupported manifest "
          "version '" +
          manifest.version + "'");
    }

    if (manifest.device_family != 0 && caps_.gpu_family != 0 &&
        manifest.device_family != caps_.gpu_family) {
      return;
    }

    std::scoped_lock lock(autotune_mu_);
    autotune_entries_.clear();
    for (const auto &entry : manifest.entries) {
      if (entry.planner_fingerprint == 0)
        continue;
      autotune_entries_[entry.planner_fingerprint] = entry;
    }
  }

  AutotuneManifest snapshot_autotune_manifest() const {
    std::scoped_lock lock(autotune_mu_);
    AutotuneManifest manifest;
    manifest.version = std::string(kAutotuneManifestVersion);
    manifest.device_name = caps_.device_name;
    manifest.device_family = caps_.gpu_family;
    manifest.entries.reserve(autotune_entries_.size());
    for (const auto &[_, entry] : autotune_entries_)
      manifest.entries.push_back(entry);
    std::sort(manifest.entries.begin(), manifest.entries.end(),
              [](const auto &a, const auto &b) {
                return a.planner_fingerprint < b.planner_fingerprint;
              });
    return manifest;
  }

  void clear_runtime_caches() { registry_.clear(); }

  bool is_gpu_available() const { return gpu_available_; }
  const MetalContextDesc &desc() const { return desc_; }
  const RuntimePolicy &policy() const { return policy_; }

  BufferArena &arena() { return arena_; }
  CommandBatchPool &batch_pool() { return batch_pool_; }
  PipelineRegistry &registry() { return registry_; }
  NumericsGuard &numerics_guard() { return numerics_guard_; }
  metal_heap::Heap *heap() { return heap_.get(); }
  void record_training_step(uint64_t numerics_reports,
                            uint64_t numerics_anomalies,
                            uint64_t bad_steps_skipped,
                            uint64_t bad_steps_rolled_back,
                            uint64_t safe_family_recoveries) {
    note_training_step(numerics_reports, numerics_anomalies, bad_steps_skipped,
                       bad_steps_rolled_back, safe_family_recoveries);
  }
  std::optional<AutotuneManifestEntry>
  lookup_autotune_entry(uint64_t planner_fingerprint) const {
    std::scoped_lock lock(autotune_mu_);
    const auto it = autotune_entries_.find(planner_fingerprint);
    if (it == autotune_entries_.end())
      return std::nullopt;
    return it->second;
  }
  void record_autotune_entry(const AutotuneManifestEntry &entry) {
    if (entry.planner_fingerprint == 0)
      return;
    std::scoped_lock lock(autotune_mu_);
    autotune_entries_[entry.planner_fingerprint] = entry;
  }

  void *raw_device() const { return device_info_.device; }
  void *raw_queue() const { return device_info_.queue; }

private:
  MetalContextDesc desc_;
  RuntimePolicy policy_;
  DeviceCapabilities caps_;
  RuntimeStats stats_;
  mutable std::mutex stats_mu_;
  metal::DeviceInfo device_info_;
  // heap_ MUST outlive arena_ (and any other member that holds OwnedBuffers
  // sourced from this Heap). Members are destroyed in reverse declaration
  // order, so heap_ is declared before arena_.
  std::unique_ptr<metal_heap::Heap> heap_;
  BufferArena arena_;
  CommandBatchPool batch_pool_;
  PipelineRegistry registry_;
  NumericsGuard numerics_guard_;
  mutable std::mutex autotune_mu_;
  std::unordered_map<uint64_t, AutotuneManifestEntry> autotune_entries_;
  bool gpu_available_ = false;
};

// ---------------------------------------------------------------------------
// MetalContext public API
// ---------------------------------------------------------------------------

std::shared_ptr<MetalContext>
MetalContext::create(const MetalContextDesc &desc) {
  return std::shared_ptr<MetalContext>(new MetalContext(desc));
}

MetalContext::MetalContext(const MetalContextDesc &desc)
    : impl_(std::make_unique<Impl>(desc)) {}

MetalContext::~MetalContext() = default;

const DeviceCapabilities &MetalContext::capabilities() const {
  return impl_->capabilities();
}

RuntimeStats MetalContext::snapshot_stats() const {
  auto stats = impl_->snapshot_stats();
  if (impl_->policy().emit_runtime_stats) {
    std::fprintf(
        stderr,
        "TMNN_STATS training_steps_completed=%llu numerics_report_count=%llu "
        "numerics_anomaly_count=%llu bad_steps_skipped=%llu "
        "bad_steps_rolled_back=%llu "
        "safe_family_recoveries=%llu compile_count=%llu "
        "pipeline_cache_hits=%llu pipeline_cache_misses=%llu "
        "async_batches_submitted=%llu persistent_bytes=%zu "
        "transient_peak_bytes=%zu\n",
        static_cast<unsigned long long>(stats.training_steps_completed),
        static_cast<unsigned long long>(stats.numerics_report_count),
        static_cast<unsigned long long>(stats.numerics_anomaly_count),
        static_cast<unsigned long long>(stats.bad_steps_skipped),
        static_cast<unsigned long long>(stats.bad_steps_rolled_back),
        static_cast<unsigned long long>(stats.safe_family_recoveries),
        static_cast<unsigned long long>(stats.compile_count),
        static_cast<unsigned long long>(stats.pipeline_cache_hits),
        static_cast<unsigned long long>(stats.pipeline_cache_misses),
        static_cast<unsigned long long>(stats.async_batches_submitted),
        stats.persistent_bytes, stats.transient_peak_bytes);
  }
  return stats;
}

void MetalContext::prewarm_autotune_manifest(const AutotuneManifest &manifest) {
  impl_->prewarm_autotune_manifest(manifest);
}

AutotuneManifest MetalContext::snapshot_autotune_manifest() const {
  return impl_->snapshot_autotune_manifest();
}

void MetalContext::clear_runtime_caches() { impl_->clear_runtime_caches(); }

bool MetalContext::is_gpu_available() const {
  return impl_->is_gpu_available();
}

const MetalContextDesc &MetalContext::desc() const { return impl_->desc(); }

const RuntimePolicy &MetalContext::policy() const { return impl_->policy(); }

// ---------------------------------------------------------------------------
// Internal accessors (detail::MetalContextAccessor)
// ---------------------------------------------------------------------------

namespace detail {

class MetalContextAccessor {
public:
  static BufferArena &arena(MetalContext &ctx) {
    return ctx.impl_->arena();
  }
  static CommandBatchPool &batch_pool(MetalContext &ctx) {
    return ctx.impl_->batch_pool();
  }
  static PipelineRegistry &registry(MetalContext &ctx) {
    return ctx.impl_->registry();
  }
  static NumericsGuard &numerics_guard(MetalContext &ctx) {
    return ctx.impl_->numerics_guard();
  }
  static metal_heap::Heap *heap(MetalContext &ctx) {
    return ctx.impl_->heap();
  }
  static void record_training_step(MetalContext &ctx, uint64_t numerics_reports,
                                   uint64_t numerics_anomalies,
                                   uint64_t bad_steps_skipped,
                                   uint64_t bad_steps_rolled_back,
                                   uint64_t safe_family_recoveries) {
    ctx.impl_->record_training_step(numerics_reports, numerics_anomalies,
                                    bad_steps_skipped, bad_steps_rolled_back,
                                    safe_family_recoveries);
  }
  static std::optional<AutotuneManifestEntry>
  lookup_autotune_entry(MetalContext &ctx, uint64_t planner_fingerprint) {
    return ctx.impl_->lookup_autotune_entry(planner_fingerprint);
  }
  static void record_autotune_entry(MetalContext &ctx,
                                    const AutotuneManifestEntry &entry) {
    ctx.impl_->record_autotune_entry(entry);
  }
  static void *raw_device(MetalContext &ctx) {
    return ctx.impl_->raw_device();
  }
  static void *raw_queue(MetalContext &ctx) {
    return ctx.impl_->raw_queue();
  }
};

BufferArena &context_arena(MetalContext &ctx) {
  return MetalContextAccessor::arena(ctx);
}

CommandBatchPool &context_batch_pool(MetalContext &ctx) {
  return MetalContextAccessor::batch_pool(ctx);
}

PipelineRegistry &context_pipeline_registry(MetalContext &ctx) {
  return MetalContextAccessor::registry(ctx);
}

metal_heap::Heap *context_heap(MetalContext &ctx) {
  return MetalContextAccessor::heap(ctx);
}

NumericsGuard &context_numerics_guard(MetalContext &ctx) {
  return MetalContextAccessor::numerics_guard(ctx);
}

void context_record_training_step(MetalContext &ctx, uint64_t numerics_reports,
                                  uint64_t numerics_anomalies,
                                  uint64_t bad_steps_skipped,
                                  uint64_t bad_steps_rolled_back,
                                  uint64_t safe_family_recoveries) {
  MetalContextAccessor::record_training_step(ctx, numerics_reports,
                                             numerics_anomalies, bad_steps_skipped,
                                             bad_steps_rolled_back,
                                             safe_family_recoveries);
}

std::optional<AutotuneManifestEntry>
context_lookup_autotune_entry(MetalContext &ctx, uint64_t planner_fingerprint) {
  return MetalContextAccessor::lookup_autotune_entry(ctx, planner_fingerprint);
}

void context_record_autotune_entry(MetalContext &ctx,
                                   const AutotuneManifestEntry &entry) {
  MetalContextAccessor::record_autotune_entry(ctx, entry);
}

void *context_raw_device(MetalContext &ctx) {
  return MetalContextAccessor::raw_device(ctx);
}

void *context_raw_queue(MetalContext &ctx) {
  return MetalContextAccessor::raw_queue(ctx);
}

void context_blit_fill(MetalContext &ctx, BufferView &view, uint8_t value) {
  if (!ctx.is_gpu_available() || !view.gpu_buffer)
    return;

  auto *cmd = metal::create_command_buffer(context_raw_queue(ctx));
  metal::encode_blit_fill(cmd, view.gpu_buffer, view.offset, view.bytes,
                          value);
  metal::commit_and_wait(cmd);
  metal::release_command_buffer(cmd);
}

void context_blit_fill_views(MetalContext &ctx,
                             std::span<const BufferView> views,
                             uint8_t value) {
  if (!ctx.is_gpu_available()) return;
  // Project to the metal_device::BlitFillRange shape and skip empty
  // entries — keeps the GPU-side encoder loop branchless and lets us
  // bail before allocating a command buffer when nothing has work.
  std::vector<metal::BlitFillRange> ranges;
  ranges.reserve(views.size());
  for (const auto &v : views) {
    if (v.gpu_buffer && v.bytes > 0) {
      ranges.push_back({v.gpu_buffer, v.offset, v.bytes});
    }
  }
  if (ranges.empty()) return;

  auto *cmd = metal::create_command_buffer(context_raw_queue(ctx));
  metal::encode_blit_fill_ranges(cmd, ranges.data(), ranges.size(), value);
  metal::commit_and_wait(cmd);
  metal::release_command_buffer(cmd);
}

void context_blit_upload(MetalContext &ctx, BufferView &dst, const void *data,
                         size_t bytes) {
  if (bytes == 0)
    return;
  if (!ctx.is_gpu_available() || !dst.gpu_buffer || !data) {
    throw std::runtime_error(
        "context_blit_upload requires a GPU-backed destination and source data");
  }

  const size_t copy_bytes = std::min(bytes, dst.bytes);
  void *staging = metal::create_buffer(context_raw_device(ctx), copy_bytes,
                                       /*shared=*/true);
  if (!staging) {
    throw std::runtime_error(
        "context_blit_upload failed to allocate staging buffer");
  }

  void *staging_data = metal::buffer_contents(staging);
  if (!staging_data) {
    metal::release_buffer(staging);
    throw std::runtime_error(
        "context_blit_upload staging buffer is not CPU-visible");
  }
  std::memcpy(staging_data, data, copy_bytes);

  auto *cmd = metal::create_command_buffer(context_raw_queue(ctx));
  metal::encode_blit_copy(cmd, staging, 0, dst.gpu_buffer, dst.offset,
                          copy_bytes);
  metal::commit_and_wait(cmd);
  metal::release_command_buffer(cmd);
  metal::release_buffer(staging);
}

void context_blit_download(MetalContext &ctx, const BufferView &src, void *out,
                           size_t bytes) {
  if (bytes == 0)
    return;
  if (!ctx.is_gpu_available() || !src.gpu_buffer || !out) {
    throw std::runtime_error(
        "context_blit_download requires a GPU-backed source and destination");
  }

  const size_t copy_bytes = std::min(bytes, src.bytes);
  void *staging = metal::create_buffer(context_raw_device(ctx), copy_bytes,
                                       /*shared=*/true);
  if (!staging) {
    throw std::runtime_error(
        "context_blit_download failed to allocate staging buffer");
  }

  auto *cmd = metal::create_command_buffer(context_raw_queue(ctx));
  metal::encode_blit_copy(cmd, src.gpu_buffer, src.offset, staging, 0,
                          copy_bytes);
  metal::commit_and_wait(cmd);
  metal::release_command_buffer(cmd);

  void *staging_data = metal::buffer_contents(staging);
  if (!staging_data) {
    metal::release_buffer(staging);
    throw std::runtime_error(
        "context_blit_download staging buffer is not CPU-visible");
  }
  std::memcpy(out, staging_data, copy_bytes);
  metal::release_buffer(staging);
}

// Helper: try to acquire a staging buffer from metal_heap::Staging
// (size-class free-list reuse); if the pool is unavailable, disabled, or
// exhausted, fall back to a fresh metal::create_buffer. Either way the
// caller gets a usable Shared MTLBuffer; the difference is whether the
// release path returns to the pool or releases outright.
//
// `heap_owned` carries the pool-backed lifetime when applicable; raw is
// the void* the caller can use uniformly. Cleanup checks `heap_owned`'s
// validity to dispatch.
struct StagingHandle {
  metal_heap::OwnedBuffer heap_owned;  // valid → dtor returns to pool
  void *raw = nullptr;                 // fallback alloc if heap_owned empty
  void *cpu = nullptr;

  StagingHandle() = default;
  StagingHandle(const StagingHandle &) = delete;
  StagingHandle &operator=(const StagingHandle &) = delete;

  // Explicit move semantics that mirror OwnedBuffer's: source is zeroed
  // out so a stale .raw / .cpu can never be mistaken for live state.
  StagingHandle(StagingHandle &&o) noexcept
      : heap_owned(std::move(o.heap_owned)), raw(o.raw), cpu(o.cpu) {
    o.raw = nullptr;
    o.cpu = nullptr;
  }
  StagingHandle &operator=(StagingHandle &&o) noexcept {
    if (this != &o) {
      heap_owned = std::move(o.heap_owned);
      raw = o.raw;
      cpu = o.cpu;
      o.raw = nullptr;
      o.cpu = nullptr;
    }
    return *this;
  }
};

static StagingHandle acquire_staging(MetalContext &ctx, std::size_t bytes) {
  StagingHandle h;
  if (auto *heap = MetalContextAccessor::heap(ctx)) {
    metal_heap::AllocDesc d;
    d.bytes = bytes;
    d.alignment = 256;
    d.lifetime = metal_heap::Lifetime::Staging;
    d.storage = metal_heap::Storage::Shared;
    d.hazard_tracking = metal_heap::HazardTracking::Untracked;
    d.debug_name = "tmnn.blit.staging";
    auto r = heap->allocate(d);
    if (r.has_value()) {
      h.heap_owned = std::move(*r);
      h.raw = h.heap_owned.mtl_buffer();
      h.cpu = h.heap_owned.cpu_data();
      return h;
    }
    // Fall through on StagingExhausted / disabled — direct alloc.
  }
  // Reached only when heap is null OR Heap::Staging returned an error.
  // Bump the observability counter so a non-zero value flags a need to
  // raise heap_config.staging_capacity_bytes.
  metal::alloc_stats().staging_fallback_count.fetch_add(
      1, std::memory_order_relaxed);
  h.raw = metal::create_buffer(context_raw_device(ctx), bytes,
                               /*shared=*/true);
  if (h.raw) h.cpu = metal::buffer_contents(h.raw);
  return h;
}

static void release_staging(StagingHandle &h) {
  if (h.heap_owned.valid()) {
    h.heap_owned = {};  // returns the bucket to the pool's free list
  } else if (h.raw) {
    metal::release_buffer(h.raw);
  }
  h.raw = nullptr;
  h.cpu = nullptr;
}

// Per-request staging — first via metal_heap::Staging (size-class free
// list, ~zero churn after warm-up), falling back to metal::create_buffer
// when the pool is disabled, exhausted, or otherwise unavailable. The
// commit_and_wait collapse stays the dominant win regardless of which
// staging source backed each request.
//
// Implementation: filter into a `Pending` vector that pairs each accepted
// request with its staging — eliminates the indices-in-lockstep fragility
// of running two predicate-filtered loops.
void context_blit_upload_views(MetalContext &ctx,
                               std::span<const BlitUploadRequest> reqs) {
  if (!ctx.is_gpu_available() || reqs.empty()) return;

  struct Pending {
    const BlitUploadRequest *req;
    StagingHandle staging;
    std::size_t copy_bytes;
  };
  std::vector<Pending> pending;
  pending.reserve(reqs.size());
  const auto release_all = [&]() {
    for (auto &p : pending) release_staging(p.staging);
  };

  for (const auto &r : reqs) {
    if (r.bytes == 0 || !r.dst.gpu_buffer || !r.src_data) continue;
    const std::size_t copy_bytes = std::min(r.bytes, r.dst.bytes);
    StagingHandle staging = acquire_staging(ctx, copy_bytes);
    if (!staging.raw) {
      release_all();
      throw std::runtime_error(
          "context_blit_upload_views: staging allocation failed");
    }
    if (!staging.cpu) {
      release_staging(staging);
      release_all();
      throw std::runtime_error(
          "context_blit_upload_views: staging is not CPU-visible");
    }
    std::memcpy(staging.cpu, r.src_data, copy_bytes);
    pending.push_back({&r, std::move(staging), copy_bytes});
  }
  if (pending.empty()) return;

  auto *cmd = metal::create_command_buffer(context_raw_queue(ctx));
  for (const auto &p : pending) {
    metal::encode_blit_copy(cmd, p.staging.raw, 0,
                            p.req->dst.gpu_buffer, p.req->dst.offset,
                            p.copy_bytes);
  }
  metal::commit_and_wait(cmd);
  metal::release_command_buffer(cmd);
  release_all();
}

void context_blit_download_views(MetalContext &ctx,
                                 std::span<const BlitDownloadRequest> reqs) {
  if (!ctx.is_gpu_available() || reqs.empty()) return;

  struct Pending {
    const BlitDownloadRequest *req;
    StagingHandle staging;
    std::size_t copy_bytes;
  };
  std::vector<Pending> pending;
  pending.reserve(reqs.size());
  const auto release_all = [&]() {
    for (auto &p : pending) release_staging(p.staging);
  };

  for (const auto &r : reqs) {
    if (r.bytes == 0 || !r.src.gpu_buffer || !r.dst_data) continue;
    const std::size_t copy_bytes = std::min(r.bytes, r.src.bytes);
    StagingHandle staging = acquire_staging(ctx, copy_bytes);
    if (!staging.raw) {
      release_all();
      throw std::runtime_error(
          "context_blit_download_views: staging allocation failed");
    }
    pending.push_back({&r, std::move(staging), copy_bytes});
  }
  if (pending.empty()) return;

  auto *cmd = metal::create_command_buffer(context_raw_queue(ctx));
  for (const auto &p : pending) {
    metal::encode_blit_copy(cmd, p.req->src.gpu_buffer, p.req->src.offset,
                            p.staging.raw, 0, p.copy_bytes);
  }
  metal::commit_and_wait(cmd);
  metal::release_command_buffer(cmd);

  for (auto &p : pending) {
    if (!p.staging.cpu) p.staging.cpu = metal::buffer_contents(p.staging.raw);
    if (!p.staging.cpu) {
      release_all();
      throw std::runtime_error(
          "context_blit_download_views: staging is not CPU-visible");
    }
    std::memcpy(p.req->dst_data, p.staging.cpu, p.copy_bytes);
  }
  release_all();
}

// ---------------------------------------------------------------------------
// Phase 5: GPU weight init via Philox-4x32-10
// ---------------------------------------------------------------------------

namespace {

// Counter-based RNG. Each thread emits 4 floats from its own (counter,
// key) seed — no shared state, no synchronization. Output mapped affinely
// to [low, high]. Reproducible across runs given the same seed.
//
// Reference: Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3"
// (SC'11). Constants match cuRAND / tinycudann's Philox-4x32-10.
constexpr const char *kInitUniformPhiloxMSL = R"MSL(
#include <metal_stdlib>
using namespace metal;

constant uint32_t PHILOX_M0 = 0xD2511F53u;
constant uint32_t PHILOX_M1 = 0xCD9E8D57u;
constant uint32_t PHILOX_W0 = 0x9E3779B9u;
constant uint32_t PHILOX_W1 = 0xBB67AE85u;

inline uint4 philox_round(uint4 ctr, uint2 key) {
  uint64_t p0 = (uint64_t)PHILOX_M0 * (uint64_t)ctr.x;
  uint64_t p1 = (uint64_t)PHILOX_M1 * (uint64_t)ctr.z;
  uint32_t hi0 = uint32_t(p0 >> 32), lo0 = uint32_t(p0);
  uint32_t hi1 = uint32_t(p1 >> 32), lo1 = uint32_t(p1);
  return uint4(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0);
}

inline uint4 philox_4x32_10(uint4 ctr, uint2 key) {
  for (int i = 0; i < 10; ++i) {
    ctr = philox_round(ctr, key);
    key = uint2(key.x + PHILOX_W0, key.y + PHILOX_W1);
  }
  return ctr;
}

struct InitParams {
  uint32_t key0;
  uint32_t key1;
  uint32_t counter_base;
  uint32_t element_count;
  float low;
  float high;
};

kernel void init_uniform_philox(
    device float *out [[buffer(0)]],
    constant InitParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
  uint4 ctr = uint4(p.counter_base + tid, 0u, 0u, 0u);
  uint2 key = uint2(p.key0, p.key1);
  uint4 r = philox_4x32_10(ctr, key);

  // u32 -> [0, 1) via 2^-32 scale, then affine to [low, high].
  const float kScale = 1.0f / 4294967296.0f;
  float4 u = float4(r) * kScale;
  float4 v = float4(p.low) + u * float4(p.high - p.low);

  uint base = tid * 4u;
  if (base + 4u <= p.element_count) {
    out[base + 0u] = v.x;
    out[base + 1u] = v.y;
    out[base + 2u] = v.z;
    out[base + 3u] = v.w;
  } else {
    if (base + 0u < p.element_count) out[base + 0u] = v.x;
    if (base + 1u < p.element_count) out[base + 1u] = v.y;
    if (base + 2u < p.element_count) out[base + 2u] = v.z;
  }
}
)MSL";

// Distinguishable PipelineKey for the init kernel. Hash chosen so it
// won't collide with the KernelCompiler's training-kernel hashes.
constexpr std::uint64_t kInitUniformPipelineHash = 0xC0FFEE5070A0501ull;

}  // namespace

void context_dispatch_init_uniform(MetalContext &ctx,
                                   BufferView dst,
                                   std::size_t element_count,
                                   float low, float high,
                                   std::uint64_t seed,
                                   std::uint32_t counter_base) {
  if (!ctx.is_gpu_available() || !dst.gpu_buffer || element_count == 0) return;

  auto &reg = MetalContextAccessor::registry(ctx);
  PipelineKey pkey{kInitUniformPipelineHash, "init_uniform_philox",
                   /*precise_math=*/false, /*binding_count=*/2,
                   /*threadgroup_memory_bytes=*/0};
  auto pso = reg.register_pipeline(pkey, kInitUniformPhiloxMSL,
                                   "init_uniform_philox");
  void *raw_pso = reg.raw_pipeline(pso);
  if (!raw_pso) {
    throw std::runtime_error(
        "context_dispatch_init_uniform: failed to compile init kernel");
  }

  // Per-call params are tiny (24 bytes) — push them through a small
  // Shared MTLBuffer instead of setBytes so we keep one dispatch path.
  struct InitParams {
    std::uint32_t key0;
    std::uint32_t key1;
    std::uint32_t counter_base;
    std::uint32_t element_count;
    float low;
    float high;
  };
  InitParams params{
      static_cast<std::uint32_t>(seed & 0xFFFFFFFFu),
      static_cast<std::uint32_t>((seed >> 32) & 0xFFFFFFFFu),
      counter_base,
      static_cast<std::uint32_t>(element_count),
      low, high,
  };
  void *param_buf = metal::create_buffer(context_raw_device(ctx),
                                         sizeof(params), /*shared=*/true);
  if (!param_buf) {
    throw std::runtime_error(
        "context_dispatch_init_uniform: param staging alloc failed");
  }
  std::memcpy(metal::buffer_contents(param_buf), &params, sizeof(params));

  // Each thread emits 4 floats; round up the grid to cover the tail.
  const std::uint32_t threads = static_cast<std::uint32_t>(
      (element_count + 3u) / 4u);
  // 256 threads/threadgroup is a safe SIMD-aligned default on Apple GPUs;
  // dispatch_threads handles non-multiple-of-tg sizing internally via
  // metal::encode_dispatch's clamp.
  const std::uint32_t tg = 256u;

  auto *cmd = metal::create_command_buffer(context_raw_queue(ctx));
  metal::DispatchDesc::BufferBind binds[2] = {
      {dst.gpu_buffer, static_cast<std::uint32_t>(dst.offset), 0u},
      {param_buf, 0u, 1u},
  };
  metal::DispatchDesc dd{
      cmd, raw_pso, binds, /*binding_count=*/2u,
      threads, 1u, 1u,
      tg,      1u, 1u,
      0u,
  };
  metal::encode_dispatch(dd);
  metal::commit_and_wait(cmd);
  metal::release_command_buffer(cmd);
  metal::release_buffer(param_buf);
}

} // namespace detail
} // namespace tmnn

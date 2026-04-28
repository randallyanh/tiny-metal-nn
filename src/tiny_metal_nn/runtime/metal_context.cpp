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
    hcfg.staging_max_bytes    = 0;

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
  // Skip zero-cost: we want exactly one cmdbuf to cover the whole batch,
  // but if no view actually has GPU bytes to fill, don't pay for an empty
  // commit_and_wait round-trip.
  bool has_work = false;
  for (const auto &v : views) {
    if (v.gpu_buffer && v.bytes > 0) { has_work = true; break; }
  }
  if (!has_work) return;

  auto *cmd = metal::create_command_buffer(context_raw_queue(ctx));
  for (const auto &v : views) {
    if (!v.gpu_buffer || v.bytes == 0) continue;
    metal::encode_blit_fill(cmd, v.gpu_buffer, v.offset, v.bytes, value);
  }
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

} // namespace detail

} // namespace tmnn

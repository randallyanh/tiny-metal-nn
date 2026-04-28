#pragma once

/**
 * @file metal_context.h
 * @brief MetalContext — tmnn NN runtime root.
 *
 * MetalContext owns the GPU device, command queue pool, pipeline registry,
 * and runtime telemetry. It is the single source of truth for all Metal
 * runtime state within a tmnn execution graph.
 *
 * Multiple trainers / runtimes may share a MetalContext via shared_ptr;
 * each MetalContext represents an isolated execution graph.
 *
 * ObjC / Metal types are hidden behind PImpl — this header is pure C++.
 */

#include "tiny-metal-nn/device_capabilities.h"
#include "tiny-metal-nn/autotune_manifest.h"
#include "tiny-metal-nn/runtime_policy.h"
#include "tiny-metal-nn/detail/runtime_stats.h"

#include <memory>

namespace tmnn {

// Forward declaration for internal accessor (see metal_context_internal.h).
namespace detail {
class MetalContextAccessor;
} // namespace detail

/// Phase 3.2: heap-budget configuration for the metal_heap::Heap that
/// MetalContext owns. By default the Heap stays available for opt-in use
/// cases (adopt_external, future TransientRing consumers) at minimal
/// wired-memory cost; set route_buffer_arena_through_heap = true to make
/// BufferArena allocate Persistent buffers from the Heap.
///
/// IMPORTANT (Apple Silicon trade-off): MTLHeap commits its full backing
/// region the moment the GPU first binds a sub-buffer, then keeps that
/// region resident for the heap's lifetime. So with
/// route_buffer_arena_through_heap = true, wired_memory delta_construct
/// rises from ~24 MiB to ~heap_capacity (≈ 1 GiB on the default config).
/// In return, per-buffer allocation latency drops from ~1180 ns to
/// ~340 ns. For tmnn's <100-buffer-per-trainer workload that's a 0.2 %
/// cold-startup win for a 44× wired-mem cost, so it's off by default —
/// turn it on if your workload has high-frequency dynamic buffer
/// creation (e.g. interactive editing, autotune sweeps, MPS interop).
struct MetalContextHeapConfig {
  // Default OFF: BufferArena uses metal::create_buffer per slot, which
  // matches the pre-3.2 wired-mem profile. The Heap is still created at
  // a small size for adopt_external + future TransientRing.
  bool route_buffer_arena_through_heap = false;

  // Explicit capacities (non-zero = use verbatim, in bytes). Used in
  // both modes; auto-derivation is skipped when these are non-zero.
  size_t persistent_shared_capacity_bytes = 0;
  size_t persistent_private_capacity_bytes = 0;

  // Auto-derivation parameters (consulted only when the corresponding
  // explicit capacity above is 0):
  //
  //   When route_buffer_arena_through_heap = true:
  //     total_budget = clamp(working_set_bytes * working_set_fraction,
  //                          min_total_bytes, max_total_bytes)
  //     shared_capacity  = total_budget * shared_share
  //     private_capacity = total_budget * (1 - shared_share)
  //
  //   When route_buffer_arena_through_heap = false:
  //     a tiny dormant heap (1 MiB / 1 MiB) — the working_set / share /
  //     bound parameters below are ignored.
  //
  // Defaults are calibrated against measured tmnn workloads on Apple
  // Silicon. Two-trainer + Huber-loss tests peak at ~470 MiB Private;
  // shared-gradient-buffer tests peak at ~268 MiB Shared. So the default
  // split is 30 / 70 (Private remains the larger binding constraint),
  // and the 1 GiB total floor delivers ~307 MiB Shared / ~716 MiB Private
  // which covers every test in the suite with margin.
  float working_set_fraction = 0.025f;
  float shared_share         = 0.3f;
  size_t min_total_bytes     = 1024ull * 1024 * 1024;         // 1 GiB
  size_t max_total_bytes     = 4ull * 1024 * 1024 * 1024;     // 4 GiB
};

/// Configuration for MetalContext creation.
struct MetalContextDesc {
  bool enable_binary_archive = true;
  bool enable_disk_cache = true;
  bool enable_precise_math_fallback = true;
  uint32_t max_inflight_batches = 2;
  RuntimePolicyOverrides policy_overrides;
  MetalContextHeapConfig heap_config;
};

/// Shared runtime root for tmnn NN execution.
class MetalContext {
public:
  /// Create a MetalContext with the given configuration.
  static std::shared_ptr<MetalContext> create(
      const MetalContextDesc &desc = {});

  ~MetalContext();

  // Non-copyable, non-moveable (shared via shared_ptr).
  MetalContext(const MetalContext &) = delete;
  MetalContext &operator=(const MetalContext &) = delete;

  /// Device capability snapshot (immutable after creation).
  [[nodiscard]] const DeviceCapabilities &capabilities() const;

  /// Snapshot of cumulative runtime stats.
  [[nodiscard]] RuntimeStats snapshot_stats() const;

  /// Hydrate planner/autotune decision state from a persisted manifest.
  void prewarm_autotune_manifest(const AutotuneManifest &manifest);

  /// Snapshot the current in-memory autotune/prewarm decision state.
  [[nodiscard]] AutotuneManifest snapshot_autotune_manifest() const;

  /// Clear all runtime cache tiers currently wired into this MetalContext.
  void clear_runtime_caches();

  /// Whether a GPU device was successfully acquired.
  [[nodiscard]] bool is_gpu_available() const;

  /// The descriptor used to create this context.
  [[nodiscard]] const MetalContextDesc &desc() const;

  /// Resolved low-level runtime policy frozen at context creation time.
  [[nodiscard]] const RuntimePolicy &policy() const;

private:
  explicit MetalContext(const MetalContextDesc &desc);

  class Impl;
  std::unique_ptr<Impl> impl_;

  // Internal accessor friends (see metal_context_internal.h).
  friend class detail::MetalContextAccessor;
};

} // namespace tmnn

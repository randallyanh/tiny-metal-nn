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

/// Configuration for MetalContext creation.
struct MetalContextDesc {
  bool enable_binary_archive = true;
  bool enable_disk_cache = true;
  bool enable_precise_math_fallback = true;
  uint32_t max_inflight_batches = 2;
  RuntimePolicyOverrides policy_overrides;
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

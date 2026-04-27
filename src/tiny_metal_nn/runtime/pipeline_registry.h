#pragma once

/**
 * @file pipeline_registry.h
 * @brief PipelineRegistry — unified NN compile cache (internal).
 *
 * Provides a KernelSpec-keyed cache for compiled pipeline state.
 * Thread-safe (mutex-protected). Stats (hits/misses/compiles) feed into
 * RuntimeStats via MetalContext.
 *
 * When a device is set via set_device(), register_pipeline() with MSL
 * source compiles real MTLComputePipelineState objects. Without a device,
 * the registry tracks metadata only.
 */

#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>

namespace tmnn {

/// Key for pipeline cache lookup.
struct PipelineKey {
  uint64_t kernel_spec_hash = 0;  ///< FNV-1a hash of KernelSpec fields.
  std::string entry_point;
  bool precise_math = false;
  uint32_t emitter_version = 0;
  uint32_t device_family = 0;

  bool operator==(const PipelineKey &o) const {
    return kernel_spec_hash == o.kernel_spec_hash &&
           entry_point == o.entry_point &&
           precise_math == o.precise_math &&
           emitter_version == o.emitter_version &&
           device_family == o.device_family;
  }
};

/// Opaque handle to a cached pipeline (index + generation).
struct PipelineHandle {
  uint32_t index = 0;
  uint32_t generation = 0;
  explicit operator bool() const { return generation != 0; }
};

/// Pipeline cache statistics.
struct PipelineCacheStats {
  uint64_t hits = 0;
  uint64_t misses = 0;
  uint64_t compile_count = 0;
  uint32_t entries = 0;
};

} // namespace tmnn

// Hash support for PipelineKey.
template <> struct std::hash<tmnn::PipelineKey> {
  size_t operator()(const tmnn::PipelineKey &k) const noexcept {
    size_t h = std::hash<uint64_t>{}(k.kernel_spec_hash);
    h ^= std::hash<std::string>{}(k.entry_point) + 0x9e3779b97f4a7c15 +
         (h << 6) + (h >> 2);
    h ^= std::hash<bool>{}(k.precise_math) + 0x9e3779b97f4a7c15 +
         (h << 6) + (h >> 2);
    h ^= std::hash<uint32_t>{}(k.emitter_version) + 0x9e3779b97f4a7c15 +
         (h << 6) + (h >> 2);
    h ^= std::hash<uint32_t>{}(k.device_family) + 0x9e3779b97f4a7c15 +
         (h << 6) + (h >> 2);
    return h;
  }
};

namespace tmnn {

class PipelineRegistry {
public:
  PipelineRegistry() = default;

  /// Set the Metal device for real pipeline compilation.
  /// Called by MetalContext after device probing.
  void set_device(void *device);

  /// Look up a pipeline by key. Returns a valid handle on cache hit,
  /// invalid handle on miss. Thread-safe.
  [[nodiscard]] PipelineHandle lookup(const PipelineKey &key);

  /// Register a pipeline for the given key (metadata-only).
  /// Returns the handle for subsequent lookups. Thread-safe.
  PipelineHandle register_pipeline(const PipelineKey &key);

  /// Register a pipeline compiled from MSL source.
  /// Compiles a real MTLComputePipelineState if device is set.
  /// Returns the handle for subsequent lookups. Thread-safe.
  PipelineHandle register_pipeline(const PipelineKey &key,
                                    const char *msl_source,
                                    const char *function_name);

  /// Get the raw id<MTLComputePipelineState> for a handle. nullptr if none.
  [[nodiscard]] void *raw_pipeline(PipelineHandle handle) const;

  /// True when the cached entry represents a sticky compile failure.
  [[nodiscard]] bool has_failure(PipelineHandle handle) const;

  /// Cached compile diagnostic for a failed entry, or empty string otherwise.
  [[nodiscard]] std::string failure_diagnostic(PipelineHandle handle) const;

  /// Max total threads per threadgroup for a pipeline.
  [[nodiscard]] uint32_t max_threads_per_tg(PipelineHandle handle) const;

  /// Cache statistics snapshot. Thread-safe.
  [[nodiscard]] PipelineCacheStats stats() const;

  /// Clear all cached entries. Thread-safe.
  void clear();

private:
  struct Entry {
    PipelineHandle handle;
    void *pipeline = nullptr; ///< id<MTLComputePipelineState>
    std::string error;
  };

  mutable std::mutex mu_;
  std::unordered_map<PipelineKey, Entry> cache_;
  uint32_t next_index_ = 1; // 0 reserved for invalid
  uint32_t generation_ = 1;
  uint64_t hits_ = 0;
  uint64_t misses_ = 0;
  uint64_t compiles_ = 0;
  void *device_ = nullptr; ///< id<MTLDevice>
};

} // namespace tmnn

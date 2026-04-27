/**
 * @file pipeline_registry.cpp
 * @brief PipelineRegistry implementation — real MTLComputePipelineState
 *        compilation when device is set, metadata-only otherwise.
 */

#include "tiny_metal_nn/runtime/pipeline_registry.h"
#include "tiny_metal_nn/runtime/metal_device.h"

namespace tmnn {

void PipelineRegistry::set_device(void *device) { device_ = device; }

PipelineHandle PipelineRegistry::lookup(const PipelineKey &key) {
  std::lock_guard lock(mu_);
  auto it = cache_.find(key);
  if (it != cache_.end()) {
    ++hits_;
    return it->second.handle;
  }
  ++misses_;
  return {}; // invalid handle
}

PipelineHandle PipelineRegistry::register_pipeline(const PipelineKey &key) {
  std::lock_guard lock(mu_);
  // If already registered, return existing handle.
  auto it = cache_.find(key);
  if (it != cache_.end())
    return it->second.handle;

  PipelineHandle h{next_index_++, generation_};
  cache_[key] = {h, nullptr, {}};
  ++compiles_;
  return h;
}

PipelineHandle PipelineRegistry::register_pipeline(const PipelineKey &key,
                                                     const char *msl_source,
                                                     const char *function_name) {
  std::lock_guard lock(mu_);
  auto it = cache_.find(key);
  if (it != cache_.end()) {
    ++hits_;
    return it->second.handle;
  }

  void *pso = nullptr;
  std::string diagnostic;
  if (device_ && msl_source && function_name) {
    pso = metal::compile_pipeline(device_, msl_source, function_name,
                                  key.precise_math, diagnostic);
    if (!pso && diagnostic.empty())
      diagnostic = "unknown pipeline compilation failure";
  }

  PipelineHandle h{next_index_++, generation_};
  cache_[key] = {h, pso, std::move(diagnostic)};
  ++compiles_;
  ++misses_;
  return h;
}

void *PipelineRegistry::raw_pipeline(PipelineHandle handle) const {
  std::lock_guard lock(mu_);
  for (const auto &[k, e] : cache_) {
    if (e.handle.index == handle.index &&
        e.handle.generation == handle.generation)
      return e.pipeline;
  }
  return nullptr;
}

bool PipelineRegistry::has_failure(PipelineHandle handle) const {
  std::lock_guard lock(mu_);
  for (const auto &[k, e] : cache_) {
    if (e.handle.index == handle.index &&
        e.handle.generation == handle.generation)
      return e.pipeline == nullptr && !e.error.empty();
  }
  return false;
}

std::string PipelineRegistry::failure_diagnostic(PipelineHandle handle) const {
  std::lock_guard lock(mu_);
  for (const auto &[k, e] : cache_) {
    if (e.handle.index == handle.index &&
        e.handle.generation == handle.generation)
      return e.error;
  }
  return {};
}

uint32_t PipelineRegistry::max_threads_per_tg(PipelineHandle handle) const {
  void *pso = raw_pipeline(handle);
  return metal::pipeline_max_threads(pso);
}

PipelineCacheStats PipelineRegistry::stats() const {
  std::lock_guard lock(mu_);
  PipelineCacheStats s;
  s.hits = hits_;
  s.misses = misses_;
  s.compile_count = compiles_;
  s.entries = static_cast<uint32_t>(cache_.size());
  return s;
}

void PipelineRegistry::clear() {
  std::lock_guard lock(mu_);
  // Release any real pipeline state objects.
  for (auto &[k, e] : cache_) {
    if (e.pipeline) {
      metal::release_pipeline(e.pipeline);
      e.pipeline = nullptr;
    }
  }
  cache_.clear();
  hits_ = 0;
  misses_ = 0;
  // Don't reset compiles_ — it's a lifetime counter.
  // Bump generation so existing handles become stale.
  ++generation_;
}

} // namespace tmnn

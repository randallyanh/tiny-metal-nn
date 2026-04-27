#pragma once

/**
 * @file runtime_authority.h
 * @brief Public C3 surfaces for runtime authority, parameter layout, storage
 *        policy, batch planning, and typed optimizer-state transport.
 */

#include "tiny-metal-nn/detail/checkpoint_contract.h"
#include "tiny-metal-nn/extension/schema.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace tmnn {

class MetalContext;

enum class StorageVisibility {
  CpuVisible,
  GpuOnly,
};

enum class ResourceLifetime {
  Persistent,
  Transient,
  Staging,
};

struct ParameterLayout {
  uint32_t hash_grid_float_count = 0;
  uint32_t mlp_weight_float_count = 0;
  TrainParamsLayout train_params_layout{};
  uint32_t adam_params_float_count = 0;
  uint32_t config_header_float_count = 8;
  uint32_t config_tail_float_count = 0;
  uint32_t target_dims = 1;
  uint32_t reduction_terms = 1;
  bool fused_adam = false;
};

struct TrainerBatchPlan {
  uint32_t max_batch_size = 0;
  uint32_t lane_count = 0;
  uint32_t input_dims = 0;
  uint32_t target_dims = 0;
  uint32_t reduction_terms = 0;
  size_t positions_bytes_per_lane = 0;
  size_t targets_bytes_per_lane = 0;
  size_t reduction_bytes_per_lane = 0;
  uint32_t threadgroup_size = 0;
  uint32_t points_per_threadgroup = 0;
};

struct RuntimeStoragePolicy {
  StorageVisibility parameter_weights = StorageVisibility::CpuVisible;
  StorageVisibility gradients = StorageVisibility::CpuVisible;
  StorageVisibility optimizer_state = StorageVisibility::CpuVisible;
  StorageVisibility control_buffers = StorageVisibility::CpuVisible;
  ResourceLifetime step_inputs = ResourceLifetime::Transient;
  ResourceLifetime step_targets = ResourceLifetime::Transient;
  ResourceLifetime step_reduction = ResourceLifetime::Transient;
  bool requires_staging_uploads = false;
};

enum class RuntimeBufferRole {
  ConfigWeights,
  HashWeights,
  MlpWeights,
  GradHash,
  GradMlp,
  TrainParams,
  AdamParams,
  FusedWeights,
  FusedFirstMoment,
  FusedSecondMoment,
};

struct RuntimeBufferView {
  size_t offset = 0;
  size_t bytes = 0;
  void *cpu_data = nullptr;
  void *gpu_buffer = nullptr;
  StorageVisibility visibility = StorageVisibility::CpuVisible;
  ResourceLifetime lifetime = ResourceLifetime::Persistent;

  [[nodiscard]] bool valid() const {
    return gpu_buffer != nullptr && bytes != 0;
  }
};

/// Canonical tmnn optimizer-only checkpoint transport.
///
/// Ownership contract:
/// - includes only logical optimizer step + optimizer payload
/// - does not include model weights, network descriptors, or trainer config
/// - callers checkpoint weights/config separately and pair them on restore
///
/// Compatibility contract:
/// - `version` must exactly match `kOptimizerStateBlobVersion`
/// - `payload` layout is version-defined and opaque at the public API level
struct OptimizerStateBlob {
  uint32_t version = kOptimizerStateBlobVersion;
  uint32_t step = 0;
  std::vector<uint8_t> payload;
};

class RuntimeAuthority {
public:
  virtual ~RuntimeAuthority() = default;

  [[nodiscard]] virtual const std::shared_ptr<MetalContext> &context() const = 0;
  [[nodiscard]] virtual ParameterLayout parameter_layout() const = 0;
  [[nodiscard]] virtual RuntimeStoragePolicy storage_policy() const = 0;
  [[nodiscard]] virtual RuntimeBufferView
  buffer(RuntimeBufferRole role) const = 0;
};

} // namespace tmnn

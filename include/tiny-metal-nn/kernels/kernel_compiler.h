#pragma once

/**
 * @file kernel_compiler.h
 * @brief Kernel compiler contract around MLPKernelEmitter.
 */

#include "tiny-metal-nn/kernels/kernel_spec.h"
#include "tiny-metal-nn/extension/kernel_compile_spec.h"
#include "tiny-metal-nn/extension/schema.h"

#include <cstdint>
#include <functional>
#include <string>

namespace tmnn {

enum class KernelRole : uint32_t {
  Eval = 0,
  Gradient = 1,
  EvalGradient = 2,
  TrainForwardBackward = 3,
  /// Backward pass with external output gradient (no internal loss).
  BackwardFromExternalGrad = 4,
  /// Forward-only for split training path. Writes output to buffer(8).
  ForwardForTraining = 5,
};

enum class SpecializationDisableReason : uint32_t {
  None = 0,
  NonStandardEncoding = 1,
  UnsupportedSpatialDims = 2,
  MultiOutputUnsupported = 3,
  HiddenDimNotSIMDAligned = 4,
  InputDimNotSIMDAligned = 5,
  ThreadgroupMemoryExceeded = 6,
  PreferenceDisabled = 7,
  SchemaIncompatible = 8,
  RoleUnsupported = 9,
};

struct SpecializationDecision {
  KernelRole role = KernelRole::TrainForwardBackward;
  bool requested_simd = false;
  bool realized_simd = false;
  bool requested_fp16 = false;
  bool realized_fp16 = false;
  bool requested_tg_weight_cache = false;
  bool realized_tg_weight_cache = false;
  SpecializationDisableReason simd_reason = SpecializationDisableReason::None;
  SpecializationDisableReason fp16_reason = SpecializationDisableReason::None;
  SpecializationDisableReason tg_cache_reason =
      SpecializationDisableReason::None;
};

struct KernelKey {
  KernelRole role = KernelRole::TrainForwardBackward;
  uint64_t resolved_spec_hash = 0;
  uint64_t schema_hash = 0;
  uint64_t compile_contract_hash = 0;
  uint32_t emitter_version = 0;

  bool operator==(const KernelKey &other) const = default;

  [[nodiscard]] uint64_t hash() const;
};

struct KernelCompileRequest {
  KernelRole role = KernelRole::TrainForwardBackward;
  KernelSpec spec;
  tmnn::extension::ExtensionSchema schema;
  tmnn::extension::KernelCompileSpec compile_spec;
};

struct KernelCompileResult {
  KernelKey key;
  KernelSpec resolved_spec;
  SpecializationDecision decision;
  std::string source;
  std::string entry_point;
};

class KernelCompiler {
public:
  static constexpr uint32_t kEmitterVersion = 1;

  [[nodiscard]] static KernelCompileResult
  compile(const KernelCompileRequest &request);

  [[nodiscard]] static tmnn::extension::ExtensionSchema
  makeDefaultSchema(const KernelSpec &spec);
};

} // namespace tmnn

namespace std {

template <> struct hash<tmnn::KernelKey> {
  size_t operator()(const tmnn::KernelKey &key) const noexcept {
    return static_cast<size_t>(key.hash());
  }
};

} // namespace std

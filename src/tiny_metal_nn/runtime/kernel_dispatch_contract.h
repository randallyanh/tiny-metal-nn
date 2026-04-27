#pragma once

/**
 * @file kernel_dispatch_contract.h
 * @brief Helpers that keep kernel specialization decisions and dispatch
 *        geometry aligned.
 */

#include "tiny-metal-nn/kernels/kernel_compiler.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace tmnn::detail {

struct KernelDispatchGeometry {
  uint32_t tg_size = 128;
  uint32_t pts_per_tg = 128;
  uint32_t threadgroup_memory_bytes = 0;
};

struct KernelSpecializationState {
  bool requested = false;
  bool realized = false;
  SpecializationDisableReason reason = SpecializationDisableReason::None;
};

struct KernelDispatchContract {
  KernelRole role = KernelRole::TrainForwardBackward;
  std::string entry_point;
  KernelSpecializationState simd;
  KernelSpecializationState fp16;
  KernelSpecializationState tg_weight_cache;
  KernelDispatchGeometry geometry;
  bool available = false;
};

[[nodiscard]] inline const char *kernel_role_name(KernelRole role) {
  switch (role) {
  case KernelRole::Eval:
    return "Eval";
  case KernelRole::Gradient:
    return "Gradient";
  case KernelRole::EvalGradient:
    return "EvalGradient";
  case KernelRole::TrainForwardBackward:
    return "TrainForwardBackward";
  case KernelRole::BackwardFromExternalGrad:
    return "BackwardFromExternalGrad";
  case KernelRole::ForwardForTraining:
    return "ForwardForTraining";
  }
  return "UnknownKernelRole";
}

[[nodiscard]] inline KernelDispatchGeometry
make_dispatch_geometry(const KernelCompileResult &result) {
  KernelDispatchGeometry geometry;
  if (result.decision.realized_simd) {
    geometry.tg_size = 32u;
    geometry.pts_per_tg = 8u;
  }
  // tmnn-generated kernels use statically-declared threadgroup storage in the
  // emitted MSL; runtime should not request extra dynamic threadgroup memory.
  geometry.threadgroup_memory_bytes = 0u;
  return geometry;
}

inline void validate_specialization_state(const char *label,
                                          const char *role_name,
                                          const KernelSpecializationState &state) {
  if (state.realized && !state.requested) {
    throw std::logic_error(std::string("KernelDispatchContract: ") + role_name +
                           " realized " + label +
                           " without it being requested");
  }
  if (state.realized && state.reason != SpecializationDisableReason::None) {
    throw std::logic_error(std::string("KernelDispatchContract: ") + role_name +
                           " realized " + label +
                           " but also recorded a disable reason");
  }
  if (!state.realized && state.requested &&
      state.reason == SpecializationDisableReason::None) {
    throw std::logic_error(std::string("KernelDispatchContract: ") + role_name +
                           " requested " + label +
                           " but did not record why realization was disabled");
  }
}

inline void validate_dispatch_contract(const KernelDispatchContract &contract) {
  if (!contract.available)
    return;

  const char *role_name = kernel_role_name(contract.role);
  if (contract.entry_point.empty()) {
    throw std::logic_error(std::string("KernelDispatchContract: ") + role_name +
                           " is missing an entry point");
  }

  validate_specialization_state("SIMD", role_name, contract.simd);
  validate_specialization_state("FP16", role_name, contract.fp16);
  validate_specialization_state("TG cache", role_name, contract.tg_weight_cache);

  const uint32_t expected_tg = contract.simd.realized ? 32u : 128u;
  const uint32_t expected_pts = contract.simd.realized ? 8u : 128u;
  if (contract.geometry.tg_size != expected_tg ||
      contract.geometry.pts_per_tg != expected_pts) {
    throw std::logic_error(std::string("KernelDispatchContract: ") + role_name +
                           " geometry does not match realized specialization");
  }
  if (contract.geometry.threadgroup_memory_bytes != 0u) {
    throw std::logic_error(std::string("KernelDispatchContract: ") + role_name +
                           " unexpectedly requested dynamic threadgroup memory");
  }
}

[[nodiscard]] inline KernelDispatchContract
make_dispatch_contract(const KernelCompileResult &result) {
  KernelDispatchContract contract;
  contract.role = result.decision.role;
  contract.entry_point = result.entry_point;
  contract.simd = {result.decision.requested_simd, result.decision.realized_simd,
                   result.decision.simd_reason};
  contract.fp16 = {result.decision.requested_fp16, result.decision.realized_fp16,
                   result.decision.fp16_reason};
  contract.tg_weight_cache = {
      result.decision.requested_tg_weight_cache,
      result.decision.realized_tg_weight_cache, result.decision.tg_cache_reason};
  contract.geometry = make_dispatch_geometry(result);
  contract.available = true;
  validate_dispatch_contract(contract);
  return contract;
}

[[nodiscard]] inline uint32_t
threadgroup_count_for_points(uint32_t point_count,
                             const KernelDispatchGeometry &geometry) {
  const uint32_t pts_per_tg = geometry.pts_per_tg == 0u ? 1u : geometry.pts_per_tg;
  const uint32_t groups = (point_count + pts_per_tg - 1u) / pts_per_tg;
  return groups == 0u ? 1u : groups;
}

[[nodiscard]] inline uint32_t
total_threads_for_points(uint32_t point_count,
                         const KernelDispatchGeometry &geometry) {
  const uint32_t tg_size = geometry.tg_size == 0u ? 1u : geometry.tg_size;
  return threadgroup_count_for_points(point_count, geometry) * tg_size;
}

} // namespace tmnn::detail

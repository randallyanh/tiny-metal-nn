#pragma once

/**
 * @file device_capabilities.h
 * @brief GPU device capability snapshot for tmnn runtime.
 */

#include <cstdint>
#include <string>

namespace tmnn {

/// Snapshot of GPU device capabilities, probed once at MetalContext creation.
struct DeviceCapabilities {
  std::string device_name;
  uint32_t gpu_family = 0;
  uint32_t max_threads_per_tg = 0;
  uint32_t max_threadgroup_memory_bytes = 0;
  bool supports_fp16 = false;
  bool supports_simdgroup_matrix = false;
  bool supports_binary_archive = false;
  bool supports_nonuniform_threadgroups = false;
};

} // namespace tmnn

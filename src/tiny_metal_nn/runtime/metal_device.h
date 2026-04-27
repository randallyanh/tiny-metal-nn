#pragma once

/**
 * @file metal_device.h
 * @brief C++ free-function declarations for Metal device operations.
 *
 * All ObjC++ code lives in metal_device.mm; non-Apple builds use
 * metal_device_stub.cpp (all no-ops). Runtime .cpp files stay pure C++
 * and call these free functions with void* handles.
 */

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

namespace tmnn::metal {

/// Process-wide counters for newBufferWithLength / blit copy / contents()
/// access. Counters are incremented unconditionally (atomics; hot-path cost
/// negligible). Reset and read by the benchmark binary's --allocation-trace
/// option to baseline hot-path memory traffic.
struct AllocStats {
  std::atomic<uint64_t> create_buffer_calls{0};
  std::atomic<uint64_t> create_buffer_bytes{0};
  std::atomic<uint64_t> blit_copy_calls{0};
  std::atomic<uint64_t> blit_copy_bytes{0};
  std::atomic<uint64_t> buffer_contents_calls{0};
};

/// Process-global stats handle.
AllocStats &alloc_stats();
/// Reset all counters to zero.
void reset_alloc_stats();

/// Information returned from device probing.
struct DeviceInfo {
  void *device = nullptr;  ///< id<MTLDevice>
  void *queue = nullptr;   ///< id<MTLCommandQueue>
  std::string name;
  uint32_t gpu_family = 0;
  uint32_t max_threads_per_tg = 0;
  uint32_t max_tg_memory_bytes = 0;
  bool supports_fp16 = false;
  bool supports_simdgroup_matrix = false;
  bool supports_binary_archive = false;
  bool supports_nonuniform_tgs = false;
};

/// Probe system default Metal device. Returns null device on failure / non-Apple.
DeviceInfo probe_default_device();

/// Release device + queue (calls [obj release] under MRC).
void release_device(DeviceInfo &info);

// ---------------------------------------------------------------------------
// Buffer operations
// ---------------------------------------------------------------------------

/// Create an MTLBuffer. Returns void* (id<MTLBuffer>).
/// shared=true -> storageModeShared, shared=false -> storageModePrivate.
void *create_buffer(void *device, size_t bytes, bool shared);

/// Release an MTLBuffer.
void release_buffer(void *buffer);

/// Get contents() pointer from a Shared MTLBuffer. nullptr for Private.
void *buffer_contents(void *buffer);

// ---------------------------------------------------------------------------
// Command buffer operations
// ---------------------------------------------------------------------------

/// Create an MTLCommandBuffer from a queue. Returns void*.
void *create_command_buffer(void *queue);

/// Commit + waitUntilCompleted on a command buffer.
void commit_and_wait(void *cmd_buf);

/// Commit async (no wait). Returns immediately.
void commit_async(void *cmd_buf);

/// Wait until a command buffer completes.
void wait_until_completed(void *cmd_buf);

/// Return the localized command-buffer error string, or empty string if none.
std::string command_buffer_error(void *cmd_buf);

/// Release a command buffer.
void release_command_buffer(void *cmd_buf);

/// GPU execution timing from a completed command buffer.
/// Returns 0.0 if the command buffer has not completed or timing unavailable.
double gpu_execution_time_us(void *cmd_buf);

// ---------------------------------------------------------------------------
// Pipeline operations
// ---------------------------------------------------------------------------

/// Compile an MTLComputePipelineState from MSL source.
/// Returns void* (id<MTLComputePipelineState>), nullptr on failure.
/// error_out receives error message on failure.
void *compile_pipeline(void *device, const char *msl_source,
                       const char *function_name, bool precise_math,
                       std::string &error_out);

/// Release a pipeline state object.
void release_pipeline(void *pipeline);

/// Get maxTotalThreadsPerThreadgroup from a pipeline.
uint32_t pipeline_max_threads(void *pipeline);

// ---------------------------------------------------------------------------
// Dispatch operations
// ---------------------------------------------------------------------------

/// Descriptor for a compute dispatch.
struct DispatchDesc {
  void *cmd_buf;
  void *pipeline;
  struct BufferBind {
    void *buffer;
    uint32_t offset;
    uint32_t index;
  };
  const BufferBind *bindings;
  uint32_t binding_count;
  uint32_t grid_x, grid_y, grid_z;
  uint32_t tg_x, tg_y, tg_z;
  uint32_t threadgroup_memory_bytes = 0; ///< Threadgroup memory (index 0).
};

/// Encode a compute dispatch on a command buffer.
void encode_dispatch(const DispatchDesc &desc);

/// Encode a blit fill (memset) on a command buffer.
void encode_blit_fill(void *cmd_buf, void *buffer, size_t offset,
                      size_t length, uint8_t value);

/// Encode a blit buffer copy on a command buffer.
void encode_blit_copy(void *cmd_buf, void *src_buffer, size_t src_offset,
                      void *dst_buffer, size_t dst_offset, size_t length);

} // namespace tmnn::metal

// =============================================================================
// metal_device.mm — ObjC++ Metal device helpers (MRC — no ARC)
// =============================================================================

#include "tiny_metal_nn/runtime/metal_device.h"
#include "tiny_metal_nn/runtime/metal_obj_ptr.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

// MRC cast macros — plain casts, no __bridge needed without ARC.
#define TO_DEVICE(p)   ((id<MTLDevice>)(p))
#define TO_QUEUE(p)    ((id<MTLCommandQueue>)(p))
#define TO_BUFFER(p)   ((id<MTLBuffer>)(p))
#define TO_CMDBUF(p)   ((id<MTLCommandBuffer>)(p))
#define TO_PIPELINE(p) ((id<MTLComputePipelineState>)(p))
#define TO_VOID(obj)   ((void*)(obj))

namespace tmnn::metal {

AllocStats &alloc_stats() {
  static AllocStats stats;
  return stats;
}

void reset_alloc_stats() {
  auto &s = alloc_stats();
  s.create_buffer_calls.store(0, std::memory_order_relaxed);
  s.create_buffer_bytes.store(0, std::memory_order_relaxed);
  s.blit_copy_calls.store(0, std::memory_order_relaxed);
  s.blit_copy_bytes.store(0, std::memory_order_relaxed);
  s.buffer_contents_calls.store(0, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Device probing
// ---------------------------------------------------------------------------

DeviceInfo probe_default_device() {
  DeviceInfo info;

  // device + queue are released via dtor on any early return; on success
  // we transfer ownership into `info` via release().
  auto device =
      tmnn::detail::MetalObjPtr<id<MTLDevice>>::adopt(MTLCreateSystemDefaultDevice());
  if (!device)
    return info;

  auto queue = tmnn::detail::MetalObjPtr<id<MTLCommandQueue>>::adopt(
      [device.get() newCommandQueue]);
  if (!queue)
    return info;  // device released by its dtor

  info.name = [[device.get() name] UTF8String];

  // MTLGPUFamilyApple7 = Apple Silicon M1+.
  if ([device.get() supportsFamily:MTLGPUFamilyApple7]) {
    info.gpu_family = 7;
  } else if ([device.get() supportsFamily:MTLGPUFamilyApple6]) {
    info.gpu_family = 6;
  } else if ([device.get() supportsFamily:MTLGPUFamilyApple5]) {
    info.gpu_family = 5;
  } else {
    info.gpu_family = 1;
  }

  info.max_threads_per_tg =
      static_cast<uint32_t>([device.get() maxThreadsPerThreadgroup].width);
  info.max_tg_memory_bytes =
      static_cast<uint32_t>([device.get() maxThreadgroupMemoryLength]);

  info.supports_fp16 = true;  // All Apple GPUs support FP16.
  info.supports_simdgroup_matrix =
      [device.get() supportsFamily:MTLGPUFamilyApple7];
  info.supports_binary_archive =
      [device.get() supportsFamily:MTLGPUFamilyApple5];
  info.supports_nonuniform_tgs =
      [device.get() supportsFamily:MTLGPUFamilyApple4];

  // Transfer +1 retain into the void* slots in info; caller releases via
  // release_device().
  info.device = TO_VOID(device.release());
  info.queue = TO_VOID(queue.release());
  return info;
}

void release_device(DeviceInfo &info) {
  if (info.queue) {
    [TO_QUEUE(info.queue) release];
    info.queue = nullptr;
  }
  if (info.device) {
    [TO_DEVICE(info.device) release];
    info.device = nullptr;
  }
}

// ---------------------------------------------------------------------------
// Buffer operations
// ---------------------------------------------------------------------------

void *create_buffer(void *device, size_t bytes, bool shared) {
  if (!device || bytes == 0)
    return nullptr;
  MTLResourceOptions opts = shared ? MTLResourceStorageModeShared
                                   : MTLResourceStorageModePrivate;
  id<MTLBuffer> buf = [TO_DEVICE(device) newBufferWithLength:bytes
                                                     options:opts];
  auto &s = alloc_stats();
  s.create_buffer_calls.fetch_add(1, std::memory_order_relaxed);
  s.create_buffer_bytes.fetch_add(bytes, std::memory_order_relaxed);
  return TO_VOID(buf); // +1 retained from newBuffer
}

void release_buffer(void *buffer) {
  if (buffer)
    [TO_BUFFER(buffer) release];
}

void *buffer_contents(void *buffer) {
  if (!buffer)
    return nullptr;
  alloc_stats().buffer_contents_calls.fetch_add(1, std::memory_order_relaxed);
  return [TO_BUFFER(buffer) contents];
}

// ---------------------------------------------------------------------------
// Command buffer operations
// ---------------------------------------------------------------------------

void *create_command_buffer(void *queue) {
  if (!queue)
    return nullptr;
  // -commandBuffer returns autoreleased; convert to +1 retained ownership.
  auto cb = tmnn::detail::MetalObjPtr<id<MTLCommandBuffer>>::retain(
      [TO_QUEUE(queue) commandBuffer]);
  return TO_VOID(cb.release());
}

void commit_and_wait(void *cmd_buf) {
  if (!cmd_buf)
    return;
  id<MTLCommandBuffer> cb = TO_CMDBUF(cmd_buf);
  [cb commit];
  [cb waitUntilCompleted];
}

void commit_async(void *cmd_buf) {
  if (!cmd_buf)
    return;
  [TO_CMDBUF(cmd_buf) commit];
}

void wait_until_completed(void *cmd_buf) {
  if (!cmd_buf)
    return;
  [TO_CMDBUF(cmd_buf) waitUntilCompleted];
}

std::string command_buffer_error(void *cmd_buf) {
  if (!cmd_buf)
    return {};
  id<MTLCommandBuffer> cb = TO_CMDBUF(cmd_buf);
  if (cb.status != MTLCommandBufferStatusError)
    return {};
  if (cb.error)
    return [[cb.error localizedDescription] UTF8String];
  return "unknown GPU error";
}

void release_command_buffer(void *cmd_buf) {
  if (cmd_buf)
    [TO_CMDBUF(cmd_buf) release];
}

double gpu_execution_time_us(void *cmd_buf) {
  if (!cmd_buf)
    return 0.0;
  id<MTLCommandBuffer> cb = TO_CMDBUF(cmd_buf);
  if (cb.status != MTLCommandBufferStatusCompleted)
    return 0.0;
  // gpuStartTime/gpuEndTime are CFAbsoluteTime (seconds).
  const double seconds = cb.GPUEndTime - cb.GPUStartTime;
  return seconds * 1e6;
}

// ---------------------------------------------------------------------------
// Pipeline operations
// ---------------------------------------------------------------------------

void *compile_pipeline(void *device, const char *msl_source,
                       const char *function_name, bool precise_math,
                       std::string &error_out) {
  if (!device || !msl_source || !function_name)
    return nullptr;

  @autoreleasepool {
    NSError *error = nil;
    NSString *src = [NSString stringWithUTF8String:msl_source];
    auto opts =
        tmnn::detail::MetalObjPtr<MTLCompileOptions *>::adopt(
            [[MTLCompileOptions alloc] init]);
    if (precise_math) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
      opts.get().fastMathEnabled = NO;
#pragma clang diagnostic pop
    }

    auto lib = tmnn::detail::MetalObjPtr<id<MTLLibrary>>::adopt(
        [TO_DEVICE(device) newLibraryWithSource:src
                                        options:opts.get()
                                          error:&error]);
    if (!lib) {
      if (error)
        error_out = [[error localizedDescription] UTF8String];
      return nullptr;
    }

    NSString *fname = [NSString stringWithUTF8String:function_name];
    auto func = tmnn::detail::MetalObjPtr<id<MTLFunction>>::adopt(
        [lib.get() newFunctionWithName:fname]);
    if (!func) {
      error_out = "Function not found: ";
      error_out += function_name;
      return nullptr;
    }

    auto pso = tmnn::detail::MetalObjPtr<id<MTLComputePipelineState>>::adopt(
        [TO_DEVICE(device) newComputePipelineStateWithFunction:func.get()
                                                         error:&error]);
    if (!pso) {
      if (error)
        error_out = [[error localizedDescription] UTF8String];
      return nullptr;
    }

    // Hand +1 retain to the caller; opts/lib/func release via their dtors.
    return TO_VOID(pso.release());
  }
}

void release_pipeline(void *pipeline) {
  if (pipeline)
    [TO_PIPELINE(pipeline) release];
}

uint32_t pipeline_max_threads(void *pipeline) {
  if (!pipeline)
    return 0;
  return static_cast<uint32_t>(
      [TO_PIPELINE(pipeline) maxTotalThreadsPerThreadgroup]);
}

// ---------------------------------------------------------------------------
// Dispatch operations
// ---------------------------------------------------------------------------

void encode_dispatch(const DispatchDesc &desc) {
  if (!desc.cmd_buf || !desc.pipeline)
    return;

  id<MTLCommandBuffer> cb = TO_CMDBUF(desc.cmd_buf);
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

  [enc setComputePipelineState:TO_PIPELINE(desc.pipeline)];

  for (uint32_t i = 0; i < desc.binding_count; ++i) {
    const auto &b = desc.bindings[i];
    [enc setBuffer:TO_BUFFER(b.buffer)
            offset:b.offset
           atIndex:b.index];
  }

  if (desc.threadgroup_memory_bytes > 0) {
    [enc setThreadgroupMemoryLength:desc.threadgroup_memory_bytes atIndex:0];
  }

  MTLSize grid = MTLSizeMake(desc.grid_x, desc.grid_y, desc.grid_z);
  MTLSize tg = MTLSizeMake(desc.tg_x, desc.tg_y, desc.tg_z);
  [enc dispatchThreads:grid threadsPerThreadgroup:tg];
  [enc endEncoding];
}

void encode_blit_fill(void *cmd_buf, void *buffer, size_t offset,
                      size_t length, uint8_t value) {
  if (!cmd_buf || !buffer || length == 0)
    return;

  id<MTLCommandBuffer> cb = TO_CMDBUF(cmd_buf);
  id<MTLBlitCommandEncoder> enc = [cb blitCommandEncoder];
  [enc fillBuffer:TO_BUFFER(buffer)
            range:NSMakeRange(offset, length)
            value:value];
  [enc endEncoding];
}

void encode_blit_copy(void *cmd_buf, void *src_buffer, size_t src_offset,
                      void *dst_buffer, size_t dst_offset, size_t length) {
  if (!cmd_buf || !src_buffer || !dst_buffer || length == 0)
    return;

  auto &s = alloc_stats();
  s.blit_copy_calls.fetch_add(1, std::memory_order_relaxed);
  s.blit_copy_bytes.fetch_add(length, std::memory_order_relaxed);

  id<MTLCommandBuffer> cb = TO_CMDBUF(cmd_buf);
  id<MTLBlitCommandEncoder> enc = [cb blitCommandEncoder];
  [enc copyFromBuffer:TO_BUFFER(src_buffer)
         sourceOffset:src_offset
             toBuffer:TO_BUFFER(dst_buffer)
    destinationOffset:dst_offset
                 size:length];
  [enc endEncoding];
}

} // namespace tmnn::metal

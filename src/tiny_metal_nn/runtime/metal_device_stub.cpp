/**
 * @file metal_device_stub.cpp
 * @brief Non-Apple fallback — all Metal operations are no-ops.
 */

#include "tiny_metal_nn/runtime/metal_device.h"

namespace tmnn::metal {

AllocStats &alloc_stats() {
  static AllocStats stats;
  return stats;
}
void reset_alloc_stats() {
  auto &s = alloc_stats();
  s.create_buffer_calls.store(0);
  s.create_buffer_bytes.store(0);
  s.blit_copy_calls.store(0);
  s.blit_copy_bytes.store(0);
  s.buffer_contents_calls.store(0);
}

DeviceInfo probe_default_device() { return {}; }
void release_device(DeviceInfo &) {}

void *create_buffer(void *, size_t, bool) { return nullptr; }
void release_buffer(void *) {}
void *buffer_contents(void *) { return nullptr; }

void *create_command_buffer(void *) { return nullptr; }
void commit_and_wait(void *) {}
void commit_async(void *) {}
void wait_until_completed(void *) {}
std::string command_buffer_error(void *) { return {}; }
void release_command_buffer(void *) {}
double gpu_execution_time_us(void *) { return 0.0; }

void *compile_pipeline(void *, const char *, const char *, bool,
                       std::string &) {
  return nullptr;
}
void release_pipeline(void *) {}
uint32_t pipeline_max_threads(void *) { return 0; }

void encode_dispatch(const DispatchDesc &) {}
void encode_blit_fill(void *, void *, size_t, size_t, uint8_t) {}
void encode_blit_copy(void *, void *, size_t, void *, size_t, size_t) {}

} // namespace tmnn::metal

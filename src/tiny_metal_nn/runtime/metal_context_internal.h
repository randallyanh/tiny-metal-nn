#pragma once

/**
 * @file metal_context_internal.h
 * @brief Internal accessors for MetalContext sub-components.
 *
 * This header is NOT part of the public API. It provides internal runtime
 * code (compat bridges, tests) with access to the BufferArena,
 * CommandBatchPool, and raw Metal handles owned by a MetalContext.
 */

#include "tiny-metal-nn/metal_context.h"
#include "tiny_metal_nn/runtime/buffer_arena.h"
#include "tiny_metal_nn/runtime/buffer_handle.h"
#include "tiny_metal_nn/runtime/command_batch.h"
#include "tiny_metal_nn/runtime/metal_heap/metal_heap.h"
#include "tiny_metal_nn/runtime/numerics_guard.h"
#include "tiny_metal_nn/runtime/pipeline_registry.h"

#include <optional>
#include <span>

namespace tmnn::detail {

/// Access the BufferArena owned by a MetalContext.
BufferArena &context_arena(MetalContext &ctx);

/// Access the CommandBatchPool owned by a MetalContext.
CommandBatchPool &context_batch_pool(MetalContext &ctx);

/// Access the PipelineRegistry owned by a MetalContext.
PipelineRegistry &context_pipeline_registry(MetalContext &ctx);

/// Access the metal_heap::Heap owned by a MetalContext. Returns nullptr if
/// no GPU device was acquired. Phase 3.1 wires this up dormant; call sites
/// migrate over in 3.2 / 3.3.
metal_heap::Heap *context_heap(MetalContext &ctx);

/// Access the NumericsGuard owned by a MetalContext.
NumericsGuard &context_numerics_guard(MetalContext &ctx);

/// Record per-step numerics/runtime telemetry on the owning MetalContext.
void context_record_training_step(MetalContext &ctx, uint64_t numerics_reports,
                                  uint64_t numerics_anomalies,
                                  uint64_t bad_steps_skipped,
                                  uint64_t bad_steps_rolled_back,
                                  uint64_t safe_family_recoveries);

/// Look up a prewarmed autotune decision by planner fingerprint.
std::optional<AutotuneManifestEntry>
context_lookup_autotune_entry(MetalContext &ctx, uint64_t planner_fingerprint);

/// Record/update an autotune decision inside the context-owned manifest state.
void context_record_autotune_entry(MetalContext &ctx,
                                   const AutotuneManifestEntry &entry);

/// Access the raw id<MTLDevice> (void*). nullptr if no GPU.
/// Internal escape hatch only: callers must not establish an independent
/// pipeline/buffer ownership root outside the owning MetalContext.
void *context_raw_device(MetalContext &ctx);

/// Access the raw id<MTLCommandQueue> (void*). nullptr if no GPU.
/// Internal escape hatch only: callers must not establish an independent
/// command submission/cache ownership root outside the owning MetalContext.
void *context_raw_queue(MetalContext &ctx);

/// Blit-fill a buffer region with a byte value via Metal blit encoder.
/// Requires gpu_available(). No-op if no device or no gpu_buffer.
void context_blit_fill(MetalContext &ctx, BufferView &view, uint8_t value);

/// Phase 4: batched blit-fill. Encodes N fillBuffer ops into ONE command
/// buffer and commits-and-waits once, instead of paying the GPU sync
/// round-trip per view. Views with no gpu_buffer or zero bytes are
/// silently skipped. Returns immediately if the resulting batch is empty.
void context_blit_fill_views(MetalContext &ctx,
                             std::span<const BufferView> views,
                             uint8_t value);

/// Phase 4 followup: batched host→Private upload. Allocates one Shared
/// staging MTLBuffer per request, memcpy's source bytes in, encodes N
/// blit-copies into ONE command buffer, commits-and-waits once, then
/// releases all staging buffers. Replaces N sequential context_blit_upload
/// calls (each of which did its own create_buffer + commit_and_wait +
/// release_buffer cycle) with one round-trip + one staging burst.
struct BlitUploadRequest {
  BufferView dst;
  const void *src_data;
  std::size_t bytes;
};
void context_blit_upload_views(MetalContext &ctx,
                               std::span<const BlitUploadRequest> reqs);

/// Phase 4 followup: batched Private→host download, mirror of upload.
struct BlitDownloadRequest {
  BufferView src;
  void *dst_data;
  std::size_t bytes;
};
void context_blit_download_views(MetalContext &ctx,
                                 std::span<const BlitDownloadRequest> reqs);

/// Phase 5: GPU-side weight init via a Philox-4x32-10 RNG kernel.
/// Fills the `dst` view with uniform samples in [low, high]. The kernel
/// is compiled once per MetalContext and cached by PipelineRegistry. The
/// counter_base offsets the per-thread counter so distinct call sites
/// (hash grid vs MLP) draw from non-overlapping streams of the same
/// seed. Caller must commit and wait downstream — this helper only
/// encodes onto a freshly-created command buffer and runs it inline.
void context_dispatch_init_uniform(MetalContext &ctx,
                                   BufferView dst,
                                   std::size_t element_count,
                                   float low, float high,
                                   std::uint64_t seed,
                                   std::uint32_t counter_base);

/// Upload host bytes into a GPU-only buffer view via a staging blit.
void context_blit_upload(MetalContext &ctx, BufferView &dst, const void *data,
                         size_t bytes);

/// Download host bytes from a GPU-only buffer view via a staging blit.
void context_blit_download(MetalContext &ctx, const BufferView &src, void *out,
                           size_t bytes);

} // namespace tmnn::detail

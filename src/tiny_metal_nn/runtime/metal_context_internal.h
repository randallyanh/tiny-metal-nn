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

/// Upload host bytes into a GPU-only buffer view via a staging blit.
void context_blit_upload(MetalContext &ctx, BufferView &dst, const void *data,
                         size_t bytes);

/// Download host bytes from a GPU-only buffer view via a staging blit.
void context_blit_download(MetalContext &ctx, const BufferView &src, void *out,
                           size_t bytes);

} // namespace tmnn::detail

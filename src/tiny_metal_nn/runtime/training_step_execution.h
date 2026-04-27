#pragma once

/**
 * @file training_step_execution.h
 * @brief Internal tmnn-owned helpers for dispatch-plan binding and
 *        enqueue/finalize step orchestration.
 */

#include "tiny-metal-nn/common.h"
#include "tiny-metal-nn/runtime_policy.h"

#include "tiny_metal_nn/runtime/buffer_handle.h"
#include "tiny_metal_nn/runtime/metal_device.h"
#include "tiny_metal_nn/runtime/pipeline_registry.h"
#include "tiny_metal_nn/runtime/training_step_lifecycle.h"

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace tmnn {

class CommandBatchPool;
class MetalContext;
class ParameterStore;
class PipelineRegistry;
class StepLaneCoordinator;

namespace detail {

static constexpr size_t kMaxDispatchBindings = 13;

enum class BindingResolutionClass : uint8_t { RuntimeStable = 0, StepLaneVarying = 1 };

enum class BindingRole : uint8_t {
  StepPositions = 0,
  StepTargets = 1,
  StepLossReduction = 2,
  ConfigWeights = 3,
  HashWeights = 4,
  GradHash = 5,
  GradMlp = 6,
  TrainParams = 7,
  AdamParams = 8,
  FusedWeights = 9,
  FusedM = 10,
  FusedV = 11,
  ActiveHashMask = 12,
  ActiveHashSummaryMask = 13,
  ActiveHashIndices = 14,
  MlpWeights = 15,
  AdamMHash = 16,
  AdamVHash = 17,
  AdamMMlp = 18,
  AdamVMlp = 19,
  ExternalGradient = 20,
  ForwardOutput = 21,
  ProbeBuffer = 22,
};

struct TrainingDispatchKernels {
  PipelineHandle fwd_bwd;
  PipelineHandle backward_ext;  ///< BackwardFromExternalGrad kernel.
  PipelineHandle forward_train; ///< ForwardForTraining kernel.
  PipelineHandle adam_fused;
  PipelineHandle adam_hash_sparse;
  PipelineHandle adam_mlp_dense;
  uint32_t tg_size = 32;
  uint32_t pts_per_tg = 64;
  uint32_t tg_memory_bytes = 0;
  /// Split-path backward kernel derives its own TG params from compile result.
  uint32_t backward_ext_tg_size = 128;
  uint32_t backward_ext_pts_per_tg = 128;
  uint32_t backward_ext_tg_memory_bytes = 0;
  /// Split-path forward kernel derives its own TG params from compile result.
  uint32_t forward_train_tg_size = 128;
  uint32_t forward_train_pts_per_tg = 128;
  uint32_t forward_train_tg_memory_bytes = 0;
  bool valid = false;
};

struct BindingTemplateEntry {
  BindingRole role = BindingRole::ConfigWeights;
  BindingResolutionClass resolution = BindingResolutionClass::RuntimeStable;
  uint32_t slot = 0;
};

struct DispatchBindingTemplate {
  std::array<BindingTemplateEntry, kMaxDispatchBindings> entries{};
  uint32_t count = 0;
};

struct ResolvedBindings {
  std::array<metal::DispatchDesc::BufferBind, kMaxDispatchBindings> binds{};
  uint32_t count = 0;
};

struct DispatchPlan {
  PipelineHandle pipeline{};
  uint32_t tg_x = 1;
  uint32_t tg_y = 1;
  uint32_t tg_z = 1;
  uint32_t threadgroup_memory_bytes = 0;
  DispatchBindingTemplate binding_template;
  ResolvedBindings resolved;
};

struct DispatchPlanDebugEntry {
  BindingRole role = BindingRole::ConfigWeights;
  uint32_t slot = 0;
  BindingResolutionClass resolution = BindingResolutionClass::RuntimeStable;
  const void *buffer = nullptr;
  uint32_t offset = 0;
};

struct DispatchPlanDebugSnapshot {
  const void *binding_template_storage = nullptr;
  const void *resolved_binding_storage = nullptr;
  uint32_t count = 0;
  std::array<DispatchPlanDebugEntry, kMaxDispatchBindings> entries{};
};

using PrepareStepLaneFn = std::function<void(StepBufferSet &)>;
using FillTrainParamsFn = std::function<void(uint32_t batch_N,
                                             uint32_t logical_step)>;
using FillAdamParamsFn = std::function<void(uint32_t logical_step)>;
using ActivateSafeFamilyFn = std::function<bool()>;
using IsSafeFamilyActiveFn = std::function<bool()>;
using RollbackCommittedStepFn = std::function<void()>;
using AppendExtraLossesFn = std::function<void(TrainingStepResult &)>;
using PrepareSparseHashAdamFn = std::function<uint32_t()>;
using NumericsOverrideHook =
    std::optional<NumericsReport> (*)(uint32_t step, bool safe_family_active);

struct EnqueueTrainingStepTimings {
  uint64_t total_ns = 0;
  uint64_t morton_sort_ns = 0;
  uint64_t drain_pending_ns = 0;
  uint64_t prepare_step_lane_ns = 0;
  uint64_t fill_train_params_ns = 0;
  uint64_t resolve_bindings_ns = 0;
  uint64_t submit_forward_backward_ns = 0;
};

struct FinalizeTrainingStepTimings {
  uint64_t total_ns = 0;
  uint64_t wait_pending_ns = 0;
  uint64_t wait_fwd_bwd_fill_ns = 0;
  uint64_t wait_fwd_bwd_dispatch_ns = 0;
  uint64_t fill_adam_params_pre_finalize_ns = 0;
  uint64_t finalize_step_readback_ns = 0;
  uint64_t numerics_report_ns = 0;
  uint64_t numerics_backward_readback_ns = 0;
  uint64_t numerics_backward_scan_ns = 0;
  uint64_t numerics_update_readback_ns = 0;
  uint64_t numerics_update_scan_ns = 0;
  uint64_t fill_adam_params_apply_ns = 0;
  uint64_t prepare_sparse_hash_adam_ns = 0;
  uint64_t submit_adam_ns = 0;
  uint64_t sync_config_weights_ns = 0;
  uint64_t append_extra_losses_ns = 0;
  uint64_t uncategorized_ns = 0;
  // GPU-side timing (from MTLCommandBuffer gpuStartTime/gpuEndTime).
  double gpu_fwd_bwd_us = 0.0;
  double gpu_adam_us = 0.0;
};

struct EnqueueTrainingStepRequest {
  const char *runtime_label = "TrainerRuntime";
  CommandBatchPool &pool;
  PipelineRegistry &registry;
  ParameterStore &parameter_store;
  StepLaneCoordinator &lane_coordinator;
  std::vector<StepBufferSet> &step_lanes;
  PendingTrainingStep &pending_to_drain;
  DispatchPlan &forward_backward_plan;
  const TrainingDispatchKernels &kernels;
  uint32_t batch_size = 0;
  uint32_t logical_step = 0;
  PrepareStepLaneFn prepare_step_lane;
  FillTrainParamsFn fill_train_params;
  bool clear_grad_buffers = true;
  EnqueueTrainingStepTimings *timings = nullptr;
};

struct FinalizeTrainingStepRequest {
  const char *runtime_label = "TrainerRuntime";
  MetalContext &context;
  CommandBatchPool &pool;
  PipelineRegistry &registry;
  ParameterStore &parameter_store;
  StepLaneCoordinator &lane_coordinator;
  std::vector<StepBufferSet> &step_lanes;
  const PendingTrainingStep &pending_step;
  DispatchPlan &forward_backward_plan;
  DispatchPlan &adam_plan;
  const TrainingDispatchKernels &kernels;
  BadStepRecoveryMode recovery_mode = BadStepRecoveryMode::SignalOnly;
  FillTrainParamsFn fill_train_params;
  FillAdamParamsFn fill_adam_params;
  ActivateSafeFamilyFn activate_safe_family;
  IsSafeFamilyActiveFn is_safe_family_active;
  RollbackCommittedStepFn rollback_committed_step;
  AppendExtraLossesFn append_extra_losses;
  NumericsOverrideHook numerics_override_hook = nullptr;
  FinalizeTrainingStepTimings *timings = nullptr;
  DispatchPlan *sparse_hash_adam_plan = nullptr;
  DispatchPlan *dense_mlp_adam_plan = nullptr;
  PrepareSparseHashAdamFn prepare_sparse_hash_adam;
};

DispatchPlan make_fwd_bwd_plan(const TrainingDispatchKernels &kernels,
                               const ParameterStore &ps,
                               bool emit_probes = false);
DispatchPlan make_forward_training_plan(const TrainingDispatchKernels &kernels,
                                        const ParameterStore &ps,
                                        bool emit_probes = false);
DispatchPlan make_backward_ext_plan(const TrainingDispatchKernels &kernels,
                                    const ParameterStore &ps,
                                    bool emit_probes = false);
DispatchPlan make_adam_plan(const ParameterStore &ps, PipelineHandle pipeline);
DispatchPlan make_sparse_hash_adam_plan(const ParameterStore &ps,
                                        PipelineHandle pipeline);
DispatchPlan make_mlp_dense_adam_plan(const ParameterStore &ps,
                                      PipelineHandle pipeline);
BatchFence submit_split_adam_batch(CommandBatchPool &pool,
                                   PipelineRegistry &reg,
                                   ParameterStore &ps,
                                   DispatchPlan &hash_adam_plan,
                                   uint32_t active_hash_count,
                                   DispatchPlan &mlp_adam_plan,
                                   uint32_t mlp_weight_count,
                                   SubmitMode mode,
                                   const char *runtime_label);
void resolve_step_lane_bindings(DispatchPlan &plan, const ParameterStore &ps,
                                const StepBufferSet &lane);
DispatchPlanDebugSnapshot snapshot_dispatch_plan(const DispatchPlan &plan);

/// Aggregate per-TG probe partials into a single ProbeResult (CPU-side).
ProbeResult aggregate_probe_partials(const BufferView &probe_buf,
                                     uint32_t num_tgs,
                                     uint32_t num_hidden_layers);

PendingTrainingStep enqueue_training_step(
    const EnqueueTrainingStepRequest &request);
TrainingStepResult finalize_training_step(
    const FinalizeTrainingStepRequest &request);

} // namespace detail
} // namespace tmnn

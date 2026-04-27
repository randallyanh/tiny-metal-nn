/**
 * @file test_training_step_execution.cpp
 * @brief Tests for tmnn-owned step execution helpers.
 */

#include <gtest/gtest.h>

#include "tiny_metal_nn/runtime/adam_params.h"
#include "tiny_metal_nn/runtime/buffer_arena.h"
#include "tiny_metal_nn/runtime/parameter_store.h"
#include "tiny_metal_nn/runtime/training_step_execution.h"

using namespace tmnn;

namespace {

ParameterStore make_store(BufferArena &arena) {
  ParameterStoreDesc desc;
  desc.hash_grid_size = 8u;
  desc.mlp_weight_count = 16u;
  desc.active_hash_mask_words = 1u;
  desc.active_hash_summary_words = 1u;
  desc.active_hash_index_capacity = 8u;
  desc.target_dims = 1u;
  return ParameterStore(desc, arena);
}

detail::TrainingDispatchKernels make_kernels() {
  detail::TrainingDispatchKernels kernels;
  kernels.fwd_bwd = {1u, 1u};
  kernels.forward_train = {3u, 1u};
  kernels.backward_ext = {4u, 1u};
  kernels.adam_fused = {2u, 1u};
  kernels.tg_size = 64u;
  kernels.pts_per_tg = 128u;
  kernels.tg_memory_bytes = 96u;
  kernels.forward_train_tg_size = 96u;
  kernels.forward_train_pts_per_tg = 192u;
  kernels.forward_train_tg_memory_bytes = 12u;
  kernels.backward_ext_tg_size = 160u;
  kernels.backward_ext_pts_per_tg = 160u;
  kernels.backward_ext_tg_memory_bytes = 20u;
  kernels.valid = true;
  return kernels;
}

StepBufferSet make_step_lane(void *positions_buffer, uint32_t positions_offset,
                             void *targets_buffer, uint32_t targets_offset,
                             void *reduction_buffer, uint32_t reduction_offset) {
  StepBufferSet lane;
  lane.positions.gpu_buffer = positions_buffer;
  lane.positions.offset = positions_offset;
  lane.targets.gpu_buffer = targets_buffer;
  lane.targets.offset = targets_offset;
  lane.loss_reduction.gpu_buffer = reduction_buffer;
  lane.loss_reduction.offset = reduction_offset;
  return lane;
}

} // namespace

TEST(TrainingStepExecution, AdamCountBitEncodingPreservesLargeCounts) {
  constexpr uint32_t kHashCount = 1u << 24;
  constexpr uint32_t kTotalCount = kHashCount + 1u;

  EXPECT_NE(static_cast<uint32_t>(static_cast<float>(kTotalCount)), kTotalCount);
  float params[kUnifiedAdamParamFloats] = {};
  params[9] = static_cast<float>(kHashCount & 0xFFFFu);
  params[10] = static_cast<float>(kHashCount >> 16u);
  EXPECT_EQ(detail::decode_split_u16_u32(params[9], params[10]), kHashCount);
}

TEST(TrainingStepExecution, FillUnifiedAdamParamsPreservesLargeCountFields) {
  TrainerConfig cfg;
  ParameterStoreDesc desc;
  desc.hash_grid_size = 1u << 24;
  desc.mlp_weight_count = 1u;

  float params[kUnifiedAdamParamFloats] = {};
  detail::fill_unified_adam_params(params, cfg, desc, 0u);

  EXPECT_EQ(detail::decode_split_u16_u32(params[9], params[10]),
            desc.hash_grid_size);
  EXPECT_FLOAT_EQ(params[0], cfg.lr_encoding);
  EXPECT_FLOAT_EQ(params[1], cfg.lr_network);
  EXPECT_FLOAT_EQ(params[11], 0.0f);
  EXPECT_FLOAT_EQ(params[12], 0.0f);
}

TEST(TrainingStepExecution, ForwardBackwardPlanLayoutUsesExpectedBindings) {
  BufferArena arena;
  auto store = make_store(arena);
  const auto kernels = make_kernels();

  const auto plan = detail::make_fwd_bwd_plan(kernels, store);
  const auto snapshot = detail::snapshot_dispatch_plan(plan);

  ASSERT_EQ(snapshot.count, 11u);
  EXPECT_EQ(plan.pipeline.index, kernels.fwd_bwd.index);
  EXPECT_EQ(plan.pipeline.generation, kernels.fwd_bwd.generation);
  EXPECT_EQ(plan.tg_x, kernels.tg_size);
  EXPECT_EQ(plan.threadgroup_memory_bytes, kernels.tg_memory_bytes);

  EXPECT_EQ(snapshot.entries[0].role, detail::BindingRole::StepPositions);
  EXPECT_EQ(snapshot.entries[0].slot, 0u);
  EXPECT_EQ(snapshot.entries[0].resolution,
            detail::BindingResolutionClass::StepLaneVarying);

  EXPECT_EQ(snapshot.entries[1].role, detail::BindingRole::StepTargets);
  EXPECT_EQ(snapshot.entries[1].slot, 1u);
  EXPECT_EQ(snapshot.entries[1].resolution,
            detail::BindingResolutionClass::StepLaneVarying);

  EXPECT_EQ(snapshot.entries[2].role, detail::BindingRole::ConfigWeights);
  EXPECT_EQ(snapshot.entries[2].slot, 2u);
  EXPECT_EQ(snapshot.entries[2].resolution,
            detail::BindingResolutionClass::RuntimeStable);

  EXPECT_EQ(snapshot.entries[6].role, detail::BindingRole::StepLossReduction);
  EXPECT_EQ(snapshot.entries[6].slot, 6u);
  EXPECT_EQ(snapshot.entries[6].resolution,
            detail::BindingResolutionClass::StepLaneVarying);

  EXPECT_EQ(snapshot.entries[7].role, detail::BindingRole::TrainParams);
  EXPECT_EQ(snapshot.entries[7].slot, 7u);
  EXPECT_EQ(snapshot.entries[7].resolution,
            detail::BindingResolutionClass::RuntimeStable);

  EXPECT_EQ(snapshot.entries[8].role, detail::BindingRole::ActiveHashMask);
  EXPECT_EQ(snapshot.entries[8].slot, 8u);
  EXPECT_EQ(snapshot.entries[8].resolution,
            detail::BindingResolutionClass::RuntimeStable);
  EXPECT_EQ(snapshot.entries[9].role, detail::BindingRole::ActiveHashSummaryMask);
  EXPECT_EQ(snapshot.entries[9].slot, 9u);
  EXPECT_EQ(snapshot.entries[9].resolution,
            detail::BindingResolutionClass::RuntimeStable);
  EXPECT_EQ(snapshot.entries[10].role, detail::BindingRole::MlpWeights);
  EXPECT_EQ(snapshot.entries[10].slot, 11u);
  EXPECT_EQ(snapshot.entries[10].resolution,
            detail::BindingResolutionClass::RuntimeStable);

}

TEST(TrainingStepExecution, StepLaneBindingReuseKeepsPlanStorageStable) {
  BufferArena arena;
  auto store = make_store(arena);
  auto plan = detail::make_fwd_bwd_plan(make_kernels(), store);

  const auto first_lane = make_step_lane(reinterpret_cast<void *>(0x1010), 16u,
                                         reinterpret_cast<void *>(0x2020), 32u,
                                         reinterpret_cast<void *>(0x3030), 48u);
  detail::resolve_step_lane_bindings(plan, store, first_lane);
  const auto first_snapshot = detail::snapshot_dispatch_plan(plan);

  const auto second_lane = make_step_lane(reinterpret_cast<void *>(0x4040), 64u,
                                          reinterpret_cast<void *>(0x5050), 80u,
                                          reinterpret_cast<void *>(0x6060), 96u);
  detail::resolve_step_lane_bindings(plan, store, second_lane);
  const auto second_snapshot = detail::snapshot_dispatch_plan(plan);

  EXPECT_EQ(first_snapshot.binding_template_storage,
            second_snapshot.binding_template_storage);
  EXPECT_EQ(first_snapshot.resolved_binding_storage,
            second_snapshot.resolved_binding_storage);
  ASSERT_EQ(second_snapshot.count, 11u);

  EXPECT_EQ(second_snapshot.entries[0].buffer, second_lane.positions.gpu_buffer);
  EXPECT_EQ(second_snapshot.entries[0].offset,
            static_cast<uint32_t>(second_lane.positions.offset));
  EXPECT_EQ(second_snapshot.entries[1].buffer, second_lane.targets.gpu_buffer);
  EXPECT_EQ(second_snapshot.entries[1].offset,
            static_cast<uint32_t>(second_lane.targets.offset));
  EXPECT_EQ(second_snapshot.entries[6].buffer,
            second_lane.loss_reduction.gpu_buffer);
  EXPECT_EQ(second_snapshot.entries[6].offset,
            static_cast<uint32_t>(second_lane.loss_reduction.offset));
  EXPECT_EQ(second_snapshot.entries[8].role, detail::BindingRole::ActiveHashMask);
  EXPECT_EQ(second_snapshot.entries[8].slot, 8u);
  EXPECT_EQ(second_snapshot.entries[9].role,
            detail::BindingRole::ActiveHashSummaryMask);
  EXPECT_EQ(second_snapshot.entries[9].slot, 9u);
  EXPECT_EQ(second_snapshot.entries[10].role, detail::BindingRole::MlpWeights);
  EXPECT_EQ(second_snapshot.entries[10].slot, 11u);
}

TEST(TrainingStepExecution, ForwardTrainingPlanUsesDedicatedKernelMetadata) {
  BufferArena arena;
  auto store = make_store(arena);
  const auto kernels = make_kernels();

  const auto plan = detail::make_forward_training_plan(kernels, store);
  const auto snapshot = detail::snapshot_dispatch_plan(plan);

  EXPECT_EQ(plan.pipeline.index, kernels.forward_train.index);
  EXPECT_EQ(plan.pipeline.generation, kernels.forward_train.generation);
  EXPECT_EQ(plan.tg_x, kernels.forward_train_tg_size);
  EXPECT_EQ(plan.threadgroup_memory_bytes,
            kernels.forward_train_tg_memory_bytes);
  EXPECT_NE(plan.tg_x, kernels.tg_size);
  EXPECT_NE(plan.threadgroup_memory_bytes, kernels.tg_memory_bytes);
  ASSERT_EQ(snapshot.count, 10u);
  EXPECT_EQ(snapshot.entries[8].role, detail::BindingRole::ForwardOutput);
  EXPECT_EQ(snapshot.entries[8].slot, 8u);
  EXPECT_EQ(snapshot.entries[9].role, detail::BindingRole::MlpWeights);
  EXPECT_EQ(snapshot.entries[9].slot, 11u);
}

TEST(TrainingStepExecution, BackwardExtPlanUsesDedicatedKernelMetadata) {
  BufferArena arena;
  auto store = make_store(arena);
  const auto kernels = make_kernels();

  const auto plan = detail::make_backward_ext_plan(kernels, store);
  const auto snapshot = detail::snapshot_dispatch_plan(plan);

  EXPECT_EQ(plan.pipeline.index, kernels.backward_ext.index);
  EXPECT_EQ(plan.pipeline.generation, kernels.backward_ext.generation);
  EXPECT_EQ(plan.tg_x, kernels.backward_ext_tg_size);
  EXPECT_EQ(plan.threadgroup_memory_bytes,
            kernels.backward_ext_tg_memory_bytes);
  EXPECT_NE(plan.tg_x, kernels.tg_size);
  EXPECT_NE(plan.threadgroup_memory_bytes, kernels.tg_memory_bytes);
  ASSERT_EQ(snapshot.count, 12u);
  EXPECT_EQ(snapshot.entries[8].role, detail::BindingRole::ExternalGradient);
  EXPECT_EQ(snapshot.entries[8].slot, 8u);
  EXPECT_EQ(snapshot.entries[9].role, detail::BindingRole::ActiveHashMask);
  EXPECT_EQ(snapshot.entries[9].slot, 9u);
  EXPECT_EQ(snapshot.entries[10].role,
            detail::BindingRole::ActiveHashSummaryMask);
  EXPECT_EQ(snapshot.entries[10].slot, 10u);
  EXPECT_EQ(snapshot.entries[11].role, detail::BindingRole::MlpWeights);
  EXPECT_EQ(snapshot.entries[11].slot, 11u);
}

TEST(TrainingStepExecution, BackwardExtProbePlanUsesSlotTwelveForProbeBuffer) {
  BufferArena arena;
  auto store = make_store(arena);
  const auto kernels = make_kernels();

  const auto plan = detail::make_backward_ext_plan(kernels, store, true);
  const auto snapshot = detail::snapshot_dispatch_plan(plan);

  ASSERT_EQ(snapshot.count, 13u);
  EXPECT_EQ(snapshot.entries[10].role,
            detail::BindingRole::ActiveHashSummaryMask);
  EXPECT_EQ(snapshot.entries[10].slot, 10u);
  EXPECT_EQ(snapshot.entries[11].role, detail::BindingRole::MlpWeights);
  EXPECT_EQ(snapshot.entries[11].slot, 11u);
  EXPECT_EQ(snapshot.entries[12].role, detail::BindingRole::ProbeBuffer);
  EXPECT_EQ(snapshot.entries[12].slot, 12u);
}

TEST(TrainingStepExecution, FinalizeTimingsGpuDurationsStartAtZero) {
  detail::FinalizeTrainingStepTimings timings;
  EXPECT_EQ(timings.gpu_fwd_bwd_us, 0.0);
  EXPECT_EQ(timings.gpu_adam_us, 0.0);
}

TEST(TrainingStepExecution, AdamPlanUsesStableFusedBindings) {
  BufferArena arena;
  auto store = make_store(arena);

  const auto plan = detail::make_adam_plan(store, PipelineHandle{7u, 1u});
  const auto snapshot = detail::snapshot_dispatch_plan(plan);

  ASSERT_EQ(snapshot.count, 6u);
  EXPECT_EQ(snapshot.entries[0].role, detail::BindingRole::FusedWeights);
  EXPECT_EQ(snapshot.entries[1].role, detail::BindingRole::GradHash);
  EXPECT_EQ(snapshot.entries[2].role, detail::BindingRole::FusedM);
  EXPECT_EQ(snapshot.entries[3].role, detail::BindingRole::FusedV);
  EXPECT_EQ(snapshot.entries[4].role, detail::BindingRole::AdamParams);
  EXPECT_EQ(snapshot.entries[5].role, detail::BindingRole::GradMlp);

  for (uint32_t i = 0; i < snapshot.count; ++i) {
    EXPECT_EQ(snapshot.entries[i].resolution,
              detail::BindingResolutionClass::RuntimeStable);
  }
}

TEST(TrainingStepExecution, SparseHashAdamPlanUsesExpectedBindings) {
  BufferArena arena;
  auto store = make_store(arena);

  const auto plan =
      detail::make_sparse_hash_adam_plan(store, PipelineHandle{8u, 1u});
  const auto snapshot = detail::snapshot_dispatch_plan(plan);

  ASSERT_EQ(snapshot.count, 6u);
  EXPECT_EQ(snapshot.entries[0].role, detail::BindingRole::HashWeights);
  EXPECT_EQ(snapshot.entries[1].role, detail::BindingRole::GradHash);
  EXPECT_EQ(snapshot.entries[2].role, detail::BindingRole::AdamMHash);
  EXPECT_EQ(snapshot.entries[3].role, detail::BindingRole::AdamVHash);
  EXPECT_EQ(snapshot.entries[4].role, detail::BindingRole::AdamParams);
  EXPECT_EQ(snapshot.entries[5].role, detail::BindingRole::ActiveHashIndices);
}

TEST(TrainingStepExecution, DenseMlpAdamPlanUsesExpectedBindings) {
  BufferArena arena;
  auto store = make_store(arena);

  const auto plan =
      detail::make_mlp_dense_adam_plan(store, PipelineHandle{9u, 1u});
  const auto snapshot = detail::snapshot_dispatch_plan(plan);

  ASSERT_EQ(snapshot.count, 5u);
  EXPECT_EQ(snapshot.entries[0].role, detail::BindingRole::MlpWeights);
  EXPECT_EQ(snapshot.entries[1].role, detail::BindingRole::GradMlp);
  EXPECT_EQ(snapshot.entries[2].role, detail::BindingRole::AdamMMlp);
  EXPECT_EQ(snapshot.entries[3].role, detail::BindingRole::AdamVMlp);
  EXPECT_EQ(snapshot.entries[4].role, detail::BindingRole::AdamParams);
}

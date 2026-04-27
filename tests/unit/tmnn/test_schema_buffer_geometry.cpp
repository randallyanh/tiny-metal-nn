/**
 * @file test_schema_buffer_geometry.cpp
 * @brief D1b tests for schema-driven runtime buffer geometry.
 *
 * Verifies that config_tail_floats, target_dims, and reduction_terms
 * correctly affect buffer allocation, step lane sizing, data reordering,
 * and loss finalization end-to-end. All tests are CPU-only (no GPU).
 */

#include "tiny_metal_nn/runtime/parameter_store.h"
#include "tiny_metal_nn/runtime/buffer_arena.h"
#include "tiny_metal_nn/runtime/morton_sort.h"
#include "tiny-metal-nn/common.h"

#include <algorithm>
#include <cstring>
#include <gtest/gtest.h>
#include <stdexcept>
#include <numeric>
#include <vector>

using namespace tmnn;

namespace {

/// Baseline desc matching legacy defaults.
ParameterStoreDesc baseline_desc() {
  ParameterStoreDesc d;
  d.hash_grid_size = 1024;
  d.mlp_weight_count = 256;
  return d;
}

} // namespace

// ---------------------------------------------------------------------------
// 1. DefaultSchemaMatchesLegacy
// ---------------------------------------------------------------------------

TEST(SchemaBufferGeometry, DefaultSchemaMatchesLegacy) {
  BufferArena arena;
  auto desc = baseline_desc();
  // Defaults: tail=0, target_dims=1, reduction_terms=1
  EXPECT_EQ(desc.config_tail_floats, 0u);
  EXPECT_EQ(desc.target_dims, 1u);
  EXPECT_EQ(desc.reduction_terms, 1u);

  ParameterStore store(desc, arena);

  // config_weights = header(8) + mlp(256) + tail(0) = 264 floats
  EXPECT_EQ(store.config_weights_bytes(),
            (kConfigPackedFloats + 256) * sizeof(float));

  // config_tail() should be empty.
  auto tail = store.config_tail();
  EXPECT_EQ(tail.bytes, 0u);
  EXPECT_EQ(tail.data, nullptr);
}

// ---------------------------------------------------------------------------
// 2. ConfigTailExtendsConfigWeights
// ---------------------------------------------------------------------------

TEST(SchemaBufferGeometry, ConfigTailExtendsConfigWeights) {
  BufferArena arena;
  auto desc = baseline_desc();
  const size_t base_bytes = (kConfigPackedFloats + 256) * sizeof(float);

  desc.config_tail_floats = 9;
  ParameterStore store(desc, arena);

  // config_weights grows by 9 * 4 = 36 bytes.
  EXPECT_EQ(store.config_weights_bytes(), base_bytes + 9 * sizeof(float));
  EXPECT_EQ(store.config_weights().bytes, base_bytes + 9 * sizeof(float));
}

// ---------------------------------------------------------------------------
// 3. ConfigTailSubView
// ---------------------------------------------------------------------------

TEST(SchemaBufferGeometry, ConfigTailSubView) {
  BufferArena arena;
  auto desc = baseline_desc();
  desc.config_tail_floats = 5;
  ParameterStore store(desc, arena);

  auto tail = store.config_tail();
  EXPECT_EQ(tail.bytes, 5 * sizeof(float));
  EXPECT_EQ(tail.offset, (kConfigPackedFloats + 256) * sizeof(float));

  // Same handle as config_weights.
  EXPECT_EQ(tail.handle, store.config_weights().handle);
}

// ---------------------------------------------------------------------------
// 4. ConfigTailPackRoundTrip
// ---------------------------------------------------------------------------

TEST(SchemaBufferGeometry, ConfigTailPackRoundTrip) {
  BufferArena arena;
  auto desc = baseline_desc();
  desc.config_tail_floats = 3;
  ParameterStore store(desc, arena);

  auto tail = store.config_tail();
  ASSERT_NE(tail.data, nullptr);
  ASSERT_EQ(tail.bytes, 3 * sizeof(float));

  // Write through tail view.
  auto *fp = static_cast<float *>(tail.data);
  fp[0] = 1.5f;
  fp[1] = -2.5f;
  fp[2] = 3.14f;

  // Read back through config_weights at the tail offset.
  auto *cfg = static_cast<const float *>(store.config_weights().data);
  EXPECT_FLOAT_EQ(cfg[kConfigPackedFloats + 256 + 0], 1.5f);
  EXPECT_FLOAT_EQ(cfg[kConfigPackedFloats + 256 + 1], -2.5f);
  EXPECT_FLOAT_EQ(cfg[kConfigPackedFloats + 256 + 2], 3.14f);
}

// ---------------------------------------------------------------------------
// 5. MultiOutputTargetBufferSizing — real lane allocation
// ---------------------------------------------------------------------------

TEST(SchemaBufferGeometry, MultiOutputTargetBufferSizing) {
  // Exercises the same allocate_step_set call that trainer_bridge uses,
  // with target_dims=4 producing a 4× larger targets buffer.
  BufferArena arena;
  auto desc = baseline_desc();
  desc.target_dims = 4;
  ParameterStore store(desc, arena);

  const uint32_t batch_size = 512;
  const uint32_t td = store.desc().target_dims;
  const size_t pos_bytes = batch_size * 3 * sizeof(float);
  const size_t tgt_bytes = batch_size * td * sizeof(float);
  const size_t red_bytes = 256 * sizeof(float);

  auto lanes = arena.allocate_step_set(pos_bytes, tgt_bytes, red_bytes, 1);
  ASSERT_EQ(lanes.size(), 1u);

  // Targets lane buffer is 4× the scalar case.
  EXPECT_EQ(lanes[0].targets.bytes, batch_size * 4 * sizeof(float));
  EXPECT_EQ(lanes[0].positions.bytes, pos_bytes);

  // Write/read round-trip through the targets buffer (multi-dim).
  auto *tp = static_cast<float *>(lanes[0].targets.data);
  ASSERT_NE(tp, nullptr);
  for (uint32_t i = 0; i < batch_size * td; ++i)
    tp[i] = static_cast<float>(i);
  EXPECT_FLOAT_EQ(tp[0], 0.0f);
  EXPECT_FLOAT_EQ(tp[batch_size * td - 1],
                   static_cast<float>(batch_size * td - 1));
}

// ---------------------------------------------------------------------------
// 6. MultiTermReductionSizing — real lane allocation
// ---------------------------------------------------------------------------

TEST(SchemaBufferGeometry, MultiTermReductionSizing) {
  // Exercises allocate_step_set with reduction_terms=3 producing
  // a 3× larger loss_reduction buffer.
  BufferArena arena;
  auto desc = baseline_desc();
  desc.reduction_terms = 3;
  ParameterStore store(desc, arena);

  const uint32_t max_tgs = 256;
  const uint32_t terms = store.desc().reduction_terms;
  const size_t red_bytes = max_tgs * terms * sizeof(float);

  auto lanes = arena.allocate_step_set(64, 64, red_bytes, 1);
  ASSERT_EQ(lanes.size(), 1u);

  // Loss reduction buffer is 3× the scalar case.
  EXPECT_EQ(lanes[0].loss_reduction.bytes, max_tgs * 3 * sizeof(float));

  // Can write all terms × TGs without out-of-bounds.
  auto *rp = static_cast<float *>(lanes[0].loss_reduction.data);
  ASSERT_NE(rp, nullptr);
  for (uint32_t i = 0; i < max_tgs * terms; ++i)
    rp[i] = static_cast<float>(i);
  EXPECT_FLOAT_EQ(rp[max_tgs * terms - 1],
                   static_cast<float>(max_tgs * terms - 1));
}

// ---------------------------------------------------------------------------
// 7. TrainingStepResultExtraLosses
// ---------------------------------------------------------------------------

TEST(SchemaBufferGeometry, TrainingStepResultExtraLosses) {
  TrainingStepResult r{};
  EXPECT_FALSE(r.numerics_reported);
  EXPECT_FALSE(r.has_numerics_anomaly);
  EXPECT_EQ(r.recovery_action, BadStepRecoveryAction::None);
  EXPECT_TRUE(r.numerics.finite_forward);
  EXPECT_TRUE(r.numerics.finite_backward);
  EXPECT_TRUE(r.numerics.finite_update);
  EXPECT_EQ(r.extra_loss_count, 0u);
  for (uint32_t i = 0; i < TrainingStepResult::kMaxExtraLosses; ++i)
    EXPECT_FLOAT_EQ(r.extra_losses[i], 0.0f);
  EXPECT_EQ(TrainingStepResult::kMaxExtraLosses, 4u);
}

// ---------------------------------------------------------------------------
// 8. FinalizeMultiTermReduction
// ---------------------------------------------------------------------------

TEST(SchemaBufferGeometry, FinalizeMultiTermReduction) {
  BufferArena arena;
  auto desc = baseline_desc();
  desc.reduction_terms = 2;
  ParameterStore store(desc, arena);

  const uint32_t num_tgs = 4;
  const uint32_t terms = 2;
  auto lanes = arena.allocate_step_set(64, 64,
      num_tgs * terms * sizeof(float), 1);
  auto &sbs = lanes[0];

  // Strided partials: partials[tg * terms + term].
  auto *partials = static_cast<float *>(sbs.loss_reduction.data);
  ASSERT_NE(partials, nullptr);

  // Term 0 (main loss): 1.0, 2.0, 3.0, 4.0 → total=10.0
  // Term 1 (extra[0]):  0.5, 1.0, 1.5, 2.0 → total=5.0
  for (uint32_t tg = 0; tg < num_tgs; ++tg) {
    partials[tg * terms + 0] = static_cast<float>(tg + 1);
    partials[tg * terms + 1] = static_cast<float>(tg + 1) * 0.5f;
  }

  const uint32_t batch_N = 10;
  auto result = store.finalize_async_step(sbs, num_tgs, batch_N);

  EXPECT_FLOAT_EQ(result.mean_loss, 1.0f);        // 10.0 / 10
  EXPECT_EQ(result.extra_loss_count, 1u);
  EXPECT_FLOAT_EQ(result.extra_losses[0], 0.5f);   // 5.0 / 10

  for (uint32_t i = 1; i < ParameterStore::kMaxExtraLosses; ++i)
    EXPECT_FLOAT_EQ(result.extra_losses[i], 0.0f);
}

// ---------------------------------------------------------------------------
// 9. MortonSortMultiDimTargetCopy — exercises the tmnn runtime Morton helper
//    with target_dims > 1.
// ---------------------------------------------------------------------------

TEST(SchemaBufferGeometry, MortonSortMultiDimTargetCopy) {
  const uint32_t td = 3;
  const size_t N = 5;

  std::vector<float> positions(N * 3);
  std::vector<float> targets(N * td);
  const std::array<float, N> xs = {0.6f, -0.8f, 0.2f, -0.2f, 0.9f};
  for (size_t i = 0; i < N; ++i) {
    positions[i * 3 + 0] = xs[i];
    positions[i * 3 + 1] = 0.0f;
    positions[i * 3 + 2] = 0.0f;
    for (uint32_t d = 0; d < td; ++d)
      targets[i * td + d] = static_cast<float>(i * 100 + d);
  }

  std::vector<uint32_t> indices;
  std::vector<uint32_t> codes;
  std::vector<float> sorted_targets(N * td);
  std::vector<float> sorted_positions(N * 3);
  tmnn::detail::morton_sort_batch(positions.data(), targets.data(),
                                  static_cast<int>(N), 3u, td, indices, codes,
                                  sorted_positions, sorted_targets);

  const std::array<size_t, N> expected = {1, 3, 2, 0, 4};
  for (size_t i = 0; i < N; ++i) {
    const size_t src = expected[i];
    EXPECT_FLOAT_EQ(sorted_positions[i * 3 + 0], positions[src * 3 + 0]);
    EXPECT_FLOAT_EQ(sorted_positions[i * 3 + 1], positions[src * 3 + 1]);
    EXPECT_FLOAT_EQ(sorted_positions[i * 3 + 2], positions[src * 3 + 2]);
    for (uint32_t d = 0; d < td; ++d) {
      EXPECT_FLOAT_EQ(sorted_targets[i * td + d], targets[src * td + d])
          << "mismatch at sorted[" << i << "][" << d << "]";
    }
  }
}

TEST(SchemaBufferGeometry, MortonSortSupports4DInputs) {
  const uint32_t td = 2;
  const size_t N = 4;

  std::vector<float> positions{
      0.0f, 0.0f, 0.0f, 0.9f,
      0.0f, 0.0f, 0.0f, -0.8f,
      0.0f, 0.0f, 0.0f, 0.1f,
      0.0f, 0.0f, 0.0f, -0.2f,
  };
  std::vector<float> targets{
      10.0f, 11.0f,
      20.0f, 21.0f,
      30.0f, 31.0f,
      40.0f, 41.0f,
  };

  std::vector<uint32_t> indices;
  std::vector<uint32_t> codes;
  std::vector<float> sorted_targets;
  std::vector<float> sorted_positions;
  tmnn::detail::morton_sort_batch(positions.data(), targets.data(),
                                  static_cast<int>(N), 4u, td, indices, codes,
                                  sorted_positions, sorted_targets);

  ASSERT_EQ(indices.size(), N);
  ASSERT_EQ(codes.size(), N);
  for (size_t i = 0; i < N; ++i) {
    const size_t src = indices[i];
    ASSERT_LT(src, N);
    if (i > 0)
      EXPECT_LE(codes[indices[i - 1]], codes[indices[i]]);
    EXPECT_FLOAT_EQ(sorted_positions[i * 4 + 3], positions[src * 4 + 3]);
    for (uint32_t d = 0; d < td; ++d) {
      EXPECT_FLOAT_EQ(sorted_targets[i * td + d], targets[src * td + d]);
    }
  }
}

// ---------------------------------------------------------------------------
// 10. MultiDimTargetUploadRoundTrip — allocates a step lane with
//     target_dims > 1, writes multi-dim sorted targets, reads back.
// ---------------------------------------------------------------------------

TEST(SchemaBufferGeometry, MultiDimTargetUploadRoundTrip) {
  BufferArena arena;
  auto desc = baseline_desc();
  desc.target_dims = 3;
  ParameterStore store(desc, arena);

  const uint32_t N = 8;
  const uint32_t td = store.desc().target_dims;
  const size_t tgt_bytes = N * td * sizeof(float);
  auto lanes = arena.allocate_step_set(N * 3 * sizeof(float), tgt_bytes,
                                        256 * sizeof(float), 1);
  auto &lane = lanes[0];

  // Simulate multi-dim sorted targets.
  std::vector<float> sorted_targets(N * td);
  for (uint32_t i = 0; i < N * td; ++i)
    sorted_targets[i] = static_cast<float>(i) * 0.1f;

  // Upload — same memcpy as trainer_bridge uses.
  std::memcpy(lane.targets.data, sorted_targets.data(),
              static_cast<size_t>(N) * td * sizeof(float));

  // Read back and verify.
  auto *uploaded = static_cast<const float *>(lane.targets.data);
  for (uint32_t i = 0; i < N * td; ++i)
    EXPECT_FLOAT_EQ(uploaded[i], sorted_targets[i])
        << "target upload mismatch at index " << i;
}

// ---------------------------------------------------------------------------
// 11. ExtraLossPropagationToTrainingStepResult — verifies the full chain:
//     multi-term finalize → AsyncStepResult → TrainingStepResult propagation
//     (same mapping as trainer_bridge::native_training_step).
// ---------------------------------------------------------------------------

TEST(SchemaBufferGeometry, ExtraLossPropagationToTrainingStepResult) {
  BufferArena arena;
  auto desc = baseline_desc();
  desc.reduction_terms = 3; // main + 2 extra
  ParameterStore store(desc, arena);

  const uint32_t num_tgs = 2;
  const uint32_t terms = 3;
  auto lanes = arena.allocate_step_set(64, 64,
      num_tgs * terms * sizeof(float), 1);
  auto &sbs = lanes[0];

  auto *partials = static_cast<float *>(sbs.loss_reduction.data);
  ASSERT_NE(partials, nullptr);

  // TG 0: [10.0, 2.0, 4.0], TG 1: [20.0, 3.0, 6.0]
  partials[0 * terms + 0] = 10.0f; partials[0 * terms + 1] = 2.0f; partials[0 * terms + 2] = 4.0f;
  partials[1 * terms + 0] = 20.0f; partials[1 * terms + 1] = 3.0f; partials[1 * terms + 2] = 6.0f;

  const uint32_t batch_N = 30;
  auto async_result = store.finalize_async_step(sbs, num_tgs, batch_N);

  // Verify AsyncStepResult.
  EXPECT_FLOAT_EQ(async_result.mean_loss, 1.0f);        // (10+20)/30
  EXPECT_EQ(async_result.extra_loss_count, 2u);
  EXPECT_FLOAT_EQ(async_result.extra_losses[0], 5.0f / 30.0f);  // (2+3)/30
  EXPECT_FLOAT_EQ(async_result.extra_losses[1], 10.0f / 30.0f); // (4+6)/30

  // Propagate to TrainingStepResult — same code as trainer_bridge.
  TrainingStepResult out{};
  out.loss = async_result.mean_loss;
  out.step = 42;
  for (uint32_t i = 0; i < async_result.extra_loss_count &&
       i < TrainingStepResult::kMaxExtraLosses; ++i)
    out.extra_losses[i] = async_result.extra_losses[i];
  out.extra_loss_count = async_result.extra_loss_count;

  // Verify TrainingStepResult.
  EXPECT_FLOAT_EQ(out.loss, 1.0f);
  EXPECT_EQ(out.step, 42u);
  EXPECT_EQ(out.extra_loss_count, 2u);
  EXPECT_FLOAT_EQ(out.extra_losses[0], 5.0f / 30.0f);
  EXPECT_FLOAT_EQ(out.extra_losses[1], 10.0f / 30.0f);
  EXPECT_FLOAT_EQ(out.extra_losses[2], 0.0f); // untouched
  EXPECT_FLOAT_EQ(out.extra_losses[3], 0.0f); // untouched
}

// ---------------------------------------------------------------------------
// 12. ValidationRejectsReductionTermsBeyondExtraLossCapacity
// ---------------------------------------------------------------------------

TEST(SchemaBufferGeometry, ValidationRejectsReductionTermsBeyondExtraLossCapacity) {
  BufferArena arena;
  auto desc = baseline_desc();
  desc.reduction_terms = 6; // 1 main + 5 extra > public TrainingStepResult capacity
  EXPECT_THROW(ParameterStore(desc, arena), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// 13. ValidationRejectsZeroTargetDims
// ---------------------------------------------------------------------------

TEST(SchemaBufferGeometry, ValidationRejectsZeroTargetDims) {
  BufferArena arena;
  auto desc = baseline_desc();
  desc.target_dims = 0;
  EXPECT_THROW(ParameterStore(desc, arena), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// 14. ValidationRejectsZeroReductionTerms
// ---------------------------------------------------------------------------

TEST(SchemaBufferGeometry, ValidationRejectsZeroReductionTerms) {
  BufferArena arena;
  auto desc = baseline_desc();
  desc.reduction_terms = 0;
  EXPECT_THROW(ParameterStore(desc, arena), std::invalid_argument);
}

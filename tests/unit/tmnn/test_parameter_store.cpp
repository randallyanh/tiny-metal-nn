/**
 * @file test_parameter_store.cpp
 * @brief C2 tests for ParameterStore.
 */

#include "tiny_metal_nn/runtime/parameter_store.h"
#include "tiny_metal_nn/runtime/buffer_arena.h"

#include <gtest/gtest.h>
#include <vector>

using namespace tmnn;

namespace {

/// Typical small config: 16 levels, 2^19 hashmap, 2 features, HD=64, 2 layers.
ParameterStoreDesc small_desc() {
  ParameterStoreDesc d;
  d.hash_grid_size = 16 * (1 << 19) * 2; // 16,777,216
  d.mlp_weight_count = 6337;              // 32*64+64 + 64*64+64 + 64*1+1
  return d;
}

/// Tiny config for fast tests.
ParameterStoreDesc tiny_desc() {
  ParameterStoreDesc d;
  d.hash_grid_size = 1024;
  d.mlp_weight_count = 256;
  return d;
}

} // namespace

// --- Construction ---

TEST(ParameterStore, Construction) {
  BufferArena arena;
  auto desc = tiny_desc();
  ParameterStore store(desc, arena);
  EXPECT_EQ(store.desc().hash_grid_size, 1024u);
  EXPECT_EQ(store.desc().mlp_weight_count, 256u);
  EXPECT_FALSE(store.desc().use_private_buffers);
  EXPECT_GT(store.total_bytes(), 0u);
}

// --- Buffer view sizes ---

TEST(ParameterStore, ViewSizes) {
  BufferArena arena;
  auto desc = tiny_desc();
  ParameterStore store(desc, arena);

  EXPECT_EQ(store.hash_weights().bytes, 1024 * sizeof(float));
  EXPECT_EQ(store.mlp_weights().bytes, 256 * sizeof(float));
  EXPECT_EQ(store.grad_hash().bytes, 1024 * sizeof(float));
  EXPECT_EQ(store.grad_mlp().bytes, 256 * sizeof(float));
  EXPECT_EQ(store.adam_m_hash().bytes, 1024 * sizeof(float));
  EXPECT_EQ(store.adam_v_hash().bytes, 1024 * sizeof(float));
  EXPECT_EQ(store.adam_m_mlp().bytes, 256 * sizeof(float));
  EXPECT_EQ(store.adam_v_mlp().bytes, 256 * sizeof(float));
}

TEST(ParameterStore, ActiveHashTrackingBuffers) {
  BufferArena arena;
  auto desc = tiny_desc();
  desc.active_hash_mask_words = 32u;
  desc.active_hash_summary_words = 8u;
  desc.active_hash_index_capacity = 96u;
  ParameterStore store(desc, arena);

  EXPECT_EQ(store.active_hash_mask().bytes, 32u * sizeof(uint32_t));
  EXPECT_EQ(store.active_hash_summary_mask().bytes, 8u * sizeof(uint32_t));
  EXPECT_EQ(store.active_hash_indices().bytes, 96u * sizeof(uint32_t));
  EXPECT_EQ(arena.slot_storage(store.active_hash_mask().handle),
            BufferStorage::Shared);
  EXPECT_EQ(arena.slot_storage(store.active_hash_summary_mask().handle),
            BufferStorage::Shared);
  EXPECT_EQ(arena.slot_storage(store.active_hash_indices().handle),
            BufferStorage::Shared);
}

TEST(ParameterStore, ActiveHashMaskCanUsePrivateStorage) {
  BufferArena arena;
  auto desc = tiny_desc();
  desc.active_hash_mask_words = 16u;
  desc.use_private_active_hash_mask = true;
  ParameterStore store(desc, arena);

  EXPECT_EQ(store.active_hash_mask().bytes, 16u * sizeof(uint32_t));
  EXPECT_EQ(arena.slot_storage(store.active_hash_mask().handle),
            BufferStorage::Private);
}

// --- Config weights layout ---

TEST(ParameterStore, ConfigWeightsLayout) {
  BufferArena arena;
  auto desc = tiny_desc();
  ParameterStore store(desc, arena);

  // config_weights = 8-float header + MLP weights.
  EXPECT_EQ(store.config_weights().bytes,
            (kConfigPackedFloats + 256) * sizeof(float));

  // Header sub-view.
  auto header = store.config_header();
  EXPECT_EQ(header.bytes, kConfigPackedFloats * sizeof(float));
  EXPECT_EQ(header.offset, 0u);

  // MLP sub-view.
  auto mlp = store.config_mlp();
  EXPECT_EQ(mlp.bytes, 256 * sizeof(float));
  EXPECT_EQ(mlp.offset, kConfigPackedFloats * sizeof(float));

  // Both share the same handle.
  EXPECT_EQ(header.handle, mlp.handle);
  EXPECT_EQ(header.handle, store.config_weights().handle);
}

// --- Control buffer sizes ---

TEST(ParameterStore, ControlBufferSizes) {
  BufferArena arena;
  ParameterStore store(tiny_desc(), arena);

  EXPECT_EQ(store.train_params().bytes, kTrainParamFloats * sizeof(float));
  EXPECT_EQ(store.adam_params().bytes, kFusedAdamParamFloats * sizeof(float));
}

// --- Private buffers ---

TEST(ParameterStore, PrivateStorage) {
  BufferArena arena;
  auto desc = tiny_desc();
  desc.use_private_buffers = true;
  ParameterStore store(desc, arena);

  // Weights are always Shared (CPU needs to upload).
  EXPECT_EQ(arena.slot_storage(store.hash_weights().handle),
            BufferStorage::Shared);
  EXPECT_EQ(arena.slot_storage(store.mlp_weights().handle),
            BufferStorage::Shared);

  // Gradients and moments are Private.
  EXPECT_EQ(arena.slot_storage(store.grad_hash().handle),
            BufferStorage::Private);
  EXPECT_EQ(arena.slot_storage(store.grad_mlp().handle),
            BufferStorage::Private);
  EXPECT_EQ(arena.slot_storage(store.adam_m_hash().handle),
            BufferStorage::Private);
  EXPECT_EQ(arena.slot_storage(store.adam_v_hash().handle),
            BufferStorage::Private);
  EXPECT_EQ(arena.slot_storage(store.adam_m_mlp().handle),
            BufferStorage::Private);
  EXPECT_EQ(arena.slot_storage(store.adam_v_mlp().handle),
            BufferStorage::Private);

  // Control buffers are always Shared.
  EXPECT_EQ(arena.slot_storage(store.config_weights().handle),
            BufferStorage::Shared);
  EXPECT_EQ(arena.slot_storage(store.train_params().handle),
            BufferStorage::Shared);
  EXPECT_EQ(arena.slot_storage(store.adam_params().handle),
            BufferStorage::Shared);
}

// --- Total bytes accounting ---

TEST(ParameterStore, TotalBytes) {
  BufferArena arena;
  auto desc = tiny_desc();
  ParameterStore store(desc, arena);

  const size_t hash_b = 1024 * sizeof(float);
  const size_t mlp_b = 256 * sizeof(float);
  const size_t config_b = (kConfigPackedFloats + 256) * sizeof(float);
  const size_t tp_b = kTrainParamFloats * sizeof(float);
  const size_t ap_b = kFusedAdamParamFloats * sizeof(float);

  // 5 × hash_b (weights, grad, m, v counted as 2*(hash+mlp) weights+grads + 2*(hash+mlp) moments)
  // Actually: hash_weights + mlp_weights + grad_hash + grad_mlp
  //           + m_hash + v_hash + m_mlp + v_mlp
  //           + config + train_params + adam_params
  size_t expected = hash_b + mlp_b           // weights
                    + hash_b + mlp_b         // grads
                    + 2 * hash_b + 2 * mlp_b // m + v
                    + config_b + tp_b + ap_b;

  EXPECT_EQ(store.total_bytes(), expected);
  EXPECT_EQ(arena.bytes_allocated(), expected);
}

TEST(ParameterStore, DestructionReleasesArenaBuffers) {
  BufferArena arena;
  const auto desc = tiny_desc();
  size_t expected = 0;

  {
    ParameterStore store(desc, arena);
    expected = store.total_bytes();
    EXPECT_EQ(arena.bytes_allocated(), expected);
    EXPECT_GT(arena.live_count(), 0u);
  }

  EXPECT_EQ(arena.bytes_allocated(), 0u);
  EXPECT_EQ(arena.live_count(), 0u);
}

TEST(ParameterStore, RepeatedConstructionReusesArenaSlots) {
  BufferArena arena;
  const auto desc = tiny_desc();
  uint32_t first_total_slots = 0;

  {
    ParameterStore store(desc, arena);
    first_total_slots = arena.total_slots();
    EXPECT_GT(first_total_slots, 0u);
  }

  EXPECT_EQ(arena.live_count(), 0u);

  {
    ParameterStore store(desc, arena);
    EXPECT_EQ(arena.total_slots(), first_total_slots);
  }

  EXPECT_EQ(arena.live_count(), 0u);
}

// --- Binding helper ---

TEST(ParameterStore, BindHelper) {
  BufferArena arena;
  ParameterStore store(tiny_desc(), arena);

  auto binding = ParameterStore::bind(store.hash_weights(), 3);
  EXPECT_EQ(binding.binding_index, 3u);
  EXPECT_EQ(binding.view.handle, store.hash_weights().handle);
  EXPECT_EQ(binding.view.bytes, store.hash_weights().bytes);
}

// --- All views are distinct ---

TEST(ParameterStore, AllViewsDistinct) {
  BufferArena arena;
  ParameterStore store(tiny_desc(), arena);

  // All top-level buffers have unique handles.
  BufferHandle handles[] = {
      store.hash_weights().handle,
      store.mlp_weights().handle,
      store.grad_hash().handle,
      store.grad_mlp().handle,
      store.adam_m_hash().handle,
      store.adam_v_hash().handle,
      store.adam_m_mlp().handle,
      store.adam_v_mlp().handle,
      store.config_weights().handle,
      store.train_params().handle,
      store.adam_params().handle,
  };

  for (size_t i = 0; i < std::size(handles); ++i)
    for (size_t j = i + 1; j < std::size(handles); ++j)
      EXPECT_NE(handles[i], handles[j])
          << "handles[" << i << "] == handles[" << j << "]";
}

// --- kConfigPackedFloats frozen contract ---

TEST(ParameterStore, ConfigPackedFloatsFrozen) {
  EXPECT_EQ(kConfigPackedFloats, 8u);
}

// --- Realistic size ---

TEST(ParameterStore, RealisticSize) {
  BufferArena arena;
  ParameterStore store(small_desc(), arena);

  // Hash grid: 16M floats × 4B = 64 MB per buffer, ×4 copies = ~256 MB
  // MLP: 6337 × 4B ≈ 25 KB per buffer, ×4 copies ≈ 100 KB
  // + config + tp + ap ≈ 25 KB
  // Total ≈ 268 MB
  EXPECT_GT(store.total_bytes(), 250'000'000u);
  EXPECT_LT(store.total_bytes(), 300'000'000u);
}

// --- C7: TrainParamsLayout ---

TEST(ParameterStore, CustomTrainParamsLayout) {
  BufferArena arena;
  ParameterStoreDesc desc;
  desc.hash_grid_size = 1024;
  desc.mlp_weight_count = 256;
  desc.train_params_layout.float_count = 5; // DNL-like
  ParameterStore store(desc, arena);
  EXPECT_EQ(store.train_params().bytes, 5 * sizeof(float));
  EXPECT_EQ(store.train_params_layout().float_count, 5u);
}

TEST(ParameterStore, CustomAdamParamsCount) {
  BufferArena arena;
  ParameterStoreDesc desc;
  desc.hash_grid_size = 1024;
  desc.mlp_weight_count = 256;
  desc.adam_params_float_count = 10;
  ParameterStore store(desc, arena);
  EXPECT_EQ(store.adam_params().bytes, 10 * sizeof(float));
}

TEST(ParameterStore, LayoutValidationThrows) {
  // Duplicate indices.
  {
    TrainParamsLayout layout;
    layout.idx_n = 0;
    layout.idx_unsigned_mode = 0; // dup!
    EXPECT_THROW(layout.validate(), std::invalid_argument);
  }
  // float_count too small.
  {
    TrainParamsLayout layout;
    layout.float_count = 0;
    EXPECT_THROW(layout.validate(), std::invalid_argument);
  }
  // Index >= float_count.
  {
    TrainParamsLayout layout;
    layout.float_count = 3; // idx_num_active_levels=3 == float_count
    EXPECT_THROW(layout.validate(), std::out_of_range);
  }
  // Valid layout should not throw.
  {
    TrainParamsLayout layout;
    EXPECT_NO_THROW(layout.validate());
  }
}

TEST(ParameterStore, FillTrainParams) {
  TrainParamsLayout layout;
  layout.float_count = 6;
  layout.idx_n = 0;
  layout.idx_unsigned_mode = 2;
  layout.idx_loss_scale = 4;
  layout.idx_num_active_levels = 5;
  layout.validate();

  float buf[6] = {-1, -1, -1, -1, -1, -1};
  fill_train_params(buf, layout, 4096, true, 128.0f, 12);

  EXPECT_FLOAT_EQ(buf[0], 4096.0f);       // idx_n
  EXPECT_FLOAT_EQ(buf[1], 0.0f);          // zero-filled
  EXPECT_FLOAT_EQ(buf[2], 1.0f);          // idx_unsigned_mode
  EXPECT_FLOAT_EQ(buf[3], 0.0f);          // zero-filled
  EXPECT_FLOAT_EQ(buf[4], 128.0f);        // idx_loss_scale
  EXPECT_FLOAT_EQ(buf[5], 12.0f);         // idx_num_active_levels
}

TEST(ParameterStore, DefaultLayoutMatchesLegacy) {
  BufferArena arena;
  ParameterStoreDesc desc;
  desc.hash_grid_size = 1024;
  desc.mlp_weight_count = 256;
  // All defaults — should produce same sizes as before C7.
  ParameterStore store(desc, arena);
  EXPECT_EQ(store.train_params().bytes, kTrainParamFloats * sizeof(float));
  EXPECT_EQ(store.adam_params().bytes, kFusedAdamParamFloats * sizeof(float));
}

// --- Hydrate weights ---

TEST(ParameterStore, HydrateWeightsRoundTrip) {
  BufferArena arena;
  auto desc = tiny_desc(); // hash=1024, mlp=256
  ParameterStore store(desc, arena);

  // Prepare known data.
  std::vector<float> hash_data(1024, 0.0f);
  hash_data[0] = 1.0f;
  hash_data[1023] = -2.0f;

  std::vector<float> mlp_data(256, 0.0f);
  mlp_data[0] = 3.0f;
  mlp_data[255] = -4.0f;

  store.hydrate_weights(hash_data.data(), hash_data.size(),
                        mlp_data.data(), mlp_data.size());

  // Verify hash weights.
  auto *hp = static_cast<const float *>(store.hash_weights().data);
  ASSERT_NE(hp, nullptr);
  EXPECT_FLOAT_EQ(hp[0], 1.0f);
  EXPECT_FLOAT_EQ(hp[1023], -2.0f);

  // Verify MLP weights.
  auto *mp = static_cast<const float *>(store.mlp_weights().data);
  ASSERT_NE(mp, nullptr);
  EXPECT_FLOAT_EQ(mp[0], 3.0f);
  EXPECT_FLOAT_EQ(mp[255], -4.0f);
}

TEST(ParameterStore, HydrateConfigWeightsLayout) {
  BufferArena arena;
  auto desc = tiny_desc();
  ParameterStore store(desc, arena);

  float header[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> mlp_data(256, 0.5f);

  store.hydrate_weights(nullptr, 0, mlp_data.data(), mlp_data.size(), header);

  // Verify config header.
  auto *cfg = static_cast<const float *>(store.config_weights().data);
  ASSERT_NE(cfg, nullptr);
  for (int i = 0; i < 8; ++i)
    EXPECT_FLOAT_EQ(cfg[i], static_cast<float>(i + 1));

  // Verify config MLP section.
  EXPECT_FLOAT_EQ(cfg[8], 0.5f);
  EXPECT_FLOAT_EQ(cfg[8 + 255], 0.5f);
}

// --- finalize_async_step ---

TEST(ParameterStore, FinalizeAsyncStepLossReadback) {
  BufferArena arena;
  ParameterStore store(tiny_desc(), arena);

  // Create a step buffer with known loss partials.
  auto lanes = arena.allocate_step_set(64, 64, 128, 1);
  auto &sbs = lanes[0];

  // Write known loss partials: 3 TGs with values 1.0, 2.0, 3.0 → total=6.0
  auto *partials = static_cast<float *>(sbs.loss_reduction.data);
  ASSERT_NE(partials, nullptr);
  partials[0] = 1.0f;
  partials[1] = 2.0f;
  partials[2] = 3.0f;

  auto result =
      store.finalize_async_step(sbs, 3, 6, 7); // mean = 6.0/6 = 1.0
  EXPECT_FLOAT_EQ(result.mean_loss, 1.0f);
  EXPECT_EQ(result.completed_step, 7u);

  auto result2 = store.finalize_async_step(sbs, 3, 6, 11);
  EXPECT_EQ(result2.completed_step, 11u);
}

TEST(ParameterStore, FinalizeAsyncStepRejectsMissingLossReductionData) {
  BufferArena arena;
  ParameterStore store(tiny_desc(), arena);

  StepBufferSet empty_sbs{};
  EXPECT_THROW((void)store.finalize_async_step(empty_sbs, 1, 1),
               std::invalid_argument);
}

TEST(ParameterStore, FinalizeAsyncStepRejectsUndersizedLossReductionBuffer) {
  BufferArena arena;
  auto desc = tiny_desc();
  desc.reduction_terms = 2;
  ParameterStore store(desc, arena);

  auto lanes = arena.allocate_step_set(64, 64, sizeof(float), 1);
  auto &sbs = lanes[0];
  ASSERT_NE(sbs.loss_reduction.data, nullptr);

  EXPECT_THROW((void)store.finalize_async_step(sbs, 1, 1),
               std::invalid_argument);
}

TEST(ParameterStore, FinalizeAsyncStepRejectsZeroBatchForNonzeroTgs) {
  BufferArena arena;
  ParameterStore store(tiny_desc(), arena);

  auto lanes = arena.allocate_step_set(64, 64, sizeof(float), 1);
  auto &sbs = lanes[0];
  auto *partials = static_cast<float *>(sbs.loss_reduction.data);
  ASSERT_NE(partials, nullptr);
  partials[0] = 1.0f;

  EXPECT_THROW((void)store.finalize_async_step(sbs, 1, 0),
               std::invalid_argument);
}

TEST(ParameterStore, FinalizeAsyncSyncsConfigMlp) {
  BufferArena arena;
  auto desc = tiny_desc();
  ParameterStore store(desc, arena);

  // Hydrate with MLP data.
  std::vector<float> mlp_data(256, 0.0f);
  mlp_data[0] = 42.0f;
  mlp_data[100] = -7.0f;
  store.hydrate_weights(nullptr, 0, mlp_data.data(), mlp_data.size());

  // Finalize triggers MLP → config_weights sync.
  StepBufferSet empty_sbs{};
  (void)store.finalize_async_step(empty_sbs, 0, 0);

  // Verify config MLP section matches mlp_weights.
  auto *cfg = static_cast<const float *>(store.config_weights().data);
  ASSERT_NE(cfg, nullptr);
  EXPECT_FLOAT_EQ(cfg[kConfigPackedFloats], 42.0f);
  EXPECT_FLOAT_EQ(cfg[kConfigPackedFloats + 100], -7.0f);
}

// --- sync_live_weights ---

TEST(ParameterStore, SyncLiveWeightsUpdatesBuffers) {
  BufferArena arena;
  ParameterStore store(tiny_desc(), arena);

  // 1. Hydrate with initial data.
  std::vector<float> init_hash(1024, 1.0f);
  std::vector<float> init_mlp(256, 2.0f);
  store.hydrate_weights(init_hash.data(), init_hash.size(),
                        init_mlp.data(), init_mlp.size());

  // Verify initial state.
  auto *hp = static_cast<const float *>(store.hash_weights().data);
  auto *mp = static_cast<const float *>(store.mlp_weights().data);
  EXPECT_FLOAT_EQ(hp[0], 1.0f);
  EXPECT_FLOAT_EQ(mp[0], 2.0f);

  // 2. Simulate training: mutate the "live" weights.
  std::vector<float> trained_hash(1024, 10.0f);
  trained_hash[500] = 99.0f;
  std::vector<float> trained_mlp(256, 20.0f);
  trained_mlp[0] = -5.0f;

  // 3. Sync live weights into ParameterStore.
  store.sync_live_weights(trained_hash.data(), trained_hash.size(),
                          trained_mlp.data(), trained_mlp.size());

  // 4. Verify ParameterStore now holds trained data, not init data.
  EXPECT_FLOAT_EQ(hp[0], 10.0f);
  EXPECT_FLOAT_EQ(hp[500], 99.0f);
  EXPECT_FLOAT_EQ(mp[0], -5.0f);
  EXPECT_FLOAT_EQ(mp[100], 20.0f);
}

// --- Fused contiguous buffers ---

TEST(ParameterStore, FusedWeightsContiguous) {
  BufferArena arena;
  ParameterStoreDesc desc;
  desc.hash_grid_size = 1024;
  desc.mlp_weight_count = 256;
  desc.use_fused_adam = true;
  ParameterStore store(desc, arena);

  // Fused view spans full [hash|mlp].
  EXPECT_EQ(store.fused_weights().bytes, (1024 + 256) * sizeof(float));
  EXPECT_TRUE(store.is_fused());

  // hash_weights is sub-view of fused_weights (same data pointer, offset=0).
  EXPECT_EQ(store.hash_weights().data, store.fused_weights().data);
  EXPECT_EQ(store.hash_weights().bytes, 1024 * sizeof(float));

  // mlp_weights is sub-view at hash_bytes offset.
  EXPECT_EQ(store.mlp_weights().data,
            static_cast<char *>(store.fused_weights().data) +
                1024 * sizeof(float));

  // Fused m/v also contiguous.
  EXPECT_EQ(store.fused_m().bytes, (1024 + 256) * sizeof(float));
  EXPECT_EQ(store.fused_v().bytes, (1024 + 256) * sizeof(float));

  // Grad buffers remain separate (different types).
  EXPECT_NE(store.grad_hash().handle, store.grad_mlp().handle);
}

TEST(ParameterStore, FusedHydrateRoundTrip) {
  BufferArena arena;
  ParameterStoreDesc desc;
  desc.hash_grid_size = 4;
  desc.mlp_weight_count = 2;
  desc.use_fused_adam = true;
  ParameterStore store(desc, arena);

  float hash[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float mlp[] = {5.0f, 6.0f};
  store.hydrate_weights(hash, 4, mlp, 2);

  // Read through fused view: [1,2,3,4,5,6].
  auto *fw = static_cast<float *>(store.fused_weights().data);
  EXPECT_FLOAT_EQ(fw[0], 1.0f);
  EXPECT_FLOAT_EQ(fw[3], 4.0f);
  EXPECT_FLOAT_EQ(fw[4], 5.0f); // mlp starts here
  EXPECT_FLOAT_EQ(fw[5], 6.0f);
}

TEST(ParameterStore, NonFusedHasEmptyFusedViews) {
  BufferArena arena;
  ParameterStore store(tiny_desc(), arena);

  EXPECT_FALSE(store.is_fused());
  EXPECT_EQ(store.fused_weights().bytes, 0u);
  EXPECT_EQ(store.fused_m().bytes, 0u);
  EXPECT_EQ(store.fused_v().bytes, 0u);
}

TEST(ParameterStore, ResetAdamStateFused) {
  BufferArena arena;
  ParameterStoreDesc desc;
  desc.hash_grid_size = 4;
  desc.mlp_weight_count = 2;
  desc.use_fused_adam = true;
  ParameterStore store(desc, arena);

  // Write non-zero values into fused m/v.
  auto *m = static_cast<float *>(store.fused_m().data);
  auto *v = static_cast<float *>(store.fused_v().data);
  ASSERT_NE(m, nullptr);
  ASSERT_NE(v, nullptr);
  for (int i = 0; i < 6; ++i) {
    m[i] = static_cast<float>(i + 1);
    v[i] = static_cast<float>(i + 10);
  }

  store.reset_adam_state();

  // All m/v should be zero.
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(m[i], 0.0f) << "m[" << i << "]";
    EXPECT_FLOAT_EQ(v[i], 0.0f) << "v[" << i << "]";
  }
}

TEST(ParameterStore, ResetAdamStateSeparate) {
  BufferArena arena;
  ParameterStoreDesc desc;
  desc.hash_grid_size = 4;
  desc.mlp_weight_count = 2;
  desc.use_fused_adam = false;
  ParameterStore store(desc, arena);

  // Write non-zero values.
  auto *mh = static_cast<float *>(store.adam_m_hash().data);
  auto *vh = static_cast<float *>(store.adam_v_hash().data);
  auto *mm = static_cast<float *>(store.adam_m_mlp().data);
  auto *vm = static_cast<float *>(store.adam_v_mlp().data);
  for (int i = 0; i < 4; ++i) mh[i] = 1.0f;
  for (int i = 0; i < 4; ++i) vh[i] = 2.0f;
  for (int i = 0; i < 2; ++i) mm[i] = 3.0f;
  for (int i = 0; i < 2; ++i) vm[i] = 4.0f;

  store.reset_adam_state();

  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(mh[i], 0.0f);
    EXPECT_FLOAT_EQ(vh[i], 0.0f);
  }
  for (int i = 0; i < 2; ++i) {
    EXPECT_FLOAT_EQ(mm[i], 0.0f);
    EXPECT_FLOAT_EQ(vm[i], 0.0f);
  }
}

TEST(ParameterStore, SyncThenFinalizePropagatesToConfigWeights) {
  BufferArena arena;
  ParameterStore store(tiny_desc(), arena);

  // 1. Hydrate with init data.
  std::vector<float> init_mlp(256, 0.0f);
  store.hydrate_weights(nullptr, 0, init_mlp.data(), init_mlp.size());

  // config_weights MLP section should be zeroed after hydrate.
  auto *cfg = static_cast<const float *>(store.config_weights().data);
  EXPECT_FLOAT_EQ(cfg[kConfigPackedFloats], 0.0f);

  // 2. Sync live trained MLP weights (simulating post-trainStepGPU).
  std::vector<float> trained_mlp(256, 0.0f);
  trained_mlp[0] = 77.0f;
  trained_mlp[255] = -33.0f;
  store.sync_live_weights(nullptr, 0, trained_mlp.data(), trained_mlp.size());

  // 3. config_weights MLP section is still stale (not updated by sync).
  EXPECT_FLOAT_EQ(cfg[kConfigPackedFloats], 0.0f);

  // 4. Finalize propagates live mlp_weights → config_weights.
  StepBufferSet empty_sbs{};
  (void)store.finalize_async_step(empty_sbs, 0, 0);

  // 5. Now config_weights reflects trained data.
  EXPECT_FLOAT_EQ(cfg[kConfigPackedFloats], 77.0f);
  EXPECT_FLOAT_EQ(cfg[kConfigPackedFloats + 255], -33.0f);
}

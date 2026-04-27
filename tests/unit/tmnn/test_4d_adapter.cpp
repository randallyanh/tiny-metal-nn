/**
 * @file test_4d_adapter.cpp
 * @brief Tests for FourDAdapter — pure SDK-level, CPU-only.
 */

#include <gtest/gtest.h>

#include "tiny-metal-nn/extension/four_d_adapter.h"

#include <cstring>
#include <vector>

using namespace tmnn;
using namespace tmnn::extension;

TEST(FourDAdapter, SchemaInputDims4) {
  FourDAdapter::Config cfg;
  FourDAdapter adapter(cfg);

  auto s = adapter.schema();
  EXPECT_EQ(s.input_dims, 4u);
  EXPECT_EQ(s.target_dims, 1u);
  EXPECT_EQ(s.reduction_terms, 1u);
  EXPECT_EQ(s.config_tail_floats, 0u);
}

TEST(FourDAdapter, CompileSpecNoSIMD) {
  FourDAdapter::Config cfg;
  FourDAdapter adapter(cfg);

  KernelCompileSpec spec;
  adapter.configure_compile_spec(spec);
  EXPECT_FALSE(spec.allow_simd);
  EXPECT_TRUE(spec.allow_tg_weight_cache);
}

TEST(FourDAdapter, CompileSpecTracksTgWeightCachePreference) {
  FourDAdapter::Config cfg;
  cfg.allow_tg_weight_cache = false;
  FourDAdapter adapter(cfg);

  KernelCompileSpec spec;
  adapter.configure_compile_spec(spec);
  EXPECT_FALSE(spec.allow_tg_weight_cache);
}

TEST(FourDAdapter, FillTrainParamsProgressive) {
  FourDAdapter::Config cfg;
  cfg.initial_active_levels = 4;
  cfg.level_activation_interval = 500;
  FourDAdapter adapter(cfg);

  auto s = adapter.schema();
  std::vector<float> buf(s.train_params_layout.float_count, -1.0f);

  // Step 0: active_levels = min(4 + 1/500, 16) = 4
  adapter.fill_train_params(buf.data(), s.train_params_layout, 512, 0);
  EXPECT_FLOAT_EQ(buf[s.train_params_layout.idx_num_active_levels], 4.0f);
  EXPECT_FLOAT_EQ(buf[s.train_params_layout.idx_n], 512.0f);

  // Step 999: active_levels = min(4 + 1000/500, 16) = 6
  adapter.fill_train_params(buf.data(), s.train_params_layout, 512, 999);
  EXPECT_FLOAT_EQ(buf[s.train_params_layout.idx_num_active_levels], 6.0f);
}

TEST(FourDAdapter, PackBatch4D) {
  FourDAdapter::Config cfg;
  FourDAdapter adapter(cfg);

  const int N = 2;
  std::vector<float> input = {1, 2, 3, 0.5f, 4, 5, 6, 0.7f};
  std::vector<float> target = {-0.1f, 0.2f};
  std::vector<float> pos_out(N * 4);
  std::vector<float> tgt_out(N);

  adapter.pack_batch(input.data(), target.data(), N,
                     pos_out.data(), tgt_out.data());

  EXPECT_EQ(pos_out, input);
  EXPECT_EQ(tgt_out, target);
}

TEST(FourDAdapter, AdamConfigDefaults) {
  FourDAdapter::Config cfg;
  FourDAdapter adapter(cfg);

  auto adam = adapter.adam_config(1);
  EXPECT_FLOAT_EQ(adam.lr_encoding, 1e-2f);
  EXPECT_FLOAT_EQ(adam.lr_network, 1e-3f);
  EXPECT_FLOAT_EQ(adam.weight_decay, 0.0f);
  EXPECT_FLOAT_EQ(adam.beta1, 0.9f);
  EXPECT_FLOAT_EQ(adam.beta2, 0.99f);
}

TEST(FourDAdapter, AdamConfigTracksCustomOptimizerFields) {
  FourDAdapter::Config cfg;
  cfg.beta1 = 0.85f;
  cfg.beta2 = 0.97f;
  cfg.epsilon = 1e-12f;
  cfg.l1_reg = 1e-4f;
  cfg.l2_reg = 2e-4f;
  FourDAdapter adapter(cfg);

  auto adam = adapter.adam_config(1);
  EXPECT_FLOAT_EQ(adam.beta1, 0.85f);
  EXPECT_FLOAT_EQ(adam.beta2, 0.97f);
  EXPECT_FLOAT_EQ(adam.epsilon, 1e-12f);
  EXPECT_FLOAT_EQ(adam.l1_reg, 1e-4f);
  EXPECT_FLOAT_EQ(adam.l2_reg, 2e-4f);
}

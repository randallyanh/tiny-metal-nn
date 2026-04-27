/**
 * @file test_standard_sdf_adapter.cpp
 * @brief Tests for StandardSDFAdapter — pure SDK-level, CPU-only.
 */

#include <gtest/gtest.h>

#include "tiny-metal-nn/extension/standard_sdf_adapter.h"

#include <vector>

using namespace tmnn;
using namespace tmnn::extension;

TEST(StandardSDFAdapter, SchemaDefaultsToStandardScalarSdf) {
  StandardSDFAdapter adapter(StandardSDFAdapter::Config{});

  auto s = adapter.schema();
  EXPECT_EQ(s.input_dims, 3u);
  EXPECT_EQ(s.target_dims, 1u);
  EXPECT_EQ(s.reduction_terms, 1u);
  EXPECT_EQ(s.config_tail_floats, 0u);
  EXPECT_EQ(s.train_params_layout.float_count, 8u);
}

TEST(StandardSDFAdapter, CompileSpecTracksPlannerPreferences) {
  StandardSDFAdapter::Config cfg;
  cfg.allow_simd = false;
  cfg.allow_fp16 = false;
  cfg.allow_tg_weight_cache = false;
  StandardSDFAdapter adapter(cfg);

  KernelCompileSpec spec;
  adapter.configure_compile_spec(spec);
  EXPECT_EQ(spec.encoding, KernelEncoding::Standard);
  EXPECT_FALSE(spec.allow_simd);
  EXPECT_FALSE(spec.allow_fp16);
  EXPECT_FALSE(spec.allow_tg_weight_cache);
}

TEST(StandardSDFAdapter, FillTrainParamsProgressive) {
  StandardSDFAdapter::Config cfg;
  cfg.initial_active_levels = 4;
  cfg.level_activation_interval = 500;
  cfg.allow_fp16 = false;
  StandardSDFAdapter adapter(cfg);

  auto s = adapter.schema();
  std::vector<float> buf(s.train_params_layout.float_count, -1.0f);
  adapter.fill_train_params(buf.data(), s.train_params_layout, 512, 0);
  EXPECT_FLOAT_EQ(buf[s.train_params_layout.idx_n], 512.0f);
  EXPECT_FLOAT_EQ(buf[s.train_params_layout.idx_loss_scale], 1.0f);
  EXPECT_FLOAT_EQ(buf[s.train_params_layout.idx_num_active_levels], 4.0f);

  adapter.fill_train_params(buf.data(), s.train_params_layout, 512, 999);
  EXPECT_FLOAT_EQ(buf[s.train_params_layout.idx_num_active_levels], 6.0f);
}

TEST(StandardSDFAdapter, AdamConfigPreservesTrainerHyperparameters) {
  StandardSDFAdapter::Config cfg;
  cfg.lr_encoding = 5e-3f;
  cfg.lr_network = 2e-4f;
  cfg.beta1 = 0.85f;
  cfg.beta2 = 0.97f;
  cfg.epsilon = 1e-12f;
  cfg.l1_reg = 1e-4f;
  cfg.l2_reg = 2e-4f;
  cfg.grad_clip = 0.5f;
  StandardSDFAdapter adapter(cfg);

  auto adam = adapter.adam_config(1);
  EXPECT_FLOAT_EQ(adam.lr_encoding, 5e-3f);
  EXPECT_FLOAT_EQ(adam.lr_network, 2e-4f);
  EXPECT_FLOAT_EQ(adam.beta1, 0.85f);
  EXPECT_FLOAT_EQ(adam.beta2, 0.97f);
  EXPECT_FLOAT_EQ(adam.epsilon, 1e-12f);
  EXPECT_FLOAT_EQ(adam.l1_reg, 1e-4f);
  EXPECT_FLOAT_EQ(adam.l2_reg, 2e-4f);
  EXPECT_FLOAT_EQ(adam.grad_clip, 0.5f);
}

TEST(StandardSDFAdapter, AdamConfigAppliesLearningRateDecay) {
  StandardSDFAdapter::Config cfg;
  cfg.lr_encoding = 1e-2f;
  cfg.lr_network = 1e-3f;
  cfg.lr_decay = 0.5f;
  cfg.lr_decay_step = 10;
  StandardSDFAdapter adapter(cfg);

  auto step1 = adapter.adam_config(1);
  auto step11 = adapter.adam_config(11);
  auto step21 = adapter.adam_config(21);

  EXPECT_FLOAT_EQ(step1.lr_encoding, 1e-2f);
  EXPECT_FLOAT_EQ(step1.lr_network, 1e-3f);
  EXPECT_FLOAT_EQ(step11.lr_encoding, 5e-3f);
  EXPECT_FLOAT_EQ(step11.lr_network, 5e-4f);
  EXPECT_FLOAT_EQ(step21.lr_encoding, 2.5e-3f);
  EXPECT_FLOAT_EQ(step21.lr_network, 2.5e-4f);
}

TEST(StandardSDFAdapter, PackBatchIdentityCopy) {
  StandardSDFAdapter adapter(StandardSDFAdapter::Config{});

  const int N = 2;
  std::vector<float> input = {1, 2, 3, 4, 5, 6};
  std::vector<float> target = {0.1f, 0.2f};
  std::vector<float> pos_out(N * 3);
  std::vector<float> tgt_out(N);

  adapter.pack_batch(input.data(), target.data(), N, pos_out.data(),
                     tgt_out.data());

  EXPECT_EQ(pos_out, input);
  EXPECT_EQ(tgt_out, target);
}

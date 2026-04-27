/**
 * @file test_multi_output_mlp_adapter.cpp
 * @brief Tests for MultiOutputMLPAdapter — pure SDK-level, CPU-only.
 */

#include <gtest/gtest.h>

#include "tiny-metal-nn/extension/multi_output_mlp_adapter.h"

#include <vector>

using namespace tmnn;
using namespace tmnn::extension;

TEST(MultiOutputMLPAdapter, SchemaMatchesOutputDims) {
  MultiOutputMLPAdapter::Config cfg;
  cfg.num_outputs = 3;
  MultiOutputMLPAdapter adapter(cfg);

  auto s = adapter.schema();
  EXPECT_EQ(s.input_dims, 3u);
  EXPECT_EQ(s.target_dims, 3u);
  EXPECT_EQ(s.reduction_terms, 1u);
  EXPECT_EQ(s.train_params_layout.float_count, 5u);
}

TEST(MultiOutputMLPAdapter, CompileSpecDisablesSIMDWithoutDnlSemantics) {
  MultiOutputMLPAdapter::Config cfg;
  cfg.num_outputs = 4;
  cfg.use_fp16 = false;
  cfg.allow_tg_weight_cache = false;
  MultiOutputMLPAdapter adapter(cfg);

  KernelCompileSpec spec;
  adapter.configure_compile_spec(spec);
  EXPECT_FALSE(spec.allow_simd);
  EXPECT_FALSE(spec.allow_fp16);
  EXPECT_FALSE(spec.allow_tg_weight_cache);
  EXPECT_EQ(spec.output_semantics, KernelOutputSemantics::Generic);
  EXPECT_EQ(spec.bc_dim_count, 0u);
}

TEST(MultiOutputMLPAdapter, AdamConfigPreservesTrainerSemantics) {
  MultiOutputMLPAdapter::Config cfg;
  cfg.num_outputs = 5;
  cfg.lr_encoding = 5e-3f;
  cfg.lr_network = 2e-4f;
  cfg.beta1 = 0.85f;
  cfg.beta2 = 0.97f;
  cfg.epsilon = 1e-10f;
  cfg.l1_reg = 0.01f;
  cfg.l2_reg = 0.02f;
  cfg.grad_clip = 0.25f;
  MultiOutputMLPAdapter adapter(cfg);

  auto adam = adapter.adam_config(1);
  EXPECT_FLOAT_EQ(adam.lr_encoding, 5e-3f);
  EXPECT_FLOAT_EQ(adam.lr_network, 2e-4f);
  EXPECT_FLOAT_EQ(adam.beta1, 0.85f);
  EXPECT_FLOAT_EQ(adam.beta2, 0.97f);
  EXPECT_FLOAT_EQ(adam.epsilon, 1e-10f);
  EXPECT_FLOAT_EQ(adam.l1_reg, 0.01f);
  EXPECT_FLOAT_EQ(adam.l2_reg, 0.02f);
  EXPECT_FLOAT_EQ(adam.grad_clip, 0.25f);
  EXPECT_FLOAT_EQ(adam.weight_decay, 0.0f);
}

TEST(MultiOutputMLPAdapter, FillTrainParamsUsesConfiguredLossScale) {
  MultiOutputMLPAdapter::Config cfg;
  cfg.num_outputs = 2;
  cfg.use_fp16 = true;
  cfg.loss_scale = 64.0f;
  cfg.initial_active_levels = 12;
  MultiOutputMLPAdapter adapter(cfg);

  auto s = adapter.schema();
  std::vector<float> buf(s.train_params_layout.float_count, -1.0f);
  adapter.fill_train_params(buf.data(), s.train_params_layout, 128, 0);

  EXPECT_FLOAT_EQ(buf[s.train_params_layout.idx_n], 128.0f);
  EXPECT_FLOAT_EQ(buf[s.train_params_layout.idx_loss_scale], 64.0f);
  EXPECT_FLOAT_EQ(buf[s.train_params_layout.idx_num_active_levels], 12.0f);
}

TEST(MultiOutputMLPAdapter, LossConfigUsesConfiguredFamily) {
  MultiOutputMLPAdapter::Config cfg;
  cfg.num_outputs = 32;
  cfg.loss_kind = LossKind::Cosine;
  MultiOutputMLPAdapter adapter(cfg);

  auto loss = adapter.loss_config();
  EXPECT_EQ(loss.kind, LossKind::Cosine);
  EXPECT_FLOAT_EQ(loss.huber_delta, 1.0f);
}

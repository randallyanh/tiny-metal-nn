/**
 * @file test_dnl_adapter.cpp
 * @brief Tests for DNLAdapter — pure SDK-level, CPU-only.
 */

#include <gtest/gtest.h>

#include "tiny-metal-nn/extension/dnl_adapter.h"

#include <cstring>
#include <vector>

using namespace tmnn;
using namespace tmnn::extension;

TEST(DNLAdapter, SchemaTargetDims) {
  DNLAdapter::Config cfg;
  cfg.num_outputs = 4;
  DNLAdapter adapter(cfg);

  auto s = adapter.schema();
  EXPECT_EQ(s.target_dims, 4u);
  EXPECT_EQ(s.input_dims, 3u);
  EXPECT_EQ(s.reduction_terms, 3u); // total + bc + piezo
  EXPECT_EQ(s.config_tail_floats, 0u);
  EXPECT_EQ(s.train_params_layout.float_count, 5u);
}

TEST(DNLAdapter, AdamConfigWeightDecay) {
  DNLAdapter::Config cfg;
  cfg.weight_decay = 0.01f;
  cfg.lr_encoding = 5e-3f;
  cfg.lr_network = 2e-4f;
  cfg.beta1 = 0.85f;
  cfg.beta2 = 0.97f;
  cfg.epsilon = 1e-10f;
  cfg.l1_reg = 0.02f;
  cfg.l2_reg = 0.03f;
  cfg.grad_clip = 0.25f;
  DNLAdapter adapter(cfg);

  auto adam = adapter.adam_config(1);
  EXPECT_FLOAT_EQ(adam.weight_decay, 0.01f);
  EXPECT_FLOAT_EQ(adam.lr_encoding, 5e-3f);
  EXPECT_FLOAT_EQ(adam.lr_network, 2e-4f);
  EXPECT_FLOAT_EQ(adam.beta1, 0.85f);
  EXPECT_FLOAT_EQ(adam.beta2, 0.97f);
  EXPECT_FLOAT_EQ(adam.epsilon, 1e-10f);
  EXPECT_FLOAT_EQ(adam.l1_reg, 0.02f);
  EXPECT_FLOAT_EQ(adam.l2_reg, 0.03f);
  EXPECT_FLOAT_EQ(adam.grad_clip, 0.25f);
}

TEST(DNLAdapter, FillTrainParams) {
  DNLAdapter::Config cfg;
  cfg.loss_scale = 64.0f;
  cfg.use_fp16 = true;
  cfg.initial_active_levels = 16;
  DNLAdapter adapter(cfg);

  auto s = adapter.schema();
  std::vector<float> buf(s.train_params_layout.float_count, -1.0f);
  adapter.fill_train_params(buf.data(), s.train_params_layout, 256, 0);

  EXPECT_FLOAT_EQ(buf[s.train_params_layout.idx_n], 256.0f);
  EXPECT_FLOAT_EQ(buf[s.train_params_layout.idx_unsigned_mode], 0.0f);
  EXPECT_FLOAT_EQ(buf[s.train_params_layout.idx_loss_scale], 64.0f);
  EXPECT_FLOAT_EQ(buf[s.train_params_layout.idx_num_active_levels], 16.0f);
}

TEST(DNLAdapter, CompileSpecNoSIMD) {
  DNLAdapter::Config cfg;
  cfg.bc_dims = 3;
  cfg.allow_tg_weight_cache = false;
  DNLAdapter adapter(cfg);

  KernelCompileSpec spec;
  adapter.configure_compile_spec(spec);
  EXPECT_FALSE(spec.allow_simd);
  EXPECT_FALSE(spec.allow_tg_weight_cache);
  EXPECT_EQ(spec.output_semantics, KernelOutputSemantics::DNL);
  EXPECT_EQ(spec.bc_dim_count, 3u);
}

TEST(DNLAdapter, PackBatchIdentityCopy) {
  DNLAdapter::Config cfg;
  cfg.num_outputs = 4;
  DNLAdapter adapter(cfg);

  const int N = 2;
  std::vector<float> input = {1, 2, 3, 4, 5, 6};
  std::vector<float> target = {0.1f, 0.2f, 0.3f, 0.4f,
                               0.5f, 0.6f, 0.7f, 0.8f};
  std::vector<float> pos_out(N * 3);
  std::vector<float> tgt_out(N * 4);

  adapter.pack_batch(input.data(), target.data(), N,
                     pos_out.data(), tgt_out.data());

  EXPECT_EQ(pos_out, input);
  EXPECT_EQ(tgt_out, target);
}

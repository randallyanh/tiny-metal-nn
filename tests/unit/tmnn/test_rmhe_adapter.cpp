/**
 * @file test_rmhe_adapter.cpp
 * @brief Tests for RMHEAdapter — pure SDK-level, CPU-only.
 */

#include <gtest/gtest.h>

#include "tiny-metal-nn/extension/rmhe_adapter.h"

#include <cstring>
#include <vector>

using namespace tmnn;
using namespace tmnn::extension;

namespace {
// Fake rotation data: 16 identity 3x3 matrices.
void make_identity_rotations(float *out) {
  for (int level = 0; level < 16; ++level) {
    float *m = out + level * 9;
    std::memset(m, 0, 9 * sizeof(float));
    m[0] = m[4] = m[8] = 1.0f;
  }
}
} // namespace

TEST(RMHEAdapter, SchemaConfigTail144) {
  RMHEAdapter::Config cfg;
  float rotations[144];
  make_identity_rotations(rotations);
  RMHEAdapter adapter(cfg, rotations);

  auto s = adapter.schema();
  EXPECT_EQ(s.config_tail_floats, 144u);
  EXPECT_EQ(s.input_dims, 3u);
  EXPECT_EQ(s.target_dims, 1u);
  EXPECT_EQ(s.reduction_terms, 1u);
  EXPECT_EQ(s.train_params_layout.float_count, 4u);
}

TEST(RMHEAdapter, CompileSpecRMHE) {
  RMHEAdapter::Config cfg;
  float rotations[144];
  make_identity_rotations(rotations);
  RMHEAdapter adapter(cfg, rotations);

  KernelCompileSpec spec;
  adapter.configure_compile_spec(spec);
  EXPECT_EQ(spec.encoding, KernelEncoding::RMHE);
  EXPECT_FALSE(spec.allow_simd);
  EXPECT_TRUE(spec.allow_tg_weight_cache);
}

TEST(RMHEAdapter, CompileSpecTracksTgWeightCachePreference) {
  RMHEAdapter::Config cfg;
  cfg.allow_tg_weight_cache = false;
  float rotations[144];
  make_identity_rotations(rotations);
  RMHEAdapter adapter(cfg, rotations);

  KernelCompileSpec spec;
  adapter.configure_compile_spec(spec);
  EXPECT_FALSE(spec.allow_tg_weight_cache);
}

TEST(RMHEAdapter, PackConfigTailRoundTrip) {
  RMHEAdapter::Config cfg;
  float rotations[144];
  make_identity_rotations(rotations);
  RMHEAdapter adapter(cfg, rotations);

  std::vector<float> tail(144, -1.0f);
  adapter.pack_config_tail(tail.data());

  // Verify identity matrices were packed.
  for (int level = 0; level < 16; ++level) {
    EXPECT_FLOAT_EQ(tail[level * 9 + 0], 1.0f); // m[0][0]
    EXPECT_FLOAT_EQ(tail[level * 9 + 4], 1.0f); // m[1][1]
    EXPECT_FLOAT_EQ(tail[level * 9 + 8], 1.0f); // m[2][2]
    EXPECT_FLOAT_EQ(tail[level * 9 + 1], 0.0f); // off-diagonal
  }
}

TEST(RMHEAdapter, AdamConfigNoWeightDecay) {
  RMHEAdapter::Config cfg;
  cfg.l1_reg = 1e-4f;
  cfg.l2_reg = 1e-5f;
  float rotations[144];
  make_identity_rotations(rotations);
  RMHEAdapter adapter(cfg, rotations);

  auto adam = adapter.adam_config(1);
  EXPECT_FLOAT_EQ(adam.weight_decay, 0.0f);
  EXPECT_FLOAT_EQ(adam.l1_reg, 1e-4f);
  EXPECT_FLOAT_EQ(adam.l2_reg, 1e-5f);
}

TEST(RMHEAdapter, AdamConfigTracksCustomOptimizerFields) {
  RMHEAdapter::Config cfg;
  cfg.beta1 = 0.85f;
  cfg.beta2 = 0.97f;
  cfg.epsilon = 1e-12f;
  float rotations[144];
  make_identity_rotations(rotations);
  RMHEAdapter adapter(cfg, rotations);

  auto adam = adapter.adam_config(1);
  EXPECT_FLOAT_EQ(adam.beta1, 0.85f);
  EXPECT_FLOAT_EQ(adam.beta2, 0.97f);
  EXPECT_FLOAT_EQ(adam.epsilon, 1e-12f);
}

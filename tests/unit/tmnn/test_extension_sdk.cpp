/**
 * @file test_extension_sdk.cpp
 * @brief Tests for the extension SDK surface types.
 *
 * IMPORTANT: This file must compile with ONLY installed SDK headers —
 * no src/tiny_metal_nn/runtime/ headers allowed.
 */

#include <gtest/gtest.h>

#include "tiny-metal-nn/extension/schema.h"
#include "tiny-metal-nn/extension/kernel_compile_spec.h"
#include "tiny-metal-nn/extension/training_adapter.h"

#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// TrainParamsLayout (tmnn:: namespace)
// ---------------------------------------------------------------------------

TEST(ExtensionSDK, TrainParamsLayoutDefaults) {
  tmnn::TrainParamsLayout layout;
  EXPECT_EQ(layout.float_count, 8u);
  EXPECT_EQ(layout.idx_n, 0u);
  EXPECT_EQ(layout.idx_unsigned_mode, 1u);
  EXPECT_EQ(layout.idx_loss_scale, 2u);
  EXPECT_EQ(layout.idx_num_active_levels, 3u);
  EXPECT_NO_THROW(layout.validate());
}

TEST(ExtensionSDK, TrainParamsLayoutCustomDNL) {
  tmnn::TrainParamsLayout layout;
  layout.float_count = 5;
  layout.idx_n = 0;
  layout.idx_unsigned_mode = 1;
  layout.idx_loss_scale = 2;
  layout.idx_num_active_levels = 3;
  EXPECT_NO_THROW(layout.validate());
  EXPECT_EQ(layout.float_count, 5u);
}

TEST(ExtensionSDK, TrainParamsLayoutRejectsDuplicateIndices) {
  tmnn::TrainParamsLayout layout;
  layout.idx_n = 0;
  layout.idx_unsigned_mode = 0;
  EXPECT_THROW(layout.validate(), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// fill_train_params (header-only, custom layout)
// ---------------------------------------------------------------------------

TEST(ExtensionSDK, FillTrainParamsCustomLayout) {
  // Non-default layout: swap idx_n and idx_loss_scale.
  tmnn::TrainParamsLayout layout;
  layout.float_count = 6;
  layout.idx_n = 4;
  layout.idx_unsigned_mode = 1;
  layout.idx_loss_scale = 0;
  layout.idx_num_active_levels = 3;
  EXPECT_NO_THROW(layout.validate());

  std::vector<float> buf(6, -1.0f);
  tmnn::fill_train_params(buf.data(), layout, /*N=*/128, /*unsigned_mode=*/true,
                          /*loss_scale=*/64.0f, /*num_active_levels=*/12);

  EXPECT_FLOAT_EQ(buf[4], 128.0f);  // idx_n
  EXPECT_FLOAT_EQ(buf[1], 1.0f);    // idx_unsigned_mode
  EXPECT_FLOAT_EQ(buf[0], 64.0f);   // idx_loss_scale
  EXPECT_FLOAT_EQ(buf[3], 12.0f);   // idx_num_active_levels
  EXPECT_FLOAT_EQ(buf[2], 0.0f);    // zeroed by memset
  EXPECT_FLOAT_EQ(buf[5], 0.0f);    // zeroed by memset
}

// ---------------------------------------------------------------------------
// ExtensionSchema (tmnn::extension:: namespace)
// ---------------------------------------------------------------------------

TEST(ExtensionSDK, ExtensionSchemaDefaults) {
  tmnn::extension::ExtensionSchema schema;
  EXPECT_EQ(schema.input_dims, 3u);
  EXPECT_EQ(schema.target_dims, 1u);
  EXPECT_EQ(schema.reduction_terms, 1u);
  EXPECT_EQ(schema.config_tail_floats, 0u);
  EXPECT_EQ(schema.train_params_layout.float_count, tmnn::kTrainParamFloats);
  EXPECT_NO_THROW(schema.validate());
}

TEST(ExtensionSDK, ExtensionSchemaRejectsInvalidInputDims) {
  tmnn::extension::ExtensionSchema schema;
  schema.input_dims = 2;
  EXPECT_THROW(schema.validate(), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// KernelCompileSpec (tmnn::extension:: namespace)
// ---------------------------------------------------------------------------

TEST(ExtensionSDK, KernelCompileSpecDefaults) {
  tmnn::extension::KernelCompileSpec spec;
  EXPECT_EQ(spec.encoding, tmnn::extension::KernelEncoding::Standard);
  EXPECT_TRUE(spec.allow_simd);
  EXPECT_TRUE(spec.allow_fp16);
  EXPECT_TRUE(spec.allow_tg_weight_cache);
  EXPECT_EQ(spec.output_semantics, tmnn::extension::KernelOutputSemantics::Generic);
}

TEST(ExtensionSDK, KernelCompileSpecRejectsRMHEForNon3D) {
  tmnn::extension::ExtensionSchema schema;
  schema.input_dims = 4;
  tmnn::extension::KernelCompileSpec spec;
  spec.encoding = tmnn::extension::KernelEncoding::RMHE;
  EXPECT_THROW(spec.validate(schema), std::invalid_argument);
}

TEST(ExtensionSDK, KernelCompileSpecRejectsInvalidBcDimCount) {
  tmnn::extension::ExtensionSchema schema;
  schema.target_dims = 4;
  schema.reduction_terms = 3;

  tmnn::extension::KernelCompileSpec spec;
  spec.bc_dim_count = 4;
  EXPECT_THROW(spec.validate(schema), std::invalid_argument);

  spec.bc_dim_count = 2;
  EXPECT_NO_THROW(spec.validate(schema));
}

TEST(ExtensionSDK, KernelCompileSpecRejectsInvalidDnlSemanticsShape) {
  tmnn::extension::ExtensionSchema schema;
  schema.input_dims = 3;
  schema.target_dims = 3;

  tmnn::extension::KernelCompileSpec spec;
  spec.output_semantics = tmnn::extension::KernelOutputSemantics::DNL;
  EXPECT_THROW(spec.validate(schema), std::invalid_argument);

  schema.target_dims = 4;
  EXPECT_NO_THROW(spec.validate(schema));

  schema.input_dims = 4;
  EXPECT_THROW(spec.validate(schema), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// configure_compile_spec: adapter patches baseline policy
// ---------------------------------------------------------------------------

namespace {

/// Mock adapter exercising the full declarative contract.
class MockRMHEAdapter : public tmnn::extension::TrainingAdapter {
public:
  tmnn::extension::ExtensionSchema schema() const override {
    tmnn::extension::ExtensionSchema s;
    s.input_dims = 3;
    s.target_dims = 1;
    s.config_tail_floats = 9; // 3x3 rotation matrix
    s.train_params_layout.float_count = 5;
    return s;
  }

  void configure_compile_spec(
      tmnn::extension::KernelCompileSpec &spec) const override {
    spec.encoding = tmnn::extension::KernelEncoding::RMHE;
    spec.allow_fp16 = false; // RMHE needs FP32 rotations
  }

  void pack_config_tail(float *dst) const override {
    // Identity rotation.
    std::memset(dst, 0, 9 * sizeof(float));
    dst[0] = dst[4] = dst[8] = 1.0f;
  }

  void pack_batch(const float *input, const float *target, int N,
                  float *positions_out, float *targets_out) const override {
    std::memcpy(positions_out, input,
                static_cast<size_t>(N) * 3 * sizeof(float));
    std::memcpy(targets_out, target,
                static_cast<size_t>(N) * 1 * sizeof(float));
  }

  void fill_train_params(float *dst, const tmnn::TrainParamsLayout &layout,
                         uint32_t N, uint32_t /*step*/) const override {
    tmnn::fill_train_params(dst, layout, N, /*unsigned_mode=*/false,
                            /*loss_scale=*/1.0f, /*num_active_levels=*/16);
  }

  tmnn::extension::AdamConfig adam_config(uint32_t next_step) const override {
    tmnn::extension::AdamConfig cfg;
    cfg.lr_encoding = 1e-2f;
    cfg.lr_network = 1e-3f;
    cfg.weight_decay = 1e-5f;
    // Simple warmup: half LR for first 100 steps.
    if (next_step < 100) {
      cfg.lr_encoding *= 0.5f;
      cfg.lr_network *= 0.5f;
    }
    return cfg;
  }

  tmnn::extension::ResultMetrics result_metrics(float mean_loss,
                                                uint32_t /*step*/) const override {
    tmnn::extension::ResultMetrics m;
    m.extra_losses[0] = mean_loss * 0.1f; // eikonal contribution
    m.extra_loss_count = 1;
    return m;
  }
};

} // namespace

TEST(ExtensionSDK, ConfigureCompileSpecPatchesBaseline) {
  MockRMHEAdapter adapter;

  // Start with default baseline.
  tmnn::extension::KernelCompileSpec spec;
  EXPECT_EQ(spec.encoding, tmnn::extension::KernelEncoding::Standard);
  EXPECT_TRUE(spec.allow_fp16);

  // Adapter patches it.
  adapter.configure_compile_spec(spec);
  EXPECT_EQ(spec.encoding, tmnn::extension::KernelEncoding::RMHE);
  EXPECT_FALSE(spec.allow_fp16);
  // Unpatched fields remain at baseline.
  EXPECT_TRUE(spec.allow_simd);
  EXPECT_TRUE(spec.allow_tg_weight_cache);
}

// ---------------------------------------------------------------------------
// pack_config_tail: RMHE-style rotation matrix tail
// ---------------------------------------------------------------------------

TEST(ExtensionSDK, PackConfigTailRMHE) {
  MockRMHEAdapter adapter;
  auto s = adapter.schema();
  EXPECT_EQ(s.config_tail_floats, 9u);

  std::vector<float> tail(s.config_tail_floats, -1.0f);
  adapter.pack_config_tail(tail.data());

  // Identity 3x3 rotation.
  EXPECT_FLOAT_EQ(tail[0], 1.0f);
  EXPECT_FLOAT_EQ(tail[4], 1.0f);
  EXPECT_FLOAT_EQ(tail[8], 1.0f);
  EXPECT_FLOAT_EQ(tail[1], 0.0f);
  EXPECT_FLOAT_EQ(tail[3], 0.0f);
}

// ---------------------------------------------------------------------------
// pack_batch: DNL-style N * target_dims
// ---------------------------------------------------------------------------

namespace {

class MockDNLAdapter : public tmnn::extension::TrainingAdapter {
public:
  tmnn::extension::ExtensionSchema schema() const override {
    tmnn::extension::ExtensionSchema s;
    s.input_dims = 3;
    s.target_dims = 4; // DNL multi-target
    return s;
  }

  void configure_compile_spec(
      tmnn::extension::KernelCompileSpec &) const override {}

  void pack_config_tail(float *) const override {}

  void pack_batch(const float *input, const float *target, int N,
                  float *positions_out, float *targets_out) const override {
    // DNL packs: interleave lattice index into positions.
    const auto in_dims = schema().input_dims;
    const auto tgt_dims = schema().target_dims;
    std::memcpy(positions_out, input,
                static_cast<size_t>(N) * in_dims * sizeof(float));
    // Target: direct copy of N * target_dims.
    std::memcpy(targets_out, target,
                static_cast<size_t>(N) * tgt_dims * sizeof(float));
  }

  void fill_train_params(float *dst, const tmnn::TrainParamsLayout &layout,
                         uint32_t N, uint32_t /*step*/) const override {
    tmnn::fill_train_params(dst, layout, N, /*unsigned_mode=*/false,
                            /*loss_scale=*/128.0f, /*num_active_levels=*/16);
  }

  tmnn::extension::AdamConfig adam_config(uint32_t) const override {
    return {};
  }

  tmnn::extension::ResultMetrics result_metrics(float, uint32_t) const override {
    return {};
  }
};

} // namespace

TEST(ExtensionSDK, PackBatchDNLMultiTarget) {
  MockDNLAdapter adapter;
  auto s = adapter.schema();
  const int N = 4;

  std::vector<float> input(N * s.input_dims);
  std::vector<float> target(N * s.target_dims);
  for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(i);
  for (size_t i = 0; i < target.size(); ++i) target[i] = static_cast<float>(i) * 0.1f;

  std::vector<float> pos_out(N * s.input_dims, -1.0f);
  std::vector<float> tgt_out(N * s.target_dims, -1.0f);

  adapter.pack_batch(input.data(), target.data(), N,
                     pos_out.data(), tgt_out.data());

  EXPECT_EQ(pos_out, input);
  EXPECT_EQ(tgt_out, target);
}

// ---------------------------------------------------------------------------
// adam_config: split LR + weight decay + warmup schedule
// ---------------------------------------------------------------------------

TEST(ExtensionSDK, AdamConfigSplitLRAndWarmup) {
  MockRMHEAdapter adapter;

  // During warmup (step < 100).
  auto cfg0 = adapter.adam_config(50);
  EXPECT_FLOAT_EQ(cfg0.lr_encoding, 5e-3f);  // 1e-2 * 0.5
  EXPECT_FLOAT_EQ(cfg0.lr_network, 5e-4f);   // 1e-3 * 0.5
  EXPECT_FLOAT_EQ(cfg0.weight_decay, 1e-5f);
  EXPECT_FLOAT_EQ(cfg0.beta1, 0.9f);
  EXPECT_FLOAT_EQ(cfg0.beta2, 0.99f);
  EXPECT_FLOAT_EQ(cfg0.epsilon, 1e-8f);
  EXPECT_FLOAT_EQ(cfg0.l1_reg, 0.0f);
  EXPECT_FLOAT_EQ(cfg0.l2_reg, 1e-6f);
  EXPECT_FLOAT_EQ(cfg0.grad_clip, 0.0f);

  // After warmup (step >= 100).
  auto cfg1 = adapter.adam_config(100);
  EXPECT_FLOAT_EQ(cfg1.lr_encoding, 1e-2f);
  EXPECT_FLOAT_EQ(cfg1.lr_network, 1e-3f);
}

// ---------------------------------------------------------------------------
// result_metrics: extra losses mapping
// ---------------------------------------------------------------------------

TEST(ExtensionSDK, ResultMetricsExtraLosses) {
  MockRMHEAdapter adapter;

  auto m = adapter.result_metrics(0.5f, 42);
  EXPECT_EQ(m.extra_loss_count, 1u);
  EXPECT_FLOAT_EQ(m.extra_losses[0], 0.05f); // 0.5 * 0.1

  auto m2 = adapter.result_metrics(1.0f, 0);
  EXPECT_FLOAT_EQ(m2.extra_losses[0], 0.1f);
}

// ---------------------------------------------------------------------------
// ResultMetrics zero-init
// ---------------------------------------------------------------------------

TEST(ExtensionSDK, ResultMetricsZeroInit) {
  tmnn::extension::ResultMetrics metrics;
  EXPECT_EQ(metrics.extra_loss_count, 0u);
  for (uint32_t i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(metrics.extra_losses[i], 0.0f);
  }
}

/**
 * @file test_tmnn_core.cpp
 * @brief Fast CPU-only tests for tmnn core types (Phase B).
 */

#include "tiny-metal-nn/cpp_api.h"
#include "tiny-metal-nn/detail/adam.h"
#include "tiny-metal-nn/common.h"
#include "tiny-metal-nn/detail/cosine_loss.h"
#include "tiny-metal-nn/encoding.h"
#include "tiny-metal-nn/factory.h"
#include "tiny-metal-nn/fully_fused_mlp.h"
#include "tiny-metal-nn/hash_grid.h"
#include "tiny-metal-nn/detail/l2_loss.h"
#include "tiny-metal-nn/detail/module.h"
#include "tiny-metal-nn/network.h"
#include "tiny-metal-nn/optimizer.h"
#include "tiny-metal-nn/trainer.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <optional>

using namespace tmnn;

// --- HashGridEncoding ---

TEST(TmnnCore, HashGridEncodingDims) {
  HashGridEncoding::Config cfg;
  HashGridEncoding enc(cfg);
  EXPECT_EQ(enc.n_input_dims(), 3);
  EXPECT_EQ(enc.n_output_dims(), 32); // 16 * 2
  EXPECT_EQ(enc.name(), "HashGridEncoding");
  // n_params = 16 * 2^19 * 2 = 16777216
  EXPECT_EQ(enc.n_params(), 16 * (1 << 19) * 2);
}

// --- FullyFusedMLP ---

TEST(TmnnCore, FullyFusedMLPParams) {
  // Default: input=32, hd=64, nhl=2, output=1
  // Layer0: 32*64 + 64 = 2112
  // Layer1: 64*64 + 64 = 4160
  // Output: 64*1 + 1   = 65
  // Total: 6337
  FullyFusedMLP mlp;
  EXPECT_EQ(mlp.n_params(), 6337);
  EXPECT_EQ(mlp.n_input_dims(), 32);
  EXPECT_EQ(mlp.n_output_dims(), 1);
}

// --- L2Loss ---

TEST(TmnnCore, L2LossCPU) {
  L2Loss loss;
  EXPECT_EQ(loss.name(), "L2");

  float pred[] = {1.0f, 2.0f, 3.0f};
  float tgt[] = {1.0f, 0.0f, 1.0f};
  // diffs: 0, 2, 2 -> squares: 0, 4, 4 -> mean: 8/3
  float val = loss.evaluate_cpu(pred, tgt, 3);
  EXPECT_NEAR(val, 8.0f / 3.0f, 1e-5f);

  // N=0 -> 0
  EXPECT_FLOAT_EQ(loss.evaluate_cpu(nullptr, nullptr, 0), 0.0f);
}

TEST(TmnnCore, CosineLossCPU) {
  CosineLoss loss;
  EXPECT_EQ(loss.name(), "Cosine");

  float same_pred[] = {1.0f, 0.0f, 0.0f};
  float same_tgt[] = {1.0f, 0.0f, 0.0f};
  EXPECT_NEAR(loss.evaluate_cpu(same_pred, same_tgt, 3), 0.0f, 1e-5f);

  float ortho_pred[] = {1.0f, 0.0f, 0.0f};
  float ortho_tgt[] = {0.0f, 1.0f, 0.0f};
  EXPECT_NEAR(loss.evaluate_cpu(ortho_pred, ortho_tgt, 3), 1.0f, 1e-5f);

  CosineLoss batched_loss(2);
  float batched_pred[] = {1.0f, 0.0f, 0.0f, 1.0f};
  float batched_tgt[] = {1.0f, 0.0f, 1.0f, 0.0f};
  EXPECT_NEAR(batched_loss.evaluate_cpu(batched_pred, batched_tgt, 4), 0.5f,
              1e-5f);
}

// --- Adam ---

TEST(TmnnCore, AdamConfig) {
  Adam adam;
  EXPECT_EQ(adam.name(), "Adam");
  EXPECT_FLOAT_EQ(adam.learning_rate(), 1e-3f);

  adam.set_learning_rate(0.01f);
  EXPECT_FLOAT_EQ(adam.learning_rate(), 0.01f);

  // Adam is a pure config holder — no step_count, increment_step, reset_step.
  EXPECT_FLOAT_EQ(adam.config().beta1, 0.9f);
  EXPECT_FLOAT_EQ(adam.config().beta2, 0.99f);
}

// --- Factory ---

TEST(TmnnCore, FactoryCreation) {
  auto enc = create_encoding();
  ASSERT_NE(enc, nullptr);
  EXPECT_EQ(enc->name(), "HashGridEncoding");

  auto net = create_network();
  ASSERT_NE(net, nullptr);
  EXPECT_EQ(net->name(), "FullyFusedMLP");

  auto loss = create_loss_l2();
  ASSERT_NE(loss, nullptr);
  EXPECT_EQ(loss->name(), "L2");

  auto cosine = create_loss_cosine(4);
  ASSERT_NE(cosine, nullptr);
  EXPECT_EQ(cosine->name(), "Cosine");

  auto opt = create_optimizer_adam();
  ASSERT_NE(opt, nullptr);
  EXPECT_EQ(opt->name(), "Adam");
}

// --- Module polymorphism ---

// --- TensorRef ---

TEST(TmnnCore, TensorRefBasics) {
  TensorShape s = TensorShape::make_2d(3, 4);
  EXPECT_EQ(s.numel(), 12u);
  EXPECT_EQ(s.rank, 2);

  TensorShape s1d = TensorShape::make_1d(7);
  EXPECT_EQ(s1d.numel(), 7u);
  EXPECT_EQ(s1d.rank, 1);

  float buf[12] = {};
  TensorRef ref;
  ref.data = buf;
  ref.shape = s;
  ref.precision = Precision::F32;
  EXPECT_EQ(ref.typed_data<float>(), buf);
}

TEST(TmnnCore, TensorRefPrecisionGuard) {
  uint16_t buf[4] = {};
  TensorRef ref;
  ref.data = buf;
  ref.shape = TensorShape::make_1d(4);
  ref.precision = Precision::F16;

  // typed_data<uint16_t>() on F16 ref should work.
  EXPECT_EQ(ref.typed_data<uint16_t>(), buf);

  // typed_data<float>() on F16 ref should assert-fail in debug.
  EXPECT_DEBUG_DEATH(ref.typed_data<float>(), "precision mismatch");
}

TEST(TmnnCore, DefaultTrainerPresetIsInteractiveSized) {
  const auto enc_cfg = default_trainer_encoding_config();
  const auto net_cfg = default_trainer_network_config(enc_cfg);
  const auto train_cfg = default_trainer_config();

  EXPECT_EQ(enc_cfg.num_levels, 16);
  EXPECT_EQ(enc_cfg.features_per_level, 2);
  EXPECT_EQ(enc_cfg.log2_hashmap_size, 14);
  EXPECT_EQ(net_cfg.hidden_dim, 32);
  EXPECT_EQ(net_cfg.n_input, enc_cfg.num_levels * enc_cfg.features_per_level);
  EXPECT_EQ(train_cfg.batch_size, 1024);

  EXPECT_LT(HashGridEncoding(enc_cfg).n_params(), HashGridEncoding().n_params());
  EXPECT_LT(FullyFusedMLP(net_cfg).n_params(), FullyFusedMLP().n_params());
}

TEST(TmnnCore, PreferredPrecisionFollowsCapabilities) {
  auto ctx = MetalContext::create();
  ASSERT_NE(ctx, nullptr);

  const auto expected =
      ctx->capabilities().supports_fp16 ? Precision::F16 : Precision::F32;
  EXPECT_EQ(preferred_precision(*ctx), expected);
  EXPECT_EQ(preferred_precision(ctx), expected);
  EXPECT_EQ(supports_fp16(*ctx), ctx->capabilities().supports_fp16);
  EXPECT_EQ(supports_fp16(ctx), ctx->capabilities().supports_fp16);
}

// --- ContextHandle ---

TEST(TmnnCore, ContextHandle) {
  ContextHandle h;
  EXPECT_FALSE(static_cast<bool>(h)); // generation=0 -> false

  h.generation = 1;
  EXPECT_TRUE(static_cast<bool>(h)); // generation!=0 -> true
}

// --- Trainer::set_learning_rate syncs to runtime ---

namespace {
class MockAuthority final : public RuntimeAuthority {
public:
  explicit MockAuthority(ParameterLayout layout) : layout_(std::move(layout)) {}

  [[nodiscard]] const std::shared_ptr<MetalContext> &context() const override {
    return context_;
  }
  [[nodiscard]] ParameterLayout parameter_layout() const override {
    return layout_;
  }
  [[nodiscard]] RuntimeStoragePolicy storage_policy() const override {
    return policy_;
  }
  [[nodiscard]] RuntimeBufferView buffer(RuntimeBufferRole) const override {
    return {};
  }

private:
  std::shared_ptr<MetalContext> context_{MetalContext::create()};
  ParameterLayout layout_{};
  RuntimeStoragePolicy policy_{};
};

/// Minimal mock runtime for testing Trainer::set_learning_rate sync.
class MockRuntime final : public ITrainerRuntime {
public:
  float last_applied_lr = 0.0f;
  int apply_count = 0;
  TrainerBatchPlan plan{77u, 2u, 3u, 1u, 1u, 77u * 3u * sizeof(float),
                        77u * sizeof(float), 256u * sizeof(float), 0u, 0u};
  OptimizerStateBlob exported_blob{1u, 9u, {1u, 2u, 3u}};
  OptimizerStateBlob imported_blob{};
  std::shared_ptr<const RuntimeAuthority> authority =
      std::make_shared<MockAuthority>(ParameterLayout{123u, 45u});

  TrainingStepResult training_step(const float *, const float *, int) override {
    return {};
  }
  void sync_weights() override {}
  [[nodiscard]] uint32_t step() const override { return 0; }
  [[nodiscard]] bool is_gpu_available() const override { return false; }
  [[nodiscard]] std::shared_ptr<const RuntimeAuthority>
  runtime_authority() const override {
    return authority;
  }
  [[nodiscard]] TrainerBatchPlan batch_plan() const override { return plan; }
  [[nodiscard]] OptimizerStateBlob export_optimizer_state() override {
    return exported_blob;
  }
  void import_optimizer_state(const OptimizerStateBlob &state) override {
    imported_blob = state;
  }
  void reset_optimizer() override {}
  void apply_optimizer_config(const Optimizer &opt) override {
    last_applied_lr = opt.learning_rate();
    ++apply_count;
  }
};
} // namespace

TEST(TmnnCore, SetLearningRateSyncsToRuntime) {
  auto net = create_network();
  auto loss = create_loss_l2();
  auto opt = create_optimizer_adam();
  auto runtime = std::make_unique<MockRuntime>();
  auto *runtime_ptr = runtime.get();

  Trainer trainer(std::move(net), std::move(loss), std::move(opt),
                  std::move(runtime));

  // Initial: optimizer lr = 1e-3 (default). No sync yet.
  EXPECT_EQ(runtime_ptr->apply_count, 0);

  // set_learning_rate syncs to runtime.
  trainer.set_learning_rate(0.05f);
  EXPECT_EQ(runtime_ptr->apply_count, 1);
  EXPECT_FLOAT_EQ(runtime_ptr->last_applied_lr, 0.05f);
  EXPECT_FLOAT_EQ(trainer.optimizer().learning_rate(), 0.05f);

  // Another change.
  trainer.set_learning_rate(0.001f);
  EXPECT_EQ(runtime_ptr->apply_count, 2);
  EXPECT_FLOAT_EQ(runtime_ptr->last_applied_lr, 0.001f);
}

TEST(TmnnCore, RuntimeAuthorityAndBatchPlanForwardFromTrainer) {
  auto net = create_network();
  auto loss = create_loss_l2();
  auto opt = create_optimizer_adam();
  auto runtime = std::make_unique<MockRuntime>();
  auto *runtime_ptr = runtime.get();

  Trainer trainer(std::move(net), std::move(loss), std::move(opt),
                  std::move(runtime));

  auto authority = trainer.runtime_authority();
  ASSERT_NE(authority, nullptr);
  auto layout = authority->parameter_layout();
  EXPECT_EQ(layout.hash_grid_float_count, 123u);
  EXPECT_EQ(layout.mlp_weight_float_count, 45u);

  auto plan = trainer.batch_plan();
  EXPECT_EQ(plan.max_batch_size, 77u);
  EXPECT_EQ(plan.lane_count, 2u);
  EXPECT_EQ(plan.input_dims, 3u);

  auto exported = trainer.export_optimizer_state();
  EXPECT_EQ(exported.version, 1u);
  EXPECT_EQ(exported.step, 9u);
  EXPECT_EQ(exported.payload.size(), 3u);

  OptimizerStateBlob imported;
  imported.version = 1;
  imported.step = 17;
  imported.payload = {9u, 8u};
  trainer.import_optimizer_state(imported);
  EXPECT_EQ(runtime_ptr->imported_blob.step, 17u);
  EXPECT_EQ(runtime_ptr->imported_blob.payload.size(), 2u);
}

TEST(TmnnCore, EvaluateFailureExposesStructuredDiagnostic) {
  auto net = create_network();
  auto loss = create_loss_l2();
  auto opt = create_optimizer_adam();
  auto runtime = std::make_unique<MockRuntime>();

  Trainer trainer(std::move(net), std::move(loss), std::move(opt),
                  std::move(runtime));

  float pos[3] = {0.0f, 0.0f, 0.0f};
  float out[1] = {};
  EXPECT_FALSE(trainer.evaluate(pos, out, 1));

  const auto diag = trainer.last_diagnostic();
  ASSERT_TRUE(diag.has_value());
  EXPECT_EQ(diag->code, DiagnosticCode::MissingHostVisibleConfigWeights);
  EXPECT_EQ(diag->operation, "Trainer::ensure_evaluator");
  EXPECT_NE(diag->message.find("host-visible config weights"), std::string::npos);
}

TEST(TmnnCore, LoggerHookObservesTryCreateEncodingFailure) {
  std::optional<DiagnosticInfo> observed;
  set_logger_hook([&](const DiagnosticInfo &diagnostic) {
    if (diagnostic.code != DiagnosticCode::None) {
      observed = diagnostic;
    }
  });

  json enc = {{"otype", "HashGrid"}, {"interpolation", "Smoothstep"}};
  auto result = try_create_encoding_from_json(3, enc);
  clear_logger_hook();

  ASSERT_FALSE(result.has_value());
  ASSERT_TRUE(observed.has_value());
  EXPECT_EQ(observed->code, DiagnosticCode::InvalidArgument);
  EXPECT_EQ(observed->operation, "create_encoding_from_json");
}

TEST(TmnnCore, LoggerHookObservesCanonicalizationNotes) {
  std::optional<DiagnosticInfo> observed;
  set_logger_hook([&](const DiagnosticInfo &diagnostic) {
    if (diagnostic.code == DiagnosticCode::None &&
        diagnostic.operation == "create_encoding_from_json") {
      observed = diagnostic;
    }
  });

  json enc = {{"otype", "MultiresolutionHashGrid"}};
  auto result = try_create_encoding_from_json(3, enc);
  clear_logger_hook();

  ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(observed.has_value());
  ASSERT_FALSE(observed->details.empty());
  EXPECT_EQ(observed->details.front().severity, DiagnosticSeverity::Info);
}

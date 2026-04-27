/**
 * @file smoke_test.cpp
 * @brief Package export smoke test and external-consumer demo for
 *        find_package(tiny_metal_nn).
 */

// tcnn-aligned umbrella header — verifies the parity entry point compiles.
#include "tiny-metal-nn/cpp_api.h"

#include <cmath>

#include "tiny-metal-nn/autotune_manifest.h"
#include "tiny-metal-nn/common.h"
#include "tiny-metal-nn/detail/adam.h"
#include "tiny-metal-nn/encoding.h"
#include "tiny-metal-nn/factory.h"
#include "tiny-metal-nn/factory_json.h"
#include "tiny-metal-nn/fully_fused_mlp.h"
#include "tiny-metal-nn/hash_grid.h"
#include "tiny-metal-nn/detail/l2_loss.h"
#include "tiny-metal-nn/loss.h"
#include "tiny-metal-nn/metal_context.h"
#include "tiny-metal-nn/detail/module.h"
#include "tiny-metal-nn/network.h"
#include "tiny-metal-nn/network_with_input_encoding.h"
#include "tiny-metal-nn/detail/network_planning.h"
#include "tiny-metal-nn/optimizer.h"
#include "tiny-metal-nn/trainer.h"

// Extension SDK headers — must compile without linking runtime.
#include "tiny-metal-nn/evaluator.h"
#include "tiny-metal-nn/extension/schema.h"
#include "tiny-metal-nn/extension/kernel_compile_spec.h"
#include "tiny-metal-nn/extension/training_adapter.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>

// ---------------------------------------------------------------------------
// Minimal FieldEvaluator mock (verifies vtable compiles without runtime).
// ---------------------------------------------------------------------------

struct SmokeEvaluator : tmnn::FieldEvaluator {
  uint32_t n_input_dims() const override { return 3; }
  uint32_t n_output_dims() const override { return 1; }
  bool evaluate(const float *, float *output, int N) override {
    for (int i = 0; i < N; ++i) output[i] = 0.0f;
    return true;
  }
  bool evaluate_with_gradient(const float *, float *output, float *gradients,
                              int N) override {
    for (int i = 0; i < N; ++i) {
      output[i] = 0.0f;
      gradients[i * 3 + 0] = gradients[i * 3 + 1] = gradients[i * 3 + 2] = 0.0f;
    }
    return true;
  }
};

// ---------------------------------------------------------------------------
// Minimal TrainingAdapter mock.
// ---------------------------------------------------------------------------

struct SmokeAdapter : tmnn::extension::TrainingAdapter {
  tmnn::extension::ExtensionSchema schema() const override {
    return {};
  }
  void configure_compile_spec(tmnn::extension::KernelCompileSpec &) const override {}
  void pack_config_tail(float *) const override {}
  void pack_batch(const float *in, const float *tgt, int N,
                  float *pos_out, float *tgt_out) const override {
    std::memcpy(pos_out, in, static_cast<size_t>(N) * 3 * sizeof(float));
    std::memcpy(tgt_out, tgt, static_cast<size_t>(N) * 1 * sizeof(float));
  }
  void fill_train_params(float *dst, const tmnn::TrainParamsLayout &layout,
                         uint32_t N, uint32_t) const override {
    tmnn::fill_train_params(dst, layout, N, false, 1.0f, 16);
  }
  tmnn::extension::AdamConfig adam_config(uint32_t) const override { return {}; }
  tmnn::extension::ResultMetrics result_metrics(float, uint32_t) const override { return {}; }
};

int main() {
  // Verify core types compile and link as a standalone library.
  auto enc = tmnn::create_encoding();
  assert(enc->name() == "HashGridEncoding");
  assert(enc->n_output_dims() == 32);
  auto rotated_enc = tmnn::create_rotated_encoding();
  assert(rotated_enc->name() == "RotatedHashGridEncoding");
  assert(rotated_enc->n_output_dims() == 32);

  auto net = tmnn::create_network();
  assert(net->n_params() == 6337);

  auto fused = tmnn::create_network_with_input_encoding(enc, net);
  assert(fused->n_input_dims() == enc->n_input_dims());
  assert(fused->n_output_dims() == net->n_output_dims());
  tmnn::NetworkFactoryOptions options;
  tmnn::MultiOutputFactoryOptions multi_output;
  multi_output.semantics = tmnn::MultiOutputSemanticProfile::DNL;
  multi_output.bc_dim_count = 1;
  options.multi_output = multi_output;
  tmnn::NetworkPlan plan;
  tmnn::AutotuneManifest manifest;
  (void)options;
  (void)plan;
  (void)manifest;

  auto loss = tmnn::create_loss_l2();
  float pred[] = {1.0f};
  float tgt[] = {0.0f};
  assert(loss->evaluate_cpu(pred, tgt, 1) == 1.0f);

  auto opt = tmnn::create_optimizer_adam();
  assert(opt->learning_rate() > 0.0f);

  // PrecisionTraits guard
  tmnn::TensorRef ref;
  float buf[1] = {42.0f};
  ref.data = buf;
  ref.precision = tmnn::Precision::F32;
  assert(ref.typed_data<float>()[0] == 42.0f);

  // TrainingStepResult.step is uint32_t
  tmnn::TrainingStepResult r;
  r.step = 100u;
  static_assert(sizeof(r.step) == sizeof(uint32_t));

  // --- Extension SDK smoke ---

  // FieldEvaluator vtable.
  SmokeEvaluator evaluator;
  tmnn::FieldEvaluator *ev_ptr = &evaluator;
  assert(ev_ptr->n_input_dims() == 3);
  assert(ev_ptr->n_output_dims() == 1);
  float pos[3] = {0.0f, 0.0f, 0.0f};
  float out[1] = {-1.0f};
  assert(ev_ptr->evaluate(pos, out, 1));
  assert(!ev_ptr->last_diagnostic().has_value());
  assert(out[0] == 0.0f);

  // TrainParamsLayout::validate() (header-only).
  tmnn::TrainParamsLayout layout;
  layout.validate(); // must not throw

  // fill_train_params (header-only).
  float tp[8] = {};
  tmnn::fill_train_params(tp, layout, 64, false, 1.0f, 16);
  assert(tp[0] == 64.0f);

  // TrainingAdapter vtable.
  SmokeAdapter adapter;
  tmnn::extension::TrainingAdapter *ad_ptr = &adapter;
  auto schema = ad_ptr->schema();
  assert(schema.input_dims == 3);
  assert(schema.target_dims == 1);
  auto adam = ad_ptr->adam_config(0);
  assert(adam.lr_encoding > 0.0f);
  auto metrics = ad_ptr->result_metrics(0.5f, 0);
  assert(metrics.extra_loss_count == 0);

  // --- C4a JSON bridge smoke ---
  nlohmann::json enc_json = {{"otype", "HashGrid"},
                             {"n_levels", 8},
                             {"n_features_per_level", 2}};
  nlohmann::json net_json = {{"otype", "FullyFusedMLP"},
                             {"n_neurons", 32},
                             {"n_hidden_layers", 2}};
  nlohmann::json model_json = {{"encoding", enc_json}, {"network", net_json}};
  auto canonical = tmnn::canonicalize_model_config(3, 1, model_json);
  assert(!canonical.has_errors());
  auto enc_from_json = tmnn::create_encoding_from_json(3, enc_json);
  auto net_from_json =
      tmnn::create_network_from_json(enc_from_json->n_output_dims(), 1, net_json);
  auto model_from_json =
      tmnn::create_network_with_input_encoding_from_json(3, 1, model_json);
  nlohmann::json rotated_enc_json = {{"otype", "RotatedMultiresHashGrid"}};
  auto rotated_from_json = tmnn::create_encoding_from_json(3, rotated_enc_json);
  auto loss_from_json = tmnn::create_loss(nlohmann::json{{"otype", "L2"}});
  auto optimizer_from_json =
      tmnn::create_optimizer(nlohmann::json{{"otype", "Adam"}});
  assert(enc_from_json->n_input_dims() == 3);
  assert(net_from_json->n_output_dims() == 1);
  assert(model_from_json->n_output_dims() == 1);
  assert(rotated_from_json->name() == "RotatedHashGridEncoding");
  assert(loss_from_json->name() == "L2");
  assert(optimizer_from_json->learning_rate() > 0.0f);

  // --- Public runtime / planner / manifest smoke ---
  auto ctx = tmnn::MetalContext::create();
  assert(ctx != nullptr);
  const auto caps = ctx->capabilities();
  assert(tmnn::supports_fp16(*ctx) == caps.supports_fp16);
  assert(tmnn::preferred_precision(*ctx) ==
         (caps.supports_fp16 ? tmnn::Precision::F16 : tmnn::Precision::F32));
  if (ctx->is_gpu_available()) {
    assert(!caps.device_name.empty());
  }

  options.metal_context = ctx;
  const auto cold_plan = model_from_json->plan(options);
  assert(cold_plan.planner_fingerprint != 0);
  assert(!cold_plan.candidate_families.empty());

  const auto cold_manifest = ctx->snapshot_autotune_manifest();
  assert(cold_manifest.version == std::string(tmnn::kAutotuneManifestVersion));
  assert(cold_manifest.entries.size() == 1);
  assert(cold_manifest.entries.front().planner_fingerprint ==
         cold_plan.planner_fingerprint);

  const auto manifest_path =
      std::filesystem::temp_directory_path() / "tmnn_smoke_manifest.json";
  tmnn::save_autotune_manifest(manifest_path.string(), cold_manifest);
  const auto reloaded_manifest =
      tmnn::load_autotune_manifest(manifest_path.string());
  std::filesystem::remove(manifest_path);
  assert(reloaded_manifest.entries.size() == cold_manifest.entries.size());

  auto prewarmed_ctx = tmnn::MetalContext::create();
  assert(prewarmed_ctx != nullptr);
  prewarmed_ctx->prewarm_autotune_manifest(reloaded_manifest);

  options.metal_context = prewarmed_ctx;
  const auto hot_plan = model_from_json->plan(options);
  assert(hot_plan.from_autotune_manifest);
  assert(hot_plan.selected_family == cold_plan.selected_family);
  assert(hot_plan.planner_fingerprint == cold_plan.planner_fingerprint);

  // --- Headline path: create_trainer() → train → evaluate ---
  if (ctx->is_gpu_available()) {
    constexpr int kSmokeBatch = 32;
    auto trainer = tmnn::create_trainer(
        tmnn::default_trainer_encoding_config(),
        tmnn::default_trainer_network_config(),
        tmnn::default_trainer_config(), ctx);
    assert(trainer.is_gpu_available());
    assert(trainer.step() == 0);
    assert(trainer.batch_plan().max_batch_size ==
           static_cast<uint32_t>(tmnn::default_trainer_config().batch_size));
    float train_pos[kSmokeBatch * 3] = {};
    float train_tgt[kSmokeBatch] = {};
    for (int i = 0; i < kSmokeBatch; ++i) {
      train_pos[i * 3] = static_cast<float>(i % 16) / 16.0f;
      train_pos[i * 3 + 1] = static_cast<float>((i / 16) % 16) / 16.0f;
      train_pos[i * 3 + 2] = 0.0f;
      train_tgt[i] = 0.1f;
    }
    auto step_result = trainer.training_step(train_pos, train_tgt, kSmokeBatch);
    assert(std::isfinite(step_result.loss));
    assert(trainer.step() == 1);
    auto opt_blob = trainer.export_optimizer_state();
    assert(opt_blob.version == tmnn::kOptimizerStateBlobVersion);
    assert(opt_blob.step == 1);
    assert(!opt_blob.payload.empty());
    trainer.reset_optimizer();
    trainer.import_optimizer_state(opt_blob);
    auto opt_blob_roundtrip = trainer.export_optimizer_state();
    assert(opt_blob_roundtrip.step == opt_blob.step);
    assert(opt_blob_roundtrip.payload == opt_blob.payload);
    float eval_out[kSmokeBatch] = {};
    bool eval_ok = trainer.evaluate(train_pos, eval_out, kSmokeBatch);
    assert(eval_ok);
    assert(!trainer.last_diagnostic().has_value());
    assert(std::isfinite(eval_out[0]));
    auto exported_evaluator_result = trainer.try_create_evaluator();
    assert(exported_evaluator_result.has_value());
    auto exported_evaluator = std::move(*exported_evaluator_result);
    assert(exported_evaluator != nullptr);
    bool exported_eval_ok =
        exported_evaluator->evaluate(train_pos, eval_out, kSmokeBatch);
    assert(exported_eval_ok);
    assert(!exported_evaluator->last_diagnostic().has_value());
    assert(trainer.runtime_authority() != nullptr);

    nlohmann::json trainer_config = {
        {"loss", {{"otype", "L2"}}},
        {"optimizer", {{"otype", "Adam"}, {"learning_rate", 1e-4f}}},
        {"encoding",
         {{"otype", "HashGrid"},
          {"n_levels", 4},
          {"n_features_per_level", 2},
          {"log2_hashmap_size", 12}}},
        {"network",
         {{"otype", "FullyFusedMLP"},
           {"n_neurons", 16},
           {"n_hidden_layers", 2}}},
    };
    auto trainer_from_config_result = tmnn::try_create_from_config(
        3, 1, trainer_config, {.batch_size = kSmokeBatch}, ctx);
    assert(trainer_from_config_result.has_value());
    auto trainer_from_config = std::move(*trainer_from_config_result);
    auto config_step =
        trainer_from_config.training_step(train_pos, train_tgt, kSmokeBatch);
    assert(std::isfinite(config_step.loss));
    bool infer_ok =
        trainer_from_config.inference(train_pos, eval_out, kSmokeBatch);
    assert(infer_ok);
    assert(!trainer_from_config.last_diagnostic().has_value());
    assert(std::isfinite(eval_out[1]));
  }

  return 0;
}

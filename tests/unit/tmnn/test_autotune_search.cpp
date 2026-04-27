/**
 * @file test_autotune_search.cpp
 * @brief Tests for tmnn-owned bounded autotune search orchestration.
 */

#include <gtest/gtest.h>

#include "tiny-metal-nn/autotune_manifest.h"
#include "tiny-metal-nn/factory.h"
#include "tiny-metal-nn/metal_context.h"
#include "tiny-metal-nn/detail/network_planning.h"
#include "tiny-metal-nn/trainer.h"

#include "tiny_metal_nn/runtime/autotune_search.h"

#include <algorithm>
#include <chrono>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace tmnn;

namespace {

std::shared_ptr<NetworkWithInputEncoding> make_model() {
  return create_network_with_input_encoding(create_encoding(), create_network());
}

class TimedRuntime final : public ITrainerRuntime {
public:
  explicit TimedRuntime(std::chrono::milliseconds step_delay)
      : step_delay_(step_delay) {
    batch_plan_.max_batch_size = 64u;
    batch_plan_.input_dims = 3u;
    batch_plan_.target_dims = 1u;
  }

  TrainingStepResult training_step(const float *, const float *, int) override {
    std::this_thread::sleep_for(step_delay_);
    TrainingStepResult out;
    out.step = ++step_;
    out.loss = 1.0f / static_cast<float>(out.step);
    return out;
  }

  void sync_weights() override {}
  [[nodiscard]] uint32_t step() const override { return step_; }
  [[nodiscard]] bool is_gpu_available() const override { return true; }
  [[nodiscard]] std::shared_ptr<const RuntimeAuthority>
  runtime_authority() const override {
    return {};
  }
  [[nodiscard]] TrainerBatchPlan batch_plan() const override {
    return batch_plan_;
  }
  [[nodiscard]] OptimizerStateBlob export_optimizer_state() override {
    return {};
  }
  void import_optimizer_state(const OptimizerStateBlob &) override {}
  void reset_optimizer() override { step_ = 0; }
  void apply_optimizer_config(const Optimizer &) override {}

private:
  TrainerBatchPlan batch_plan_{};
  uint32_t step_ = 0;
  std::chrono::milliseconds step_delay_;
};

class TimedSequenceRuntime final : public ITrainerRuntime {
public:
  explicit TimedSequenceRuntime(std::vector<std::chrono::milliseconds> step_delays)
      : step_delays_(std::move(step_delays)) {
    batch_plan_.max_batch_size = 64u;
    batch_plan_.input_dims = 3u;
    batch_plan_.target_dims = 1u;
  }

  TrainingStepResult training_step(const float *, const float *, int) override {
    const size_t delay_index = std::min<size_t>(
        step_, step_delays_.empty() ? 0u : step_delays_.size() - 1u);
    if (!step_delays_.empty())
      std::this_thread::sleep_for(step_delays_[delay_index]);
    TrainingStepResult out;
    out.step = ++step_;
    out.loss = 1.0f / static_cast<float>(out.step);
    return out;
  }

  void sync_weights() override {}
  [[nodiscard]] uint32_t step() const override { return step_; }
  [[nodiscard]] bool is_gpu_available() const override { return true; }
  [[nodiscard]] std::shared_ptr<const RuntimeAuthority>
  runtime_authority() const override {
    return {};
  }
  [[nodiscard]] TrainerBatchPlan batch_plan() const override {
    return batch_plan_;
  }
  [[nodiscard]] OptimizerStateBlob export_optimizer_state() override {
    return {};
  }
  void import_optimizer_state(const OptimizerStateBlob &) override {}
  void reset_optimizer() override { step_ = 0; }
  void apply_optimizer_config(const Optimizer &) override {}

private:
  TrainerBatchPlan batch_plan_{};
  uint32_t step_ = 0;
  std::vector<std::chrono::milliseconds> step_delays_;
};

detail::AutotuneSearchRuntimeFactory make_runtime_factory(
    std::unordered_map<NetworkFamily, std::chrono::milliseconds> delays,
    int &call_count) {
  return [delays, &call_count](
             const NetworkPlan &plan,
             const std::shared_ptr<MetalContext> &)
             -> std::unique_ptr<ITrainerRuntime> {
    ++call_count;
    return std::make_unique<TimedRuntime>(delays.at(plan.selected_family));
  };
}

detail::AutotuneSearchRuntimeFactory make_sequence_runtime_factory(
    std::unordered_map<NetworkFamily, std::vector<std::chrono::milliseconds>>
        delays,
    int &call_count) {
  return [delays = std::move(delays), &call_count](
             const NetworkPlan &plan,
             const std::shared_ptr<MetalContext> &)
             -> std::unique_ptr<ITrainerRuntime> {
     ++call_count;
     return std::make_unique<TimedSequenceRuntime>(delays.at(plan.selected_family));
   };
}

detail::AutotuneSearchRuntimeFactory make_build_and_step_runtime_factory(
    std::unordered_map<NetworkFamily, std::chrono::milliseconds> build_delays,
    std::unordered_map<NetworkFamily, std::chrono::milliseconds> step_delays,
    int &call_count) {
  return [build_delays = std::move(build_delays),
          step_delays = std::move(step_delays), &call_count](
             const NetworkPlan &plan,
             const std::shared_ptr<MetalContext> &)
             -> std::unique_ptr<ITrainerRuntime> {
    ++call_count;
    std::this_thread::sleep_for(build_delays.at(plan.selected_family));
    return std::make_unique<TimedRuntime>(step_delays.at(plan.selected_family));
  };
}

} // namespace

TEST(AutotuneSearch, SelectsFastestEligibleFamilyAndPersistsMeasurement) {
  auto ctx = MetalContext::create();
  if (!ctx->is_gpu_available())
    GTEST_SKIP() << "Autotune search orchestration requires a GPU-backed context";

  auto model = make_model();

  NetworkFactoryOptions options;
  options.metal_context = ctx;
  options.enable_bounded_autotune_search = true;
  options.autotune_search_batch_size = 64u;
  options.autotune_search_warmup_steps = 2u;
  options.autotune_search_measure_steps = 2u;

  const auto baseline_plan = model->plan(options);
  ASSERT_GE(baseline_plan.candidate_families.size(), 2u);

  std::unordered_map<NetworkFamily, std::chrono::milliseconds> delays{
      {NetworkFamily::FullyFusedMetal, std::chrono::milliseconds(6)},
      {NetworkFamily::TiledMetal, std::chrono::milliseconds(1)},
      {NetworkFamily::SafeDebugMetal, std::chrono::milliseconds(8)},
  };
  int runtime_calls = 0;

  const auto measured = detail::run_bounded_autotune_search(
      *model, baseline_plan, options, ctx, 64, "AutotuneSearch",
      make_runtime_factory(delays, runtime_calls));

  EXPECT_EQ(runtime_calls,
            static_cast<int>(baseline_plan.candidate_families.size()));
  EXPECT_EQ(measured.selected_family, NetworkFamily::TiledMetal);
  EXPECT_FALSE(measured.from_autotune_manifest);
  EXPECT_TRUE(measured.selected_by_autotune_search);
  EXPECT_GT(measured.autotune_measured_step_ns, 0u);
  EXPECT_FALSE(measured.fused_train);

  const auto manifest = ctx->snapshot_autotune_manifest();
  ASSERT_EQ(manifest.entries.size(), 1u);
  EXPECT_EQ(manifest.entries[0].selected_family, NetworkFamily::TiledMetal);
  EXPECT_TRUE(manifest.entries[0].selected_by_measurement);
  EXPECT_EQ(manifest.entries[0].measurement_batch_size, 64u);
  EXPECT_EQ(manifest.entries[0].measurement_warmup_steps, 2u);
  EXPECT_EQ(manifest.entries[0].measurement_steps, 2u);
  EXPECT_EQ(manifest.entries[0].measurement_objective,
            AutotuneSearchObjective::SteadyStateStep);
  EXPECT_EQ(measured.autotune_measured_objective_ns,
            measured.autotune_measured_step_ns);
  EXPECT_EQ(manifest.entries[0].selected_family_objective_ns,
            measured.autotune_measured_objective_ns);
}

TEST(AutotuneSearch, ReusesMeasuredManifestWithoutRerunningCandidates) {
  auto ctx = MetalContext::create();
  if (!ctx->is_gpu_available())
    GTEST_SKIP() << "Autotune search orchestration requires a GPU-backed context";

  auto model = make_model();

  NetworkFactoryOptions options;
  options.metal_context = ctx;
  options.enable_bounded_autotune_search = true;
  options.autotune_search_batch_size = 64u;
  options.autotune_search_warmup_steps = 2u;
  options.autotune_search_measure_steps = 2u;

  const auto baseline_plan = model->plan(options);
  ASSERT_GE(baseline_plan.candidate_families.size(), 2u);

  AutotuneManifest manifest;
  manifest.entries.push_back(AutotuneManifestEntry{
      baseline_plan.planner_fingerprint,
      NetworkFamily::TiledMetal,
      baseline_plan.candidate_families,
      baseline_plan.reasons,
      false,
      true,
      64u,
      2u,
      2u,
      4321u,
      {
          AutotuneCandidateMeasurement{NetworkFamily::FullyFusedMetal, 6000u,
                                       5000u, true, ""},
          AutotuneCandidateMeasurement{NetworkFamily::TiledMetal, 1000u, 4321u,
                                       true, ""},
      },
  });
  ctx->prewarm_autotune_manifest(manifest);

  int runtime_calls = 0;
  const auto measured = detail::run_bounded_autotune_search(
      *model, baseline_plan, options, ctx, 64, "AutotuneSearch",
      make_runtime_factory(
          {{NetworkFamily::FullyFusedMetal, std::chrono::milliseconds(6)},
           {NetworkFamily::TiledMetal, std::chrono::milliseconds(1)},
           {NetworkFamily::SafeDebugMetal, std::chrono::milliseconds(8)}},
          runtime_calls));

  EXPECT_EQ(runtime_calls, 0);
  EXPECT_EQ(measured.selected_family, NetworkFamily::TiledMetal);
  EXPECT_TRUE(measured.from_autotune_manifest);
  EXPECT_TRUE(measured.selected_by_autotune_search);
  EXPECT_EQ(measured.autotune_measured_step_ns, 4321u);
}

TEST(AutotuneSearch, WarmupPolicyMismatchRerunsCandidates) {
  auto ctx = MetalContext::create();
  if (!ctx->is_gpu_available())
    GTEST_SKIP() << "Autotune search orchestration requires a GPU-backed context";

  auto model = make_model();

  NetworkFactoryOptions options;
  options.metal_context = ctx;
  options.enable_bounded_autotune_search = true;
  options.autotune_search_batch_size = 64u;
  options.autotune_search_warmup_steps = 2u;
  options.autotune_search_measure_steps = 2u;

  const auto baseline_plan = model->plan(options);
  ASSERT_GE(baseline_plan.candidate_families.size(), 2u);

  AutotuneManifest manifest;
  manifest.entries.push_back(AutotuneManifestEntry{
      baseline_plan.planner_fingerprint,
      NetworkFamily::TiledMetal,
      baseline_plan.candidate_families,
      baseline_plan.reasons,
      false,
      true,
      64u,
      0u,
      2u,
      4321u,
      {
          AutotuneCandidateMeasurement{NetworkFamily::FullyFusedMetal, 6000u,
                                       5000u, true, ""},
          AutotuneCandidateMeasurement{NetworkFamily::TiledMetal, 1000u, 4321u,
                                       true, ""},
      },
  });
  ctx->prewarm_autotune_manifest(manifest);

  int runtime_calls = 0;
  const auto measured = detail::run_bounded_autotune_search(
      *model, baseline_plan, options, ctx, 64, "AutotuneSearch",
      make_runtime_factory(
          {{NetworkFamily::FullyFusedMetal, std::chrono::milliseconds(6)},
           {NetworkFamily::TiledMetal, std::chrono::milliseconds(1)},
           {NetworkFamily::SafeDebugMetal, std::chrono::milliseconds(8)}},
          runtime_calls));

  EXPECT_GT(runtime_calls, 0);
  EXPECT_FALSE(measured.from_autotune_manifest);
}

TEST(AutotuneSearch, ObjectivePolicyMismatchRerunsCandidates) {
  auto ctx = MetalContext::create();
  if (!ctx->is_gpu_available())
    GTEST_SKIP() << "Autotune search orchestration requires a GPU-backed context";

  auto model = make_model();

  NetworkFactoryOptions options;
  options.metal_context = ctx;
  options.enable_bounded_autotune_search = true;
  options.autotune_search_batch_size = 64u;
  options.autotune_search_warmup_steps = 0u;
  options.autotune_search_measure_steps = 2u;
  options.autotune_search_objective =
      AutotuneSearchObjective::BuildPlusMeasureWindow;

  const auto baseline_plan = model->plan(options);
  ASSERT_GE(baseline_plan.candidate_families.size(), 2u);

  AutotuneManifest manifest;
  manifest.entries.push_back(AutotuneManifestEntry{
      baseline_plan.planner_fingerprint,
      NetworkFamily::TiledMetal,
      baseline_plan.candidate_families,
      baseline_plan.reasons,
      false,
      true,
      64u,
      0u,
      2u,
      4321u,
      {
          AutotuneCandidateMeasurement{NetworkFamily::FullyFusedMetal, 6000u,
                                       5000u, true, ""},
          AutotuneCandidateMeasurement{NetworkFamily::TiledMetal, 1000u, 4321u,
                                       true, ""},
      },
      AutotuneSearchObjective::SteadyStateStep,
      4321u,
  });
  ctx->prewarm_autotune_manifest(manifest);

  int runtime_calls = 0;
  const auto measured = detail::run_bounded_autotune_search(
      *model, baseline_plan, options, ctx, 64, "AutotuneSearch",
      make_runtime_factory(
          {{NetworkFamily::FullyFusedMetal, std::chrono::milliseconds(6)},
           {NetworkFamily::TiledMetal, std::chrono::milliseconds(1)},
           {NetworkFamily::SafeDebugMetal, std::chrono::milliseconds(8)}},
          runtime_calls));

  EXPECT_GT(runtime_calls, 0);
  EXPECT_FALSE(measured.from_autotune_manifest);
}

TEST(AutotuneSearch, BuildAwareObjectiveSelectsFastestStartupWindow) {
  auto ctx = MetalContext::create();
  if (!ctx->is_gpu_available())
    GTEST_SKIP() << "Autotune search orchestration requires a GPU-backed context";

  auto model = make_model();

  NetworkFactoryOptions options;
  options.metal_context = ctx;
  options.enable_bounded_autotune_search = true;
  options.autotune_search_batch_size = 64u;
  options.autotune_search_warmup_steps = 0u;
  options.autotune_search_measure_steps = 2u;
  options.autotune_search_objective =
      AutotuneSearchObjective::BuildPlusMeasureWindow;

  const auto baseline_plan = model->plan(options);
  ASSERT_GE(baseline_plan.candidate_families.size(), 2u);
  if (std::find(baseline_plan.candidate_families.begin(),
                baseline_plan.candidate_families.end(),
                NetworkFamily::FullyFusedMetal) ==
          baseline_plan.candidate_families.end() ||
      std::find(baseline_plan.candidate_families.begin(),
                baseline_plan.candidate_families.end(),
                NetworkFamily::TiledMetal) ==
          baseline_plan.candidate_families.end()) {
    GTEST_SKIP() << "Build-aware flush proof requires fused/tiled candidate coverage";
  }

  int runtime_calls = 0;
  const auto measured = detail::run_bounded_autotune_search(
      *model, baseline_plan, options, ctx, 64, "AutotuneSearch",
      make_build_and_step_runtime_factory(
          {{NetworkFamily::FullyFusedMetal, std::chrono::milliseconds(6)},
           {NetworkFamily::TiledMetal, std::chrono::milliseconds(1)},
           {NetworkFamily::SafeDebugMetal, std::chrono::milliseconds(8)}},
          {{NetworkFamily::FullyFusedMetal, std::chrono::milliseconds(1)},
           {NetworkFamily::TiledMetal, std::chrono::milliseconds(2)},
           {NetworkFamily::SafeDebugMetal, std::chrono::milliseconds(9)}},
          runtime_calls));

  EXPECT_EQ(runtime_calls,
            static_cast<int>(baseline_plan.candidate_families.size()));
  EXPECT_EQ(measured.selected_family, NetworkFamily::TiledMetal);
  EXPECT_EQ(measured.autotune_search_objective,
            AutotuneSearchObjective::BuildPlusMeasureWindow);
  EXPECT_GT(measured.autotune_measured_objective_ns,
            measured.autotune_measured_step_ns);

  const auto manifest = ctx->snapshot_autotune_manifest();
  ASSERT_EQ(manifest.entries.size(), 1u);
  const auto &entry = manifest.entries.front();
  EXPECT_EQ(entry.measurement_objective,
            AutotuneSearchObjective::BuildPlusMeasureWindow);
  EXPECT_EQ(entry.selected_family, NetworkFamily::TiledMetal);
  EXPECT_EQ(entry.selected_family_objective_ns,
            measured.autotune_measured_objective_ns);
  const auto selected =
      std::find_if(entry.measurements.begin(), entry.measurements.end(),
                   [&](const auto &candidate) {
                     return candidate.family == NetworkFamily::TiledMetal;
                   });
  ASSERT_NE(selected, entry.measurements.end());
  EXPECT_GT(selected->build_ns, 0u);
  EXPECT_GT(selected->objective_ns, selected->measured_step_ns);
}

TEST(AutotuneSearch, IgnoresBootstrapNoiseAndSelectsSteadyStateWinner) {
  auto ctx = MetalContext::create();
  if (!ctx->is_gpu_available())
    GTEST_SKIP() << "Autotune search orchestration requires a GPU-backed context";

  auto model = make_model();

  NetworkFactoryOptions options;
  options.metal_context = ctx;
  options.enable_bounded_autotune_search = true;
  options.autotune_search_batch_size = 64u;
  options.autotune_search_warmup_steps = 2u;
  options.autotune_search_measure_steps = 2u;

  const auto baseline_plan = model->plan(options);
  ASSERT_GE(baseline_plan.candidate_families.size(), 2u);

  int runtime_calls = 0;
  const auto measured = detail::run_bounded_autotune_search(
      *model, baseline_plan, options, ctx, 64, "AutotuneSearch",
      make_sequence_runtime_factory(
          {
              {NetworkFamily::FullyFusedMetal,
               {std::chrono::milliseconds(8), std::chrono::milliseconds(8),
                std::chrono::milliseconds(1), std::chrono::milliseconds(1)}},
              {NetworkFamily::TiledMetal,
               {std::chrono::milliseconds(3), std::chrono::milliseconds(3),
                std::chrono::milliseconds(3), std::chrono::milliseconds(3)}},
              {NetworkFamily::SafeDebugMetal,
               {std::chrono::milliseconds(9), std::chrono::milliseconds(9),
                std::chrono::milliseconds(9), std::chrono::milliseconds(9)}},
          },
          runtime_calls));

  EXPECT_EQ(runtime_calls,
            static_cast<int>(baseline_plan.candidate_families.size()));
  EXPECT_EQ(measured.selected_family, NetworkFamily::FullyFusedMetal);

  const auto manifest = ctx->snapshot_autotune_manifest();
  ASSERT_EQ(manifest.entries.size(), 1u);
  const auto &entry = manifest.entries.front();
  EXPECT_EQ(entry.measurement_warmup_steps, 2u);
  EXPECT_EQ(entry.measurement_objective,
            AutotuneSearchObjective::SteadyStateStep);
  ASSERT_EQ(entry.measurements.size(), baseline_plan.candidate_families.size());
  const auto selected =
      std::find_if(entry.measurements.begin(), entry.measurements.end(),
                   [&](const auto &candidate) {
                     return candidate.family == NetworkFamily::FullyFusedMetal;
                   });
  ASSERT_NE(selected, entry.measurements.end());
  EXPECT_TRUE(selected->succeeded);
  EXPECT_LT(selected->measured_step_ns, 3000000u);
}

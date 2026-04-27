/**
 * @file tmnn_runtime_benchmarks.cpp
 * @brief Standalone tmnn-owned benchmark entry point.
 */

#include "tiny-metal-nn/factory.h"
#include "tiny-metal-nn/default_trainer.h"
#include "tiny-metal-nn/metal_context.h"
#include "tiny-metal-nn/detail/network_planning.h"
#include "tiny-metal-nn/trainer.h"

#include "tiny_metal_nn/runtime/autotune_search.h"
#include "tiny_metal_nn/runtime/metal_device.h"
#include "tiny_metal_nn/runtime/morton_sort.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

using namespace tmnn;

DeviceCapabilities make_synthetic_caps() {
  DeviceCapabilities caps;
  caps.device_name = "SyntheticMetal";
  caps.gpu_family = 7;
  caps.max_threads_per_tg = 1024;
  caps.max_threadgroup_memory_bytes = 32768;
  caps.supports_fp16 = true;
  caps.supports_simdgroup_matrix = true;
  caps.supports_nonuniform_threadgroups = true;
  caps.supports_binary_archive = true;
  return caps;
}

std::shared_ptr<NetworkWithInputEncoding> make_model() {
  return create_network_with_input_encoding(create_encoding(), create_network());
}

constexpr int kNeulatHiddenDim = 64;
constexpr int kNeulatLog2HashmapSize = 19;
constexpr int kNeulatBatchSize = 1024;
constexpr int kNeulatEvalCount = 4096;
constexpr double kNeulatTrainingTargetMs = 2.0;
constexpr double kNeulatEvaluateTargetMs = 0.5;
constexpr double kNeulatGradientTargetMs = 1.0;
constexpr double kNeulatStartupTargetMs = 50.0;

double median_ns(std::vector<uint64_t> samples) {
  if (samples.empty())
    return 0.0;
  std::sort(samples.begin(), samples.end());
  return static_cast<double>(samples[samples.size() / 2]);
}

struct SampleSummary {
  double p50_ns = 0.0;
  double p99_ns = 0.0;
  double min_ns = 0.0;
  double max_ns = 0.0;
};

struct DoubleSampleSummary {
  double p50 = 0.0;
  double p99 = 0.0;
  double min = 0.0;
  double max = 0.0;
};

SampleSummary summarize_ns(std::vector<uint64_t> samples) {
  SampleSummary out;
  if (samples.empty())
    return out;
  std::sort(samples.begin(), samples.end());
  out.min_ns = static_cast<double>(samples.front());
  out.max_ns = static_cast<double>(samples.back());
  out.p50_ns = static_cast<double>(samples[samples.size() / 2]);
  const size_t p99_index =
      std::min(samples.size() - 1u,
               static_cast<size_t>(std::ceil(samples.size() * 0.99)) - 1u);
  out.p99_ns = static_cast<double>(samples[p99_index]);
  return out;
}

DoubleSampleSummary summarize_double(std::vector<double> samples) {
  DoubleSampleSummary out;
  if (samples.empty())
    return out;
  std::sort(samples.begin(), samples.end());
  out.min = samples.front();
  out.max = samples.back();
  out.p50 = samples[samples.size() / 2];
  const size_t p99_index =
      std::min(samples.size() - 1u,
               static_cast<size_t>(std::ceil(samples.size() * 0.99)) - 1u);
  out.p99 = samples[p99_index];
  return out;
}

void print_ms_summary(const char *group,
                      const char *metric,
                      const SampleSummary &summary,
                      int runs,
                      double target_ms) {
  const double p50_ms = summary.p50_ns / 1.0e6;
  const double p99_ms = summary.p99_ns / 1.0e6;
  std::printf(
      "tmnn %s [%s]: p50=%.3f ms p99=%.3f ms min=%.3f ms max=%.3f ms "
      "target=%.3f ms p50_ok=%s p99_ok=%s runs=%d\n",
      group, metric, p50_ms, p99_ms, summary.min_ns / 1.0e6,
      summary.max_ns / 1.0e6, target_ms, p50_ms <= target_ms ? "true" : "false",
      p99_ms <= target_ms ? "true" : "false", runs);
}

void print_breakdown_ms_summary(const char *group,
                                const char *metric,
                                const SampleSummary &summary,
                                const SampleSummary &total_summary,
                                int runs) {
  const double share_p50 =
      total_summary.p50_ns > 0.0 ? (summary.p50_ns * 100.0 / total_summary.p50_ns)
                                 : 0.0;
  std::printf(
      "tmnn %s [%s]: p50=%.3f ms p99=%.3f ms share_p50=%.1f%% runs=%d\n",
      group, metric, summary.p50_ns / 1.0e6, summary.p99_ns / 1.0e6, share_p50,
      runs);
}

void print_breakdown_us_summary(const char *group,
                                const char *metric,
                                const DoubleSampleSummary &summary,
                                const SampleSummary &total_summary,
                                int runs) {
  const double share_p50 = total_summary.p50_ns > 0.0
                               ? (summary.p50 * 1.0e3 * 100.0 / total_summary.p50_ns)
                               : 0.0;
  std::printf(
      "tmnn %s [%s]: p50=%.1f us p99=%.1f us share_p50=%.1f%% runs=%d\n",
      group, metric, summary.p50, summary.p99, share_p50, runs);
}

void print_training_step_profile_breakdown(
    const char *group, const std::vector<TrainingStepProfile> &profiles) {
  if (profiles.empty()) {
    std::printf("tmnn %s breakdown: unavailable\n", group);
    return;
  }

  std::vector<uint64_t> total_samples;
  total_samples.reserve(profiles.size());
  for (const auto &profile : profiles)
    total_samples.push_back(profile.total_ns);
  const auto total_summary = summarize_ns(std::move(total_samples));
  print_breakdown_ms_summary(group, "profile_total", total_summary, total_summary,
                             static_cast<int>(profiles.size()));

  struct NsMetricSpec {
    const char *name;
    uint64_t TrainingStepProfile::*member;
  };
  const NsMetricSpec kNsMetrics[] = {
      {"morton_sort", &TrainingStepProfile::morton_sort_ns},
      {"enqueue_total", &TrainingStepProfile::enqueue_total_ns},
      {"drain_pending", &TrainingStepProfile::drain_pending_ns},
      {"prepare_step_lane", &TrainingStepProfile::prepare_step_lane_ns},
      {"fill_train_params", &TrainingStepProfile::fill_train_params_ns},
      {"resolve_bindings", &TrainingStepProfile::resolve_bindings_ns},
      {"submit_forward_backward", &TrainingStepProfile::submit_forward_backward_ns},
      {"finalize_total", &TrainingStepProfile::finalize_total_ns},
      {"wait_pending", &TrainingStepProfile::wait_pending_ns},
      {"wait_fwd_bwd_fill", &TrainingStepProfile::wait_fwd_bwd_fill_ns},
      {"wait_fwd_bwd_dispatch",
       &TrainingStepProfile::wait_fwd_bwd_dispatch_ns},
      {"fill_adam_params_pre_finalize",
       &TrainingStepProfile::fill_adam_params_pre_finalize_ns},
      {"finalize_step_readback", &TrainingStepProfile::finalize_step_readback_ns},
      {"numerics_report", &TrainingStepProfile::numerics_report_ns},
      {"numerics_backward_readback",
       &TrainingStepProfile::numerics_backward_readback_ns},
      {"numerics_backward_scan",
       &TrainingStepProfile::numerics_backward_scan_ns},
      {"numerics_update_readback",
       &TrainingStepProfile::numerics_update_readback_ns},
      {"numerics_update_scan", &TrainingStepProfile::numerics_update_scan_ns},
      {"fill_adam_params_apply",
       &TrainingStepProfile::fill_adam_params_apply_ns},
      {"prepare_sparse_hash_adam",
       &TrainingStepProfile::prepare_sparse_hash_adam_ns},
      {"submit_adam", &TrainingStepProfile::submit_adam_ns},
      {"sync_config_weights", &TrainingStepProfile::sync_config_weights_ns},
      {"append_extra_losses", &TrainingStepProfile::append_extra_losses_ns},
      {"probe_aggregation", &TrainingStepProfile::probe_aggregation_ns},
      {"uncategorized", &TrainingStepProfile::uncategorized_ns},
  };
  for (const auto &spec : kNsMetrics) {
    std::vector<uint64_t> samples;
    samples.reserve(profiles.size());
    bool any_nonzero = false;
    for (const auto &profile : profiles) {
      const auto value = profile.*(spec.member);
      samples.push_back(value);
      any_nonzero = any_nonzero || value != 0;
    }
    if (!any_nonzero)
      continue;
    print_breakdown_ms_summary(group, spec.name, summarize_ns(std::move(samples)),
                               total_summary, static_cast<int>(profiles.size()));
  }

  struct UsMetricSpec {
    const char *name;
    double TrainingStepProfile::*member;
  };
  const UsMetricSpec kUsMetrics[] = {
      {"gpu_fwd_bwd", &TrainingStepProfile::gpu_fwd_bwd_us},
      {"gpu_adam", &TrainingStepProfile::gpu_adam_us},
  };
  for (const auto &spec : kUsMetrics) {
    std::vector<double> samples;
    samples.reserve(profiles.size());
    bool any_nonzero = false;
    for (const auto &profile : profiles) {
      const auto value = profile.*(spec.member);
      samples.push_back(value);
      any_nonzero = any_nonzero || value != 0.0;
    }
    if (!any_nonzero)
      continue;
    print_breakdown_us_summary(group, spec.name,
                               summarize_double(std::move(samples)),
                               total_summary, static_cast<int>(profiles.size()));
  }
}

void print_training_route(const char *label, const Trainer &trainer) {
  const auto inspection = trainer.inspect_runtime();
  if (!inspection.has_value()) {
    std::printf("tmnn %s route: unavailable\n", label);
    return;
  }

  const auto &step = inspection->training_step;
  std::printf(
      "tmnn %s route: entry=%s available=%s safe_family=%s "
      "simd=%s->%s fp16=%s->%s tg_cache=%s->%s tg=%u points_per_tg=%u "
      "tg_mem=%u batch=%u\n",
      label, step.entry_point.empty() ? "<none>" : step.entry_point.c_str(),
      step.available ? "true" : "false",
      inspection->safe_family_active ? "true" : "false",
      step.requested_simd ? "true" : "false",
      step.realized_simd ? "true" : "false",
      step.requested_fp16 ? "true" : "false",
      step.realized_fp16 ? "true" : "false",
      step.requested_tg_weight_cache ? "true" : "false",
      step.realized_tg_weight_cache ? "true" : "false", step.threadgroup_size,
      step.points_per_threadgroup, step.threadgroup_memory_bytes,
      inspection->batch_size);
}

HashGridEncoding::Config make_neulat_encoding_config() {
  HashGridEncoding::Config cfg;
  cfg.log2_hashmap_size = kNeulatLog2HashmapSize;
  return cfg;
}

FullyFusedMLP::Config
make_neulat_network_config(const HashGridEncoding::Config &enc_cfg) {
  FullyFusedMLP::Config cfg;
  cfg.hidden_dim = kNeulatHiddenDim;
  cfg.n_input = enc_cfg.num_levels * enc_cfg.features_per_level;
  return cfg;
}

TrainerConfig make_neulat_trainer_config() {
  TrainerConfig cfg;
  cfg.batch_size = kNeulatBatchSize;
  return cfg;
}

void make_sphere_batch(uint32_t batch_size,
                       uint32_t input_dims,
                       uint32_t target_dims,
                       std::vector<float> &positions,
                       std::vector<float> &targets,
                       uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  positions.assign(static_cast<size_t>(batch_size) * input_dims, 0.0f);
  targets.assign(static_cast<size_t>(batch_size) * target_dims, 0.0f);
  for (uint32_t i = 0; i < batch_size; ++i) {
    const float x = dist(rng);
    const float y = dist(rng);
    const float z = dist(rng);
    positions[static_cast<size_t>(i) * input_dims + 0u] = x;
    if (input_dims > 1u)
      positions[static_cast<size_t>(i) * input_dims + 1u] = y;
    if (input_dims > 2u)
      positions[static_cast<size_t>(i) * input_dims + 2u] = z;
    targets[static_cast<size_t>(i) * target_dims + 0u] =
        std::sqrt(x * x + y * y + z * z) - 0.5f;
  }
}

void make_eval_positions(uint32_t batch_size,
                         uint32_t input_dims,
                         std::vector<float> &positions,
                         uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  positions.assign(static_cast<size_t>(batch_size) * input_dims, 0.0f);
  for (uint32_t i = 0; i < batch_size; ++i) {
    positions[static_cast<size_t>(i) * input_dims + 0u] = dist(rng);
    if (input_dims > 1u)
      positions[static_cast<size_t>(i) * input_dims + 1u] = dist(rng);
    if (input_dims > 2u)
      positions[static_cast<size_t>(i) * input_dims + 2u] = dist(rng);
  }
}

template <typename StepFn>
std::vector<uint64_t> collect_timing_samples(int warmup_runs,
                                             int timed_runs,
                                             StepFn &&step_fn) {
  for (int i = 0; i < warmup_runs; ++i) {
    if (!step_fn(i, false)) {
      std::fprintf(stderr, "benchmark warmup failed at iteration %d\n", i);
      std::exit(1);
    }
  }

  std::vector<uint64_t> samples;
  samples.reserve(static_cast<size_t>(timed_runs));
  for (int i = 0; i < timed_runs; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    if (!step_fn(i, true)) {
      std::fprintf(stderr, "benchmark timed run failed at iteration %d\n", i);
      std::exit(1);
    }
    const auto t1 = std::chrono::steady_clock::now();
    samples.push_back(static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()));
  }
  return samples;
}

class TimedRuntime final : public ITrainerRuntime {
public:
  explicit TimedRuntime(std::chrono::microseconds step_delay)
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
  std::chrono::microseconds step_delay_;
};

detail::AutotuneSearchRuntimeFactory make_runtime_factory() {
  const std::unordered_map<NetworkFamily, std::chrono::microseconds> delays{
      {NetworkFamily::FullyFusedMetal, std::chrono::microseconds(900)},
      {NetworkFamily::TiledMetal, std::chrono::microseconds(450)},
      {NetworkFamily::SafeDebugMetal, std::chrono::microseconds(1800)},
  };
  return [delays](const NetworkPlan &plan,
                  const std::shared_ptr<MetalContext> &)
             -> std::unique_ptr<ITrainerRuntime> {
    return std::make_unique<TimedRuntime>(delays.at(plan.selected_family));
  };
}

void run_planner_benchmark(int iterations) {
  auto model = make_model();
  NetworkFactoryOptions options;
  std::vector<uint64_t> samples;
  samples.reserve(static_cast<size_t>(iterations));
  for (int i = 0; i < iterations; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto plan = plan_network(*model, make_synthetic_caps(), options);
    const auto t1 = std::chrono::steady_clock::now();
    if (plan.selected_family == NetworkFamily::SafeDebugMetal &&
        !plan.fused_eval && !plan.fused_train) {
      std::fprintf(stderr, "unexpected planner result\n");
      std::exit(1);
    }
    samples.push_back(static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()));
  }

  std::printf("tmnn planner [synthetic-standard-3d]: median=%.0f ns runs=%d\n",
              median_ns(std::move(samples)), iterations);
}

void run_morton_benchmark(int iterations, uint32_t batch_size) {
  std::mt19937 rng(42u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> positions(static_cast<size_t>(batch_size) * 3u);
  std::vector<float> targets(static_cast<size_t>(batch_size), 0.0f);
  for (auto &value : positions)
    value = dist(rng);
  for (auto &value : targets)
    value = dist(rng);

  std::vector<uint32_t> indices;
  std::vector<uint32_t> codes;
  std::vector<float> sorted_positions;
  std::vector<float> sorted_targets;
  std::vector<uint64_t> samples;
  samples.reserve(static_cast<size_t>(iterations));
  for (int i = 0; i < iterations; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    tmnn::detail::morton_sort_batch(positions.data(), targets.data(),
                                    static_cast<int>(batch_size), 3u, 1u, indices,
                                    codes, sorted_positions, sorted_targets);
    const auto t1 = std::chrono::steady_clock::now();
    samples.push_back(static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()));
  }

  std::printf("tmnn morton-sort [batch=%u, dims=3]: median=%.0f ns runs=%d\n",
              batch_size, median_ns(std::move(samples)), iterations);
}

void run_autotune_benchmark(bool smoke) {
  auto ctx = MetalContext::create();
  if (!ctx->is_gpu_available()) {
    std::printf("tmnn autotune-search [synthetic-runtimes]: skipped (no GPU)\n");
    return;
  }

  auto model = make_model();
  NetworkFactoryOptions options;
  options.metal_context = ctx;
  options.enable_bounded_autotune_search = true;
  options.autotune_search_batch_size = 64u;
  options.autotune_search_measure_steps = smoke ? 1u : 2u;

  const auto baseline_plan = model->plan(options);

  const auto cold_t0 = std::chrono::steady_clock::now();
  const auto cold = tmnn::detail::run_bounded_autotune_search(
      *model, baseline_plan, options, ctx, 64, "tmnn_runtime_benchmarks",
      make_runtime_factory());
  const auto cold_t1 = std::chrono::steady_clock::now();

  const auto hot_t0 = std::chrono::steady_clock::now();
  const auto hot = tmnn::detail::run_bounded_autotune_search(
      *model, baseline_plan, options, ctx, 64, "tmnn_runtime_benchmarks",
      make_runtime_factory());
  const auto hot_t1 = std::chrono::steady_clock::now();

  const auto cold_ns = static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(cold_t1 - cold_t0)
          .count());
  const auto hot_ns = static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(hot_t1 - hot_t0)
          .count());

  std::printf("tmnn autotune-search [synthetic-runtimes]: cold=%.0f ns hot=%.0f "
              "ns family=%s manifest=%s measured_step=%llu\n",
              cold_ns, hot_ns,
              std::string(std::string_view(to_string(cold.selected_family))).c_str(),
              hot.from_autotune_manifest ? "true" : "false",
              static_cast<unsigned long long>(hot.autotune_measured_step_ns));
}

void run_default_trainer_hot_step_benchmark(bool smoke) {
  auto ctx = MetalContext::create();
  if (!ctx->is_gpu_available()) {
    std::printf("tmnn default-trainer hot-step: skipped (no GPU)\n");
    return;
  }

  const int timed_steps = smoke ? 5 : 10;

  auto run_case = [&](const char *label,
                      const HashGridEncoding::Config &enc_cfg,
                      const FullyFusedMLP::Config &net_cfg,
                      const TrainerConfig &train_cfg) {
    auto trainer = create_trainer(enc_cfg, net_cfg, train_cfg, ctx);
    print_training_route(label, trainer);
    const auto plan = trainer.batch_plan();
    const int N = static_cast<int>(plan.max_batch_size);

    std::mt19937 rng(42u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> positions(static_cast<size_t>(N) * plan.input_dims);
    std::vector<float> targets(static_cast<size_t>(N) * plan.target_dims);
    for (int i = 0; i < N; ++i) {
      const float x = dist(rng);
      const float y = dist(rng);
      const float z = dist(rng);
      positions[static_cast<size_t>(i) * plan.input_dims + 0] = x;
      positions[static_cast<size_t>(i) * plan.input_dims + 1] = y;
      positions[static_cast<size_t>(i) * plan.input_dims + 2] = z;
      targets[static_cast<size_t>(i)] =
          std::sqrt(x * x + y * y + z * z) - 0.5f;
    }

    auto warmup = trainer.training_step(positions.data(), targets.data(), N);
    if (!std::isfinite(warmup.loss)) {
      std::fprintf(stderr, "%s warmup produced non-finite loss\n", label);
      std::exit(1);
    }

    std::vector<uint64_t> samples;
    samples.reserve(static_cast<size_t>(timed_steps));
    for (int i = 0; i < timed_steps; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      auto result = trainer.training_step(positions.data(), targets.data(), N);
      const auto t1 = std::chrono::steady_clock::now();
      if (!std::isfinite(result.loss)) {
        std::fprintf(stderr, "%s timed step produced non-finite loss\n", label);
        std::exit(1);
      }
      samples.push_back(static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
              .count()));
    }

    std::printf(
        "tmnn default-trainer hot-step [%s, batch=%d]: median=%.3f ms runs=%d\n",
        label, N, median_ns(std::move(samples)) / 1.0e6, timed_steps);
  };

  auto sparse_cfg = default_trainer_config();
  const auto default_enc_cfg = default_trainer_encoding_config();
  const auto default_net_cfg = default_trainer_network_config(default_enc_cfg);
  run_case("default-path", default_enc_cfg, default_net_cfg, sparse_cfg);

  auto large_enc_cfg = default_enc_cfg;
  large_enc_cfg.log2_hashmap_size = 19;
  const auto large_net_cfg = default_trainer_network_config(large_enc_cfg);
  run_case("large-hash default-path", large_enc_cfg, large_net_cfg, sparse_cfg);
}

void run_neulat_latency_benchmark(bool smoke) {
  auto ctx = MetalContext::create();
  if (!ctx->is_gpu_available()) {
    std::printf("tmnn neulat-latency: skipped (no GPU)\n");
    return;
  }

  const int warmup_runs = smoke ? 10 : 1000;
  const int timed_runs = smoke ? 10 : 100;
  const int startup_runs = smoke ? 5 : 20;
  const int eval_count = smoke ? 1024 : kNeulatEvalCount;

  const auto enc_cfg = make_neulat_encoding_config();
  const auto net_cfg = make_neulat_network_config(enc_cfg);
  const auto train_cfg = make_neulat_trainer_config();
  const char *training_metric = "training_step hd=64 log2=19 batch=1024";
  auto trainer = create_trainer(enc_cfg, net_cfg, train_cfg, ctx);
  print_training_route("neulat-latency", trainer);
  TrainingStepProfilingOptions profiling_options;
  profiling_options.enabled = true;
  trainer.set_training_step_profiling(profiling_options);
  const auto plan = trainer.batch_plan();
  const auto batch_size = static_cast<uint32_t>(plan.max_batch_size);
  const auto input_dims = plan.input_dims;
  const auto target_dims = plan.target_dims;

  std::vector<float> train_positions;
  std::vector<float> train_targets;
  make_sphere_batch(batch_size, input_dims, target_dims, train_positions,
                    train_targets, 42u);

  for (int i = 0; i < warmup_runs; ++i) {
    const auto warmup = trainer.training_step(train_positions.data(),
                                              train_targets.data(),
                                              static_cast<int>(batch_size));
    if (!std::isfinite(warmup.loss)) {
      std::fprintf(stderr, "neulat training_step warmup failed at iteration %d\n",
                   i);
      std::exit(1);
    }
  }
  std::vector<uint64_t> training_samples;
  std::vector<TrainingStepProfile> training_profiles;
  training_samples.reserve(static_cast<size_t>(timed_runs));
  training_profiles.reserve(static_cast<size_t>(timed_runs));
  for (int i = 0; i < timed_runs; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto result = trainer.training_step(train_positions.data(),
                                              train_targets.data(),
                                              static_cast<int>(batch_size));
    const auto t1 = std::chrono::steady_clock::now();
    if (!std::isfinite(result.loss)) {
      std::fprintf(stderr, "neulat training_step timed run failed at iteration %d\n",
                   i);
      std::exit(1);
    }
    const auto profile = trainer.last_training_step_profile();
    if (!profile.has_value()) {
      std::fprintf(stderr,
                   "neulat training_step profile unavailable at iteration %d\n",
                   i);
      std::exit(1);
    }
    training_samples.push_back(static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()));
    training_profiles.push_back(*profile);
  }
  print_ms_summary("neulat-latency",
                   training_metric,
                   summarize_ns(std::move(training_samples)), timed_runs,
                   kNeulatTrainingTargetMs);
  print_training_step_profile_breakdown("neulat-latency-breakdown",
                                        training_profiles);

  std::vector<float> eval_positions;
  make_eval_positions(static_cast<uint32_t>(eval_count), input_dims,
                      eval_positions, 123u);
  std::vector<float> eval_output(static_cast<size_t>(eval_count) * target_dims,
                                 0.0f);
  auto evaluate_samples = collect_timing_samples(
      warmup_runs, timed_runs, [&](int, bool) {
        return trainer.evaluate(eval_positions.data(), eval_output.data(),
                                eval_count);
      });
  print_ms_summary("neulat-latency", "evaluate N=4096",
                   summarize_ns(std::move(evaluate_samples)), timed_runs,
                   kNeulatEvaluateTargetMs);

  std::vector<float> eval_grad_output(
      static_cast<size_t>(eval_count) * target_dims, 0.0f);
  std::vector<float> eval_grad(
      static_cast<size_t>(eval_count) * input_dims, 0.0f);
  auto gradient_samples = collect_timing_samples(
      warmup_runs, timed_runs, [&](int, bool) {
        return trainer.evaluate_with_gradient(eval_positions.data(),
                                              eval_grad_output.data(),
                                              eval_grad.data(), eval_count);
      });
  print_ms_summary("neulat-latency", "evaluate_with_gradient N=4096",
                   summarize_ns(std::move(gradient_samples)), timed_runs,
                   kNeulatGradientTargetMs);

  auto startup_positions = train_positions;
  auto startup_targets = train_targets;
  std::vector<uint64_t> startup_samples;
  startup_samples.reserve(static_cast<size_t>(startup_runs));
  for (int i = 0; i < startup_runs; ++i) {
    auto startup_ctx = MetalContext::create();
    if (!startup_ctx->is_gpu_available()) {
      std::printf("tmnn neulat-latency startup: skipped (no GPU)\n");
      return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    auto startup_trainer =
        create_trainer(enc_cfg, net_cfg, train_cfg, startup_ctx);
    const auto first_step = startup_trainer.training_step(
        startup_positions.data(), startup_targets.data(),
        static_cast<int>(batch_size));
    const auto t1 = std::chrono::steady_clock::now();
    if (!std::isfinite(first_step.loss)) {
      std::fprintf(stderr, "neulat startup first step produced non-finite loss\n");
      std::exit(1);
    }
    startup_samples.push_back(static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()));
  }
  print_ms_summary("neulat-latency",
                   "startup create_trainer->first_step (trainer-cold, process-warm)",
                   summarize_ns(std::move(startup_samples)), startup_runs,
                   kNeulatStartupTargetMs);
}

} // namespace

int main(int argc, char **argv) {
  bool smoke = false;
  bool only_neulat_latency = false;
  bool run_neulat_latency = false;
  bool allocation_trace = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string_view(argv[i]) == "--smoke")
      smoke = true;
    else if (std::string_view(argv[i]) == "--neulat-latency")
      run_neulat_latency = true;
    else if (std::string_view(argv[i]) == "--only-neulat-latency") {
      run_neulat_latency = true;
      only_neulat_latency = true;
    } else if (std::string_view(argv[i]) == "--allocation-trace")
      allocation_trace = true;
  }

  if (!only_neulat_latency) {
    run_planner_benchmark(smoke ? 32 : 1024);
    run_morton_benchmark(smoke ? 8 : 128, smoke ? 256u : 1024u);
    run_autotune_benchmark(smoke);
    // Reset counters here so the hot-step benchmark's own warmup is excluded
    // from the trace; the inner benchmark function takes care of warmup.
    if (allocation_trace) {
      tmnn::metal::reset_alloc_stats();
    }
    run_default_trainer_hot_step_benchmark(smoke);
  }
  if (run_neulat_latency)
    run_neulat_latency_benchmark(smoke);

  if (allocation_trace) {
    const auto &s = tmnn::metal::alloc_stats();
    std::printf("alloc-trace: create_buffer_calls=%llu bytes=%llu "
                "blit_copy_calls=%llu bytes=%llu "
                "buffer_contents_calls=%llu\n",
                (unsigned long long)s.create_buffer_calls.load(),
                (unsigned long long)s.create_buffer_bytes.load(),
                (unsigned long long)s.blit_copy_calls.load(),
                (unsigned long long)s.blit_copy_bytes.load(),
                (unsigned long long)s.buffer_contents_calls.load());
  }
  return 0;
}

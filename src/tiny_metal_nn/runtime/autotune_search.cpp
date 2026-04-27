/**
 * @file autotune_search.cpp
 * @brief Internal bounded-autotune search orchestration for tmnn runtimes.
 */

#include "tiny_metal_nn/runtime/autotune_search.h"

#include "tiny_metal_nn/runtime/family_policy.h"
#include "tiny_metal_nn/runtime/metal_context_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace tmnn::detail {
namespace {

constexpr double kAutotuneMinimumRelativeGain = 0.02;
struct AutotuneSearchCandidateResult {
  NetworkPlan plan;
  AutotuneCandidateMeasurement measurement;
};

uint64_t resolve_autotune_objective_ns(AutotuneSearchObjective objective,
                                       uint64_t build_ns,
                                       uint64_t total_measured_ns,
                                       uint64_t measured_step_ns) {
  switch (objective) {
  case AutotuneSearchObjective::SteadyStateStep:
    return measured_step_ns;
  case AutotuneSearchObjective::BuildPlusMeasureWindow:
    return build_ns + total_measured_ns;
  }
  return measured_step_ns;
}

std::vector<float> make_autotune_search_inputs(uint32_t batch_size,
                                               uint32_t input_dims) {
  if (input_dims == 0) {
    throw std::invalid_argument(
        "make_autotune_search_inputs: input_dims must be > 0");
  }
  std::vector<float> inputs(static_cast<size_t>(batch_size) * input_dims, 0.0f);
  for (size_t i = 0; i < inputs.size(); ++i) {
    const int centered = static_cast<int>(i % 23u) - 11;
    inputs[i] = static_cast<float>(centered) * 0.03125f;
  }
  return inputs;
}

std::vector<float> make_autotune_search_targets(uint32_t batch_size,
                                                uint32_t target_dims) {
  if (target_dims == 0) {
    throw std::invalid_argument(
        "make_autotune_search_targets: target_dims must be > 0");
  }
  std::vector<float> targets(static_cast<size_t>(batch_size) * target_dims,
                             0.0f);
  for (size_t i = 0; i < targets.size(); ++i) {
    const int centered = static_cast<int>(i % 19u) - 9;
    targets[i] = static_cast<float>(centered) * 0.015625f;
  }
  return targets;
}

} // namespace

NetworkPlan run_bounded_autotune_search(
    const NetworkWithInputEncoding &model, const NetworkPlan &baseline_plan,
    const NetworkFactoryOptions &factory_options,
    const std::shared_ptr<MetalContext> &ctx, int configured_batch_size,
    const char *surface,
    const AutotuneSearchRuntimeFactory &make_runtime_for_plan,
    const AutotuneSearchExtraBuildCostFn &measure_extra_build_cost) {
  if (!factory_options.enable_bounded_autotune_search ||
      factory_options.forced_family.has_value() || !ctx ||
      !ctx->is_gpu_available() || baseline_plan.candidate_families.size() < 2) {
    return baseline_plan;
  }

  if (factory_options.autotune_search_batch_size == 0) {
    throw std::invalid_argument(std::string(surface) +
                                ": autotune_search_batch_size must be > 0 "
                                "when bounded autotune search is enabled");
  }
  if (factory_options.autotune_search_measure_steps == 0) {
    throw std::invalid_argument(std::string(surface) +
                                ": autotune_search_measure_steps must be > 0 "
                                "when bounded autotune search is enabled");
  }
  if (configured_batch_size <= 0) {
    throw std::invalid_argument(std::string(surface) +
                                ": trainer batch_size must be > 0 when bounded "
                                "autotune search is enabled");
  }

  const uint32_t measurement_batch_size =
      std::min(static_cast<uint32_t>(configured_batch_size),
               factory_options.autotune_search_batch_size);
  const uint32_t warmup_steps = factory_options.autotune_search_warmup_steps;
  const uint32_t measurement_steps =
      factory_options.autotune_search_measure_steps;
  const AutotuneSearchObjective objective =
      factory_options.autotune_search_objective;

  if (auto existing = detail::context_lookup_autotune_entry(
          *ctx, baseline_plan.planner_fingerprint)) {
    const bool batch_matches =
        existing->measurement_batch_size == measurement_batch_size;
    const bool warmup_matches =
        existing->measurement_warmup_steps == warmup_steps;
    const bool steps_match = existing->measurement_steps == measurement_steps;
    const bool objective_matches =
        existing->measurement_objective == objective;
    const bool family_still_eligible =
        std::find(baseline_plan.candidate_families.begin(),
                  baseline_plan.candidate_families.end(),
                  existing->selected_family) !=
        baseline_plan.candidate_families.end();
    if (existing->selected_by_measurement && batch_matches && warmup_matches &&
        steps_match && objective_matches &&
        family_still_eligible &&
        existing->selected_family_measured_step_ns > 0) {
      const uint64_t objective_ns =
          existing->selected_family_objective_ns > 0
              ? existing->selected_family_objective_ns
              : existing->selected_family_measured_step_ns;
      return resolved_plan_with_selected_family(
          baseline_plan, existing->selected_family, true, true,
          existing->selected_family_measured_step_ns, objective, objective_ns);
    }
  }

  std::vector<AutotuneSearchCandidateResult> candidate_results;
  candidate_results.reserve(baseline_plan.candidate_families.size());
  size_t best_index = std::numeric_limits<size_t>::max();

  for (const auto family : baseline_plan.candidate_families) {
    AutotuneSearchCandidateResult result;
    result.measurement.family = family;

    try {
      auto measurement_ctx = MetalContext::create(ctx->desc());
      if (!measurement_ctx->is_gpu_available()) {
        throw std::runtime_error(
            "autotune search requires a GPU-backed measurement context");
      }

      NetworkFactoryOptions candidate_options = factory_options;
      candidate_options.metal_context = measurement_ctx;
      candidate_options.forced_family = family;
      candidate_options.enable_bounded_autotune_search = false;

      const auto build_begin = std::chrono::steady_clock::now();
      result.plan = model.plan(candidate_options);
      if (result.plan.selected_family != family) {
        throw std::runtime_error(
            "planner rejected an autotune candidate family that was marked "
            "eligible");
      }

      auto runtime = make_runtime_for_plan(result.plan, measurement_ctx);
      auto authority = runtime->runtime_authority();
      const auto build_end = std::chrono::steady_clock::now();
      result.measurement.build_ns = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(build_end -
                                                               build_begin)
              .count());
      if (objective == AutotuneSearchObjective::BuildPlusMeasureWindow &&
          measure_extra_build_cost) {
        result.measurement.build_ns +=
            measure_extra_build_cost(result.plan, measurement_ctx, authority);
      }
      const auto batch_plan = runtime->batch_plan();
      const uint32_t effective_batch_size =
          std::min(measurement_batch_size, batch_plan.max_batch_size);
      if (effective_batch_size == 0 || batch_plan.input_dims == 0 ||
          batch_plan.target_dims == 0) {
        throw std::runtime_error(
            "autotune search received an invalid runtime batch plan");
      }

      auto inputs =
          make_autotune_search_inputs(effective_batch_size, batch_plan.input_dims);
      auto targets = make_autotune_search_targets(effective_batch_size,
                                                  batch_plan.target_dims);

      for (uint32_t warmup_step = 0; warmup_step < warmup_steps;
           ++warmup_step) {
        const auto warmup_begin = std::chrono::steady_clock::now();
        const auto warmup =
            runtime->training_step(inputs.data(), targets.data(),
                                   static_cast<int>(effective_batch_size));
        const auto warmup_end = std::chrono::steady_clock::now();
        if (!std::isfinite(warmup.loss)) {
          throw std::runtime_error(
              "autotune warmup produced a non-finite training loss");
        }
        result.measurement.warmup_step_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(warmup_end -
                                                                 warmup_begin)
                .count());
      }

      uint64_t total_measured_ns = 0;
      for (uint32_t step = 0; step < measurement_steps; ++step) {
        const auto measure_begin = std::chrono::steady_clock::now();
        const auto measured =
            runtime->training_step(inputs.data(), targets.data(),
                                   static_cast<int>(effective_batch_size));
        const auto measure_end = std::chrono::steady_clock::now();
        if (!std::isfinite(measured.loss)) {
          throw std::runtime_error(
              "autotune measurement produced a non-finite training loss");
        }
        total_measured_ns +=
            static_cast<uint64_t>(std::chrono::duration_cast<
                                      std::chrono::nanoseconds>(measure_end -
                                                                measure_begin)
                                      .count());
      }

      result.measurement.measured_step_ns =
          total_measured_ns / measurement_steps;
      result.measurement.objective_ns =
          resolve_autotune_objective_ns(objective, result.measurement.build_ns,
                                        total_measured_ns,
                                        result.measurement.measured_step_ns);
      result.measurement.succeeded = true;
    } catch (const std::exception &e) {
      result.measurement.failure = e.what();
    }

    candidate_results.push_back(std::move(result));
    const auto &measured = candidate_results.back().measurement;
    if (measured.succeeded &&
        (best_index == std::numeric_limits<size_t>::max() ||
         measured.objective_ns <
             candidate_results[best_index].measurement.objective_ns)) {
      best_index = candidate_results.size() - 1;
    }
  }

  if (best_index == std::numeric_limits<size_t>::max()) {
    std::string failure = std::string(surface) +
                          ": bounded autotune search failed for all candidate "
                          "families";
    for (const auto &candidate : candidate_results) {
      failure += "; ";
      failure += std::string(to_string(candidate.measurement.family));
      failure += ": ";
      failure += candidate.measurement.failure.empty()
                     ? "unknown failure"
                     : candidate.measurement.failure;
    }
    throw std::runtime_error(failure);
  }

  size_t selected_index = best_index;
  const auto baseline_index =
      std::find_if(candidate_results.begin(), candidate_results.end(),
                   [&](const auto &candidate) {
                     return candidate.plan.selected_family ==
                            baseline_plan.selected_family;
                   });
  if (baseline_index != candidate_results.end() &&
      baseline_index->measurement.succeeded &&
      baseline_index->measurement.objective_ns > 0 &&
      candidate_results[best_index].measurement.succeeded &&
      candidate_results[best_index].measurement.objective_ns > 0 &&
      best_index != static_cast<size_t>(
                        std::distance(candidate_results.begin(), baseline_index))) {
    const double baseline_ns =
        static_cast<double>(baseline_index->measurement.objective_ns);
    const double best_ns =
        static_cast<double>(candidate_results[best_index].measurement.objective_ns);
    const double relative_gain = 1.0 - (best_ns / baseline_ns);
    if (relative_gain <= kAutotuneMinimumRelativeGain) {
      selected_index =
          static_cast<size_t>(std::distance(candidate_results.begin(),
                                            baseline_index));
    }
  }

  const auto measured = resolved_plan_with_selected_family(
      baseline_plan, candidate_results[selected_index].plan.selected_family,
      false, true,
      candidate_results[selected_index].measurement.measured_step_ns, objective,
      candidate_results[selected_index].measurement.objective_ns);

  std::vector<AutotuneCandidateMeasurement> measurements;
  measurements.reserve(candidate_results.size());
  for (const auto &candidate : candidate_results)
    measurements.push_back(candidate.measurement);

  detail::context_record_autotune_entry(
      *ctx,
      AutotuneManifestEntry{
          baseline_plan.planner_fingerprint,
          measured.selected_family,
          baseline_plan.candidate_families,
          baseline_plan.reasons,
          false,
          true,
          measurement_batch_size,
          warmup_steps,
          measurement_steps,
          measured.autotune_measured_step_ns,
          std::move(measurements),
          objective,
          measured.autotune_measured_objective_ns,
      });

  return measured;
}

} // namespace tmnn::detail

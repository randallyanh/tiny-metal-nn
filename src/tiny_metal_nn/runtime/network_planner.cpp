/**
 * @file network_planner.cpp
 * @brief Planner implementation for tmnn network family selection.
 */

#include "tiny-metal-nn/detail/network_planning.h"

#include "tiny-metal-nn/fully_fused_mlp.h"
#include "tiny-metal-nn/hash_grid.h"
#include "tiny-metal-nn/metal_context.h"
#include "tiny-metal-nn/network_with_input_encoding.h"
#include "tiny-metal-nn/rotated_hash_grid.h"
#include "tiny_metal_nn/runtime/family_policy.h"
#include "tiny_metal_nn/runtime/metal_context_internal.h"

#include <algorithm>
#include <optional>

namespace tmnn {
namespace {

constexpr uint32_t kSimdTrainThreadgroupLimitBytes = 32768u;
constexpr uint64_t kFnvOffset = 14695981039346656037ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

void add_reason(std::vector<PlannerFallbackReason> &reasons,
                PlannerFallbackReason reason) {
  if (reason == PlannerFallbackReason::None)
    return;
  if (std::find(reasons.begin(), reasons.end(), reason) == reasons.end())
    reasons.push_back(reason);
}

void hash_u64(uint64_t &state, uint64_t value) {
  state ^= value;
  state *= kFnvPrime;
}

void hash_bool(uint64_t &state, bool value) {
  hash_u64(state, value ? 1u : 0u);
}

void hash_str(uint64_t &state, std::string_view value) {
  for (const char c : value)
    hash_u64(state, static_cast<unsigned char>(c));
}

void add_candidate(std::vector<NetworkFamily> &candidates, NetworkFamily family) {
  if (std::find(candidates.begin(), candidates.end(), family) ==
      candidates.end()) {
    candidates.push_back(family);
  }
}

bool is_gpu_available(const DeviceCapabilities &caps) {
  return !caps.device_name.empty() || caps.max_threads_per_tg != 0 ||
         caps.max_threadgroup_memory_bytes != 0 || caps.supports_fp16 ||
         caps.supports_simdgroup_matrix ||
         caps.supports_nonuniform_threadgroups ||
         caps.supports_binary_archive;
}

uint32_t effective_simd_train_limit(const DeviceCapabilities &caps) {
  if (caps.max_threadgroup_memory_bytes == 0)
    return 0;
  return std::min(caps.max_threadgroup_memory_bytes,
                  kSimdTrainThreadgroupLimitBytes);
}

uint32_t simd_train_tg_bytes(const FullyFusedMLP &network) {
  const auto &cfg = network.config();
  const uint32_t mlp_weight_bytes =
      static_cast<uint32_t>(network.n_params()) * sizeof(float);
  const uint32_t act_buffers =
      static_cast<uint32_t>((cfg.num_hidden_layers >= 2) ? 3 : 2);
  const uint32_t activation_bytes =
      act_buffers * 8u * static_cast<uint32_t>(cfg.hidden_dim) *
      static_cast<uint32_t>(sizeof(float));
  const uint32_t feature_bytes =
      8u * static_cast<uint32_t>(cfg.n_input) *
      static_cast<uint32_t>(sizeof(float));
  return mlp_weight_bytes + activation_bytes + feature_bytes;
}

uint64_t planner_fingerprint(const Encoding &encoding, const Network &network,
                             const NetworkFactoryOptions &options) {
  uint64_t state = kFnvOffset;
  hash_str(state, encoding.name());
  hash_str(state, network.name());
  hash_u64(state, static_cast<uint64_t>(options.backend));
  hash_bool(state, options.enable_safe_debug_family);
  hash_bool(state, options.forced_family.has_value());
  if (options.forced_family) {
    hash_u64(state, static_cast<uint64_t>(*options.forced_family));
  }
  hash_bool(state, options.multi_output.has_value());
  if (options.multi_output) {
    hash_u64(state, static_cast<uint64_t>(options.multi_output->semantics));
    hash_u64(state, options.multi_output->bc_dim_count);
  }

  if (const auto *hash_grid = dynamic_cast<const HashGridEncoding *>(&encoding)) {
    const auto &cfg = hash_grid->config();
    hash_u64(state, static_cast<uint64_t>(cfg.num_levels));
    hash_u64(state, static_cast<uint64_t>(cfg.features_per_level));
    hash_u64(state, static_cast<uint64_t>(cfg.log2_hashmap_size));
    hash_u64(state, static_cast<uint64_t>(cfg.input_dims));
    hash_u64(state, static_cast<uint64_t>(cfg.base_resolution));
    hash_u64(state, static_cast<uint64_t>(cfg.per_level_scale * 1000000.0f));
  } else if (const auto *rotated =
                 dynamic_cast<const RotatedHashGridEncoding *>(&encoding)) {
    const auto &cfg = rotated->config();
    hash_u64(state, static_cast<uint64_t>(cfg.num_levels));
    hash_u64(state, static_cast<uint64_t>(cfg.features_per_level));
    hash_u64(state, static_cast<uint64_t>(cfg.log2_hashmap_size));
    hash_u64(state, static_cast<uint64_t>(cfg.input_dims));
    hash_u64(state, static_cast<uint64_t>(cfg.base_resolution));
    hash_u64(state, static_cast<uint64_t>(cfg.per_level_scale * 1000000.0f));
  }

  if (const auto *mlp = dynamic_cast<const FullyFusedMLP *>(&network)) {
    const auto &cfg = mlp->config();
    hash_u64(state, static_cast<uint64_t>(cfg.n_input));
    hash_u64(state, static_cast<uint64_t>(cfg.hidden_dim));
    hash_u64(state, static_cast<uint64_t>(cfg.n_output));
    hash_u64(state, static_cast<uint64_t>(cfg.num_hidden_layers));
  }

  return state;
}

NetworkPlan plan_with_capabilities(const Encoding &encoding,
                                   const Network &network,
                                   const DeviceCapabilities &capabilities,
                                   const NetworkFactoryOptions &options) {
  NetworkPlan plan;
  std::optional<AutotuneManifestEntry> matched_manifest_entry;
  plan.capabilities = capabilities;
  plan.planner_fingerprint = planner_fingerprint(encoding, network, options);

  if (options.backend == ExecutionBackendPreference::RequireCPUReference) {
    add_reason(plan.reasons,
               PlannerFallbackReason::UnsupportedBackendPreference);
    return plan;
  }

  if (!is_gpu_available(capabilities)) {
    add_reason(plan.reasons, PlannerFallbackReason::NoGPUContext);
    return plan;
  }

  const auto *hash_grid = dynamic_cast<const HashGridEncoding *>(&encoding);
  const auto *rotated_hash_grid =
      dynamic_cast<const RotatedHashGridEncoding *>(&encoding);
  const auto *mlp = dynamic_cast<const FullyFusedMLP *>(&network);
  if ((!hash_grid && !rotated_hash_grid) || !mlp) {
    add_reason(plan.reasons, PlannerFallbackReason::UnsupportedEncoding);
    return plan;
  }

  const bool rotated_encoding = rotated_hash_grid != nullptr;
  const int spatial_dims = hash_grid ? hash_grid->config().input_dims
                                     : rotated_hash_grid->config().input_dims;
  const int encoded_width = mlp->config().n_input;
  const int hidden_dim = mlp->config().hidden_dim;
  const int num_outputs = mlp->config().n_output;

  if (spatial_dims != 3 && spatial_dims != 4) {
    add_reason(plan.reasons, PlannerFallbackReason::UnsupportedInputDims);
    return plan;
  }
  if (rotated_encoding && spatial_dims != 3) {
    add_reason(plan.reasons, PlannerFallbackReason::UnsupportedInputDims);
    return plan;
  }
  if (num_outputs < 1) {
    add_reason(plan.reasons, PlannerFallbackReason::UnsupportedOutputDims);
    return plan;
  }

  const bool fused_encoding = !rotated_encoding;
  const bool fused_spatial_dims = spatial_dims == 3;
  const bool fused_output_dims = num_outputs == 1;
  const bool hidden_dim_supported = (hidden_dim % 8) == 0;
  const bool encoded_width_supported = (encoded_width % 8) == 0;
  const bool simd_capable = capabilities.supports_simdgroup_matrix;
  const bool eval_fused_eligible =
      fused_encoding && fused_spatial_dims && fused_output_dims &&
      hidden_dim_supported &&
      encoded_width_supported && simd_capable;
  const bool train_tg_supported =
      eval_fused_eligible &&
      simd_train_tg_bytes(*mlp) <= effective_simd_train_limit(capabilities);

  plan.fused_eval = eval_fused_eligible;
  plan.fused_train = train_tg_supported;

  if (!fused_encoding) {
    add_reason(plan.reasons,
               PlannerFallbackReason::EncodingRequiresTiledFamily);
  }
  if (!fused_spatial_dims)
    add_reason(plan.reasons, PlannerFallbackReason::UnsupportedInputDims);
  if (!fused_output_dims)
    add_reason(plan.reasons,
               PlannerFallbackReason::MultiOutputFusedUnsupported);
  if (!hidden_dim_supported)
    add_reason(plan.reasons, PlannerFallbackReason::HiddenDimUnsupported);
  if (!encoded_width_supported)
    add_reason(plan.reasons, PlannerFallbackReason::UnsupportedInputDims);
  if (!simd_capable)
    add_reason(plan.reasons,
               PlannerFallbackReason::MissingSIMDGroupMatrix);
  if (eval_fused_eligible && !train_tg_supported)
    add_reason(plan.reasons,
               PlannerFallbackReason::ThreadgroupMemoryExceeded);

  if (options.forced_family == NetworkFamily::SafeDebugMetal) {
    plan.candidate_families = {NetworkFamily::SafeDebugMetal};
  } else if (options.forced_family == NetworkFamily::TiledMetal) {
    plan.candidate_families = {NetworkFamily::TiledMetal};
  } else {
    if (plan.fused_train)
      add_candidate(plan.candidate_families, NetworkFamily::FullyFusedMetal);
    add_candidate(plan.candidate_families, NetworkFamily::TiledMetal);
  }

  if (options.forced_family == NetworkFamily::SafeDebugMetal) {
    add_reason(plan.reasons, PlannerFallbackReason::ForcedSafeDebug);
    plan.selected_family = NetworkFamily::SafeDebugMetal;
    plan.fused_eval = false;
    plan.fused_train = false;
  } else if (options.forced_family == NetworkFamily::TiledMetal) {
    add_reason(plan.reasons, PlannerFallbackReason::ForcedTiled);
    plan.selected_family = NetworkFamily::TiledMetal;
    plan.fused_eval = false;
    plan.fused_train = false;
  } else if (options.forced_family == NetworkFamily::FullyFusedMetal &&
             plan.fused_train) {
    plan.selected_family = NetworkFamily::FullyFusedMetal;
  } else if (plan.fused_train) {
    plan.selected_family = NetworkFamily::FullyFusedMetal;
  } else {
    plan.selected_family = NetworkFamily::TiledMetal;
  }
  plan.autotune_search_objective = options.autotune_search_objective;

  if (options.metal_context && !options.forced_family.has_value()) {
    if (auto entry = detail::context_lookup_autotune_entry(
            *options.metal_context, plan.planner_fingerprint)) {
      const bool family_still_eligible =
          std::find(plan.candidate_families.begin(), plan.candidate_families.end(),
                    entry->selected_family) != plan.candidate_families.end();
      const bool measurement_policy_matches =
          !entry->selected_by_measurement ||
          (entry->measurement_warmup_steps == options.autotune_search_warmup_steps &&
           entry->measurement_steps == options.autotune_search_measure_steps &&
           entry->measurement_objective == options.autotune_search_objective);
      if (family_still_eligible && measurement_policy_matches) {
        matched_manifest_entry = entry;
        const uint64_t objective_ns =
            entry->selected_family_objective_ns > 0
                ? entry->selected_family_objective_ns
                : entry->selected_family_measured_step_ns;
        plan = detail::resolved_plan_with_selected_family(
            plan, entry->selected_family, true, entry->selected_by_measurement,
            entry->selected_family_measured_step_ns,
            entry->measurement_objective, objective_ns);
      }
    }
  }

  if (options.metal_context) {
    if (matched_manifest_entry &&
        matched_manifest_entry->selected_family == plan.selected_family) {
      detail::context_record_autotune_entry(*options.metal_context,
                                            *matched_manifest_entry);
    } else {
      detail::context_record_autotune_entry(
          *options.metal_context,
          AutotuneManifestEntry{plan.planner_fingerprint, plan.selected_family,
                                plan.candidate_families, plan.reasons,
                                options.forced_family.has_value(), false, 0u,
                                options.autotune_search_warmup_steps, 0u, 0u,
                                {}, options.autotune_search_objective, 0u});
    }
  }

  return plan;
}

std::shared_ptr<MetalContext>
resolve_context_for_planning(const NetworkFactoryOptions &options) {
  if (options.metal_context)
    return options.metal_context;
  if (options.backend == ExecutionBackendPreference::RequireCPUReference)
    return nullptr;
  return MetalContext::create();
}

} // namespace

NetworkPlan plan_network(const Encoding &encoding, const Network &network,
                         const DeviceCapabilities &capabilities,
                         const NetworkFactoryOptions &options) {
  return plan_with_capabilities(encoding, network, capabilities, options);
}

NetworkPlan plan_network(const NetworkWithInputEncoding &model,
                         const DeviceCapabilities &capabilities,
                         const NetworkFactoryOptions &options) {
  return plan_network(*model.encoding(), *model.network(), capabilities, options);
}

NetworkPlan plan_network(const Encoding &encoding, const Network &network,
                         const NetworkFactoryOptions &options) {
  auto ctx = resolve_context_for_planning(options);
  NetworkFactoryOptions resolved = options;
  resolved.metal_context = ctx;
  if (!ctx)
    return plan_network(encoding, network, DeviceCapabilities{}, resolved);
  return plan_network(encoding, network, ctx->capabilities(), resolved);
}

NetworkPlan plan_network(const NetworkWithInputEncoding &model,
                         const NetworkFactoryOptions &options) {
  return plan_network(*model.encoding(), *model.network(), options);
}

NetworkPlan
NetworkWithInputEncoding::plan(const NetworkFactoryOptions &options) const {
  return tmnn::plan_network(*this, options);
}

} // namespace tmnn

/**
 * @file kernel_compiler.cpp
 * @brief Kernel compiler contract around MLPKernelEmitter.
 */

#include "tiny-metal-nn/kernels/kernel_compiler.h"

#include "tiny-metal-nn/kernels/mlp_kernel_emitter.h"

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace tmnn {
namespace {

constexpr uint64_t kFnvBasis = 14695981039346656037ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

void mix_hash(uint64_t &hash, uint64_t value) {
  hash ^= value;
  hash *= kFnvPrime;
}

uint64_t hash_train_params_layout(const tmnn::TrainParamsLayout &layout) {
  uint64_t hash = kFnvBasis;
  mix_hash(hash, static_cast<uint64_t>(layout.float_count));
  mix_hash(hash, static_cast<uint64_t>(layout.idx_n));
  mix_hash(hash, static_cast<uint64_t>(layout.idx_unsigned_mode));
  mix_hash(hash, static_cast<uint64_t>(layout.idx_loss_scale));
  mix_hash(hash, static_cast<uint64_t>(layout.idx_num_active_levels));
  return hash;
}

uint64_t hash_schema(const tmnn::extension::ExtensionSchema &schema) {
  uint64_t hash = kFnvBasis;
  mix_hash(hash, static_cast<uint64_t>(schema.input_dims));
  mix_hash(hash, static_cast<uint64_t>(schema.target_dims));
  mix_hash(hash, static_cast<uint64_t>(schema.reduction_terms));
  mix_hash(hash, static_cast<uint64_t>(schema.config_tail_floats));
  mix_hash(hash, hash_train_params_layout(schema.train_params_layout));
  return hash;
}

uint64_t hash_compile_contract(KernelRole role,
                               const tmnn::extension::KernelCompileSpec &spec) {
  uint64_t hash = kFnvBasis;
  mix_hash(hash, static_cast<uint64_t>(role));
  mix_hash(hash, static_cast<uint64_t>(spec.output_semantics));
  mix_hash(hash, static_cast<uint64_t>(spec.bc_dim_count));
  mix_hash(hash, static_cast<uint64_t>(spec.loss_kind));
  if (spec.loss_kind == tmnn::extension::LossKind::Huber) {
    uint32_t hd_bits;
    std::memcpy(&hd_bits, &spec.huber_delta, sizeof(uint32_t));
    mix_hash(hash, static_cast<uint64_t>(hd_bits));
  }
  mix_hash(hash, spec.enable_probes ? 1ULL : 0ULL);
  return hash;
}

std::string entry_point_for(KernelRole role, const KernelSpec &spec) {
  const bool simd =
      spec.use_simd && spec.encoding == KernelSpec::Standard &&
      spec.spatial_dims == 3 && spec.num_outputs == 1;
  switch (role) {
  case KernelRole::Eval:
    return simd ? "neural_sdf_eval_simd" : "neural_sdf_eval_points";
  case KernelRole::Gradient:
    return simd ? "neural_sdf_analytical_gradient_simd"
                : "neural_sdf_analytical_gradient_points";
  case KernelRole::EvalGradient:
    return simd ? "neural_sdf_analytical_eval_gradient_simd"
                : "neural_sdf_analytical_eval_gradient_points";
  case KernelRole::TrainForwardBackward:
    return "neural_sdf_train_forward_backward";
  case KernelRole::BackwardFromExternalGrad:
    return "neural_sdf_train_external_grad";
  case KernelRole::ForwardForTraining:
    return "neural_sdf_forward_for_training";
  }
  throw std::invalid_argument("KernelCompiler: unsupported kernel role");
}

SpecializationDisableReason
simd_disable_reason(KernelRole role, const KernelSpec &spec, bool requested) {
  if (!requested)
    return SpecializationDisableReason::PreferenceDisabled;
  if (spec.spatial_dims != 3)
    return SpecializationDisableReason::UnsupportedSpatialDims;
  if (spec.encoding != KernelSpec::Standard)
    return SpecializationDisableReason::NonStandardEncoding;
  if (spec.num_outputs != 1)
    return SpecializationDisableReason::MultiOutputUnsupported;
  if (!spec.canUseSIMD())
    return SpecializationDisableReason::HiddenDimNotSIMDAligned;
  if ((spec.input_dim % 8) != 0)
    return SpecializationDisableReason::InputDimNotSIMDAligned;
  if (role == KernelRole::TrainForwardBackward &&
      spec.simdTrainTGBytes() > 32768)
    return SpecializationDisableReason::ThreadgroupMemoryExceeded;
  return SpecializationDisableReason::None;
}

SpecializationDisableReason scalar_fp16_eval_disable_reason(
    KernelRole role, const KernelSpec &spec, bool requested) {
  if (!requested)
    return SpecializationDisableReason::PreferenceDisabled;
  if (role != KernelRole::Eval)
    return SpecializationDisableReason::RoleUnsupported;
  if (spec.spatial_dims != 3)
    return SpecializationDisableReason::UnsupportedSpatialDims;
  if (spec.encoding != KernelSpec::Standard)
    return SpecializationDisableReason::NonStandardEncoding;
  if (spec.num_outputs != 1)
    return SpecializationDisableReason::MultiOutputUnsupported;
  if (spec.features_per_level != 2 || spec.num_hidden_layers <= 1 ||
      (spec.hidden_dim % 4) != 0)
    return SpecializationDisableReason::SchemaIncompatible;
  return SpecializationDisableReason::None;
}

KernelSpec resolve_spec(const KernelCompileRequest &request,
                        SpecializationDecision &decision) {
  KernelSpec resolved = request.spec;
  if (request.compile_spec.encoding == tmnn::extension::KernelEncoding::RMHE)
    resolved.encoding = KernelSpec::RMHE;

  resolved.use_simd = false;
  resolved.use_fp16 = false;
  resolved.use_tg_weight_cache = false;
  resolved.use_fp16_hash_grid = false;
  resolved.use_fp16_simd = false;

  // Lower loss config from compile spec to kernel spec.
  switch (request.compile_spec.loss_kind) {
  case tmnn::extension::LossKind::L2:
    resolved.loss = KernelSpec::L2;
    break;
  case tmnn::extension::LossKind::L1:
    resolved.loss = KernelSpec::L1;
    break;
  case tmnn::extension::LossKind::Huber:
    resolved.loss = KernelSpec::Huber;
    break;
  case tmnn::extension::LossKind::Cosine:
    resolved.loss = KernelSpec::Cosine;
    break;
  }
  resolved.huber_delta = request.compile_spec.huber_delta;
  // ForwardOnly kernels don't have backward pass — probes are meaningless.
  resolved.emit_probes = request.compile_spec.enable_probes &&
                         request.role != KernelRole::ForwardForTraining;
  resolved.emit_active_hash_mask =
      request.role == KernelRole::TrainForwardBackward ||
      request.role == KernelRole::BackwardFromExternalGrad;

  decision.role = request.role;
  decision.requested_simd = request.compile_spec.allow_simd;
  decision.requested_fp16 = request.compile_spec.allow_fp16;
  decision.requested_tg_weight_cache = request.compile_spec.allow_tg_weight_cache;

  // Probes force scalar path — SIMD probe instrumentation not implemented.
  const bool simd_allowed = decision.requested_simd && !resolved.emit_probes;
  decision.simd_reason =
      resolved.emit_probes
          ? SpecializationDisableReason::PreferenceDisabled
          : simd_disable_reason(request.role, resolved, decision.requested_simd);
  decision.realized_simd =
      simd_allowed && decision.simd_reason == SpecializationDisableReason::None;
  resolved.use_simd = decision.realized_simd;

  if (!decision.requested_fp16) {
    decision.fp16_reason = SpecializationDisableReason::PreferenceDisabled;
  } else if (decision.realized_simd) {
    if (request.role != KernelRole::Eval &&
        request.role != KernelRole::TrainForwardBackward) {
      decision.fp16_reason = SpecializationDisableReason::RoleUnsupported;
    } else {
      decision.fp16_reason = SpecializationDisableReason::None;
      decision.realized_fp16 = true;
      resolved.use_fp16 = true;
      if (request.role != KernelRole::TrainForwardBackward)
        resolved.use_fp16_simd = true;
    }
  } else {
    decision.fp16_reason = scalar_fp16_eval_disable_reason(
        request.role, resolved, decision.requested_fp16);
    if (decision.fp16_reason == SpecializationDisableReason::None) {
      decision.realized_fp16 = true;
      resolved.use_fp16 = true;
    }
  }

  if (decision.realized_simd && decision.fp16_reason == SpecializationDisableReason::None) {
    decision.fp16_reason = SpecializationDisableReason::None;
  }

  if (!decision.requested_tg_weight_cache) {
    decision.tg_cache_reason = SpecializationDisableReason::PreferenceDisabled;
  } else if (!resolved.canUseTGCache()) {
    decision.tg_cache_reason =
        SpecializationDisableReason::ThreadgroupMemoryExceeded;
  } else {
    decision.tg_cache_reason = SpecializationDisableReason::None;
    decision.realized_tg_weight_cache = true;
    resolved.use_tg_weight_cache = true;
  }

  resolved.validate();
  return resolved;
}

std::string emit_train_param_macros(const tmnn::TrainParamsLayout &layout) {
  return "#define TMNN_TRAIN_PARAMS_IDX_N " + std::to_string(layout.idx_n) +
         "\n#define TMNN_TRAIN_PARAMS_IDX_UNSIGNED_MODE " +
         std::to_string(layout.idx_unsigned_mode) +
         "\n#define TMNN_TRAIN_PARAMS_IDX_LOSS_SCALE " +
         std::to_string(layout.idx_loss_scale) +
         "\n#define TMNN_TRAIN_PARAMS_IDX_NUM_ACTIVE_LEVELS " +
         std::to_string(layout.idx_num_active_levels) + "\n";
}

std::string emit_output_semantics_macros(
    const tmnn::extension::KernelCompileSpec &compile_spec) {
  if (compile_spec.output_semantics ==
      tmnn::extension::KernelOutputSemantics::DNL) {
    return "#define TMNN_OUTPUT_SEMANTICS_DNL 1\n";
  }
  return {};
}

std::string emit_source(const KernelCompileRequest &request,
                        const KernelSpec &resolved_spec) {
  MLPKernelEmitter emitter;
  switch (request.role) {
  case KernelRole::Eval:
    return emit_output_semantics_macros(request.compile_spec) +
           emitter.emitEvalKernel(resolved_spec);
  case KernelRole::Gradient:
    return emit_output_semantics_macros(request.compile_spec) +
           emitter.emitGradientKernel(resolved_spec);
  case KernelRole::EvalGradient:
    return emit_output_semantics_macros(request.compile_spec) +
           emitter.emitEvalGradientKernel(resolved_spec);
  case KernelRole::TrainForwardBackward: {
    std::string source = emit_train_param_macros(request.schema.train_params_layout);
    if (request.schema.reduction_terms > 1) {
      source += "#define REDUCTION_TERMS " +
                std::to_string(request.schema.reduction_terms) + "\n";
    }
    source += emit_output_semantics_macros(request.compile_spec);
    if (request.compile_spec.bc_dim_count > 0) {
      source += "#define BC_DIM_COUNT " +
                std::to_string(request.compile_spec.bc_dim_count) + "\n";
    }
    source += emitter.emitTrainKernel(resolved_spec);
    return source;
  }
  case KernelRole::BackwardFromExternalGrad: {
    std::string source = emit_train_param_macros(request.schema.train_params_layout);
    if (request.schema.reduction_terms > 1) {
      source += "#define REDUCTION_TERMS " +
                std::to_string(request.schema.reduction_terms) + "\n";
    }
    source += emit_output_semantics_macros(request.compile_spec);
    source += emitter.emitBackwardExternalGradKernel(resolved_spec);
    return source;
  }
  case KernelRole::ForwardForTraining: {
    std::string source = emit_train_param_macros(request.schema.train_params_layout);
    source += emit_output_semantics_macros(request.compile_spec);
    source += emitter.emitForwardForTrainingKernel(resolved_spec);
    return source;
  }
  }
  throw std::invalid_argument("KernelCompiler: unsupported kernel role");
}

class SourceCache {
public:
  template <typename Fn> std::string get_or_create(const KernelKey &key, Fn &&fn) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto it = cache_.find(key);
      if (it != cache_.end())
        return it->second;
    }

    std::string source = fn();

    std::lock_guard<std::mutex> lock(mu_);
    auto [it, inserted] = cache_.emplace(key, std::move(source));
    if (!inserted)
      return it->second;
    return it->second;
  }

private:
  std::mutex mu_;
  std::unordered_map<KernelKey, std::string> cache_;
};

SourceCache &source_cache() {
  static SourceCache cache;
  return cache;
}

} // namespace

uint64_t KernelKey::hash() const {
  uint64_t hash = kFnvBasis;
  mix_hash(hash, static_cast<uint64_t>(role));
  mix_hash(hash, resolved_spec_hash);
  mix_hash(hash, schema_hash);
  mix_hash(hash, compile_contract_hash);
  mix_hash(hash, static_cast<uint64_t>(emitter_version));
  return hash;
}

tmnn::extension::ExtensionSchema
KernelCompiler::makeDefaultSchema(const KernelSpec &spec) {
  tmnn::extension::ExtensionSchema schema;
  schema.input_dims = static_cast<uint32_t>(spec.spatial_dims);
  schema.target_dims = static_cast<uint32_t>(spec.num_outputs);
  return schema;
}

KernelCompileResult KernelCompiler::compile(const KernelCompileRequest &request) {
  request.spec.validate();
  request.schema.validate();
  request.compile_spec.validate(request.schema);

  if (request.spec.spatial_dims != static_cast<int>(request.schema.input_dims)) {
    throw std::invalid_argument(
        "KernelCompiler: spec.spatial_dims must match schema.input_dims");
  }
  if (request.spec.num_outputs != static_cast<int>(request.schema.target_dims)) {
    throw std::invalid_argument(
        "KernelCompiler: spec.num_outputs must match schema.target_dims");
  }
  if (request.role != KernelRole::TrainForwardBackward &&
      request.role != KernelRole::BackwardFromExternalGrad &&
      request.compile_spec.bc_dim_count > 0) {
    throw std::invalid_argument(
        "KernelCompiler: bc_dim_count is only valid for TrainForwardBackward");
  }
  if (request.compile_spec.output_semantics ==
          tmnn::extension::KernelOutputSemantics::DNL &&
      request.role != KernelRole::Eval &&
      request.role != KernelRole::TrainForwardBackward) {
    throw std::invalid_argument(
        "KernelCompiler: DNL output semantics are only valid for Eval and "
        "TrainForwardBackward");
  }

  SpecializationDecision decision;
  KernelSpec resolved_spec = resolve_spec(request, decision);

  KernelKey key;
  key.role = request.role;
  key.resolved_spec_hash = resolved_spec.hash();
  key.schema_hash = hash_schema(request.schema);
  key.compile_contract_hash =
      hash_compile_contract(request.role, request.compile_spec);
  key.emitter_version = kEmitterVersion;

  KernelCompileResult result;
  result.key = key;
  result.resolved_spec = resolved_spec;
  result.decision = decision;
  result.entry_point = entry_point_for(request.role, resolved_spec);
  result.source =
      source_cache().get_or_create(key, [&]() { return emit_source(request, resolved_spec); });
  return result;
}

} // namespace tmnn

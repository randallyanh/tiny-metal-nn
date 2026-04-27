#pragma once

/**
 * @file extension/kernel_compile_spec.h
 * @brief KernelEncoding + KernelCompileSpec — SDK surface for specifying
 *        kernel compilation preferences.
 *
 * Dependencies: <cstdint>, <stdexcept>, schema.h.
 */

#include "tiny-metal-nn/extension/schema.h"

#include <cstdint>
#include <stdexcept>

namespace tmnn {
namespace extension {

/// Encoding scheme used by the hash grid / input layer.
enum class KernelEncoding : uint32_t {
  Standard = 0, ///< Multi-resolution hash encoding (default).
  RMHE = 1,     ///< Rotation-augmented multi-resolution hash encoding.
};

/// Output semantic profile for emitted kernels.
enum class KernelOutputSemantics : uint32_t {
  Generic = 0, ///< Generic MLP outputs with no implicit activation profile.
  DNL = 1,     ///< DNL 4-output activation profile (sigmoid/id/softplus/softplus).
};

/// Loss function family for training kernels.
enum class LossKind : uint32_t {
  L2 = 0,     ///< Mean squared error (default).
  L1 = 1,     ///< Mean absolute error.
  Huber = 2,  ///< Huber loss (smooth L1).
  Cosine = 3, ///< Mean per-sample cosine distance on generic multi-output paths.
};

/// Compile-time preferences for kernel code generation.
struct KernelCompileSpec {
  KernelEncoding encoding = KernelEncoding::Standard;
  bool allow_simd = true;            ///< Allow SIMD cooperative kernels.
  bool allow_fp16 = true;            ///< Allow FP16 mixed-precision kernels.
  bool allow_tg_weight_cache = true; ///< Allow threadgroup weight caching.
  KernelOutputSemantics output_semantics = KernelOutputSemantics::Generic;

  /// Loss function family. Lowered to KernelSpec::loss at compile time.
  LossKind loss_kind = LossKind::L2;

  /// Huber delta parameter. Only used when loss_kind == Huber.
  float huber_delta = 1.0f;

  /// Opt-in per-step probe telemetry. Forces scalar training path.
  bool enable_probes = false;

  /// Boundary-condition output dimension count for diagnostic loss decomposition.
  /// When > 0 and num_outputs > 1: kernel writes decomposed reduction terms
  /// (total, bc, piezo) instead of a single loss value.
  /// 0 = disabled (default).
  uint32_t bc_dim_count = 0;

  /// Validate compile preferences against a schema. Throws on error.
  inline void validate(const ExtensionSchema &schema) const {
    schema.validate();

    if (encoding == KernelEncoding::RMHE && schema.input_dims != 3)
      throw std::invalid_argument(
          "KernelCompileSpec: RMHE requires schema.input_dims == 3");

    if (output_semantics == KernelOutputSemantics::DNL) {
      if (schema.input_dims != 3)
        throw std::invalid_argument(
            "KernelCompileSpec: DNL output semantics require schema.input_dims == 3");
      if (schema.target_dims != 4)
        throw std::invalid_argument(
            "KernelCompileSpec: DNL output semantics require schema.target_dims == 4");
    }

    if (bc_dim_count > 0) {
      if (schema.target_dims <= 1)
        throw std::invalid_argument(
            "KernelCompileSpec: bc_dim_count > 0 requires schema.target_dims > 1");
      if (bc_dim_count >= schema.target_dims)
        throw std::invalid_argument(
            "KernelCompileSpec: bc_dim_count must be < schema.target_dims");
      if (schema.reduction_terms < 3)
        throw std::invalid_argument(
            "KernelCompileSpec: bc_dim_count requires schema.reduction_terms >= 3");
    }

    if (loss_kind == LossKind::Huber && huber_delta <= 0.0f)
      throw std::invalid_argument(
          "KernelCompileSpec: Huber loss requires huber_delta > 0");
    if (loss_kind == LossKind::Cosine) {
      if (schema.target_dims < 2) {
        throw std::invalid_argument(
            "KernelCompileSpec: Cosine loss requires schema.target_dims >= 2");
      }
      if (schema.reduction_terms != 1) {
        throw std::invalid_argument(
            "KernelCompileSpec: Cosine loss requires schema.reduction_terms == 1");
      }
      if (bc_dim_count != 0) {
        throw std::invalid_argument(
            "KernelCompileSpec: Cosine loss does not support bc_dim_count");
      }
    }
  }
};

} // namespace extension
} // namespace tmnn

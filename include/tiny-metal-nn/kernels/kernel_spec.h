#pragma once

/**
 * @file kernels/kernel_spec.h
 * @brief KernelSpec: fully describes an MSL kernel's compile-time configuration.
 *
 * All architecture dimensions (hidden_dim, num_layers, num_outputs, hash grid
 * parameters) are captured here and baked as integer constants into generated
 * MSL code by MLPKernelEmitter.
 */

#include <cstddef>
#include <cstdint>

namespace tmnn {

struct KernelSpec {
  /// Number of floats output by packConfig() — the GPU config header.
  static constexpr int kConfigPackedFloats = 8;

  // Network structure
  int input_dim = 32;         // num_levels * features_per_level
  int hidden_dim = 64;
  int num_hidden_layers = 2;
  int num_outputs = 1;        // 1 (SDF) or 4 (DNL)

  // Encoding
  int num_levels = 16;
  int features_per_level = 2;
  int log2_hashmap_size = 19;
  float base_resolution = 16.0f;
  float per_level_scale = 1.447f;
  int spatial_dims = 3;       // 3 or 4

  // Kernel variant flags
  bool use_fp16 = false;
  bool use_int_atomics = true;
  bool use_tg_weight_cache = true;
  bool use_simd = false;
  bool use_fp16_hash_grid = false;
  bool use_fp16_simd = false;

  // Encoding type
  enum EncodingType { Standard, RMHE, FourD };
  EncodingType encoding = Standard;

  // Loss type (training kernels)
  enum LossType { L2, L1, Huber, Cosine };
  LossType loss = L2;
  float huber_delta = 1.0f; ///< Huber loss delta (only used when loss == Huber).

  // Activation function
  enum Activation { ReLU };
  Activation activation = ReLU;

  // Probe mode (opt-in diagnostic instrumentation)
  bool emit_probes = false;
  bool emit_active_hash_mask = false;

  /// Construct from packed 8-float config header + schema overrides.
  static KernelSpec fromConfigHeader(const float header[8],
                                     uint32_t target_dims = 1,
                                     uint32_t input_dims = 3);

  /// Total MLP parameter count (all weights + biases, all layers).
  [[nodiscard]] int mlpWeightCount() const;

  [[nodiscard]] int tgBufferFloats() const { return mlpWeightCount(); }

  [[nodiscard]] bool canUseTGCache() const {
    return tgBufferFloats() * 4 <= 25600;
  }

  [[nodiscard]] bool canUseTGGrad() const {
    return mlpWeightCount() * 4 <= 32768;
  }

  [[nodiscard]] bool canUseSIMD() const { return hidden_dim % 8 == 0; }

  [[nodiscard]] int simdTrainTGBytes() const {
    int bytes = mlpWeightCount() * 4;
    int act_bufs = (num_hidden_layers >= 2) ? 3 : 2;
    bytes += act_bufs * 8 * hidden_dim * 4;
    bytes += 8 * input_dim * 4;
    return bytes;
  }

  /// Validate spec fields. Throws std::invalid_argument on bad values.
  void validate() const;

  /// FNV-1a hash for cache key.
  [[nodiscard]] uint64_t hash() const;
};

} // namespace tmnn

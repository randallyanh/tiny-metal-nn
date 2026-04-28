#pragma once

/**
 * @file weight_init.h
 * @brief Weight initialization configuration for tmnn trainers.
 *
 * Industry-standard initializers for hash-grid encodings and fused MLPs.
 * Driven by a Philox-4x32-10 counter-based RNG dispatched on the GPU,
 * so cold-startup weight init scales with GPU memory bandwidth instead
 * of CPU single-thread RNG (~120-240× speedup on default-encoding
 * configurations with log2_hashmap >= 17).
 *
 * Defaults follow the literature:
 *   - hash grid: uniform [-1e-4, 1e-4] (instant-NGP §3 init)
 *   - MLP weights: Kaiming uniform with leaky-ReLU a=0 default
 *
 * Override on TrainerConfig::weight_init at construction time, or use
 * Trainer::set_initial_weights(...) post-construction for transfer
 * learning / pre-trained loads.
 */

#include <cstdint>

namespace tmnn {

/// Initialization strategy for the hash-grid encoding table.
enum class HashGridInit : uint32_t {
  /// Uniform [-hash_grid_range, +hash_grid_range]. Default; matches the
  /// instant-NGP literature (range = 1e-4).
  Uniform = 0,
  /// All zeros — useful for fine-tuning from a pre-trained backbone, or
  /// for ablations that disable hash-grid contribution.
  Zero = 1,
};

/// Initialization strategy for the fully-fused MLP weight tensor.
enum class MlpInit : uint32_t {
  /// Kaiming uniform: U[-bound, bound], bound = sqrt(6 / ((1+a²) * fan_in)).
  /// Default. Suited for ReLU / leaky-ReLU activations (a is the negative
  /// slope; 0 for plain ReLU).
  KaimingUniform = 0,
  /// Kaiming normal: N(0, sqrt(2 / ((1+a²) * fan_in))).
  KaimingNormal = 1,
  /// Xavier (Glorot) uniform: U[-bound, bound],
  /// bound = sqrt(6 / (fan_in + fan_out)). For tanh/sigmoid activations.
  XavierUniform = 2,
  /// Xavier (Glorot) normal: N(0, sqrt(2 / (fan_in + fan_out))).
  XavierNormal = 3,
  /// Plain uniform [-mlp_uniform_range, +mlp_uniform_range].
  Uniform = 4,
  /// Plain normal N(0, mlp_normal_stddev).
  Normal = 5,
  /// All zeros.
  Zero = 6,
};

/// Weight initialization configuration. Defaults match industry practice.
struct WeightInitConfig {
  /// Hash-grid table.
  HashGridInit hash_grid_mode = HashGridInit::Uniform;
  /// Range used when hash_grid_mode == Uniform. Defaults to instant-NGP's
  /// 1e-4 (small enough that early gradients stay well-scaled).
  float hash_grid_range = 1.0e-4f;

  /// MLP weight tensor.
  MlpInit mlp_mode = MlpInit::KaimingUniform;
  /// Used when mlp_mode == Uniform.
  float mlp_uniform_range = 1.0e-2f;
  /// Used when mlp_mode == Normal.
  float mlp_normal_stddev = 1.0e-2f;
  /// Negative slope used in Kaiming modes (0 = plain ReLU).
  float mlp_kaiming_a = 0.0f;

  /// 64-bit RNG seed split into two 32-bit Philox keys at dispatch time.
  /// Change this per run for reproducibility across multi-seed sweeps.
  uint64_t seed = 42u;
};

}  // namespace tmnn

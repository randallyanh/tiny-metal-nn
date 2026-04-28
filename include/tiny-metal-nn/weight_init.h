#pragma once

/**
 * @file weight_init.h
 * @brief Weight initialization configuration for tmnn trainers.
 *
 * Industry-standard initializers for hash-grid encodings and fused MLPs.
 * Driven by a Philox-4x32-10 counter-based RNG dispatched on the GPU,
 * so cold-startup weight init scales with GPU memory bandwidth instead
 * of CPU single-thread RNG.
 *
 * Defaults follow the literature:
 *   - hash grid: uniform [-1e-4, 1e-4] (instant-NGP §3 init)
 *   - MLP weights: fan-in-scaled uniform with the gain set by
 *     mlp_nonlinearity (defaults to ReLU ⇒ gain = sqrt(2))
 *   - MLP biases: U[-1/sqrt(fan_in), 1/sqrt(fan_in)], matching
 *     PyTorch's torch.nn.Linear.reset_parameters() default exactly
 *
 * Per-layer dispatch:
 *   The MLP weight tensor is sliced into its constituent layers
 *   (W₀: input × hidden, W₁..W_{L-1}: hidden × hidden, W_out:
 *   hidden × n_outputs) and each weight slice is initialized with its
 *   own fan_in. Pre-P5.5 used a single `hidden_dim` approximation;
 *   P5.5 ships per-layer correctness.
 *
 * Mode (fan_in vs fan_out):
 *   All Kaiming/Xavier formulas use mode='fan_in' (preserves the
 *   forward-pass activation variance through the network). This
 *   matches the PyTorch nn.init defaults and tinycudann. Mode
 *   'fan_out' (preserves backward-pass gradient variance) is future
 *   work; in the current implementation `fan_out` only enters the
 *   Xavier formulas via `(fan_in + fan_out)`.
 *
 * Out-of-scope activations:
 *   The Kaiming-gain table covers ReLU / LeakyReLU / Linear / Tanh /
 *   Sigmoid (see MlpNonlinearity below). Networks that use Snake,
 *   SIREN's sine activation, or GELU need different gains and are NOT
 *   automatically scaled correctly by the Kaiming modes here. For
 *   those, set mlp_mode = MlpInit::Uniform or MlpInit::Normal with an
 *   explicit mlp_uniform_range / mlp_normal_stddev (e.g. SIREN's
 *   first-layer special init), or supply weights via
 *   Trainer::set_initial_weights(...) directly.
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
  /// Kaiming uniform: U[-bound, bound], bound = gain · sqrt(3 / fan_in)
  /// where `gain` is computed from `mlp_nonlinearity` (see MlpNonlinearity).
  /// For ReLU (default) gain = sqrt(2) ⇒ bound = sqrt(6 / fan_in).
  KaimingUniform = 0,
  /// Kaiming normal: N(0, gain / sqrt(fan_in)). For ReLU (default)
  /// stddev = sqrt(2 / fan_in).
  KaimingNormal = 1,
  /// Xavier (Glorot) uniform: U[-bound, bound],
  /// bound = sqrt(6 / (fan_in + fan_out)). Activation-agnostic.
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

/// Activation function the MLP weights feed into. Used by the
/// fan-in-scaled init modes to compute the per-layer gain (matches
/// PyTorch's torch.nn.init.calculate_gain table). For the
/// FullyFusedMLP path, hidden layers feed into ReLU and the output
/// layer feeds into a linear identity — the current implementation
/// applies the SAME nonlinearity to all layer slices. When the network
/// downstream is something this enum does not enumerate (Snake /
/// SIREN / GELU / etc.), use MlpInit::Uniform or MlpInit::Normal with
/// an explicit range / stddev instead of one of the fan-in-scaled
/// modes.
enum class MlpNonlinearity : uint32_t {
  /// gain = 1. Identity / no activation. Matches the output layer of a
  /// regression-style MLP when paired with MSE loss.
  Linear = 0,
  /// gain = sqrt(2). Default. Matches FullyFusedMLP's hidden-layer
  /// activation.
  ReLU = 1,
  /// gain = sqrt(2 / (1 + a²)). Use when the network's hidden activation
  /// is leaky-ReLU; `a` is the negative slope (0 for plain ReLU).
  LeakyReLU = 2,
  /// gain = 5/3. PyTorch convention.
  Tanh = 3,
  /// gain = 1. PyTorch convention (sigmoid does not need a special gain
  /// for stable variance propagation in deep networks).
  Sigmoid = 4,
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
  /// Activation the MLP feeds into; sets the fan-in-scaled init gain
  /// for KaimingUniform / KaimingNormal modes.
  MlpNonlinearity mlp_nonlinearity = MlpNonlinearity::ReLU;
  /// Used when mlp_mode == Uniform.
  float mlp_uniform_range = 1.0e-2f;
  /// Used when mlp_mode == Normal.
  float mlp_normal_stddev = 1.0e-2f;
  /// Negative slope used when mlp_nonlinearity == LeakyReLU. Ignored
  /// otherwise. (0 ⇒ plain ReLU.)
  float mlp_kaiming_a = 0.0f;

  // All fan-in-scaled / Xavier formulas use mode='fan_in' (forward-pass
  // variance preservation). Matches PyTorch nn.init defaults and
  // tinycudann. fan_out mode is future work and would simply substitute
  // fan_out for fan_in in the bound / stddev formulas.

  /// 64-bit RNG seed split into two 32-bit Philox keys at dispatch time.
  /// Change this per run for reproducibility across multi-seed sweeps.
  uint64_t seed = 42u;
};

}  // namespace tmnn

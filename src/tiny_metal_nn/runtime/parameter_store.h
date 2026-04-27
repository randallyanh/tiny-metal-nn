#pragma once

/**
 * @file parameter_store.h
 * @brief ParameterStore — converged parameter/gradient/optimizer buffer
 *        management for tmnn NN runtime (internal).
 *
 * Replaces the scattered buffer management in individual trainers.
 * Allocates all persistent buffers (weights, gradients, Adam m/v,
 * config_weights, train_params, adam_params) from a BufferArena,
 * and provides typed accessors + binding helpers.
 *
 * Frozen contracts:
 * - kConfigPackedFloats = 8 (header size in config_weights buffer)
 * - config_weights = [8-float header | MLP weights]
 * - train_params schema is consumed from TrainParamsLayout
 * - Hash grid training is FP32, optimizer moments are always FP32
 */

#include "tiny-metal-nn/extension/schema.h"

#include "tiny_metal_nn/runtime/buffer_arena.h"
#include "tiny_metal_nn/runtime/buffer_handle.h"

#include <cstdint>

namespace tmnn {

/// Frozen constant: config_weights header size.
static constexpr uint32_t kConfigPackedFloats = 8;

/// Frozen constant: fused Adam params buffer size.
static constexpr uint32_t kFusedAdamParamFloats = 12;

/// Frozen constant: legacy Adam params buffer size.
static constexpr uint32_t kAdamParamFloats = 10;

/// Unified Adam/AdamW params buffer size (adds weight_decay slot).
static constexpr uint32_t kUnifiedAdamParamFloats = 13;

/// Descriptor for ParameterStore creation.
struct ParameterStoreDesc {
  uint32_t hash_grid_size = 0;   ///< Number of float elements in hash grid.
  uint32_t mlp_weight_count = 0; ///< Number of float elements in MLP.
  bool use_private_buffers = false; ///< Use Private storage for grads/moments.
  bool use_fused_adam = false;   ///< Allocate contiguous [hash|mlp] for weights/m/v.
  bool use_private_active_hash_mask = false; ///< Keep active-hash bitset GPU-private.
  uint32_t active_hash_mask_words = 0; ///< 32-bit words in the active-hash bitset.
  uint32_t active_hash_summary_words = 0; ///< 32-bit words in the coarse active-hash summary.
  uint32_t active_hash_index_capacity = 0; ///< Max compacted active hash slots.
  TrainParamsLayout train_params_layout; ///< Layout for train_params buffer.
  uint32_t adam_params_float_count = kFusedAdamParamFloats; ///< Adam params size.
  uint32_t config_tail_floats = 0;  ///< Extra floats after MLP in config_weights.
  uint32_t target_dims = 1;         ///< Output dimensionality per sample.
  uint32_t reduction_terms = 1;     ///< Loss reduction terms per threadgroup.
};

/// Converged parameter/gradient/optimizer buffer store.
class ParameterStore {
public:
  /// Create a ParameterStore backed by the given arena.
  ParameterStore(const ParameterStoreDesc &desc, BufferArena &arena);
  ~ParameterStore();

  ParameterStore(const ParameterStore &) = delete;
  ParameterStore &operator=(const ParameterStore &) = delete;
  ParameterStore(ParameterStore &&) = delete;
  ParameterStore &operator=(ParameterStore &&) = delete;

  /// The descriptor used to create this store.
  [[nodiscard]] const ParameterStoreDesc &desc() const { return desc_; }

  /// The train_params layout used by this store.
  [[nodiscard]] const TrainParamsLayout &train_params_layout() const {
    return desc_.train_params_layout;
  }

  // --- Weight views ---
  [[nodiscard]] BufferView hash_weights() const { return hash_weights_; }
  [[nodiscard]] BufferView mlp_weights() const { return mlp_weights_; }

  // --- Gradient views ---
  [[nodiscard]] BufferView grad_hash() const { return grad_hash_; }
  [[nodiscard]] BufferView grad_mlp() const { return grad_mlp_; }
  [[nodiscard]] BufferView active_hash_mask() const { return active_hash_mask_; }
  [[nodiscard]] BufferView active_hash_summary_mask() const {
    return active_hash_summary_mask_;
  }
  [[nodiscard]] BufferView active_hash_indices() const {
    return active_hash_indices_;
  }

  // --- Optimizer moment views ---
  [[nodiscard]] BufferView adam_m_hash() const { return adam_m_hash_; }
  [[nodiscard]] BufferView adam_v_hash() const { return adam_v_hash_; }
  [[nodiscard]] BufferView adam_m_mlp() const { return adam_m_mlp_; }
  [[nodiscard]] BufferView adam_v_mlp() const { return adam_v_mlp_; }

  // --- Control buffer views ---
  /// Config weights = [8-float header | MLP weights].
  [[nodiscard]] BufferView config_weights() const { return config_weights_; }
  /// Training hyperparameters (per-step).
  [[nodiscard]] BufferView train_params() const { return train_params_; }
  /// Adam optimizer hyperparameters.
  [[nodiscard]] BufferView adam_params() const { return adam_params_; }

  // --- Binding helpers ---

  /// Create a binding from any view to a pipeline slot index.
  [[nodiscard]] static BufferBinding bind(const BufferView &view,
                                          uint32_t slot) {
    return BufferArena::bind(view, slot);
  }

  /// Config weights header sub-view (the 8-float packed header).
  [[nodiscard]] BufferView config_header() const;

  /// Config weights MLP sub-view (the MLP weights after the header).
  [[nodiscard]] BufferView config_mlp() const;

  /// Config weights tail sub-view (extra floats after MLP, for adapter use).
  [[nodiscard]] BufferView config_tail() const;

  // --- Async step finalization ---

  /// Result of finalizing an async training step.
  static constexpr uint32_t kMaxExtraLosses = 4;
  struct AsyncStepResult {
    float mean_loss = 0.0f;
    float extra_losses[kMaxExtraLosses] = {};
    uint32_t extra_loss_count = 0;
    uint32_t completed_step = 0; ///< Caller-supplied completed-step echo.
  };

  /// Hydrate buffers with real weight data (initial population).
  /// Copies hash grid and MLP weights into CPU-backed buffers,
  /// populates config_weights header+MLP section.
  void hydrate_weights(const float *hash_data, size_t hash_count,
                       const float *mlp_data, size_t mlp_count,
                       const float *config_header_8f = nullptr);

  /// Sync live trained weights into the store (called after each training step).
  /// Only updates hash_weights_ and mlp_weights_. The packed config buffer keeps
  /// its 8-float header stable; refresh its MLP section only via the explicit
  /// sync helpers when a packed snapshot is required.
  void sync_live_weights(const float *hash_data, size_t hash_count,
                         const float *mlp_data, size_t mlp_count);

  /// Finalize an async step: read back loss from the lane buffer and,
  /// when requested, sync the MLP weights into config_weights.
  [[nodiscard]] AsyncStepResult finalize_async_step(
      const StepBufferSet &completed, uint32_t num_tgs, uint32_t batch_N,
      uint32_t completed_step = 0, bool sync_config_weights = true);

  /// Refresh the config_weights MLP section from the live MLP weight buffer.
  void sync_config_mlp_from_live_weights();

  // --- Fused buffer views (for Adam kernel) ---
  [[nodiscard]] BufferView fused_weights() const { return fused_weights_; }
  [[nodiscard]] BufferView fused_m() const { return fused_m_; }
  [[nodiscard]] BufferView fused_v() const { return fused_v_; }
  [[nodiscard]] bool is_fused() const { return desc_.use_fused_adam; }

  /// Zero all CPU-visible Adam m/v buffers (fused or separate).
  void reset_adam_state();

  // --- Size queries ---

  /// Total persistent bytes allocated by this store.
  [[nodiscard]] size_t total_bytes() const { return total_bytes_; }

  /// Hash grid size in bytes.
  [[nodiscard]] size_t hash_bytes() const {
    return desc_.hash_grid_size * sizeof(float);
  }

  /// MLP weight size in bytes.
  [[nodiscard]] size_t mlp_bytes() const {
    return desc_.mlp_weight_count * sizeof(float);
  }

  /// Config weights total size in bytes.
  [[nodiscard]] size_t config_weights_bytes() const {
    return (kConfigPackedFloats + desc_.mlp_weight_count +
            desc_.config_tail_floats) * sizeof(float);
  }

private:
  BufferArena *arena_ = nullptr;
  ParameterStoreDesc desc_;
  size_t total_bytes_ = 0;

  // Weight buffers (Persistent, Shared).
  BufferView hash_weights_;
  BufferView mlp_weights_;

  // Gradient buffers (Persistent, may be Private).
  BufferView grad_hash_;
  BufferView grad_mlp_;
  BufferView active_hash_mask_;
  BufferView active_hash_summary_mask_;
  BufferView active_hash_indices_;

  // Optimizer moment buffers (Persistent, may be Private).
  BufferView adam_m_hash_;
  BufferView adam_v_hash_;
  BufferView adam_m_mlp_;
  BufferView adam_v_mlp_;

  // Control buffers (Persistent, Shared).
  BufferView config_weights_;
  BufferView train_params_;
  BufferView adam_params_;

  // Fused contiguous buffers (when use_fused_adam=true).
  BufferView fused_weights_; // [hash_grid | mlp] contiguous
  BufferView fused_m_;       // [m_hash | m_mlp] contiguous
  BufferView fused_v_;       // [v_hash | v_mlp] contiguous
};

} // namespace tmnn

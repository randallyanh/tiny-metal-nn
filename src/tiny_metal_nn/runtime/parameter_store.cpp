/**
 * @file parameter_store.cpp
 * @brief ParameterStore implementation.
 */

#include "tiny_metal_nn/runtime/parameter_store.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace tmnn {
namespace {

void release_owned_view(BufferArena *arena, BufferView &view) {
  if (!arena || view.bytes == 0)
    return;
  if (arena->is_valid(view.handle))
    arena->release(view.handle);
  view = {};
}

void sync_config_mlp_section(const ParameterStoreDesc &desc,
                             const BufferView &mlp_weights,
                             const BufferView &config_weights) {
  if (!mlp_weights.data || !config_weights.data)
    return;
  auto *dst = static_cast<char *>(config_weights.data) +
              kConfigPackedFloats * sizeof(float);
  std::memcpy(dst, mlp_weights.data, desc.mlp_weight_count * sizeof(float));
}

} // namespace

// ---------------------------------------------------------------------------
// ParameterStore
// ---------------------------------------------------------------------------

ParameterStore::ParameterStore(const ParameterStoreDesc &desc,
                                BufferArena &arena)
    : arena_(&arena), desc_(desc) {
  if (desc_.target_dims < 1)
    throw std::invalid_argument(
        "ParameterStoreDesc: target_dims must be >= 1");
  if (desc_.reduction_terms < 1)
    throw std::invalid_argument(
        "ParameterStoreDesc: reduction_terms must be >= 1");
  if (desc_.reduction_terms > kMaxExtraLosses + 1)
    throw std::invalid_argument(
        "ParameterStoreDesc: reduction_terms must be <= 5 "
        "(1 mean loss + up to 4 extra losses)");
  desc_.train_params_layout.validate();

  const size_t hash_bytes = desc.hash_grid_size * sizeof(float);
  const size_t mlp_bytes = desc.mlp_weight_count * sizeof(float);
  const size_t active_hash_mask_bytes =
      static_cast<size_t>(desc.active_hash_mask_words) * sizeof(uint32_t);
  const size_t active_hash_summary_bytes =
      static_cast<size_t>(desc.active_hash_summary_words) * sizeof(uint32_t);
  const size_t active_hash_index_bytes =
      static_cast<size_t>(desc.active_hash_index_capacity) * sizeof(uint32_t);
  const size_t config_bytes =
      (kConfigPackedFloats + desc.mlp_weight_count +
       desc.config_tail_floats) * sizeof(float);

  auto storage_for_grads =
      desc.use_private_buffers ? BufferStorage::Private : BufferStorage::Shared;
  const auto active_hash_mask_storage =
      desc.use_private_active_hash_mask ? BufferStorage::Private
                                        : BufferStorage::Shared;

  if (desc.use_fused_adam) {
    // --- Fused contiguous allocation: [hash_grid | mlp] ---
    const size_t total_bytes = hash_bytes + mlp_bytes;

    // Single contiguous buffer for weights (Shared, CPU needs upload).
    fused_weights_ = arena.view(arena.allocate(
        {total_bytes, 256, BufferStorage::Shared,
         BufferLifetime::Persistent, "fused_weights"}));
    hash_weights_ = BufferArena::sub_view(fused_weights_, 0, hash_bytes);
    mlp_weights_ = BufferArena::sub_view(fused_weights_, hash_bytes, mlp_bytes);

    // Single contiguous buffer for first moment (m).
    fused_m_ = arena.view(arena.allocate(
        {total_bytes, 256, storage_for_grads,
         BufferLifetime::Persistent, "fused_m"}));
    adam_m_hash_ = BufferArena::sub_view(fused_m_, 0, hash_bytes);
    adam_m_mlp_ = BufferArena::sub_view(fused_m_, hash_bytes, mlp_bytes);

    // Single contiguous buffer for second moment (v).
    fused_v_ = arena.view(arena.allocate(
        {total_bytes, 256, storage_for_grads,
         BufferLifetime::Persistent, "fused_v"}));
    adam_v_hash_ = BufferArena::sub_view(fused_v_, 0, hash_bytes);
    adam_v_mlp_ = BufferArena::sub_view(fused_v_, hash_bytes, mlp_bytes);
  } else {
    // --- Separate allocations (legacy) ---

    // Weights (Shared, Persistent).
    hash_weights_ = arena.view(arena.allocate(
        {hash_bytes, 256, BufferStorage::Shared,
         BufferLifetime::Persistent, "hash_weights"}));

    mlp_weights_ = arena.view(arena.allocate(
        {mlp_bytes, 256, BufferStorage::Shared,
         BufferLifetime::Persistent, "mlp_weights"}));

    // Optimizer moments (may be Private, always FP32, Persistent).
    adam_m_hash_ = arena.view(arena.allocate(
        {hash_bytes, 256, storage_for_grads,
         BufferLifetime::Persistent, "adam_m_hash"}));

    adam_v_hash_ = arena.view(arena.allocate(
        {hash_bytes, 256, storage_for_grads,
         BufferLifetime::Persistent, "adam_v_hash"}));

    adam_m_mlp_ = arena.view(arena.allocate(
        {mlp_bytes, 256, storage_for_grads,
         BufferLifetime::Persistent, "adam_m_mlp"}));

    adam_v_mlp_ = arena.view(arena.allocate(
        {mlp_bytes, 256, storage_for_grads,
         BufferLifetime::Persistent, "adam_v_mlp"}));
  }

  // --- Gradients (always separate: int atomics for hash, float for MLP) ---
  grad_hash_ = arena.view(arena.allocate(
      {hash_bytes, 256, storage_for_grads,
       BufferLifetime::Persistent, "grad_hash"}));

  grad_mlp_ = arena.view(arena.allocate(
      {mlp_bytes, 256, storage_for_grads,
       BufferLifetime::Persistent, "grad_mlp"}));

  if (active_hash_mask_bytes > 0) {
    active_hash_mask_ = arena.view(arena.allocate(
        {active_hash_mask_bytes, 256, active_hash_mask_storage,
         BufferLifetime::Persistent, "active_hash_mask"}));
  }

  if (active_hash_summary_bytes > 0) {
    active_hash_summary_mask_ = arena.view(arena.allocate(
        {active_hash_summary_bytes, 256, BufferStorage::Shared,
         BufferLifetime::Persistent, "active_hash_summary_mask"}));
  }

  if (active_hash_index_bytes > 0) {
    active_hash_indices_ = arena.view(arena.allocate(
        {active_hash_index_bytes, 256, BufferStorage::Shared,
         BufferLifetime::Persistent, "active_hash_indices"}));
  }

  // --- Control buffers (Shared, Persistent) ---
  config_weights_ = arena.view(arena.allocate(
      {config_bytes, 256, BufferStorage::Shared,
       BufferLifetime::Persistent, "config_weights"}));

  const size_t train_param_bytes =
      desc_.train_params_layout.float_count * sizeof(float);
  train_params_ = arena.view(arena.allocate(
      {train_param_bytes, 256, BufferStorage::Shared,
       BufferLifetime::Persistent, "train_params"}));

  const size_t adam_param_bytes =
      desc_.adam_params_float_count * sizeof(float);
  adam_params_ = arena.view(arena.allocate(
      {adam_param_bytes, 256, BufferStorage::Shared,
       BufferLifetime::Persistent, "adam_params"}));

  // Total persistent bytes: weights + grads + m + v + control buffers.
  // Fused mode: 3 contiguous buffers (weights, m, v) + 2 grad buffers.
  // Separate mode: 8 individual buffers (2 weights, 2 grads, 4 moments).
  total_bytes_ = 2 * (hash_bytes + mlp_bytes) // weights + grads
                 + 2 * (hash_bytes + mlp_bytes) // m + v
                 + active_hash_mask_bytes + active_hash_summary_bytes
                 + active_hash_index_bytes
                 + config_bytes + train_param_bytes + adam_param_bytes;
}

ParameterStore::~ParameterStore() {
  if (!arena_)
    return;

  if (desc_.use_fused_adam) {
    release_owned_view(arena_, fused_weights_);
    release_owned_view(arena_, fused_m_);
    release_owned_view(arena_, fused_v_);
  } else {
    release_owned_view(arena_, hash_weights_);
    release_owned_view(arena_, mlp_weights_);
    release_owned_view(arena_, adam_m_hash_);
    release_owned_view(arena_, adam_v_hash_);
    release_owned_view(arena_, adam_m_mlp_);
    release_owned_view(arena_, adam_v_mlp_);
  }

  release_owned_view(arena_, grad_hash_);
  release_owned_view(arena_, grad_mlp_);
  release_owned_view(arena_, active_hash_mask_);
  release_owned_view(arena_, active_hash_summary_mask_);
  release_owned_view(arena_, active_hash_indices_);
  release_owned_view(arena_, config_weights_);
  release_owned_view(arena_, train_params_);
  release_owned_view(arena_, adam_params_);

  hash_weights_ = {};
  mlp_weights_ = {};
  adam_m_hash_ = {};
  adam_v_hash_ = {};
  adam_m_mlp_ = {};
  adam_v_mlp_ = {};
  grad_hash_ = {};
  grad_mlp_ = {};
  active_hash_mask_ = {};
  active_hash_summary_mask_ = {};
  active_hash_indices_ = {};
  config_weights_ = {};
  train_params_ = {};
  adam_params_ = {};
  fused_weights_ = {};
  fused_m_ = {};
  fused_v_ = {};
}

void ParameterStore::hydrate_weights(const float *hash_data, size_t hash_count,
                                      const float *mlp_data, size_t mlp_count,
                                      const float *config_header_8f) {
  // Copy hash grid weights.
  if (hash_weights_.data && hash_data && hash_count > 0) {
    const size_t bytes = std::min(hash_count, static_cast<size_t>(desc_.hash_grid_size))
                         * sizeof(float);
    std::memcpy(hash_weights_.data, hash_data, bytes);
  }

  // Copy MLP weights.
  if (mlp_weights_.data && mlp_data && mlp_count > 0) {
    const size_t bytes = std::min(mlp_count, static_cast<size_t>(desc_.mlp_weight_count))
                         * sizeof(float);
    std::memcpy(mlp_weights_.data, mlp_data, bytes);
  }

  // Populate config_weights = [8-float header | MLP weights].
  if (config_weights_.data) {
    auto *dst = static_cast<char *>(config_weights_.data);

    // Header (8 floats).
    if (config_header_8f) {
      std::memcpy(dst, config_header_8f, kConfigPackedFloats * sizeof(float));
    } else {
      std::memset(dst, 0, kConfigPackedFloats * sizeof(float));
    }

    // MLP section.
    if (mlp_data && mlp_count > 0) {
      const size_t bytes = std::min(mlp_count, static_cast<size_t>(desc_.mlp_weight_count))
                           * sizeof(float);
      std::memcpy(dst + kConfigPackedFloats * sizeof(float), mlp_data, bytes);
    }
  }
}

void ParameterStore::sync_live_weights(const float *hash_data, size_t hash_count,
                                        const float *mlp_data, size_t mlp_count) {
  if (hash_weights_.data && hash_data && hash_count > 0) {
    const size_t bytes = std::min(hash_count, static_cast<size_t>(desc_.hash_grid_size))
                         * sizeof(float);
    std::memcpy(hash_weights_.data, hash_data, bytes);
  }
  if (mlp_weights_.data && mlp_data && mlp_count > 0) {
    const size_t bytes = std::min(mlp_count, static_cast<size_t>(desc_.mlp_weight_count))
                         * sizeof(float);
    std::memcpy(mlp_weights_.data, mlp_data, bytes);
  }
}

ParameterStore::AsyncStepResult ParameterStore::finalize_async_step(
    const StepBufferSet &completed, uint32_t num_tgs, uint32_t batch_N,
    uint32_t completed_step, bool sync_config_weights) {
  const uint32_t terms = desc_.reduction_terms;
  AsyncStepResult result{};

  if (num_tgs > 0) {
    if (!completed.loss_reduction.data) {
      throw std::invalid_argument(
          "ParameterStore::finalize_async_step: loss_reduction data is required "
          "when num_tgs > 0");
    }
    if (batch_N == 0) {
      throw std::invalid_argument(
          "ParameterStore::finalize_async_step: batch_N must be > 0 when "
          "num_tgs > 0");
    }
    const size_t required_bytes =
        static_cast<size_t>(num_tgs) * static_cast<size_t>(terms) *
        sizeof(float);
    if (completed.loss_reduction.bytes < required_bytes) {
      throw std::invalid_argument(
          "ParameterStore::finalize_async_step: loss_reduction buffer is too "
          "small for num_tgs * reduction_terms");
    }
  }

  // Sum loss reduction partials (strided layout: partials[tg * terms + term]).
  if (completed.loss_reduction.data && num_tgs > 0) {
    auto *partials = static_cast<const float *>(completed.loss_reduction.data);

    // Term 0 → mean_loss.
    float total0 = 0.0f;
    for (uint32_t i = 0; i < num_tgs; ++i)
      total0 += partials[i * terms];
    result.mean_loss = (batch_N > 0) ? total0 / static_cast<float>(batch_N) : 0.0f;

    // Terms 1..N → extra_losses[].
    const uint32_t extra_count = (terms > 1) ? (terms - 1) : 0u;
    result.extra_loss_count = extra_count;
    for (uint32_t t = 0; t < extra_count; ++t) {
      float total_t = 0.0f;
      for (uint32_t i = 0; i < num_tgs; ++i)
        total_t += partials[i * terms + t + 1];
      result.extra_losses[t] = (batch_N > 0) ? total_t / static_cast<float>(batch_N) : 0.0f;
    }
  }

  if (sync_config_weights)
    sync_config_mlp_section(desc_, mlp_weights_, config_weights_);

  result.completed_step = completed_step;
  return result;
}

void ParameterStore::sync_config_mlp_from_live_weights() {
  sync_config_mlp_section(desc_, mlp_weights_, config_weights_);
}

void ParameterStore::reset_adam_state() {
  if (desc_.use_fused_adam) {
    if (fused_m_.data)
      std::memset(fused_m_.data, 0, fused_m_.bytes);
    if (fused_v_.data)
      std::memset(fused_v_.data, 0, fused_v_.bytes);
  } else {
    if (adam_m_hash_.data)
      std::memset(adam_m_hash_.data, 0, adam_m_hash_.bytes);
    if (adam_v_hash_.data)
      std::memset(adam_v_hash_.data, 0, adam_v_hash_.bytes);
    if (adam_m_mlp_.data)
      std::memset(adam_m_mlp_.data, 0, adam_m_mlp_.bytes);
    if (adam_v_mlp_.data)
      std::memset(adam_v_mlp_.data, 0, adam_v_mlp_.bytes);
  }
}

BufferView ParameterStore::config_header() const {
  return BufferArena::sub_view(config_weights_, 0,
                               kConfigPackedFloats * sizeof(float));
}

BufferView ParameterStore::config_mlp() const {
  return BufferArena::sub_view(config_weights_,
                               kConfigPackedFloats * sizeof(float),
                               desc_.mlp_weight_count * sizeof(float));
}

BufferView ParameterStore::config_tail() const {
  if (desc_.config_tail_floats == 0)
    return BufferView{};
  const size_t tail_offset =
      (kConfigPackedFloats + desc_.mlp_weight_count) * sizeof(float);
  return BufferArena::sub_view(config_weights_, tail_offset,
                               desc_.config_tail_floats * sizeof(float));
}

} // namespace tmnn

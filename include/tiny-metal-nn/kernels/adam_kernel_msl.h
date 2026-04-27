#pragma once

namespace tmnn {

// ---------------------------------------------------------------------------
// Unified Adam/AdamW Update kernel — processes both hash grid AND MLP weights
// ---------------------------------------------------------------------------
// Extends train_adam_fused_msl.h with decoupled weight decay (13th param).
// When weight_decay > 0: AdamW semantics (w -= lr * weight_decay * w before
// Adam step; L1/L2 regularization suppressed).
// When weight_decay == 0: identical to fused Adam (L1 hash + L2 MLP active).
//
// Buffer layout:
//   [0] weights      — (hash_sz + mlp_sz) floats (device read/write)
//   [1] gradients    — (hash_sz + mlp_sz) floats/ints (device read/write)
//   [2] m            — (hash_sz + mlp_sz) floats (first moment)
//   [3] v            — (hash_sz + mlp_sz) floats (second moment)
//   [4] adam_params   — 13 floats (constant):
//        [0] lr_hash, [1] lr_mlp, [2] beta1, [3] beta2, [4] eps,
//        [5] bc1, [6] bc2, [7] l1_hash, [8] l2_mlp,
//        [9] hash_sz_lo16, [10] hash_sz_hi16, [11] grad_clip, [12] weight_decay
//        `hash_sz` is reconstructed as lo16 | (hi16 << 16), which preserves the
//        exact uint32 count while keeping the legacy 13-float buffer ABI.

constexpr const char* kNeuralSDFAdamUnifiedMSL = R"msl(
#include <metal_stdlib>
using namespace metal;

#ifndef INV_GRAD_SCALE
#define INV_GRAD_SCALE (1.0f / 65536.0f)
#endif

#ifndef SPARSE_HASH_ENTRY_WIDTH
#define SPARSE_HASH_ENTRY_WIDTH 1u
#endif

kernel void neural_sdf_adam_unified(
    device float*       weights     [[buffer(0)]],
#ifdef USE_INT_ATOMICS_HASH
    device int*         grad_hash   [[buffer(1)]],
    device float*       grad_mlp    [[buffer(5)]],
#else
    device float*       gradients   [[buffer(1)]],
#endif
    device float*       m           [[buffer(2)]],
    device float*       v           [[buffer(3)]],
    constant float*     adam_params [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    float lr_hash     = adam_params[0];
    float lr_mlp      = adam_params[1];
    float beta1       = adam_params[2];
    float beta2       = adam_params[3];
    float eps         = adam_params[4];
    float bc1         = adam_params[5];
    float bc2         = adam_params[6];
    float l1_hash     = adam_params[7];
    float l2_mlp      = adam_params[8];
    uint hash_sz      = uint(adam_params[9]) | (uint(adam_params[10]) << 16);
    float grad_clip   = adam_params[11];
    float weight_decay = adam_params[12];

    bool is_mlp = (tid >= hash_sz);
    float lr = is_mlp ? lr_mlp : lr_hash;

    // Read gradient (do NOT clear — zero_gradients() is called explicitly
    // at the start of each accumulation window, not inside Adam).
    float g;
#ifdef USE_INT_ATOMICS_HASH
    if (is_mlp) {
        uint mlp_idx = tid - hash_sz;
        g = grad_mlp[mlp_idx];
    } else {
        g = float(grad_hash[tid]) * INV_GRAD_SCALE;
    }
#else
    g = gradients[tid];
#endif

    // Optional gradient clipping
    if (grad_clip > 0.0f)
        g = clamp(g, -grad_clip, grad_clip);

    // Decoupled weight decay (AdamW) OR traditional regularization
    if (weight_decay > 0.0f) {
        // AdamW: apply weight decay directly to weights, suppress L1/L2
        weights[tid] -= lr * weight_decay * weights[tid];
    } else {
        // Traditional Adam: L1 for hash grid, L2 for MLP
        if (!is_mlp && l1_hash > 0.0f) {
            float w = weights[tid];
            g += l1_hash * (w > 0.0f ? 1.0f : (w < 0.0f ? -1.0f : 0.0f));
        }
        if (is_mlp && l2_mlp > 0.0f) {
            g += l2_mlp * weights[tid];
        }
    }

    // Adam update
    float m_new = beta1 * m[tid] + (1.0f - beta1) * g;
    float v_new = beta2 * v[tid] + (1.0f - beta2) * g * g;
    m[tid] = m_new;
    v[tid] = v_new;

    float m_hat = m_new / bc1;
    float v_hat = v_new / bc2;

    weights[tid] -= lr * m_hat / (sqrt(v_hat) + eps);
}
)msl";

constexpr const char* kNeuralSDFAdamSplitMSL = R"msl(
#include <metal_stdlib>
using namespace metal;

#ifndef INV_GRAD_SCALE
#define INV_GRAD_SCALE (1.0f / 65536.0f)
#endif

kernel void neural_sdf_adam_hash_sparse(
    device float*       hash_weights  [[buffer(0)]],
    device int*         grad_hash     [[buffer(1)]],
    device float*       m_hash        [[buffer(2)]],
    device float*       v_hash        [[buffer(3)]],
    constant float*     adam_params   [[buffer(4)]],
    device const uint*  active_indices [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    float lr_hash      = adam_params[0];
    float beta1        = adam_params[2];
    float beta2        = adam_params[3];
    float eps          = adam_params[4];
    float bc1          = adam_params[5];
    float bc2          = adam_params[6];
    float l1_hash      = adam_params[7];
    float grad_clip    = adam_params[11];
    float weight_decay = adam_params[12];

    uint base_idx = active_indices[tid];
    for (uint lane = 0; lane < SPARSE_HASH_ENTRY_WIDTH; ++lane) {
        uint idx = base_idx + lane;

        float g = float(grad_hash[idx]) * INV_GRAD_SCALE;

        if (grad_clip > 0.0f)
            g = clamp(g, -grad_clip, grad_clip);

        if (weight_decay > 0.0f) {
            hash_weights[idx] -= lr_hash * weight_decay * hash_weights[idx];
        } else if (l1_hash > 0.0f) {
            float w = hash_weights[idx];
            g += l1_hash * (w > 0.0f ? 1.0f : (w < 0.0f ? -1.0f : 0.0f));
        }

        float m_new = beta1 * m_hash[idx] + (1.0f - beta1) * g;
        float v_new = beta2 * v_hash[idx] + (1.0f - beta2) * g * g;
        m_hash[idx] = m_new;
        v_hash[idx] = v_new;

        float m_hat = m_new / bc1;
        float v_hat = v_new / bc2;
        hash_weights[idx] -= lr_hash * m_hat / (sqrt(v_hat) + eps);
    }
}

kernel void neural_sdf_adam_mlp_dense(
    device float*       mlp_weights  [[buffer(0)]],
    device float*       grad_mlp     [[buffer(1)]],
    device float*       m_mlp        [[buffer(2)]],
    device float*       v_mlp        [[buffer(3)]],
    constant float*     adam_params  [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    float lr_mlp       = adam_params[1];
    float beta1        = adam_params[2];
    float beta2        = adam_params[3];
    float eps          = adam_params[4];
    float bc1          = adam_params[5];
    float bc2          = adam_params[6];
    float l2_mlp       = adam_params[8];
    float grad_clip    = adam_params[11];
    float weight_decay = adam_params[12];

    float g = grad_mlp[tid];

    if (grad_clip > 0.0f)
        g = clamp(g, -grad_clip, grad_clip);

    if (weight_decay > 0.0f) {
        mlp_weights[tid] -= lr_mlp * weight_decay * mlp_weights[tid];
    } else if (l2_mlp > 0.0f) {
        g += l2_mlp * mlp_weights[tid];
    }

    float m_new = beta1 * m_mlp[tid] + (1.0f - beta1) * g;
    float v_new = beta2 * v_mlp[tid] + (1.0f - beta2) * g * g;
    m_mlp[tid] = m_new;
    v_mlp[tid] = v_new;

    float m_hat = m_new / bc1;
    float v_hat = v_new / bc2;
    mlp_weights[tid] -= lr_mlp * m_hat / (sqrt(v_hat) + eps);
}
)msl";

} // namespace tmnn

/**
 * @file mlp_kernel_emitter.cpp
 * @brief MLPKernelEmitter: generates specialized MSL kernel source from KernelSpec.
 */

#include "tiny-metal-nn/kernels/mlp_kernel_emitter.h"

#include <cassert>
#include <iomanip>
#include <vector>

namespace tmnn {

// Code-generation constants (MSL FNV hash primes, matching Instant-NGP)
namespace {
  constexpr uint32_t kFNV_PRIME_Y = 2654435761u;
  constexpr uint32_t kFNV_PRIME_Z = 805459861u;
  constexpr uint32_t kFNV_PRIME_W = 56711039u;
  constexpr int kRotationMatrixFloats = 9;    // 3x3 row-major
  constexpr float kDefaultGradScale = 65536.0f;

  bool use_scalar_vec4_eval_fastpath(const KernelSpec& spec) {
    return !spec.use_simd && spec.encoding == KernelSpec::Standard &&
           spec.spatial_dims == 3 && spec.num_outputs == 1 &&
           spec.features_per_level == 2 && (spec.hidden_dim % 4) == 0;
  }

  bool use_scalar_fp16_hidden_fastpath(const KernelSpec& spec) {
    return use_scalar_vec4_eval_fastpath(spec) && spec.use_fp16 &&
           spec.num_hidden_layers > 1;
  }

  void emit_train_param_macros(std::ostringstream& o) {
    o << "#ifndef TMNN_TRAIN_PARAMS_IDX_N\n"
      << "#define TMNN_TRAIN_PARAMS_IDX_N 0\n"
      << "#endif\n"
      << "#ifndef TMNN_TRAIN_PARAMS_IDX_UNSIGNED_MODE\n"
      << "#define TMNN_TRAIN_PARAMS_IDX_UNSIGNED_MODE 1\n"
      << "#endif\n"
      << "#ifndef TMNN_TRAIN_PARAMS_IDX_LOSS_SCALE\n"
      << "#define TMNN_TRAIN_PARAMS_IDX_LOSS_SCALE 2\n"
      << "#endif\n"
      << "#ifndef TMNN_TRAIN_PARAMS_IDX_NUM_ACTIVE_LEVELS\n"
      << "#define TMNN_TRAIN_PARAMS_IDX_NUM_ACTIVE_LEVELS 3\n"
      << "#endif\n\n";
  }

  void emit_active_hash_tracking(std::ostringstream& o,
                                 const char* tracked_index_expr,
                                 bool grouped_entry_pairs,
                                 const char* indent) {
    if (grouped_entry_pairs) {
      o << indent << "uint entry_idx = " << tracked_index_expr << " >> 1u;\n"
        << indent << "uint word_idx = entry_idx >> 5u;\n"
        << indent << "uint word_mask = (1u << (entry_idx & 31u));\n";
    } else {
      o << indent << "uint word_idx = " << tracked_index_expr << " >> 5u;\n"
        << indent << "uint word_mask = (1u << (" << tracked_index_expr
        << " & 31u));\n";
    }
    o << indent
      << "uint word_prev = atomic_fetch_or_explicit(&active_hash_mask[word_idx],\n"
      << indent << "    word_mask, memory_order_relaxed);\n"
      << indent << "if (word_prev == 0u) {\n"
      << indent << "    uint child_word_idx = word_idx >> 1u;\n"
      << indent
      << "    atomic_fetch_or_explicit(&active_hash_summary_mask[child_word_idx >> 5u],\n"
      << indent
      << "        (1u << (child_word_idx & 31u)), memory_order_relaxed);\n"
      << indent << "}\n";
  }
} // namespace

// ===========================================================================
// MLPKernelEmitter — shared fragments
// ===========================================================================

void MLPKernelEmitter::emitPreamble(std::ostringstream& o, const KernelSpec& spec) {
  o << "#include <metal_stdlib>\n"
    << "using namespace metal;\n\n";

  // Bake architecture constants
  o << "// --- Architecture constants (baked from KernelSpec) ---\n"
    << "constant int INPUT_DIM = " << spec.input_dim << ";\n"
    << "constant int HIDDEN_DIM = " << spec.hidden_dim << ";\n"
    << "constant int NUM_HIDDEN_LAYERS = " << spec.num_hidden_layers << ";\n"
    << "constant int NUM_OUTPUTS = " << spec.num_outputs << ";\n"
    << "constant int NUM_LEVELS = " << spec.num_levels << ";\n"
    << "constant int FEATURES_PER_LEVEL = " << spec.features_per_level << ";\n"
    << "constant int LOG2_HASHMAP_SIZE = " << spec.log2_hashmap_size << ";\n"
    << "constant float BASE_RESOLUTION = " << std::fixed << std::setprecision(6) << spec.base_resolution << "f;\n"
    << "constant float PER_LEVEL_SCALE = " << std::fixed << std::setprecision(6) << spec.per_level_scale << "f;\n"
    << "constant int MLP_WEIGHT_COUNT = " << spec.mlpWeightCount() << ";\n"
    << "\n";

  // Hash grid type alias (float or half depending on FP16 hash grid mode)
  if (spec.use_fp16_hash_grid) {
    o << "typedef half HASH_T;\n"
      << "typedef half2 HASH2_T;\n\n";
  } else {
    o << "typedef float HASH_T;\n"
      << "typedef float2 HASH2_T;\n\n";
  }

  // SIMD activation / matrix type aliases (only emitted for SIMD kernels)
  if (spec.use_simd) {
    if (spec.use_fp16_simd) {
      o << "typedef half SIMD_ACT;\n"
        << "typedef simdgroup_half8x8 SIMD_MAT;\n\n";
    } else {
      o << "typedef float SIMD_ACT;\n"
        << "typedef simdgroup_float8x8 SIMD_MAT;\n\n";
    }
  }

  // Config struct (still used for load_config compatibility)
  o << "struct NeuralSDFConfig {\n"
    << "    int num_levels;\n"
    << "    int features_per_level;\n"
    << "    int log2_hashmap_size;\n"
    << "    float base_resolution;\n"
    << "    float per_level_scale;\n"
    << "    int hidden_dim;\n"
    << "    int num_hidden_layers;\n"
    << "    int num_points;\n"
    << "};\n\n"
    << "inline NeuralSDFConfig load_config(device const float* cw) {\n"
    << "    NeuralSDFConfig c;\n"
    << "    c.num_levels         = int(cw[0]);\n"
    << "    c.features_per_level = int(cw[1]);\n"
    << "    c.log2_hashmap_size  = int(cw[2]);\n"
    << "    c.base_resolution    = cw[3];\n"
    << "    c.per_level_scale    = cw[4];\n"
    << "    c.hidden_dim         = int(cw[5]);\n"
    << "    c.num_hidden_layers  = int(cw[6]);\n"
    << "    c.num_points         = int(cw[7]);\n"
    << "    return c;\n"
    << "}\n\n";

  // ReLU
  o << "inline float nn_relu(float x) { return x > 0.0 ? x : 0.0; }\n\n";
}

void MLPKernelEmitter::emitHashFunctions(std::ostringstream& o, const KernelSpec& spec) {
  o << "inline uint hash_coords(int3 coords, uint table_size) {\n"
    << "    uint h = uint(coords.x) * 1u\n"
    << "           ^ uint(coords.y) * " << kFNV_PRIME_Y << "u\n"
    << "           ^ uint(coords.z) * " << kFNV_PRIME_Z << "u;\n"
    << "    return h % table_size;\n"
    << "}\n\n";

  if (spec.encoding == KernelSpec::FourD || spec.spatial_dims == 4) {
    o << "inline uint hash_coords_4d(int4 coords, uint table_size) {\n"
      << "    uint h = uint(coords.x) * 1u\n"
      << "           ^ uint(coords.y) * " << kFNV_PRIME_Y << "u\n"
      << "           ^ uint(coords.z) * " << kFNV_PRIME_Z << "u\n"
      << "           ^ uint(coords.w) * " << kFNV_PRIME_W << "u;\n"
      << "    return h % table_size;\n"
      << "}\n\n";
  }
}

void MLPKernelEmitter::emitWeightOffsets(std::ostringstream& o, const KernelSpec& spec) {
  // Compute weight offsets for each layer
  int offset = 0;
  int hidden_half_offset = 0;
  o << "    // MLP weight offsets (baked constants)\n";

  // Layer 0: input_dim → hidden_dim
  o << "    const int w0_off = " << offset << ";\n";
  offset += spec.input_dim * spec.hidden_dim;
  o << "    const int b0_off = " << offset << ";\n";
  offset += spec.hidden_dim;

  // Hidden layers 1..N-1
  for (int i = 1; i < spec.num_hidden_layers; ++i) {
    o << "    const int w" << i << "_off = " << offset << ";\n";
    if (use_scalar_fp16_hidden_fastpath(spec))
      o << "    const int w" << i << "_half_off = " << hidden_half_offset << ";\n";
    offset += spec.hidden_dim * spec.hidden_dim;
    hidden_half_offset += spec.hidden_dim * spec.hidden_dim;
    o << "    const int b" << i << "_off = " << offset << ";\n";
    offset += spec.hidden_dim;
  }

  // Output layer
  o << "    const int wO_off = " << offset << ";\n";
  offset += spec.hidden_dim * spec.num_outputs;
  o << "    const int bO_off = " << offset << ";\n";
  o << "\n";
}

void MLPKernelEmitter::emitHashEncode3D(std::ostringstream& o, const KernelSpec& spec) {
  emitHashEncodeUnified(o, spec, {});
}

void MLPKernelEmitter::emitHashEncode4D(std::ostringstream& o, const KernelSpec& spec) {
  emitHashEncodeUnified(o, spec, {.is_4d = true});
}

void MLPKernelEmitter::emitRMHEHashEncode(std::ostringstream& o, const KernelSpec& spec) {
  emitHashEncodeUnified(o, spec, {.is_rmhe = true});
}

void MLPKernelEmitter::emitHashEncodeUnified(std::ostringstream& o, const KernelSpec& spec,
                                             const HashEncodeOpts& opts) {
  if (opts.is_cooperative) {
    // Cooperative path: separate inline function emitted by emitHashEncodeCooperative()
    // This method handles the scalar/inline hash encoding paths.
    return;
  }

  const char* label = opts.is_4d ? "4D quadrilinear" :
                      opts.is_rmhe ? "rotated 3D trilinear" : "3D trilinear";
  o << "    // === Hash grid encode (" << label << ") fused with W0 matmul ===\n"
    << "    uint table_size = 1u << uint(LOG2_HASHMAP_SIZE);\n";

  if (opts.is_rmhe) {
    o << "    device const float* rotations = mlp + MLP_WEIGHT_COUNT;\n";
  }

  o << "    float h0[HIDDEN_DIM];\n"
    << "    for (int j = 0; j < HIDDEN_DIM; j++) h0[j] = 0.0f;\n\n";
  o << "    float resolution = BASE_RESOLUTION;\n"
    << "    for (int l = 0; l < NUM_LEVELS; l++) {\n";

  if (opts.is_rmhe) {
    o << "\n"
      << "        // Apply rotation for this level\n"
      << "        constant float* R = rotations + l * " << kRotationMatrixFloats << ";\n"
      << "        float rx = R[0]*pos.x + R[1]*pos.y + R[2]*pos.z;\n"
      << "        float ry = R[3]*pos.x + R[4]*pos.y + R[5]*pos.z;\n"
      << "        float rz = R[6]*pos.x + R[7]*pos.y + R[8]*pos.z;\n\n"
      << "        float sx = rx * resolution;\n"
      << "        float sy = ry * resolution;\n"
      << "        float sz = rz * resolution;\n\n";
  } else if (opts.is_4d) {
    o << "        float sx = pos.x * resolution;\n"
      << "        float sy = pos.y * resolution;\n"
      << "        float sz = pos.z * resolution;\n"
      << "        float sw = pos.w * resolution;\n\n";
  } else {
    o << "        float sx = pos.x * resolution;\n"
      << "        float sy = pos.y * resolution;\n"
      << "        float sz = pos.z * resolution;\n\n";
  }

  // Base coords and fractional positions
  if (opts.is_4d) {
    o << "        int4 bc = int4(int(floor(sx)), int(floor(sy)), int(floor(sz)), int(floor(sw)));\n"
      << "        float4 frac = float4(sx - floor(sx), sy - floor(sy), sz - floor(sz), sw - floor(sw));\n\n";
  } else {
    o << "        int3 bc = int3(int(floor(sx)), int(floor(sy)), int(floor(sz)));\n"
      << "        float3 frac = float3(sx - floor(sx), sy - floor(sy), sz - floor(sz));\n\n";
  }

  // Feature accumulation with trilinear/quadrilinear interpolation
  // Vectorized path: when features_per_level == 2 and 3D, use float2 loads
  bool vec2_path = (spec.features_per_level == 2 && !opts.is_4d && !opts.is_rmhe);
  const bool scalar_vec4_fastpath =
      use_scalar_vec4_eval_fastpath(spec) && !opts.is_4d && !opts.is_rmhe &&
      opts.fuse_w0;

  if (vec2_path) {
    // float2 vectorized: load 2 features at once
    o << "        {\n"
      << "            float2 feat2 = float2(0.0f);\n"
      << "            for (int dz = 0; dz < 2; dz++)\n"
      << "                for (int dy = 0; dy < 2; dy++)\n"
      << "                    for (int dx = 0; dx < 2; dx++) {\n"
      << "                        uint h = hash_coords(bc + int3(dx, dy, dz), table_size);\n"
      << "                        uint grid_off = uint(l) * table_size * 2u + h * 2u;\n"
      << "                        float wx = (dx == 0) ? (1.0f - frac.x) : frac.x;\n"
      << "                        float wy = (dy == 0) ? (1.0f - frac.y) : frac.y;\n"
      << "                        float wz = (dz == 0) ? (1.0f - frac.z) : frac.z;\n"
      << "                        float w = wx * wy * wz;\n"
      << "                        feat2 += w * float2(*((device const HASH2_T*)(hash_grid + grid_off)));\n"
      << "                    }\n\n";
    if (opts.fuse_w0) {
      if (scalar_vec4_fastpath) {
        o << "            int fb = l * 2;\n"
          << "            for (int j = 0; j < HIDDEN_DIM; j += 4) {\n"
        << "                float4 acc4 = feat2.x * *((device const float4*)(mlp + w0_off + fb * HIDDEN_DIM + j))\n"
        << "                            + feat2.y * *((device const float4*)(mlp + w0_off + (fb + 1) * HIDDEN_DIM + j));\n"
          << "                h0[j + 0] += acc4.x;\n"
          << "                h0[j + 1] += acc4.y;\n"
          << "                h0[j + 2] += acc4.z;\n"
          << "                h0[j + 3] += acc4.w;\n"
          << "            }\n";
      } else {
        o << "            int fb = l * 2;\n"
          << "            for (int j = 0; j < HIDDEN_DIM; j++)\n"
          << "                h0[j] += feat2.x * W_SRC(w0_off + fb * HIDDEN_DIM + j)\n"
          << "                       + feat2.y * W_SRC(w0_off + (fb + 1) * HIDDEN_DIM + j);\n";
      }
    }
    o << "        }\n"
      << "        resolution *= PER_LEVEL_SCALE;\n"
      << "    }\n\n";
  } else {
    // Scalar path (original)
    o << "        for (int f = 0; f < FEATURES_PER_LEVEL; f++) {\n"
      << "            float feat = 0.0f;\n";

    if (opts.is_4d) {
      o << "            for (int dw = 0; dw < 2; dw++)\n"
        << "                for (int dz = 0; dz < 2; dz++)\n"
        << "                    for (int dy = 0; dy < 2; dy++)\n"
        << "                        for (int dx = 0; dx < 2; dx++) {\n"
        << "                            uint h = hash_coords_4d(bc + int4(dx, dy, dz, dw), table_size);\n"
        << "                            uint grid_off = uint(l) * table_size * FEATURES_PER_LEVEL + uint(h) * FEATURES_PER_LEVEL + f;\n"
        << "                            float wx = (dx == 0) ? (1.0f - frac.x) : frac.x;\n"
        << "                            float wy = (dy == 0) ? (1.0f - frac.y) : frac.y;\n"
        << "                            float wz = (dz == 0) ? (1.0f - frac.z) : frac.z;\n"
        << "                            float wt = (dw == 0) ? (1.0f - frac.w) : frac.w;\n"
        << "                            feat += wx * wy * wz * wt * hash_grid[grid_off];\n"
        << "                        }\n\n";
    } else {
      o << "            for (int dz = 0; dz < 2; dz++)\n"
        << "                for (int dy = 0; dy < 2; dy++)\n"
        << "                    for (int dx = 0; dx < 2; dx++) {\n"
        << "                        uint h = hash_coords(bc + int3(dx, dy, dz), table_size);\n"
        << "                        uint grid_off = uint(l) * table_size * FEATURES_PER_LEVEL + uint(h) * FEATURES_PER_LEVEL + f;\n"
        << "                        float wx = (dx == 0) ? (1.0f - frac.x) : frac.x;\n"
        << "                        float wy = (dy == 0) ? (1.0f - frac.y) : frac.y;\n"
        << "                        float wz = (dz == 0) ? (1.0f - frac.z) : frac.z;\n"
        << "                        feat += wx * wy * wz * hash_grid[grid_off];\n"
        << "                    }\n\n";
    }

    if (opts.fuse_w0) {
      o << "            int feat_idx = l * FEATURES_PER_LEVEL + f;\n"
        << "            int w_row = w0_off + feat_idx * HIDDEN_DIM;\n"
        << "            for (int j = 0; j < HIDDEN_DIM; j++)\n"
        << "                h0[j] += feat * W_SRC(w_row + j);\n";
    }

    o << "        }\n"
      << "        resolution *= PER_LEVEL_SCALE;\n"
      << "    }\n\n";
  }

  if (opts.fuse_w0) {
    o << "    // Bias + ReLU for layer 0\n";
    if (scalar_vec4_fastpath) {
      o << "    for (int j = 0; j < HIDDEN_DIM; j += 4) {\n"
        << "        float4 h4 = float4(h0[j + 0], h0[j + 1], h0[j + 2], h0[j + 3]);\n"
        << "        float4 b4 = *((device const float4*)(mlp + b0_off + j));\n"
        << "        h4 = max(h4 + b4, float4(0.0f));\n"
        << "        h0[j + 0] = h4.x;\n"
        << "        h0[j + 1] = h4.y;\n"
        << "        h0[j + 2] = h4.z;\n"
        << "        h0[j + 3] = h4.w;\n"
        << "    }\n\n";
    } else {
      o << "    for (int j = 0; j < HIDDEN_DIM; j++)\n"
        << "        h0[j] = max(h0[j] + W_SRC(b0_off + j), 0.0f);\n\n";
    }
  }
}

void MLPKernelEmitter::emitMLPForward(std::ostringstream& o, const KernelSpec& spec) {
  int hd = spec.hidden_dim;
  int nhl = spec.num_hidden_layers;
  int nout = spec.num_outputs;
  const bool scalar_vec4_fastpath = use_scalar_vec4_eval_fastpath(spec);
  const bool scalar_fp16_hidden_fastpath =
      use_scalar_fp16_hidden_fastpath(spec);

  o << "    // === MLP forward: hidden layers 1..N-1 ===\n";

  // For layers 1..N-1, each is hidden_dim → hidden_dim with ReLU
  // We alternate between arrays: even layers use hEven, odd use hOdd.
  // Layer 0 output is in h0 already.
  // For N=2: layer 1 reads h0, writes h1. Then output reads h1.
  // For N=3: layer 1 reads h0, writes h1. Layer 2 reads h1, writes h0. Output reads h0.
  // General: layer i input is h{(i-1)%2}, output is h{i%2}.
  // We need two arrays: h0 (already declared) and h1.

  if (nhl > 1) {
    o << "    float h1[HIDDEN_DIM];\n";
  }
  if (scalar_fp16_hidden_fastpath) {
    o << "    const int HIDDEN_HALF_TILE_WORDS = 16;\n"
      << "    const int HIDDEN_HALF_TILES_PER_AXIS = HIDDEN_DIM / 4;\n";
  }

  for (int layer = 1; layer < nhl; ++layer) {
    const char* src_arr = ((layer - 1) % 2 == 0) ? "h0" : "h1";
    const char* dst_arr = (layer % 2 == 0) ? "h0" : "h1";
    o << "    // Hidden layer " << layer << ": " << src_arr << " -> " << dst_arr << "\n";
    if (scalar_fp16_hidden_fastpath) {
      o << "    for (int j = 0; j < HIDDEN_DIM; j += 4) {\n"
        << "        float4 acc4 = float4(0.0f);\n"
        << "        int j_tile = j / 4;\n"
        << "        for (int i = 0; i < HIDDEN_DIM; i += 4) {\n"
        << "            int tile_off = w" << layer
        << "_half_off + (j_tile * HIDDEN_HALF_TILES_PER_AXIS + (i / 4)) * "
        << "HIDDEN_HALF_TILE_WORDS;\n"
        << "            device const half4* tile = (device const half4*)"
        << "(mlp_hidden_half + tile_off);\n"
        << "            float4 src4 = float4(" << src_arr << "[i + 0], " << src_arr
        << "[i + 1], " << src_arr << "[i + 2], " << src_arr << "[i + 3]);\n"
        << "            acc4 += src4.x * float4(tile[0])\n"
        << "                 +  src4.y * float4(tile[1])\n"
        << "                 +  src4.z * float4(tile[2])\n"
        << "                 +  src4.w * float4(tile[3]);\n"
        << "        }\n"
        << "        float4 b4 = *((device const float4*)(mlp + b" << layer
        << "_off + j));\n"
        << "        acc4 = max(acc4 + b4, float4(0.0f));\n"
        << "        " << dst_arr << "[j + 0] = acc4.x;\n"
        << "        " << dst_arr << "[j + 1] = acc4.y;\n"
        << "        " << dst_arr << "[j + 2] = acc4.z;\n"
        << "        " << dst_arr << "[j + 3] = acc4.w;\n"
        << "    }\n\n";
    } else if (scalar_vec4_fastpath) {
      o << "    for (int j = 0; j < HIDDEN_DIM; j += 4) {\n"
        << "        float4 acc4 = float4(0.0f);\n"
        << "        for (int i = 0; i < HIDDEN_DIM; i++) {\n"
        << "            acc4 += " << src_arr << "[i] * *((device const float4*)(mlp + w"
        << layer << "_off + i * HIDDEN_DIM + j));\n"
        << "        }\n"
        << "        float4 b4 = *((device const float4*)(mlp + b" << layer
        << "_off + j));\n"
        << "        acc4 = max(acc4 + b4, float4(0.0f));\n"
        << "        " << dst_arr << "[j + 0] = acc4.x;\n"
        << "        " << dst_arr << "[j + 1] = acc4.y;\n"
        << "        " << dst_arr << "[j + 2] = acc4.z;\n"
        << "        " << dst_arr << "[j + 3] = acc4.w;\n"
        << "    }\n\n";
    } else {
      o << "    for (int j = 0; j < HIDDEN_DIM; j++) {\n"
        << "        float acc = 0.0f;\n"
        << "        for (int i = 0; i < HIDDEN_DIM; i++)\n"
        << "            acc += " << src_arr << "[i] * W_SRC(w" << layer << "_off + i * HIDDEN_DIM + j);\n"
        << "        " << dst_arr << "[j] = max(acc + W_SRC(b" << layer << "_off + j), 0.0f);\n"
        << "    }\n\n";
    }
  }

  // Output layer: last hidden layer output → num_outputs
  // Last hidden array: if nhl==1, it's h0. If nhl==2, layer 1 output is h1.
  // General: layer nhl-1 output is h{(nhl-1)%2}.
  const char* last_h = ((nhl - 1) % 2 == 0) ? "h0" : "h1";

  if (nout == 1) {
    o << "    // Output layer: " << last_h << " -> sdf\n"
      << "    float sdf = W_SRC(bO_off);\n";
    if (scalar_vec4_fastpath) {
      o << "    for (int j = 0; j < HIDDEN_DIM; j += 4)\n"
        << "        sdf += dot(float4(" << last_h << "[j + 0], " << last_h
        << "[j + 1], " << last_h << "[j + 2], " << last_h
        << "[j + 3]), *((device const float4*)(mlp + wO_off + j)));\n\n";
    } else {
      o << "    for (int j = 0; j < HIDDEN_DIM; j++)\n"
        << "        sdf += " << last_h << "[j] * W_SRC(wO_off + j);\n\n";
    }
  } else {
    o << "    // Output layer: " << last_h << " -> outputs[" << nout << "]\n"
      << "    float outputs[NUM_OUTPUTS];\n"
      << "    for (int m = 0; m < NUM_OUTPUTS; m++) {\n"
      << "        float acc = W_SRC(bO_off + m);\n"
      << "        for (int j = 0; j < HIDDEN_DIM; j++)\n"
      << "            acc += " << last_h << "[j] * W_SRC(wO_off + j * NUM_OUTPUTS + m);\n"
      << "        outputs[m] = acc;\n"
      << "    }\n\n";

    // DNL-specific activations
    if (nout == 4) {
      o << "#ifdef TMNN_OUTPUT_SEMANTICS_DNL\n"
        << "    // DNL activations: sigmoid, identity, softplus, softplus\n"
        << "    outputs[0] = 1.0f / (1.0f + exp(-outputs[0])); // density (sigmoid)\n"
        << "    // outputs[1] = outputs[1]; // TPMS logit (identity)\n"
        << "    outputs[2] = log(1.0f + exp(outputs[2])); // frequency (softplus)\n"
        << "    outputs[3] = log(1.0f + exp(outputs[3])); // Young's modulus (softplus)\n"
        << "#endif\n\n";
    }
  }
}

void MLPKernelEmitter::emitMLPBackward(std::ostringstream& o, const KernelSpec& spec) {
  int hd = spec.hidden_dim;
  int nhl = spec.num_hidden_layers;
  int nout = spec.num_outputs;

  // Backward pass — compute d_pre_h for each layer in reverse order.
  // We reuse the h0/h1 arrays in-place for gradients (saves registers).
  // The last hidden layer output is in h{(nhl-1)%2}.
  const char* last_h = ((nhl - 1) % 2 == 0) ? "h0" : "h1";

  o << "    // === MLP backward (in-place) ===\n";

  if (nout == 1) {
    // Output layer backward: d_sdf is already computed.
    // d_pre_h[j] = d_sdf * W_out[j] * relu_mask(h[j])
    o << "    // Output layer backward: compute d_pre_last_h in-place\n"
      << "    for (int j = 0; j < HIDDEN_DIM; j++) {\n"
      << "        float h_val = " << last_h << "[j];\n"
      << "        tg_grad_mlp[wO_off + j] += d_sdf * h_val;\n"
      << "        float relu_mask = (h_val > 0.0f) ? 1.0f : 0.0f;\n"
      << "        " << last_h << "[j] = d_sdf * mlp[wO_off + j] * relu_mask;\n"
      << "    }\n"
      << "    tg_grad_mlp[bO_off] += d_sdf;\n\n";
  } else {
    // Multi-output backward
    o << "    // Output layer backward (multi-output)\n"
      << "    for (int j = 0; j < HIDDEN_DIM; j++) {\n"
      << "        float h_val = " << last_h << "[j];\n"
      << "        float d_pre = 0.0f;\n"
      << "        for (int m = 0; m < NUM_OUTPUTS; m++) {\n"
      << "            tg_grad_mlp[wO_off + j * NUM_OUTPUTS + m] += d_out[m] * h_val;\n"
      << "            d_pre += d_out[m] * mlp[wO_off + j * NUM_OUTPUTS + m];\n"
      << "        }\n"
      << "        float relu_mask = (h_val > 0.0f) ? 1.0f : 0.0f;\n"
      << "        " << last_h << "[j] = d_pre * relu_mask;\n"
      << "    }\n"
      << "    for (int m = 0; m < NUM_OUTPUTS; m++)\n"
      << "        tg_grad_mlp[bO_off + m] += d_out[m];\n\n";
  }

  // Hidden layers backward: nhl-1 down to 1
  // After output backward, last_h contains d_pre for the last hidden layer.
  // Now propagate: layer i backward uses d_pre from layer i's output.
  for (int layer = nhl - 1; layer >= 1; --layer) {
    // d_pre is in h{layer%2} (which was the output of layer = the d_pre after output backward)
    const char* d_arr = (layer % 2 == 0) ? "h0" : "h1"; // holds d_pre for this layer
    const char* h_arr = ((layer - 1) % 2 == 0) ? "h0" : "h1"; // holds activation from prev layer

    // W grad: tg_grad_mlp[w_off + i*hd + j] += d_arr[j] * h_arr[i]
    // Then compute d_pre for the previous layer:
    // d_pre_prev[i] = (sum_j d_arr[j] * W[i,j]) * relu_mask(h_arr[i])
    o << "    // Layer " << layer << " backward\n"
      << "    for (int i = 0; i < HIDDEN_DIM; i++) {\n"
      << "        float h_val = " << h_arr << "[i];\n"
      << "        for (int j = 0; j < HIDDEN_DIM; j++)\n"
      << "            tg_grad_mlp[w" << layer << "_off + i * HIDDEN_DIM + j] += " << d_arr << "[j] * h_val;\n"
      << "        float acc = 0.0f;\n"
      << "        for (int j = 0; j < HIDDEN_DIM; j++)\n"
      << "            acc += " << d_arr << "[j] * mlp[w" << layer << "_off + i * HIDDEN_DIM + j];\n"
      << "        " << h_arr << "[i] = acc * ((h_val > 0.0f) ? 1.0f : 0.0f);\n"
      << "    }\n"
      << "    for (int j = 0; j < HIDDEN_DIM; j++)\n"
      << "        tg_grad_mlp[b" << layer << "_off + j] += " << d_arr << "[j];\n\n";
  }

  // After all hidden backwards, h0 contains d_pre_h0 (gradient w.r.t. layer 0 pre-activation)
}

void MLPKernelEmitter::emitFP16Scale(std::ostringstream& o,
                                     const KernelSpec& spec,
                                     const char* var, bool forward,
                                     int indent) {
  if (!spec.use_fp16) return;
  for (int i = 0; i < indent; ++i) o << ' ';
  if (forward)
    o << var << " *= loss_scale;\n";
  else
    o << var << " *= inv_loss_scale;\n";
}

void MLPKernelEmitter::emitLoss(std::ostringstream& o, const KernelSpec& spec) {
  // Loss scale (FP16 loss scaling; hoisted inv_loss_scale used by hash scatter)
  o << "        float loss_scale = train_params[TMNN_TRAIN_PARAMS_IDX_LOSS_SCALE];\n"
    << "        inv_loss_scale = 1.0f / loss_scale;\n\n";

  if (spec.num_outputs == 1) {
    o << "        // Loss computation\n"
      << "        float target = targets[tid];\n";

    if (spec.loss == KernelSpec::L2) {
      o << "        if (unsigned_mode != 0) {\n"
        << "            float abs_sdf = abs(sdf);\n"
        << "            float residual = abs_sdf - target;\n"
        << "            thread_loss = residual * residual;\n"
        << "            float sign_sdf = (sdf >= 0.0f) ? 1.0f : -1.0f;\n"
        << "            d_sdf = 2.0f * residual * sign_sdf * inv_N;\n"
        << "        } else {\n"
        << "            float residual = sdf - target;\n"
        << "            thread_loss = residual * residual;\n"
        << "            d_sdf = 2.0f * residual * inv_N;\n"
        << "        }\n";
    } else if (spec.loss == KernelSpec::L1) {
      o << "        {\n"
        << "            float residual = sdf - target;\n"
        << "            thread_loss = abs(residual);\n"
        << "            d_sdf = ((residual > 0.0f) ? 1.0f : ((residual < 0.0f) ? -1.0f : 0.0f)) * inv_N;\n"
        << "        }\n";
    } else { // Huber
      o << "        {\n"
        << "            float residual = sdf - target;\n"
        << "            float abs_r = abs(residual);\n"
        << "            float delta = " << std::fixed << std::setprecision(6) << spec.huber_delta << "f;\n"
        << "            if (abs_r <= delta) {\n"
        << "                thread_loss = 0.5f * residual * residual;\n"
        << "                d_sdf = residual * inv_N;\n"
        << "            } else {\n"
        << "                thread_loss = delta * (abs_r - 0.5f * delta);\n"
        << "                d_sdf = delta * ((residual > 0.0f) ? 1.0f : -1.0f) * inv_N;\n"
        << "            }\n"
        << "        }\n";
    }

    emitFP16Scale(o, spec, "d_sdf", true);
    o << "\n";
  } else {
    // Multi-output loss with optional bc/piezo decomposition.
    o << "        // Multi-output loss\n"
      << "        float d_out[NUM_OUTPUTS];\n"
      << "        for (int m = 0; m < NUM_OUTPUTS; m++) {\n"
      << "            float target = targets[tid * NUM_OUTPUTS + m];\n"
      << "            float residual = outputs[m] - target;\n";

    if (spec.loss == KernelSpec::L2) {
      o << "            float term = residual * residual;\n"
        << "            d_out[m] = 2.0f * residual * inv_N;\n";
    } else if (spec.loss == KernelSpec::L1) {
      o << "            float term = abs(residual);\n"
        << "            d_out[m] = ((residual > 0.0f) ? 1.0f : "
        << "((residual < 0.0f) ? -1.0f : 0.0f)) * inv_N;\n";
    } else if (spec.loss == KernelSpec::Huber) {
      o << "            float abs_r = abs(residual);\n"
        << "            float delta = " << std::fixed << std::setprecision(6)
        << spec.huber_delta << "f;\n"
        << "            float term = (abs_r <= delta) ? "
        << "0.5f * residual * residual : "
        << "delta * (abs_r - 0.5f * delta);\n"
        << "            d_out[m] = (abs_r <= delta) ? "
        << "residual * inv_N : "
        << "delta * ((residual > 0.0f) ? 1.0f : -1.0f) * inv_N;\n";
    } else { // Cosine
      o << "            float term = 0.0f;\n"
        << "            d_out[m] = 0.0f;\n";
    }

    o << "            thread_loss += term;\n"
      << "#ifdef BC_DIM_COUNT\n"
      << "            if (m < BC_DIM_COUNT) thread_loss_bc += term;\n"
      << "            else thread_loss_piezo += term;\n"
      << "#endif\n"
      << "        }\n";

    if (spec.loss == KernelSpec::Cosine) {
      o << "        float dot = 0.0f;\n"
        << "        float pred_norm_sq = 0.0f;\n"
        << "        float target_norm_sq = 0.0f;\n"
        << "        for (int m = 0; m < NUM_OUTPUTS; m++) {\n"
        << "            float target = targets[tid * NUM_OUTPUTS + m];\n"
        << "            dot += outputs[m] * target;\n"
        << "            pred_norm_sq += outputs[m] * outputs[m];\n"
        << "            target_norm_sq += target * target;\n"
        << "        }\n"
        << "        float pred_norm = sqrt(pred_norm_sq + 1e-8f);\n"
        << "        float target_norm = sqrt(target_norm_sq + 1e-8f);\n"
        << "        float denom = pred_norm * target_norm;\n"
        << "        float cosine = dot / denom;\n"
        << "        float pred_norm_denom = pred_norm_sq + 1e-8f;\n"
        << "        thread_loss = 1.0f - cosine;\n"
        << "        for (int m = 0; m < NUM_OUTPUTS; m++) {\n"
        << "            float target = targets[tid * NUM_OUTPUTS + m];\n"
        << "            d_out[m] = (cosine * outputs[m] / pred_norm_denom - target / denom) * inv_N;\n"
        << "        }\n";
    }

    if (spec.use_fp16) {
      o << "        for (int m = 0; m < NUM_OUTPUTS; m++) d_out[m] *= loss_scale;\n";
    }
    o << "\n";
  }
}

void MLPKernelEmitter::emitHashScatter(std::ostringstream& o, const KernelSpec& spec) {
  bool is4d = (spec.encoding == KernelSpec::FourD || spec.spatial_dims == 4);

  o << "    // === Layer 0 backward: W0+b0 grads + hash grid scatter ===\n"
    << "    for (int l = 0; l < num_active_levels; l++) {\n"
    << "        float resolution = BASE_RESOLUTION * pow(PER_LEVEL_SCALE, float(l));\n";

  if (is4d) {
    o << "        float sx = px * resolution;\n"
      << "        float sy = py * resolution;\n"
      << "        float sz = pz * resolution;\n"
      << "        float sw = pw * resolution;\n\n"
      << "        int4 bc = int4(int(floor(sx)), int(floor(sy)), int(floor(sz)), int(floor(sw)));\n"
      << "        float4 frac = float4(sx - floor(sx), sy - floor(sy), sz - floor(sz), sw - floor(sw));\n\n";
  } else if (spec.encoding == KernelSpec::RMHE) {
    // RMHE: apply per-level rotation before hash scatter
    o << "        constant float* R = rotations + l * " << kRotationMatrixFloats << ";\n"
      << "        float sx = (R[0]*px + R[1]*py + R[2]*pz) * resolution;\n"
      << "        float sy = (R[3]*px + R[4]*py + R[5]*pz) * resolution;\n"
      << "        float sz = (R[6]*px + R[7]*py + R[8]*pz) * resolution;\n"
      << "        int3 bc = int3(int(floor(sx)), int(floor(sy)), int(floor(sz)));\n"
      << "        float3 frac = float3(sx - floor(sx), sy - floor(sy), sz - floor(sz));\n\n";
  } else {
    o << "        float sx = px * resolution;\n"
      << "        float sy = py * resolution;\n"
      << "        float sz = pz * resolution;\n\n"
      << "        int3 bc = int3(int(floor(sx)), int(floor(sy)), int(floor(sz)));\n"
      << "        float3 frac = float3(sx - floor(sx), sy - floor(sy), sz - floor(sz));\n\n";
  }

  // Vectorized path: when features_per_level == 2 and 3D non-RMHE
  bool vec2_scatter = (spec.features_per_level == 2 && !is4d && spec.encoding != KernelSpec::RMHE);

  if (vec2_scatter) {
    o << "        {\n"
      << "            const uint level_base = uint(l) * table_size * 2u;\n"
      << "            const float wx0 = 1.0f - frac.x;\n"
      << "            const float wx1 = frac.x;\n"
      << "            const float wy0 = 1.0f - frac.y;\n"
      << "            const float wy1 = frac.y;\n"
      << "            const float wz0 = 1.0f - frac.z;\n"
      << "            const float wz1 = frac.z;\n"
      << "            uint corner_grid_off[8];\n"
      << "            float corner_weight[8];\n"
      << "            uint corner_idx = 0u;\n"
      << "            // Recompute features (vectorized float2)\n"
      << "            float2 feat2 = float2(0.0f);\n"
      << "            for (int dz = 0; dz < 2; dz++)\n"
      << "                for (int dy = 0; dy < 2; dy++)\n"
      << "                    for (int dx = 0; dx < 2; dx++) {\n"
      << "                        uint h = hash_coords(bc + int3(dx, dy, dz), table_size);\n"
      << "                        uint grid_off = level_base + h * 2u;\n"
      << "                        float wx = (dx == 0) ? wx0 : wx1;\n"
      << "                        float wy = (dy == 0) ? wy0 : wy1;\n"
      << "                        float wz = (dz == 0) ? wz0 : wz1;\n"
      << "                        float w = wx * wy * wz;\n"
      << "                        corner_grid_off[corner_idx] = grid_off;\n"
      << "                        corner_weight[corner_idx] = w;\n"
      << "                        feat2 += w * float2(*((device const HASH2_T*)(hash_grid + grid_off)));\n"
      << "                        corner_idx++;\n"
      << "                    }\n\n"
      << "            // W0 grads (2 features)\n"
      << "            int fb = l * 2;\n"
      << "            int w_row0 = w0_off + fb * HIDDEN_DIM;\n"
      << "            int w_row1 = w0_off + (fb + 1) * HIDDEN_DIM;\n"
      << "            for (int j = 0; j < HIDDEN_DIM; j++) {\n"
      << "                MLP_GRAD_ADD(w_row0 + j, h0[j] * feat2.x);\n"
      << "                MLP_GRAD_ADD(w_row1 + j, h0[j] * feat2.y);\n"
      << "            }\n\n"
      << "            // d_feat (vectorized)\n"
      << "            float2 d_feat2 = float2(0.0f);\n"
      << "            for (int j = 0; j < HIDDEN_DIM; j++) {\n"
      << "                d_feat2.x += h0[j] * W_SRC(w_row0 + j);\n"
      << "                d_feat2.y += h0[j] * W_SRC(w_row1 + j);\n"
      << "            }\n";

    if (spec.use_fp16) {
      o << "            d_feat2 *= inv_loss_scale;\n\n";
      } else {
      o << "\n";
    }

    o << "            // Hash scatter (vectorized d_feat2)\n"
      << "            corner_idx = 0u;\n"
      << "            for (int dz = 0; dz < 2; dz++)\n"
      << "                for (int dy = 0; dy < 2; dy++)\n"
      << "                    for (int dx = 0; dx < 2; dx++) {\n"
      << "                        uint grid_off = corner_grid_off[corner_idx];\n"
      << "                        float w = corner_weight[corner_idx];\n"
      << "                        corner_idx++;\n";

    if (spec.use_int_atomics) {
      o << "                        atomic_fetch_add_explicit(&grad_hash[grid_off],\n"
        << "                            int(d_feat2.x * w * GRAD_SCALE), memory_order_relaxed);\n"
        << "                        atomic_fetch_add_explicit(&grad_hash[grid_off + 1],\n"
        << "                            int(d_feat2.y * w * GRAD_SCALE), memory_order_relaxed);\n";
    } else {
      o << "                        atomic_fetch_add_explicit(&grad_hash[grid_off],\n"
        << "                            d_feat2.x * w, memory_order_relaxed);\n"
        << "                        atomic_fetch_add_explicit(&grad_hash[grid_off + 1],\n"
        << "                            d_feat2.y * w, memory_order_relaxed);\n";
    }
    if (spec.emit_active_hash_mask) {
      emit_active_hash_tracking(o, "grid_off", true, "                        ");
    }

    o << "                    }\n"
      << "        }\n";

  } else {
    // Original scalar path
    o << "        for (int f = 0; f < FEATURES_PER_LEVEL; f++) {\n"
      << "            int feat_idx = l * FEATURES_PER_LEVEL + f;\n\n"
      << "            // Recompute feature\n"
      << "            float feat = 0.0f;\n";

    if (is4d) {
      o << "            for (int dw = 0; dw < 2; dw++)\n"
        << "                for (int dz = 0; dz < 2; dz++)\n"
        << "                    for (int dy = 0; dy < 2; dy++)\n"
        << "                        for (int dx = 0; dx < 2; dx++) {\n"
        << "                            uint h = hash_coords_4d(bc + int4(dx, dy, dz, dw), table_size);\n"
        << "                            uint grid_off = uint(l) * table_size * FEATURES_PER_LEVEL + uint(h) * FEATURES_PER_LEVEL + f;\n"
        << "                            float wx = (dx == 0) ? (1.0f - frac.x) : frac.x;\n"
        << "                            float wy = (dy == 0) ? (1.0f - frac.y) : frac.y;\n"
        << "                            float wz = (dz == 0) ? (1.0f - frac.z) : frac.z;\n"
        << "                            float wt = (dw == 0) ? (1.0f - frac.w) : frac.w;\n"
        << "                            feat += wx * wy * wz * wt * hash_grid[grid_off];\n"
        << "                        }\n\n";
    } else {
      o << "            for (int dz = 0; dz < 2; dz++)\n"
        << "                for (int dy = 0; dy < 2; dy++)\n"
        << "                    for (int dx = 0; dx < 2; dx++) {\n"
        << "                        uint h = hash_coords(bc + int3(dx, dy, dz), table_size);\n"
        << "                        uint grid_off = uint(l) * table_size * FEATURES_PER_LEVEL + uint(h) * FEATURES_PER_LEVEL + f;\n"
        << "                        float wx = (dx == 0) ? (1.0f - frac.x) : frac.x;\n"
        << "                        float wy = (dy == 0) ? (1.0f - frac.y) : frac.y;\n"
        << "                        float wz = (dz == 0) ? (1.0f - frac.z) : frac.z;\n"
        << "                        feat += wx * wy * wz * hash_grid[grid_off];\n"
        << "                    }\n\n";
    }

    o << "            // W0 grad\n"
      << "            int w_row = w0_off + feat_idx * HIDDEN_DIM;\n"
      << "            for (int j = 0; j < HIDDEN_DIM; j++)\n"
      << "                MLP_GRAD_ADD(w_row + j, h0[j] * feat);\n\n"
      << "            // d_feat\n"
      << "            float d_feat = 0.0f;\n"
      << "            for (int j = 0; j < HIDDEN_DIM; j++)\n"
      << "                d_feat += h0[j] * W_SRC(w_row + j);\n\n";

    emitFP16Scale(o, spec, "d_feat", false, 12);
    if (spec.use_fp16) o << "\n";

    // Hash grid scatter (device atomics)
    o << "            // Hash grid scatter (device atomics)\n";
    if (is4d) {
      o << "            for (int dw = 0; dw < 2; dw++)\n"
        << "                for (int dz = 0; dz < 2; dz++)\n"
        << "                    for (int dy = 0; dy < 2; dy++)\n"
        << "                        for (int dx = 0; dx < 2; dx++) {\n"
        << "                            uint h = hash_coords_4d(bc + int4(dx, dy, dz, dw), table_size);\n"
        << "                            uint grid_off = uint(l) * table_size * FEATURES_PER_LEVEL + uint(h) * FEATURES_PER_LEVEL + f;\n"
        << "                            float wx = (dx == 0) ? (1.0f - frac.x) : frac.x;\n"
        << "                            float wy = (dy == 0) ? (1.0f - frac.y) : frac.y;\n"
        << "                            float wz = (dz == 0) ? (1.0f - frac.z) : frac.z;\n"
        << "                            float wt = (dw == 0) ? (1.0f - frac.w) : frac.w;\n";
    } else {
      o << "            for (int dz = 0; dz < 2; dz++)\n"
        << "                for (int dy = 0; dy < 2; dy++)\n"
        << "                    for (int dx = 0; dx < 2; dx++) {\n"
        << "                        uint h = hash_coords(bc + int3(dx, dy, dz), table_size);\n"
        << "                        uint grid_off = uint(l) * table_size * FEATURES_PER_LEVEL + uint(h) * FEATURES_PER_LEVEL + f;\n"
        << "                        float wx = (dx == 0) ? (1.0f - frac.x) : frac.x;\n"
        << "                        float wy = (dy == 0) ? (1.0f - frac.y) : frac.y;\n"
        << "                        float wz = (dz == 0) ? (1.0f - frac.z) : frac.z;\n";
    }

    if (spec.use_int_atomics) {
      if (is4d) {
        o << "                            atomic_fetch_add_explicit(&grad_hash[grid_off],\n"
          << "                                int(d_feat * wx * wy * wz * wt * GRAD_SCALE), memory_order_relaxed);\n"
          << "                        }\n";
      } else {
        o << "                        atomic_fetch_add_explicit(&grad_hash[grid_off],\n"
          << "                            int(d_feat * wx * wy * wz * GRAD_SCALE), memory_order_relaxed);\n"
          << "                    }\n";
      }
    } else {
      if (is4d) {
        o << "                            atomic_fetch_add_explicit(&grad_hash[grid_off],\n"
          << "                                d_feat * wx * wy * wz * wt, memory_order_relaxed);\n"
          << "                        }\n";
     } else {
       o << "                        atomic_fetch_add_explicit(&grad_hash[grid_off],\n"
         << "                            d_feat * wx * wy * wz, memory_order_relaxed);\n"
         << "                    }\n";
    }
    if (spec.emit_active_hash_mask) {
      if (is4d) {
        if (spec.features_per_level == 2) {
          emit_active_hash_tracking(o, "grid_off", true, "                            ");
          o << "                        }\n";
        } else {
          emit_active_hash_tracking(o, "grid_off", false,
                                    "                            ");
          o << "                        }\n";
        }
      } else {
        if (spec.features_per_level == 2) {
          emit_active_hash_tracking(o, "grid_off", true, "                        ");
          o << "                    }\n";
        } else {
          emit_active_hash_tracking(o, "grid_off", false, "                        ");
          o << "                    }\n";
        }
      }
    }
  }
  } // close else (scalar path)

  if (!vec2_scatter) {
    // Close the for-f loop (scalar path)
    o << "        }\n";
  }
  o << "    }\n\n"
    << "    // b0 grads\n"
    << "    for (int j = 0; j < HIDDEN_DIM; j++)\n"
    << "        MLP_GRAD_ADD(b0_off + j, h0[j]);\n\n";
}

void MLPKernelEmitter::emitProbeReduction(std::ostringstream& o, const KernelSpec& spec) {
  // Sequential TG reductions for each probe field → probe_partials[tgid*PROBE_STRIDE+field].
  // Reuses LOSS_BUF (same threadgroup memory as loss reduction).
  auto reduce_max = [&](const char* thread_var, int field_offset) {
    o << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      << "    LOSS_BUF[lid] = " << thread_var << ";\n"
      << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      << "    for (uint s = TG_SIZE / 2; s > 0; s >>= 1) {\n"
      << "        if (lid < s) LOSS_BUF[lid] = max(LOSS_BUF[lid], LOSS_BUF[lid + s]);\n"
      << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      << "    }\n"
      << "    if (lid == 0) probe_partials[tgid * PROBE_STRIDE + " << field_offset << "] = LOSS_BUF[0];\n\n";
  };

  auto reduce_sum = [&](const char* thread_var, int field_offset) {
    o << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      << "    LOSS_BUF[lid] = " << thread_var << ";\n"
      << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      << "    for (uint s = TG_SIZE / 2; s > 0; s >>= 1) {\n"
      << "        if (lid < s) LOSS_BUF[lid] += LOSS_BUF[lid + s];\n"
      << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      << "    }\n"
      << "    if (lid == 0) probe_partials[tgid * PROBE_STRIDE + " << field_offset << "] = LOSS_BUF[0];\n\n";
  };

  o << "\n    // === Probe TG reduction ===\n";

  const int L = spec.num_hidden_layers;
  // Layout: [fwd_nan, bwd_nan, hash_grad_l2, act_max[0..L-1], output_abs_max, output_neg_min, grad_l2[0..L-1]]
  reduce_max("probe_fwd_nan", 0);
  reduce_max("probe_bwd_nan", 1);
  reduce_sum("probe_hash_grad_l2", 2);
  for (int l = 0; l < L; ++l) {
    std::string var = "probe_act_max_" + std::to_string(l);
    reduce_max(var.c_str(), 3 + l);
  }
  reduce_max("probe_output_abs_max", 3 + L);
  reduce_max("probe_output_neg_min", 3 + L + 1);
  for (int l = 0; l < L; ++l) {
    std::string var = "probe_grad_l2_" + std::to_string(l);
    reduce_sum(var.c_str(), 3 + L + 2 + l);
  }
}

void MLPKernelEmitter::emitTGFlush(std::ostringstream& o, const KernelSpec& spec) {
  o << "    // === Flush TG MLP grads → device via atomic adds ===\n"
    << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    << "    for (int i = int(lid); i < MLP_WEIGHT_COUNT; i += TG_SIZE) {\n"
    << "        float val = tg_grad_mlp[i];\n"
    << "        if (val != 0.0f)\n";
  if (spec.use_fp16) {
    o << "            atomic_fetch_add_explicit(&grad_mlp[i], val * inv_loss_scale, memory_order_relaxed);\n";
  } else {
    o << "            atomic_fetch_add_explicit(&grad_mlp[i], val, memory_order_relaxed);\n";
  }
  o << "    }\n\n";
}

void MLPKernelEmitter::emitLossReduction(std::ostringstream& o, const KernelSpec& spec) {
  o << "    // === Loss reduction — threadgroup tree reduction ===\n"
    << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    << "    LOSS_BUF[lid] = thread_loss;\n"
    << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n"
    << "    for (uint s = TG_SIZE / 2; s > 0; s >>= 1) {\n"
    << "        if (lid < s)\n"
    << "            LOSS_BUF[lid] += LOSS_BUF[lid + s];\n"
    << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    << "    }\n\n"
    << "    if (lid == 0) {\n"
    << "#ifdef REDUCTION_TERMS\n"
    << "        loss_partials[tgid * REDUCTION_TERMS] = LOSS_BUF[0];\n"
    << "#else\n"
    << "        loss_partials[tgid] = LOSS_BUF[0];\n"
    << "#endif\n"
    << "    }\n\n"
    << "#ifdef BC_DIM_COUNT\n"
    << "    // BC loss reduction (term 1)\n"
    << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    << "    LOSS_BUF[lid] = thread_loss_bc;\n"
    << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    << "    for (uint s = TG_SIZE / 2; s > 0; s >>= 1) {\n"
    << "        if (lid < s) LOSS_BUF[lid] += LOSS_BUF[lid + s];\n"
    << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    << "    }\n"
    << "    if (lid == 0) loss_partials[tgid * REDUCTION_TERMS + 1] = LOSS_BUF[0];\n\n"
    << "    // Piezo loss reduction (term 2)\n"
    << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    << "    LOSS_BUF[lid] = thread_loss_piezo;\n"
    << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    << "    for (uint s = TG_SIZE / 2; s > 0; s >>= 1) {\n"
    << "        if (lid < s) LOSS_BUF[lid] += LOSS_BUF[lid + s];\n"
    << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    << "    }\n"
    << "    if (lid == 0) loss_partials[tgid * REDUCTION_TERMS + 2] = LOSS_BUF[0];\n"
    << "#endif\n";
}

// ===========================================================================
// SIMD cooperative hash encoding (32 threads, 8 points, 4 threads/point)
// ===========================================================================

void MLPKernelEmitter::emitHashEncodeCooperative(std::ostringstream& o, const KernelSpec& spec) {
  int fpl = spec.features_per_level;
  int nl = spec.num_levels;
  int levels_per_thread = (nl + 3) / 4; // ceil(NUM_LEVELS / 4)

  o << "// Cooperative hash encode: 32 threads → 8 points, 4 threads per point\n"
    << "inline void encode_hash_grid_coop(\n"
    << "    uint lid, uint base_point,\n"
    << "    device const float* positions,\n"
    << "    device const HASH_T* hash_grid,\n"
    << "    int num_points,\n"
    << "    threadgroup SIMD_ACT act[8][HIDDEN_DIM])\n"
    << "{\n"
    << "    uint local_point = lid / 4;\n"
    << "    uint level_group = lid % 4;\n"
    << "    uint table_size = 1u << uint(LOG2_HASHMAP_SIZE);\n\n"
    << "    // Zero this thread's feature slots\n"
    << "    int feat_start = int(level_group) * " << levels_per_thread * fpl << ";\n"
    << "    int feat_end = min(feat_start + " << levels_per_thread * fpl << ", INPUT_DIM);\n"
    << "    for (int j = feat_start; j < feat_end; j++)\n"
    << "        act[local_point][j] = 0.0;\n\n"
    << "    int start_level = int(level_group) * " << levels_per_thread << ";\n"
    << "    float resolution = BASE_RESOLUTION;\n"
    << "    for (int s = 0; s < start_level; s++)\n"
    << "        resolution *= PER_LEVEL_SCALE;\n\n"
    << "    uint point_idx = base_point + local_point;\n"
    << "    if (int(point_idx) < num_points) {\n"
    << "        float3 pos = float3(positions[point_idx*3],\n"
    << "                            positions[point_idx*3+1],\n"
    << "                            positions[point_idx*3+2]);\n\n"
    << "        for (int l_off = 0; l_off < " << levels_per_thread << "; l_off++) {\n"
    << "            int l = start_level + l_off;\n"
    << "            if (l >= NUM_LEVELS) break;\n"
    << "            float3 scaled = pos * resolution;\n"
    << "            int3 bc = int3(int(floor(scaled.x)), int(floor(scaled.y)), int(floor(scaled.z)));\n"
    << "            float3 frac = float3(scaled.x - floor(scaled.x),\n"
    << "                                 scaled.y - floor(scaled.y),\n"
    << "                                 scaled.z - floor(scaled.z));\n"
    << "            ";
  if (fpl == 2) {
    // Vectorized float2 path for features_per_level == 2
    o << "{\n"
      << "                float2 feat2 = float2(0.0f);\n"
      << "                for (int dz = 0; dz < 2; dz++)\n"
      << "                    for (int dy = 0; dy < 2; dy++)\n"
      << "                        for (int dx = 0; dx < 2; dx++) {\n"
      << "                            uint h = hash_coords(bc + int3(dx, dy, dz), table_size);\n"
      << "                            uint grid_off = uint(l) * table_size * 2u + h * 2u;\n"
      << "                            float wx = (dx == 0) ? (1.0f - frac.x) : frac.x;\n"
      << "                            float wy = (dy == 0) ? (1.0f - frac.y) : frac.y;\n"
      << "                            float wz = (dz == 0) ? (1.0f - frac.z) : frac.z;\n"
      << "                            float w = wx * wy * wz;\n"
      << "                            feat2 += w * float2(*((device const HASH2_T*)(hash_grid + grid_off)));\n"
      << "                        }\n"
      << "                act[local_point][l * 2] = feat2.x;\n"
      << "                act[local_point][l * 2 + 1] = feat2.y;\n"
      << "            }\n";
  } else {
    o << "for (int f = 0; f < FEATURES_PER_LEVEL; f++) {\n"
      << "                float feat = 0.0f;\n"
      << "                for (int dz = 0; dz < 2; dz++)\n"
      << "                    for (int dy = 0; dy < 2; dy++)\n"
      << "                        for (int dx = 0; dx < 2; dx++) {\n"
      << "                            uint h = hash_coords(bc + int3(dx, dy, dz), table_size);\n"
      << "                            uint grid_off = uint(l) * table_size * FEATURES_PER_LEVEL + uint(h) * FEATURES_PER_LEVEL + f;\n"
      << "                            float wx = (dx == 0) ? (1.0f - frac.x) : frac.x;\n"
      << "                            float wy = (dy == 0) ? (1.0f - frac.y) : frac.y;\n"
      << "                            float wz = (dz == 0) ? (1.0f - frac.z) : frac.z;\n"
      << "                            feat += wx * wy * wz * hash_grid[grid_off];\n"
      << "                        }\n"
      << "                act[local_point][l * FEATURES_PER_LEVEL + f] = feat;\n"
      << "            }\n";
  }
  o
    << "            resolution *= PER_LEVEL_SCALE;\n"
    << "        }\n"
    << "    }\n"
    << "}\n\n";
}

void MLPKernelEmitter::emitHashEncodeCoopScalarW0(std::ostringstream& o,
                                                  const KernelSpec& spec) {
  const int w0_bias_off = spec.input_dim * spec.hidden_dim;
  const bool vec2_path = (spec.features_per_level == 2);

  o << "// Scalar hash encode + fused W0 for 8 points, one thread per point\n"
    << "inline void encode_hash_grid_scalar_w0(\n"
    << "    uint lid, uint base_point,\n"
    << "    device const float* positions,\n"
    << "    device const HASH_T* hash_grid,\n"
    << "    device const SIMD_ACT* mlp,\n"
    << "    int num_points,\n"
    << "    threadgroup SIMD_ACT act_b[8][HIDDEN_DIM])\n"
    << "{\n"
    << "    if (lid >= 8) return;\n"
    << "    uint point_idx = base_point + lid;\n"
    << "    for (int j = 0; j < HIDDEN_DIM; ++j)\n"
    << "        act_b[lid][j] = SIMD_ACT(0);\n"
    << "    if (int(point_idx) >= num_points) return;\n\n"
    << "    uint table_size = 1u << uint(LOG2_HASHMAP_SIZE);\n"
    << "    float3 pos = float3(positions[point_idx * 3],\n"
    << "                        positions[point_idx * 3 + 1],\n"
    << "                        positions[point_idx * 3 + 2]);\n"
    << "    float resolution = BASE_RESOLUTION;\n"
    << "    for (int l = 0; l < NUM_LEVELS; ++l) {\n"
    << "        float3 scaled = pos * resolution;\n"
    << "        int3 bc = int3(int(floor(scaled.x)), int(floor(scaled.y)), int(floor(scaled.z)));\n"
    << "        float3 frac = float3(scaled.x - floor(scaled.x),\n"
    << "                             scaled.y - floor(scaled.y),\n"
    << "                             scaled.z - floor(scaled.z));\n";

  if (vec2_path) {
    o << "        float2 feat2 = float2(0.0f);\n"
      << "        for (int dz = 0; dz < 2; ++dz)\n"
      << "            for (int dy = 0; dy < 2; ++dy)\n"
      << "                for (int dx = 0; dx < 2; ++dx) {\n"
      << "                    uint h = hash_coords(bc + int3(dx, dy, dz), table_size);\n"
      << "                    uint grid_off = uint(l) * table_size * 2u + h * 2u;\n"
      << "                    float wx = (dx == 0) ? (1.0f - frac.x) : frac.x;\n"
      << "                    float wy = (dy == 0) ? (1.0f - frac.y) : frac.y;\n"
      << "                    float wz = (dz == 0) ? (1.0f - frac.z) : frac.z;\n"
      << "                    float w = wx * wy * wz;\n"
      << "                    feat2 += w * float2(*((device const HASH2_T*)(hash_grid + grid_off)));\n"
      << "                }\n"
      << "        int feat_base = l * 2;\n"
      << "        for (int j = 0; j < HIDDEN_DIM; ++j) {\n"
      << "            float acc = float(act_b[lid][j]);\n"
      << "            acc += feat2.x * float(mlp[feat_base * HIDDEN_DIM + j]);\n"
      << "            acc += feat2.y * float(mlp[(feat_base + 1) * HIDDEN_DIM + j]);\n"
      << "            act_b[lid][j] = SIMD_ACT(acc);\n"
      << "        }\n";
  } else {
    o << "        for (int f = 0; f < FEATURES_PER_LEVEL; ++f) {\n"
      << "            float feat = 0.0f;\n"
      << "            for (int dz = 0; dz < 2; ++dz)\n"
      << "                for (int dy = 0; dy < 2; ++dy)\n"
      << "                    for (int dx = 0; dx < 2; ++dx) {\n"
      << "                        uint h = hash_coords(bc + int3(dx, dy, dz), table_size);\n"
      << "                        uint grid_off = uint(l) * table_size * FEATURES_PER_LEVEL + uint(h) * FEATURES_PER_LEVEL + f;\n"
      << "                        float wx = (dx == 0) ? (1.0f - frac.x) : frac.x;\n"
      << "                        float wy = (dy == 0) ? (1.0f - frac.y) : frac.y;\n"
      << "                        float wz = (dz == 0) ? (1.0f - frac.z) : frac.z;\n"
      << "                        feat += wx * wy * wz * hash_grid[grid_off];\n"
      << "                    }\n"
      << "            int feat_idx = l * FEATURES_PER_LEVEL + f;\n"
      << "            int w_row = feat_idx * HIDDEN_DIM;\n"
      << "            for (int j = 0; j < HIDDEN_DIM; ++j)\n"
      << "                act_b[lid][j] = SIMD_ACT(float(act_b[lid][j]) + feat * float(mlp[w_row + j]));\n"
      << "        }\n";
  }

  o << "        resolution *= PER_LEVEL_SCALE;\n"
    << "    }\n"
    << "    for (int j = 0; j < HIDDEN_DIM; ++j)\n"
    << "        act_b[lid][j] = max(SIMD_ACT(0), act_b[lid][j] + mlp[" << w0_bias_off
    << " + j]);\n"
    << "}\n\n";
}

void MLPKernelEmitter::emitHashEncodeCoopScalarW0Training(
    std::ostringstream& o, const KernelSpec& spec) {
  const int w0_bias_off = spec.input_dim * spec.hidden_dim;
  const bool vec2_path = (spec.features_per_level == 2);

  o << "// Scalar hash encode + fused W0 for training hot path.\n"
    << "// Also writes raw encoded features into features_tg for W0 grads and\n"
    << "// any later checkpoint-style recompute.\n"
    << "inline void encode_hash_grid_scalar_w0_train(\n"
    << "    uint lid, uint base_point,\n"
    << "    device const float* positions,\n"
    << "    device const HASH_T* hash_grid,\n"
    << "    device const float* mlp,\n"
    << "    int num_points,\n"
    << "    threadgroup float act_b[8][HIDDEN_DIM],\n"
    << "    threadgroup float features_tg[8 * INPUT_DIM])\n"
    << "{\n"
    << "    if (lid >= 8) return;\n"
    << "    uint point_idx = base_point + lid;\n"
    << "    for (int j = 0; j < HIDDEN_DIM; ++j)\n"
    << "        act_b[lid][j] = 0.0f;\n"
    << "    for (int j = 0; j < INPUT_DIM; ++j)\n"
    << "        features_tg[lid * INPUT_DIM + j] = 0.0f;\n"
    << "    if (int(point_idx) >= num_points) return;\n\n"
    << "    uint table_size = 1u << uint(LOG2_HASHMAP_SIZE);\n"
    << "    float3 pos = float3(positions[point_idx * 3],\n"
    << "                        positions[point_idx * 3 + 1],\n"
    << "                        positions[point_idx * 3 + 2]);\n"
    << "    float resolution = BASE_RESOLUTION;\n"
    << "    for (int l = 0; l < NUM_LEVELS; ++l) {\n"
    << "        float3 scaled = pos * resolution;\n"
    << "        float3 scaled_floor = floor(scaled);\n"
    << "        int3 bc = int3(scaled_floor);\n"
    << "        float3 frac = scaled - scaled_floor;\n";

  if (vec2_path) {
    o << "        float2 feat2 = float2(0.0f);\n"
      << "        for (int dz = 0; dz < 2; ++dz)\n"
      << "            for (int dy = 0; dy < 2; ++dy)\n"
      << "                for (int dx = 0; dx < 2; ++dx) {\n"
      << "                    uint h = hash_coords(bc + int3(dx, dy, dz), table_size);\n"
      << "                    uint grid_off = uint(l) * table_size * 2u + h * 2u;\n"
      << "                    float wx = (dx == 0) ? (1.0f - frac.x) : frac.x;\n"
      << "                    float wy = (dy == 0) ? (1.0f - frac.y) : frac.y;\n"
      << "                    float wz = (dz == 0) ? (1.0f - frac.z) : frac.z;\n"
      << "                    float w = wx * wy * wz;\n"
      << "                    feat2 += w * float2(*((device const HASH2_T*)(hash_grid + grid_off)));\n"
      << "                }\n"
      << "        int feat_base = l * 2;\n"
      << "        features_tg[lid * INPUT_DIM + feat_base] = feat2.x;\n"
      << "        features_tg[lid * INPUT_DIM + feat_base + 1] = feat2.y;\n"
      << "        for (int j = 0; j < HIDDEN_DIM; ++j) {\n"
      << "            act_b[lid][j] += feat2.x * mlp[feat_base * HIDDEN_DIM + j];\n"
      << "            act_b[lid][j] += feat2.y * mlp[(feat_base + 1) * HIDDEN_DIM + j];\n"
      << "        }\n";
  } else {
    o << "        for (int f = 0; f < FEATURES_PER_LEVEL; ++f) {\n"
      << "            float feat = 0.0f;\n"
      << "            for (int dz = 0; dz < 2; ++dz)\n"
      << "                for (int dy = 0; dy < 2; ++dy)\n"
      << "                    for (int dx = 0; dx < 2; ++dx) {\n"
      << "                        uint h = hash_coords(bc + int3(dx, dy, dz), table_size);\n"
      << "                        uint grid_off = uint(l) * table_size * FEATURES_PER_LEVEL + uint(h) * FEATURES_PER_LEVEL + f;\n"
      << "                        float wx = (dx == 0) ? (1.0f - frac.x) : frac.x;\n"
      << "                        float wy = (dy == 0) ? (1.0f - frac.y) : frac.y;\n"
      << "                        float wz = (dz == 0) ? (1.0f - frac.z) : frac.z;\n"
      << "                        feat += wx * wy * wz * hash_grid[grid_off];\n"
      << "                    }\n"
      << "            int feat_idx = l * FEATURES_PER_LEVEL + f;\n"
      << "            features_tg[lid * INPUT_DIM + feat_idx] = feat;\n"
      << "            int w_row = feat_idx * HIDDEN_DIM;\n"
      << "            for (int j = 0; j < HIDDEN_DIM; ++j)\n"
      << "                act_b[lid][j] += feat * mlp[w_row + j];\n"
      << "        }\n";
  }

  o << "        resolution *= PER_LEVEL_SCALE;\n"
    << "    }\n"
    << "    for (int j = 0; j < HIDDEN_DIM; ++j)\n"
    << "        act_b[lid][j] = max(0.0f, act_b[lid][j] + mlp[" << w0_bias_off
    << " + j]);\n"
    << "}\n\n";
}

// ===========================================================================
// SIMD MLP forward pass (simdgroup_float8x8)
// ===========================================================================

void MLPKernelEmitter::emitMLPForwardSIMD(std::ostringstream& o, const KernelSpec& spec) {
  int hd = spec.hidden_dim;
  int nhl = spec.num_hidden_layers;
  int id = spec.input_dim;
  bool tg_cache = spec.use_tg_weight_cache && spec.canUseTGCache();

  // Determine weight source: device or threadgroup
  const char* w_src = tg_cache ? "mlp_tg" : "mlp";
  const char* w_addr = tg_cache ? "threadgroup const" : "device const";

  o << "// SIMD MLP forward: " << nhl << " hidden layers, hidden_dim=" << hd << "\n"
    << "inline void mlp_forward_simd(\n"
    << "    uint lid,\n"
    << "    " << w_addr << " SIMD_ACT* " << w_src << ",\n"
    << "    threadgroup SIMD_ACT act_a[8][HIDDEN_DIM],\n"
    << "    threadgroup SIMD_ACT act_b[8][HIDDEN_DIM])\n"
    << "{\n";

  // Compute weight offsets
  int offset = 0;

  // Layer 0: INPUT_DIM → HIDDEN_DIM
  int w0_off = offset;
  offset += id * hd;
  int b0_off = offset;
  offset += hd;

  int outer0 = hd / 8;
  int inner0 = id / 8;

  o << "    // Layer 0: act_a[8][" << id << "] × W0[" << id << "×" << hd << "] + b0 → ReLU → act_b[8][" << hd << "]\n";
  o << "    for (int oj = 0; oj < " << outer0 << "; oj++) {\n"
    << "        SIMD_MAT acc, a_tile, w_tile;\n"
    << "        simdgroup_load(a_tile, &act_a[0][0], HIDDEN_DIM);\n"
    << "        simdgroup_load(w_tile, " << w_src << " + " << w0_off << " + oj * 8, HIDDEN_DIM);\n"
    << "        simdgroup_multiply(acc, a_tile, w_tile);\n"
    << "        for (int ik = 1; ik < " << inner0 << "; ik++) {\n"
    << "            simdgroup_load(a_tile, &act_a[0][ik * 8], HIDDEN_DIM);\n"
    << "            simdgroup_load(w_tile, " << w_src << " + " << w0_off << " + ik * 8 * HIDDEN_DIM + oj * 8, HIDDEN_DIM);\n"
    << "            simdgroup_multiply_accumulate(acc, a_tile, w_tile, acc);\n"
    << "        }\n"
    << "        simdgroup_store(acc, &act_b[0][oj * 8], HIDDEN_DIM);\n"
    << "    }\n"
    << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n"
    << "    // Bias + ReLU for Layer 0\n"
    << "    for (int j = int(lid); j < " << 8 * hd << "; j += 32) {\n"
    << "        int row = j / HIDDEN_DIM;\n"
    << "        int col = j % HIDDEN_DIM;\n"
    << "        act_b[row][col] = max(SIMD_ACT(0), act_b[row][col] + " << w_src << "[" << b0_off << " + col]);\n"
    << "    }\n"
    << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n";

  // Hidden layers 1..N-1
  int inner_h = hd / 8;
  int outer_h = hd / 8;
  for (int layer = 1; layer < nhl; ++layer) {
    int w_off = offset;
    offset += hd * hd;
    int b_off = offset;
    offset += hd;

    // Alternate: even layers read act_b write act_a, odd read act_a write act_b
    // Layer 0 wrote to act_b. Layer 1 reads act_b writes act_a, etc.
    const char* src_arr = (layer % 2 == 1) ? "act_b" : "act_a";
    const char* dst_arr = (layer % 2 == 1) ? "act_a" : "act_b";

    o << "    // Layer " << layer << ": " << src_arr << " × W" << layer << "[" << hd << "×" << hd << "] + b → ReLU → " << dst_arr << "\n";
    o << "    for (int oj = 0; oj < " << outer_h << "; oj++) {\n"
      << "        SIMD_MAT acc, a_tile, w_tile;\n"
      << "        simdgroup_load(a_tile, &" << src_arr << "[0][0], HIDDEN_DIM);\n"
      << "        simdgroup_load(w_tile, " << w_src << " + " << w_off << " + oj * 8, HIDDEN_DIM);\n"
      << "        simdgroup_multiply(acc, a_tile, w_tile);\n"
      << "        for (int ik = 1; ik < " << inner_h << "; ik++) {\n"
      << "            simdgroup_load(a_tile, &" << src_arr << "[0][ik * 8], HIDDEN_DIM);\n"
      << "            simdgroup_load(w_tile, " << w_src << " + " << w_off << " + ik * 8 * HIDDEN_DIM + oj * 8, HIDDEN_DIM);\n"
      << "            simdgroup_multiply_accumulate(acc, a_tile, w_tile, acc);\n"
      << "        }\n"
      << "        simdgroup_store(acc, &" << dst_arr << "[0][oj * 8], HIDDEN_DIM);\n"
      << "    }\n"
      << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n"
      << "    // Bias + ReLU for Layer " << layer << "\n"
      << "    for (int j = int(lid); j < " << 8 * hd << "; j += 32) {\n"
      << "        int row = j / HIDDEN_DIM;\n"
      << "        int col = j % HIDDEN_DIM;\n"
      << "        " << dst_arr << "[row][col] = max(SIMD_ACT(0), " << dst_arr << "[row][col] + " << w_src << "[" << b_off << " + col]);\n"
      << "    }\n"
      << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n";
  }

  o << "}\n\n";
}

void MLPKernelEmitter::emitMLPForwardSIMDHiddenOnly(std::ostringstream& o,
                                                    const KernelSpec& spec) {
  const int hd = spec.hidden_dim;
  const int nhl = spec.num_hidden_layers;

  if (nhl <= 1) {
    o << "inline void mlp_forward_simd_hidden_only(\n"
      << "    uint /*lid*/,\n"
      << "    device const SIMD_ACT* /*mlp*/,\n"
      << "    threadgroup SIMD_ACT act_a[8][HIDDEN_DIM],\n"
      << "    threadgroup SIMD_ACT act_b[8][HIDDEN_DIM])\n"
      << "{\n"
      << "    (void)act_a;\n"
      << "    (void)act_b;\n"
      << "}\n\n";
    return;
  }

  const int w0_bytes = spec.input_dim * hd + hd;
  std::ostringstream body;
  body << "// SIMD hidden layers only: layer 0 already computed into act_b\n"
       << "inline void mlp_forward_simd_hidden_only(\n"
       << "    uint lid,\n"
       << "    device const SIMD_ACT* mlp,\n"
       << "    threadgroup SIMD_ACT act_a[8][HIDDEN_DIM],\n"
       << "    threadgroup SIMD_ACT act_b[8][HIDDEN_DIM])\n"
       << "{\n";

  int offset = w0_bytes;
  const int inner_h = hd / 8;
  const int outer_h = hd / 8;
  for (int layer = 1; layer < nhl; ++layer) {
    const int w_off = offset;
    offset += hd * hd;
    const int b_off = offset;
    offset += hd;

    const char* src_arr = (layer % 2 == 1) ? "act_b" : "act_a";
    const char* dst_arr = (layer % 2 == 1) ? "act_a" : "act_b";

    body << "    // Layer " << layer << ": " << src_arr << " × W" << layer
         << " + b → ReLU → " << dst_arr << "\n"
         << "    for (int oj = 0; oj < " << outer_h << "; ++oj) {\n"
         << "        SIMD_MAT acc, a_tile, w_tile;\n"
         << "        simdgroup_load(a_tile, &" << src_arr << "[0][0], HIDDEN_DIM);\n"
         << "        simdgroup_load(w_tile, mlp + " << w_off << " + oj * 8, HIDDEN_DIM);\n"
         << "        simdgroup_multiply(acc, a_tile, w_tile);\n"
         << "        for (int ik = 1; ik < " << inner_h << "; ++ik) {\n"
         << "            simdgroup_load(a_tile, &" << src_arr << "[0][ik * 8], HIDDEN_DIM);\n"
         << "            simdgroup_load(w_tile, mlp + " << w_off
         << " + ik * 8 * HIDDEN_DIM + oj * 8, HIDDEN_DIM);\n"
         << "            simdgroup_multiply_accumulate(acc, a_tile, w_tile, acc);\n"
         << "        }\n"
         << "        simdgroup_store(acc, &" << dst_arr << "[0][oj * 8], HIDDEN_DIM);\n"
         << "    }\n"
         << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n"
         << "    for (int j = int(lid); j < " << 8 * hd << "; j += 32) {\n"
         << "        int row = j / HIDDEN_DIM;\n"
         << "        int col = j % HIDDEN_DIM;\n"
         << "        " << dst_arr << "[row][col] = max(SIMD_ACT(0), "
         << dst_arr << "[row][col] + mlp[" << b_off << " + col]);\n"
         << "    }\n"
         << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n";
  }
  body << "}\n\n";
  o << body.str();
}

// ===========================================================================
// SIMD MLP backward: scalar per-thread backward for gradient computation.
// Uses saved activations from SIMD forward to compute d_pre_h0.
// This is NOT used for training (training has its own backward in emitTrainKernel).
// ===========================================================================

void MLPKernelEmitter::emitMLPBackwardSIMD(std::ostringstream& o, const KernelSpec& spec) {
  int hd = spec.hidden_dim;
  int nhl = spec.num_hidden_layers;
  bool tg_cache = spec.use_tg_weight_cache && spec.canUseTGCache();
  const char* w_src = tg_cache ? "mlp_tg" : "mlp";

  // Compute weight offsets inline (no std::vector needed)
  int offset = spec.input_dim * hd + hd; // skip W0 + b0
  int layer_w_offs[16]; // max 16 hidden layers
  for (int i = 1; i < nhl; ++i) {
    layer_w_offs[i] = offset;
    offset += hd * hd + hd;
  }
  int wO_off = offset;

  // Output layer backward: d_sdf=1.0 → grad_pre_last
  std::string last_h = "h" + std::to_string(nhl - 1);
  o << "        // Output layer backward\n"
    << "        float grad_pre_last[HIDDEN_DIM];\n"
    << "        for (int j = 0; j < HIDDEN_DIM; j++)\n"
    << "            grad_pre_last[j] = " << w_src << "[" << wO_off << " + j] * (("
    << last_h << "[j] > 0.0f) ? 1.0f : 0.0f);\n\n";

  // Propagate through hidden layers in reverse
  std::string cur_grad = "grad_pre_last";
  for (int layer = nhl - 1; layer >= 1; --layer) {
    std::string h_prev = "h" + std::to_string(layer - 1);
    std::string next_grad = "grad_pre_" + std::to_string(layer - 1);
    int w_off = layer_w_offs[layer];

    o << "        float " << next_grad << "[HIDDEN_DIM];\n"
      << "        for (int i = 0; i < HIDDEN_DIM; i++) {\n"
      << "            float acc = 0.0f;\n"
      << "            for (int j = 0; j < HIDDEN_DIM; j++)\n"
      << "                acc += " << cur_grad << "[j] * " << w_src << "[" << w_off << " + i * HIDDEN_DIM + j];\n"
      << "            " << next_grad << "[i] = acc * ((" << h_prev << "[i] > 0.0f) ? 1.0f : 0.0f);\n"
      << "        }\n\n";
    cur_grad = next_grad;
  }
}

// ===========================================================================
// Eval kernel
// ===========================================================================

std::string MLPKernelEmitter::emitEvalKernel(const KernelSpec& spec) {
  spec.validate();
  std::ostringstream o;
  bool is4d = (spec.encoding == KernelSpec::FourD || spec.spatial_dims == 4);

  // SIMD path: 32 threads/TG, 8 points per TG, simdgroup matmul
  if (spec.use_simd && !is4d && spec.num_outputs == 1
      && spec.encoding == KernelSpec::Standard) {
    bool tg_cache = spec.use_tg_weight_cache && spec.canUseTGCache();
    const bool hybrid_scalar_w0 = !spec.use_fp16_simd && !tg_cache;

    emitPreamble(o, spec);
    o << "#include <metal_simdgroup_matrix>\n\n";
    emitHashFunctions(o, spec);
    if (hybrid_scalar_w0) {
      emitHashEncodeCoopScalarW0(o, spec);
      emitMLPForwardSIMDHiddenOnly(o, spec);
    } else {
      emitHashEncodeCooperative(o, spec);
      emitMLPForwardSIMD(o, spec);
    }

    // Output layer weight offset
    int wO_off = spec.input_dim * spec.hidden_dim + spec.hidden_dim;
    for (int i = 1; i < spec.num_hidden_layers; ++i)
      wO_off += spec.hidden_dim * spec.hidden_dim + spec.hidden_dim;
    int bO_off = wO_off + spec.hidden_dim * spec.num_outputs;

    // Determine which act buffer has the last hidden layer's output
    // Layer 0 → act_b, Layer 1 → act_a, Layer 2 → act_b, ...
    // For nhl hidden layers, last forward layer output is in:
    // nhl % 2 == 1 → act_b (if nhl=1: act_b. if nhl=2: layer1→act_a)
    // Actually: layer 0 → act_b. layer 1 → act_a. So after nhl layers:
    // nhl=1: act_b. nhl=2: act_a. nhl=3: act_b.
    // Pattern: nhl % 2 == 1 → act_b, nhl % 2 == 0 → act_a
    const char* last_act = (spec.num_hidden_layers % 2 == 1) ? "act_b" : "act_a";

    // Kernel entry
    o << "kernel void neural_sdf_eval_simd(\n"
      << "    device const float*  positions     [[buffer(0)]],\n"
      << "    device float*        output        [[buffer(1)]],\n"
      << "    device const float*  config_weights [[buffer(2)]],\n"
      << "    device const HASH_T*  hash_grid     [[buffer(3)]],\n";
    if (spec.use_fp16_simd) {
      o << "    device const SIMD_ACT* mlp_half      [[buffer(4)]],\n";
    } else {
      o << "    device const SIMD_ACT* mlp_direct    [[buffer(5)]],\n";
    }
    o
      << "    uint tgid [[threadgroup_position_in_grid]],\n"
      << "    uint lid  [[thread_position_in_threadgroup]])\n"
      << "{\n"
      << "    int num_pts = int(config_weights[7]);\n";
    if (spec.use_fp16_simd) {
      o << "    device const SIMD_ACT* mlp = mlp_half;\n";
    } else {
      o << "    device const SIMD_ACT* mlp = mlp_direct;\n";
    }
    o
      << "    uint base_point = tgid * 8;\n\n";

    if (tg_cache) {
      o << "    // TG MLP weight cache (cooperative load)\n"
        << "    threadgroup SIMD_ACT mlp_tg[MLP_WEIGHT_COUNT];\n"
        << "    for (int i = int(lid); i < MLP_WEIGHT_COUNT; i += 32)\n"
        << "        mlp_tg[i] = SIMD_ACT(mlp[i]);\n\n";
    }

    o << "    threadgroup SIMD_ACT act_a[8][HIDDEN_DIM];\n"
      << "    threadgroup SIMD_ACT act_b[8][HIDDEN_DIM];\n\n"
      << "    ";
    if (hybrid_scalar_w0) {
      o << "encode_hash_grid_scalar_w0(lid, base_point, positions, hash_grid, mlp, num_pts, act_b);\n"
        << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n"
        << "    mlp_forward_simd_hidden_only(lid, mlp, act_a, act_b);\n\n";
    } else {
      o << "encode_hash_grid_coop(lid, base_point, positions, hash_grid, num_pts, act_a);\n"
        << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n"
        << "    mlp_forward_simd(lid, " << (tg_cache ? "mlp_tg" : "mlp") << ", act_a, act_b);\n\n";
    }

    // Layer out: scalar dot product (8 threads, 1 per point)
    o << "    // Output layer: scalar dot product\n"
      << "    if (lid < 8) {\n"
      << "        uint pt = base_point + lid;\n"
      << "        if (int(pt) < num_pts) {\n"
      << "            " << (tg_cache ? "threadgroup const SIMD_ACT" : "device const SIMD_ACT") << "* wO = " << (tg_cache ? "mlp_tg" : "mlp") << " + " << wO_off << ";\n"
      << "            float sdf = float(" << (tg_cache ? "mlp_tg" : "mlp") << "[" << bO_off << "]);\n"
      << "            for (int j = 0; j < HIDDEN_DIM; j++)\n"
      << "                sdf += float(" << last_act << "[lid][j]) * float(wO[j]);\n"
      << "            output[pt] = sdf;\n"
      << "        }\n"
      << "    }\n"
      << "}\n";

    return o.str();
  }

  // --- Scalar path (original) ---
  emitPreamble(o, spec);
  emitHashFunctions(o, spec);
  const bool scalar_fp16_hidden_fastpath =
      use_scalar_fp16_hidden_fastpath(spec);

  // Define W_SRC macro for eval (reads from constant address space)
  o << "#define W_SRC(idx) mlp[idx]\n\n";

  // Inline eval function
  if (is4d) {
      o << "inline float neural_sdf_eval(\n"
      << "    float4 pos,\n"
      << "    device const HASH_T* hash_grid,\n"
      << "    device const float* mlp,\n"
      << "    NeuralSDFConfig cfg)\n"
      << "{\n";
  } else {
    o << "inline float neural_sdf_eval(\n"
      << "    float3 pos,\n"
      << "    device const HASH_T* hash_grid,\n"
      << "    device const float* mlp,\n";
    if (scalar_fp16_hidden_fastpath) {
      o << "    device const half* mlp_hidden_half,\n";
    }
    o
      << "    NeuralSDFConfig cfg)\n"
      << "{\n";
  }

  emitWeightOffsets(o, spec);

  if (spec.encoding == KernelSpec::RMHE)
    emitRMHEHashEncode(o, spec);
  else if (is4d)
    emitHashEncode4D(o, spec);
  else
    emitHashEncode3D(o, spec);

  emitMLPForward(o, spec);

  if (spec.num_outputs == 1) {
    o << "    return sdf;\n";
  } else {
    o << "    return outputs[0];\n";
  }
  o << "}\n\n";

  // Entry point
  o << "kernel void neural_sdf_eval_points(\n"
    << "    device const float*  positions     [[buffer(0)]],\n"
    << "    device float*        output        [[buffer(1)]],\n"
    << "    device const float*  config_weights [[buffer(2)]],\n"
    << "    device const HASH_T*  hash_grid     [[buffer(3)]],\n";
  if (scalar_fp16_hidden_fastpath) {
    o << "    device const half*   mlp_hidden_half [[buffer(4)]],\n";
  }
  o << "    device const float*  mlp_direct     [[buffer(5)]],\n";
  o
    << "    uint tid [[thread_position_in_grid]])\n"
    << "{\n"
    << "    int num_pts = int(config_weights[7]);\n"
    << "    if (int(tid) >= num_pts) return;\n\n"
    << "    NeuralSDFConfig cfg = load_config(config_weights);\n"
    << "    device const float* mlp = mlp_direct;\n\n";

  if (is4d) {
    o << "    float4 pos = float4(positions[tid * 4], positions[tid * 4 + 1], "
      << "positions[tid * 4 + 2], positions[tid * 4 + 3]);\n";
  } else {
    o << "    float3 pos = float3(positions[tid * 3], positions[tid * 3 + 1], "
      << "positions[tid * 3 + 2]);\n";
  }

  if (spec.num_outputs == 1) {
    o << "    output[tid] = neural_sdf_eval(pos, hash_grid, mlp, ";
    if (scalar_fp16_hidden_fastpath) {
      o << "mlp_hidden_half, ";
    }
    o << "cfg);\n";
  } else {
    o << "\n";
    emitWeightOffsets(o, spec);
    if (spec.encoding == KernelSpec::RMHE)
      emitRMHEHashEncode(o, spec);
    else if (is4d)
      emitHashEncode4D(o, spec);
    else
      emitHashEncode3D(o, spec);
    emitMLPForward(o, spec);
    o << "    for (int m = 0; m < NUM_OUTPUTS; m++)\n"
      << "        output[tid * NUM_OUTPUTS + m] = outputs[m];\n";
  }
  o << "}\n";

  return o.str();
}

// ===========================================================================
// Gradient kernel
// ===========================================================================

void MLPKernelEmitter::emitTrilinearFeatureGrad3D(std::ostringstream& o) {
  o << "inline float hash_grid_lookup(device const HASH_T* hg,\n"
    << "                              int level, int feature_idx,\n"
    << "                              int3 corner, uint table_size,\n"
    << "                              int features_per_level) {\n"
    << "    uint h = hash_coords(corner, table_size);\n"
    << "    uint grid_offset = uint(level) * table_size * features_per_level\n"
    << "                    + int(h) * features_per_level\n"
    << "                    + feature_idx;\n"
    << "    return hg[grid_offset];\n"
    << "}\n\n"
    << "inline float trilinear_feature_grad(float3 frac_pos,\n"
    << "                                     device const HASH_T* hg,\n"
    << "                                     int level, int feature_idx,\n"
    << "                                     int3 base_coord, uint table_size,\n"
    << "                                     int features_per_level,\n"
    << "                                     thread float3& grad_frac) {\n"
    << "    float fx = frac_pos.x;\n"
    << "    float fy = frac_pos.y;\n"
    << "    float fz = frac_pos.z;\n\n"
    << "    float c000 = hash_grid_lookup(hg, level, feature_idx, base_coord + int3(0,0,0), table_size, features_per_level);\n"
    << "    float c001 = hash_grid_lookup(hg, level, feature_idx, base_coord + int3(0,0,1), table_size, features_per_level);\n"
    << "    float c010 = hash_grid_lookup(hg, level, feature_idx, base_coord + int3(0,1,0), table_size, features_per_level);\n"
    << "    float c011 = hash_grid_lookup(hg, level, feature_idx, base_coord + int3(0,1,1), table_size, features_per_level);\n"
    << "    float c100 = hash_grid_lookup(hg, level, feature_idx, base_coord + int3(1,0,0), table_size, features_per_level);\n"
    << "    float c101 = hash_grid_lookup(hg, level, feature_idx, base_coord + int3(1,0,1), table_size, features_per_level);\n"
    << "    float c110 = hash_grid_lookup(hg, level, feature_idx, base_coord + int3(1,1,0), table_size, features_per_level);\n"
    << "    float c111 = hash_grid_lookup(hg, level, feature_idx, base_coord + int3(1,1,1), table_size, features_per_level);\n\n"
    << "    float c00 = c000 * (1.0 - fx) + c100 * fx;\n"
    << "    float c01 = c001 * (1.0 - fx) + c101 * fx;\n"
    << "    float c10 = c010 * (1.0 - fx) + c110 * fx;\n"
    << "    float c11 = c011 * (1.0 - fx) + c111 * fx;\n"
    << "    float c0 = c00 * (1.0 - fy) + c10 * fy;\n"
    << "    float c1 = c01 * (1.0 - fy) + c11 * fy;\n"
    << "    float val = c0 * (1.0 - fz) + c1 * fz;\n\n"
    << "    float dvdx = (1.0 - fz) * ((1.0 - fy) * (c100 - c000) + fy * (c110 - c010))\n"
    << "               + fz          * ((1.0 - fy) * (c101 - c001) + fy * (c111 - c011));\n"
    << "    float dvdy = (1.0 - fz) * ((1.0 - fx) * (c010 - c000) + fx * (c110 - c100))\n"
    << "               + fz          * ((1.0 - fx) * (c011 - c001) + fx * (c111 - c101));\n"
    << "    float dvdz = (1.0 - fy) * ((1.0 - fx) * (c001 - c000) + fx * (c101 - c100))\n"
    << "               + fy          * ((1.0 - fx) * (c011 - c010) + fx * (c111 - c110));\n\n"
    << "    grad_frac = float3(dvdx, dvdy, dvdz);\n"
    << "    return val;\n"
    << "}\n\n";
}

std::string MLPKernelEmitter::emitGradientKernel(const KernelSpec& spec) {
  return emitGradientKernelImpl(spec, /*include_sdf=*/false);
}

std::string MLPKernelEmitter::emitEvalGradientKernel(const KernelSpec& spec) {
  return emitGradientKernelImpl(spec, /*include_sdf=*/true);
}

std::string MLPKernelEmitter::emitGradientKernelImpl(const KernelSpec& spec,
                                                     bool include_sdf) {
  spec.validate();
  std::ostringstream o;
  bool is4d = (spec.encoding == KernelSpec::FourD || spec.spatial_dims == 4);
  const int out_floats = include_sdf ? 4 : (is4d ? 4 : 3);
  const char* simd_kernel_name = include_sdf
      ? "neural_sdf_analytical_eval_gradient_simd"
      : "neural_sdf_analytical_gradient_simd";
  const char* scalar_kernel_name = include_sdf
      ? "neural_sdf_analytical_eval_gradient_points"
      : "neural_sdf_analytical_gradient_points";

  // SIMD path: 32 threads/TG, 8 points, SIMD forward + scalar backward
  if (spec.use_simd && !is4d && spec.num_outputs == 1
      && spec.encoding == KernelSpec::Standard) {
    bool tg_cache = spec.use_tg_weight_cache && spec.canUseTGCache();
    const char* w_src = tg_cache ? "mlp_tg" : "mlp";
    int hd = spec.hidden_dim;
    int nhl = spec.num_hidden_layers;

    emitPreamble(o, spec);
    o << "#include <metal_simdgroup_matrix>\n\n";
    emitHashFunctions(o, spec);
    emitTrilinearFeatureGrad3D(o);
    emitHashEncodeCooperative(o, spec);
    emitMLPForwardSIMD(o, spec);

    // Compute output layer weight offset
    int wO_off_g = spec.input_dim * hd + hd;
    for (int i = 1; i < nhl; ++i) wO_off_g += hd * hd + hd;
    int bO_off_g = wO_off_g + hd;

    o << "kernel void " << simd_kernel_name << "(\n"
      << "    device const float*  positions     [[buffer(0)]],\n"
      << "    device float*        output        [[buffer(1)]],\n"
      << "    device const float*  config_weights [[buffer(2)]],\n"
      << "    device const HASH_T*  hash_grid     [[buffer(3)]],\n";
    if (spec.use_fp16_simd) {
      o << "    device const SIMD_ACT* mlp_half      [[buffer(4)]],\n";
    }
    o << "    device const SIMD_ACT* mlp_direct    [[buffer(5)]],\n";
    o
      << "    uint tgid [[threadgroup_position_in_grid]],\n"
      << "    uint lid  [[thread_position_in_threadgroup]])\n"
      << "{\n"
      << "    int num_pts = int(config_weights[7]);\n";
    if (spec.use_fp16_simd) {
      o << "    device const SIMD_ACT* mlp = mlp_half;\n";
    } else {
      o << "    device const SIMD_ACT* mlp = mlp_direct;\n";
    }
    o
      << "    uint base_point = tgid * 8;\n"
      << "    uint table_size = 1u << uint(LOG2_HASHMAP_SIZE);\n\n";

    if (tg_cache) {
      o << "    threadgroup SIMD_ACT mlp_tg[MLP_WEIGHT_COUNT];\n"
        << "    for (int i = int(lid); i < MLP_WEIGHT_COUNT; i += 32)\n"
        << "        mlp_tg[i] = SIMD_ACT(mlp[i]);\n\n";
    }

    o << "    threadgroup SIMD_ACT act_a[8][HIDDEN_DIM];\n"
      << "    threadgroup SIMD_ACT act_b[8][HIDDEN_DIM];\n\n"
      << "    encode_hash_grid_coop(lid, base_point, positions, hash_grid, num_pts, act_a);\n"
      << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n";

    // Save layer activations for backward: copy from TG to per-thread registers
    // We need to save all hidden layer activations before SIMD forward clobbers them.
    // Do SIMD forward, then save act from TG to per-thread arrays.
    o << "    mlp_forward_simd(lid, " << w_src << ", act_a, act_b);\n\n";

    // Only 8 threads do backward (1 per point)
    o << "    if (lid < 8) {\n"
      << "        uint pt = base_point + lid;\n"
      << "        if (int(pt) < num_pts) {\n";

    // Copy activations from TG to per-thread registers
    // After forward: Layer 0 result in act_b, Layer 1 result in act_a (for nhl=2)
    // We need to read them before they're clobbered (but they won't be — only 8 threads do backward)
    // For nhl=2: h0 = act_b[lid], h1 = act_a[lid] (last layer)
    // General: After layer k (0-indexed): result is in act_b if k%2==0, act_a if k%2==1
    // Actually: layer 0 → act_b. layer 1 → act_a. layer 2 → act_b...
    // So layer k result: k%2==0 → act_b, k%2==1 → act_a
    for (int layer = 0; layer < nhl; ++layer) {
      const char* src_tg = (layer % 2 == 0) ? "act_b" : "act_a";
      o << "            float h" << layer << "[HIDDEN_DIM];\n"
        << "            for (int j = 0; j < HIDDEN_DIM; j++)\n"
        << "                h" << layer << "[j] = float(" << src_tg << "[lid][j]);\n\n";
    }

    // Backward pass
    emitMLPBackwardSIMD(o, spec);

    // Determine cur_grad name after backward
    std::string cur_grad = (nhl > 1) ? "grad_pre_0" : "grad_pre_last";

    // Position gradient via trilinear gradient
    o << "            float3 pos = float3(positions[pt*3], positions[pt*3+1], positions[pt*3+2]);\n"
      << "            float3 grad_pos = float3(0.0f);\n\n"
      << "            for (int l = 0; l < NUM_LEVELS; l++) {\n"
      << "                float resolution = BASE_RESOLUTION * pow(PER_LEVEL_SCALE, float(l));\n"
      << "                float3 sc = pos * resolution;\n"
      << "                int3 bc = int3(int(floor(sc.x)), int(floor(sc.y)), int(floor(sc.z)));\n"
      << "                float3 frac_pos = float3(sc.x-floor(sc.x), sc.y-floor(sc.y), sc.z-floor(sc.z));\n\n"
      << "                for (int f = 0; f < FEATURES_PER_LEVEL; f++) {\n"
      << "                    int feat_idx = l * FEATURES_PER_LEVEL + f;\n"
      << "                    float grad_feat = 0.0f;\n"
      << "                    int w_row = feat_idx * HIDDEN_DIM;\n"
      << "                    for (int j = 0; j < HIDDEN_DIM; j++)\n"
      << "                        grad_feat += " << cur_grad << "[j] * " << w_src << "[w_row + j];\n\n"
      << "                    float3 tri_grad;\n"
      << "                    trilinear_feature_grad(frac_pos, hash_grid, l, f, bc,\n"
      << "                                           table_size, FEATURES_PER_LEVEL, tri_grad);\n"
      << "                    grad_pos += grad_feat * resolution * tri_grad;\n"
      << "                }\n"
      << "            }\n\n";
    if (include_sdf) {
      // Compute SDF value before writing output
      std::string last_h_sdf = "h" + std::to_string(nhl - 1);
      o << "            float sdf = " << w_src << "[" << bO_off_g << "];\n"
        << "            for (int j = 0; j < HIDDEN_DIM; j++)\n"
        << "                sdf += " << last_h_sdf << "[j] * " << w_src << "[" << wO_off_g << " + j];\n\n";
      o << "            output[pt*4+0] = sdf;\n"
        << "            output[pt*4+1] = grad_pos.x;\n"
        << "            output[pt*4+2] = grad_pos.y;\n"
        << "            output[pt*4+3] = grad_pos.z;\n";
    } else {
      o << "            output[pt*3+0] = grad_pos.x;\n"
        << "            output[pt*3+1] = grad_pos.y;\n"
        << "            output[pt*3+2] = grad_pos.z;\n";
    }
    o << "        }\n"
      << "    }\n"
      << "}\n";

    return o.str();
  }

  // --- Scalar path ---
  emitPreamble(o, spec);
  emitHashFunctions(o, spec);

  // Emit helper functions
  if (!is4d) {
    emitTrilinearFeatureGrad3D(o);
  }

  // Entry point
  o << "kernel void " << scalar_kernel_name << "(\n"
    << "    device const float*  positions     [[buffer(0)]],\n"
    << "    device float*        output        [[buffer(1)]],\n"
    << "    device const float*  config_weights [[buffer(2)]],\n"
    << "    device const HASH_T*  hash_grid     [[buffer(3)]],\n"
    << "    device const float*  mlp_direct     [[buffer(5)]],\n"
    << "    uint tid [[thread_position_in_grid]])\n"
    << "{\n"
    << "    int num_pts = int(config_weights[7]);\n"
    << "    if (int(tid) >= num_pts) return;\n\n"
    << "    device const float* mlp = mlp_direct;\n"
    << "    uint table_size = 1u << uint(LOG2_HASHMAP_SIZE);\n\n";

  emitWeightOffsets(o, spec);

  // Read position
  if (is4d) {
    o << "    float4 pos = float4(positions[tid*4], positions[tid*4+1], positions[tid*4+2], positions[tid*4+3]);\n";
  } else {
    o << "    float3 pos = float3(positions[tid*3], positions[tid*3+1], positions[tid*3+2]);\n";
  }

  // === Forward pass: fused hash encode + W0 matmul (matches eval kernel) ===
  o << "\n    // === Forward pass: hash encode fused with W0 matmul ===\n"
    << "    float h0[HIDDEN_DIM];\n"
    << "    for (int j = 0; j < HIDDEN_DIM; j++) h0[j] = 0.0f;\n\n";

  // Store per-level info for backward pass
  o << "    float resolutions[NUM_LEVELS];\n";
  if (is4d) {
    o << "    int4 base_coords[NUM_LEVELS];\n"
      << "    float4 frac_positions[NUM_LEVELS];\n\n";
  } else {
    o << "    int3 base_coords[NUM_LEVELS];\n"
      << "    float3 frac_positions[NUM_LEVELS];\n\n";
  }

  o << "    for (int l = 0; l < NUM_LEVELS; l++) {\n"
    << "        float resolution = BASE_RESOLUTION * pow(PER_LEVEL_SCALE, float(l));\n"
    << "        resolutions[l] = resolution;\n";

  if (is4d) {
    o << "        float4 sc = pos * resolution;\n"
      << "        int4 bc = int4(int(floor(sc.x)), int(floor(sc.y)), int(floor(sc.z)), int(floor(sc.w)));\n"
      << "        float4 frac = float4(sc.x-floor(sc.x), sc.y-floor(sc.y), sc.z-floor(sc.z), sc.w-floor(sc.w));\n"
      << "        base_coords[l] = bc;\n"
      << "        frac_positions[l] = frac;\n\n"
      << "        for (int f = 0; f < FEATURES_PER_LEVEL; f++) {\n"
      << "            float feat = 0.0f;\n"
      << "            for (int dw = 0; dw < 2; dw++)\n"
      << "                for (int dz = 0; dz < 2; dz++)\n"
      << "                    for (int dy = 0; dy < 2; dy++)\n"
      << "                        for (int dx = 0; dx < 2; dx++) {\n"
      << "                            uint h = hash_coords_4d(bc + int4(dx,dy,dz,dw), table_size);\n"
      << "                            uint off = uint(l)*table_size*FEATURES_PER_LEVEL + uint(h)*FEATURES_PER_LEVEL + f;\n"
      << "                            float wx=(dx==0)?(1.0f-frac.x):frac.x;\n"
      << "                            float wy=(dy==0)?(1.0f-frac.y):frac.y;\n"
      << "                            float wz=(dz==0)?(1.0f-frac.z):frac.z;\n"
      << "                            float wt=(dw==0)?(1.0f-frac.w):frac.w;\n"
      << "                            feat += wx*wy*wz*wt*hash_grid[off];\n"
      << "                        }\n"
      << "            int feat_idx = l * FEATURES_PER_LEVEL + f;\n"
      << "            int w_row = w0_off + feat_idx * HIDDEN_DIM;\n"
      << "            for (int j = 0; j < HIDDEN_DIM; j++)\n"
      << "                h0[j] += feat * mlp[w_row + j];\n"
      << "        }\n";
  } else {
    o << "        float3 sc = pos * resolution;\n"
      << "        int3 bc = int3(int(floor(sc.x)), int(floor(sc.y)), int(floor(sc.z)));\n"
      << "        float3 frac = float3(sc.x-floor(sc.x), sc.y-floor(sc.y), sc.z-floor(sc.z));\n"
      << "        base_coords[l] = bc;\n"
      << "        frac_positions[l] = frac;\n\n"
      << "        for (int f = 0; f < FEATURES_PER_LEVEL; f++) {\n"
      << "            float3 dummy;\n"
      << "            float feat = trilinear_feature_grad(frac, hash_grid, l, f, bc,\n"
      << "                                                 table_size, FEATURES_PER_LEVEL, dummy);\n"
      << "            int feat_idx = l * FEATURES_PER_LEVEL + f;\n"
      << "            int w_row = w0_off + feat_idx * HIDDEN_DIM;\n"
      << "            for (int j = 0; j < HIDDEN_DIM; j++)\n"
      << "                h0[j] += feat * mlp[w_row + j];\n"
      << "        }\n";
  }

  o << "    }\n\n"
    << "    // Bias + ReLU for layer 0\n"
    << "    for (int j = 0; j < HIDDEN_DIM; j++)\n"
    << "        h0[j] = nn_relu(h0[j] + mlp[b0_off + j]);\n\n";

  // Hidden layers — save each layer's activations in separate arrays
  // (unlike eval kernel's ping-pong, gradient needs all activations for backward)
  int nhl = spec.num_hidden_layers;
  for (int layer = 1; layer < nhl; ++layer) {
    std::string src = "h" + std::to_string(layer - 1);
    std::string dst = "h" + std::to_string(layer);
    o << "    float " << dst << "[HIDDEN_DIM];\n"
      << "    for (int j = 0; j < HIDDEN_DIM; j++) {\n"
      << "        float acc = 0.0f;\n"
      << "        for (int i = 0; i < HIDDEN_DIM; i++)\n"
      << "            acc += " << src << "[i] * mlp[w" << layer << "_off + i * HIDDEN_DIM + j];\n"
      << "        " << dst << "[j] = nn_relu(acc + mlp[b" << layer << "_off + j]);\n"
      << "    }\n\n";
  }

  // Backward: d_sdf = 1.0 (gradient of output w.r.t. itself)
  std::string last_h = "h" + std::to_string(nhl - 1);
  o << "    // === Backward: gradient w.r.t. position ===\n"
    << "    float grad_pre_last[HIDDEN_DIM];\n"
    << "    for (int j = 0; j < HIDDEN_DIM; j++)\n"
    << "        grad_pre_last[j] = mlp[wO_off + j] * ((" << last_h << "[j] > 0.0f) ? 1.0f : 0.0f);\n\n";

  // Propagate through hidden layers (reverse)
  std::string cur_grad = "grad_pre_last";
  for (int layer = nhl - 1; layer >= 1; --layer) {
    std::string h_prev = "h" + std::to_string(layer - 1);
    std::string next_grad = "grad_pre_" + std::to_string(layer - 1);

    o << "    float " << next_grad << "[HIDDEN_DIM];\n"
      << "    for (int i = 0; i < HIDDEN_DIM; i++) {\n"
      << "        float acc = 0.0f;\n"
      << "        for (int j = 0; j < HIDDEN_DIM; j++)\n"
      << "            acc += " << cur_grad << "[j] * mlp[w" << layer << "_off + i * HIDDEN_DIM + j];\n"
      << "        " << next_grad << "[i] = acc * ((" << h_prev << "[i] > 0.0f) ? 1.0f : 0.0f);\n"
      << "    }\n\n";

    cur_grad = next_grad;
  }

  // Position gradient via trilinear gradient
  o << "    // === Position gradient via trilinear gradient ===\n";
  if (is4d) {
    o << "    float4 grad_pos = float4(0.0f);\n\n";
  } else {
    o << "    float3 grad_pos = float3(0.0f);\n\n";
  }

  o << "    for (int l = 0; l < NUM_LEVELS; l++) {\n"
    << "        float resolution = resolutions[l];\n";

  if (is4d) {
    o << "        int4 bc = base_coords[l];\n"
      << "        float4 frac_pos = frac_positions[l];\n\n";
  } else {
    o << "        int3 bc = base_coords[l];\n"
      << "        float3 frac_pos = frac_positions[l];\n\n";
  }

  o << "        for (int f = 0; f < FEATURES_PER_LEVEL; f++) {\n"
    << "            int feat_idx = l * FEATURES_PER_LEVEL + f;\n"
    << "            float grad_feat = 0.0f;\n"
    << "            int w_row = w0_off + feat_idx * HIDDEN_DIM;\n"
    << "            for (int j = 0; j < HIDDEN_DIM; j++)\n"
    << "                grad_feat += " << cur_grad << "[j] * mlp[w_row + j];\n\n";

  // Trilinear gradient
  if (is4d) {
    // 4D quadrilinear gradient — lookup 16 corners
    o << "            float c[2][2][2][2];\n"
      << "            for (int dw=0;dw<2;dw++) for (int dz=0;dz<2;dz++) for (int dy=0;dy<2;dy++) for (int dx=0;dx<2;dx++) {\n"
      << "                uint h = hash_coords_4d(bc + int4(dx,dy,dz,dw), table_size);\n"
      << "                uint off = uint(l)*table_size*FEATURES_PER_LEVEL + uint(h)*FEATURES_PER_LEVEL + f;\n"
      << "                c[dw][dz][dy][dx] = hash_grid[off];\n"
      << "            }\n\n"
      << "            float dvdx=0,dvdy=0,dvdz=0,dvdt=0;\n"
      << "            for (int dw=0;dw<2;dw++) { float wt=(dw==0)?(1.0f-frac_pos.w):frac_pos.w;\n"
      << "              for (int dz=0;dz<2;dz++) { float wz=(dz==0)?(1.0f-frac_pos.z):frac_pos.z;\n"
      << "                for (int dy=0;dy<2;dy++) { float wy=(dy==0)?(1.0f-frac_pos.y):frac_pos.y;\n"
      << "                  dvdx += wy*wz*wt*(c[dw][dz][dy][1]-c[dw][dz][dy][0]);\n"
      << "            }}}\n"
      << "            for (int dw=0;dw<2;dw++) { float wt=(dw==0)?(1.0f-frac_pos.w):frac_pos.w;\n"
      << "              for (int dz=0;dz<2;dz++) { float wz=(dz==0)?(1.0f-frac_pos.z):frac_pos.z;\n"
      << "                for (int dx=0;dx<2;dx++) { float wx=(dx==0)?(1.0f-frac_pos.x):frac_pos.x;\n"
      << "                  dvdy += wx*wz*wt*(c[dw][dz][1][dx]-c[dw][dz][0][dx]);\n"
      << "            }}}\n"
      << "            for (int dw=0;dw<2;dw++) { float wt=(dw==0)?(1.0f-frac_pos.w):frac_pos.w;\n"
      << "              for (int dy=0;dy<2;dy++) { float wy=(dy==0)?(1.0f-frac_pos.y):frac_pos.y;\n"
      << "                for (int dx=0;dx<2;dx++) { float wx=(dx==0)?(1.0f-frac_pos.x):frac_pos.x;\n"
      << "                  dvdz += wx*wy*wt*(c[dw][1][dy][dx]-c[dw][0][dy][dx]);\n"
      << "            }}}\n"
      << "            for (int dz=0;dz<2;dz++) { float wz=(dz==0)?(1.0f-frac_pos.z):frac_pos.z;\n"
      << "              for (int dy=0;dy<2;dy++) { float wy=(dy==0)?(1.0f-frac_pos.y):frac_pos.y;\n"
      << "                for (int dx=0;dx<2;dx++) { float wx=(dx==0)?(1.0f-frac_pos.x):frac_pos.x;\n"
      << "                  dvdt += wx*wy*wz*(c[1][dz][dy][dx]-c[0][dz][dy][dx]);\n"
      << "            }}}\n\n"
      << "            grad_pos.x += grad_feat * resolution * dvdx;\n"
      << "            grad_pos.y += grad_feat * resolution * dvdy;\n"
      << "            grad_pos.z += grad_feat * resolution * dvdz;\n"
      << "            grad_pos.w += grad_feat * resolution * dvdt;\n";
  } else {
    // 3D — use trilinear_feature_grad helper (matches working kernel exactly)
    o << "            float3 tri_grad;\n"
      << "            trilinear_feature_grad(frac_pos, hash_grid, l, f, bc,\n"
      << "                                   table_size, FEATURES_PER_LEVEL, tri_grad);\n"
      << "            grad_pos += grad_feat * resolution * tri_grad;\n";
  }

  o << "        }\n"
    << "    }\n\n";

  // Write output
  if (include_sdf) {
    // Compute SDF value
    std::string last_h_out = "h" + std::to_string(nhl - 1);
    o << "    // SDF output\n"
      << "    float sdf = mlp[bO_off];\n"
      << "    for (int j = 0; j < HIDDEN_DIM; j++)\n"
      << "        sdf += " << last_h_out << "[j] * mlp[wO_off + j];\n\n";

    o << "    output[tid*4+0] = sdf;\n"
      << "    output[tid*4+1] = grad_pos.x;\n"
      << "    output[tid*4+2] = grad_pos.y;\n"
      << "    output[tid*4+3] = grad_pos.z;\n";
  } else if (is4d) {
    o << "    output[tid*4+0] = grad_pos.x;\n"
      << "    output[tid*4+1] = grad_pos.y;\n"
      << "    output[tid*4+2] = grad_pos.z;\n"
      << "    output[tid*4+3] = grad_pos.w;\n";
  } else {
    o << "    output[tid*3+0] = grad_pos.x;\n"
      << "    output[tid*3+1] = grad_pos.y;\n"
      << "    output[tid*3+2] = grad_pos.z;\n";
  }

  o << "}\n";

  return o.str();
}

// ===========================================================================
// Training kernel
// ===========================================================================

std::string MLPKernelEmitter::emitTrainKernel(const KernelSpec& spec) {
  spec.validate();
  // =========================================================================
  // SIMD training path: 32 threads/TG, 8 batches × 8 points = 64 pts/TG
  // Uses gradient checkpointing for nhl > 2: saves hash features to TG memory
  // and recomputes forward activations during backward.
  // Falls through to scalar path otherwise.
  // =========================================================================
  if (spec.use_simd && spec.canUseSIMD()
      && spec.simdTrainTGBytes() <= 32768
      && spec.input_dim % 8 == 0
      && spec.num_outputs == 1
      && spec.encoding == KernelSpec::Standard
      && spec.spatial_dims == 3) {
    std::ostringstream o;
    const int hd = spec.hidden_dim;
    const int id = spec.input_dim;
    const int nhl = spec.num_hidden_layers;
    const int mlp_wc = spec.mlpWeightCount();
    const int fpl = spec.features_per_level;
    const int nl = spec.num_levels;
    const int levels_per_thread = (nl + 3) / 4;
    const bool hybrid_scalar_w0 = (fpl == 2);
    static const char* buf_names[] = {"act_a", "act_b", "act_c"};

    // Preamble
    o << "#include <metal_stdlib>\n"
      << "#include <metal_simdgroup_matrix>\n"
      << "using namespace metal;\n\n"
      << "constant int INPUT_DIM = " << id << ";\n"
      << "constant int HIDDEN_DIM = " << hd << ";\n"
      << "constant int NUM_LEVELS = " << nl << ";\n"
      << "constant int FEATURES_PER_LEVEL = " << fpl << ";\n"
      << "constant int LOG2_HASHMAP_SIZE = " << spec.log2_hashmap_size << ";\n"
      << "constant float BASE_RESOLUTION = " << std::fixed << std::setprecision(6) << spec.base_resolution << "f;\n"
      << "constant float PER_LEVEL_SCALE = " << std::fixed << std::setprecision(6) << spec.per_level_scale << "f;\n"
      << "constant int MLP_WEIGHT_COUNT = " << mlp_wc << ";\n"
      << "#define SIMD_TG_SIZE 32\n"
      << "#define TG_SIZE 32\n"
      << "#define POINTS_PER_TG 8\n"
      << "#define BATCHES_PER_TG 8\n"
      << "#define TOTAL_PTS_PER_TG (BATCHES_PER_TG * POINTS_PER_TG)\n\n";
    emit_train_param_macros(o);

    // Hash grid type alias
    if (spec.use_fp16_hash_grid) {
      o << "typedef half HASH_T;\n"
        << "typedef half2 HASH2_T;\n\n";
    } else {
      o << "typedef float HASH_T;\n"
        << "typedef float2 HASH2_T;\n\n";
    }

    // Training kernel: always float SIMD (mlp_tg reused for gradient accumulation)
    o << "typedef float SIMD_ACT;\n"
      << "typedef simdgroup_float8x8 SIMD_MAT;\n\n";

    if (spec.use_int_atomics) {
      o << "#define GRAD_SCALE " << kDefaultGradScale << "f\n"
        << "#define INV_GRAD_SCALE (1.0f / " << kDefaultGradScale << "f)\n\n";
    }

    emitHashFunctions(o, spec);
    if (hybrid_scalar_w0) {
      emitHashEncodeCoopScalarW0Training(o, spec);
    }

    // Kernel entry
    o << "kernel void neural_sdf_train_forward_backward(\n"
      << "    device const float*       positions      [[buffer(0)]],\n"
      << "    device const float*       targets        [[buffer(1)]],\n"
      << "    device const float*       config_weights [[buffer(2)]],\n"
      << "    device const HASH_T*       hash_grid      [[buffer(3)]],\n";
    if (spec.use_int_atomics) {
      o << "    device atomic_int*        grad_hash      [[buffer(4)]],\n";
    } else {
      o << "    device atomic_float*      grad_hash      [[buffer(4)]],\n";
    }
    o << "    device atomic_float*      grad_mlp       [[buffer(5)]],\n"
      << "    device float*             loss_partials  [[buffer(6)]],\n"
      << "    constant float*           train_params   [[buffer(7)]],\n";
    if (spec.emit_active_hash_mask) {
      o << "    device atomic_uint*       active_hash_mask [[buffer(8)]],\n";
      o << "    device atomic_uint*       active_hash_summary_mask [[buffer(9)]],\n";
    }
    o << "    device const float*       mlp_weights    [[buffer(11)]],\n";
    o << "    uint tgid [[threadgroup_position_in_grid]],\n"
      << "    uint lid  [[thread_position_in_threadgroup]])\n"
      << "{\n"
      << "    int N = int(train_params[TMNN_TRAIN_PARAMS_IDX_N]);\n"
      << "    int unsigned_mode = int(train_params[TMNN_TRAIN_PARAMS_IDX_UNSIGNED_MODE]);\n"
      << "    float loss_scale = train_params[TMNN_TRAIN_PARAMS_IDX_LOSS_SCALE];\n"
      << "    float inv_loss_scale = 1.0f / loss_scale;\n"
      << "    int num_active_levels = int(train_params[TMNN_TRAIN_PARAMS_IDX_NUM_ACTIVE_LEVELS]);\n"
      << "    if (num_active_levels <= 0) num_active_levels = NUM_LEVELS;\n\n"
      << "    device const float* mlp = mlp_weights;\n"
      << "    uint table_size = 1u << uint(LOG2_HASHMAP_SIZE);\n"
      << "    float inv_N = 1.0f / float(N);\n\n";

    // Weight offsets (loop-based for N layers)
    std::vector<int> w_offs(nhl), b_offs(nhl);
    int ofs = 0;
    for (int layer = 0; layer < nhl; ++layer) {
      w_offs[layer] = ofs;
      ofs += (layer == 0) ? id * hd : hd * hd;
      b_offs[layer] = ofs;
      ofs += hd;
      o << "    const int w" << layer << "_off = " << w_offs[layer] << ";\n"
        << "    const int b" << layer << "_off = " << b_offs[layer] << ";\n";
    }
    o << "    const int wO_off = " << ofs << ";\n";
    ofs += hd;
    o << "    const int bO_off = " << ofs << ";\n\n";

    // TG memory
    o << "    threadgroup float mlp_tg[MLP_WEIGHT_COUNT];\n"
      << "    threadgroup float act_a[8][HIDDEN_DIM];\n"
      << "    threadgroup float act_b[8][HIDDEN_DIM];\n";
    if (nhl >= 2)
      o << "    threadgroup float act_c[8][HIDDEN_DIM];\n";
    o << "    threadgroup float features_tg[8 * INPUT_DIM];\n";
    o << "\n";

    // Zero mlp_tg
    o << "    for (int i = int(lid); i < MLP_WEIGHT_COUNT; i += SIMD_TG_SIZE)\n"
      << "        mlp_tg[i] = 0.0f;\n"
      << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n"
      << "    float total_tg_loss = 0.0f;\n\n";

    // ================================================================
    // Multi-batch loop
    // ================================================================
    if (fpl == 2 && !hybrid_scalar_w0) {
      o << "    uint scatter_corner_grid_off[" << levels_per_thread * 8 << "];\n"
        << "    float scatter_corner_weight[" << levels_per_thread * 8 << "];\n\n";
    }
    o << "    for (int batch = 0; batch < BATCHES_PER_TG; batch++) {\n"
      << "        uint base_point = tgid * TOTAL_PTS_PER_TG + batch * POINTS_PER_TG;\n\n";

    if (hybrid_scalar_w0) {
      o << "        // Scalar hash encode + fused W0 (training hot path)\n"
        << "        encode_hash_grid_scalar_w0_train(\n"
        << "            lid, base_point, positions, hash_grid, mlp, N, act_b,\n"
        << "            features_tg);\n"
        << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n\n";
    } else {
      // -- Cooperative hash encode → act_a --
      int feats_per_thread = levels_per_thread * fpl;
      o << "        // Hash encoding (cooperative, 4 threads/point)\n"
        << "        {\n"
        << "            uint local_point = lid / 4;\n"
        << "            uint level_group = lid % 4;\n"
        << "            int feat_start = int(level_group) * " << feats_per_thread << ";\n"
        << "            int feat_end = min(feat_start + " << feats_per_thread << ", INPUT_DIM);\n"
        << "            for (int j = feat_start; j < feat_end; j++)\n"
        << "                act_a[local_point][j] = 0.0f;\n\n"
        << "            uint point_idx = base_point + local_point;\n"
        << "            if (int(point_idx) < N) {\n"
        << "                float3 pos = float3(positions[point_idx*3],\n"
        << "                                    positions[point_idx*3+1],\n"
        << "                                    positions[point_idx*3+2]);\n"
        << "                for (int l_off = 0; l_off < " << levels_per_thread << "; l_off++) {\n"
        << "                    int l = int(level_group) * " << levels_per_thread << " + l_off;\n"
        << "                    if (l >= NUM_LEVELS) break;\n"
        << "                    float resolution = BASE_RESOLUTION * pow(PER_LEVEL_SCALE, float(l));\n"
        << "                    float3 scaled = pos * resolution;\n"
        << "                    float3 scaled_floor = floor(scaled);\n"
        << "                    int3 bc = int3(scaled_floor);\n"
        << "                    float3 frac = scaled - scaled_floor;\n"
        << "                    ";
      if (fpl == 2) {
        // Vectorized float2 path
        o << "{\n"
          << "                        float2 feat2 = float2(0.0f);\n"
          << "                        int corner_base = l_off * 8;\n"
          << "                        int corner_idx = 0;\n"
          << "                        for (int dz = 0; dz < 2; dz++)\n"
          << "                            for (int dy = 0; dy < 2; dy++)\n"
          << "                                for (int dx = 0; dx < 2; dx++) {\n"
          << "                                    uint h = hash_coords(bc + int3(dx, dy, dz), table_size);\n"
          << "                                    uint grid_off = uint(l) * table_size * 2u + h * 2u;\n"
          << "                                    float wx = (dx == 0) ? (1.0f - frac.x) : frac.x;\n"
          << "                                    float wy = (dy == 0) ? (1.0f - frac.y) : frac.y;\n"
          << "                                    float wz = (dz == 0) ? (1.0f - frac.z) : frac.z;\n"
          << "                                    float w = wx * wy * wz;\n"
          << "                                    scatter_corner_grid_off[corner_base + corner_idx] = grid_off;\n"
          << "                                    scatter_corner_weight[corner_base + corner_idx] = w;\n"
          << "                                    feat2 += w * float2(*((device const HASH2_T*)(hash_grid + grid_off)));\n"
          << "                                    corner_idx++;\n"
          << "                                }\n"
          << "                        act_a[local_point][l * 2] = feat2.x;\n"
          << "                        act_a[local_point][l * 2 + 1] = feat2.y;\n"
          << "                    }\n";
      } else {
        o << "for (int f = 0; f < FEATURES_PER_LEVEL; f++) {\n"
          << "                        float feat = 0.0f;\n"
          << "                        for (int dz = 0; dz < 2; dz++)\n"
          << "                            for (int dy = 0; dy < 2; dy++)\n"
          << "                                for (int dx = 0; dx < 2; dx++) {\n"
          << "                                    uint h = hash_coords(bc + int3(dx, dy, dz), table_size);\n"
          << "                                    uint grid_off = uint(l) * table_size * FEATURES_PER_LEVEL + uint(h) * FEATURES_PER_LEVEL + f;\n"
          << "                                    float wx = (dx == 0) ? (1.0f - frac.x) : frac.x;\n"
          << "                                    float wy = (dy == 0) ? (1.0f - frac.y) : frac.y;\n"
          << "                                    float wz = (dz == 0) ? (1.0f - frac.z) : frac.z;\n"
          << "                                    feat += wx * wy * wz * hash_grid[grid_off];\n"
          << "                                }\n"
          << "                        act_a[local_point][l * FEATURES_PER_LEVEL + f] = feat;\n"
          << "                    }\n";
      }
      o << "                }\n"
        << "            }\n"
        << "        }\n"
        << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n\n";

      // -- Save features for SIMD W0 backward grads (+ gradient checkpointing) --
      o << "        // Save features for SIMD W0 backward\n"
        << "        for (int j = int(lid); j < 8 * INPUT_DIM; j += SIMD_TG_SIZE) {\n"
        << "            features_tg[j] = act_a[j / INPUT_DIM][j % INPUT_DIM];\n"
        << "        }\n"
        << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n\n";
    }

    // -- SIMD Forward loop --
    // Helper lambda to emit one SIMD forward layer
    auto emitSimdFwdLayer = [&](int layer, const char* src, const char* dst,
                                 int in_tiles, int out_tiles) {
      o << "        // SIMD forward: Layer " << layer << "\n"
        << "        for (int oj = 0; oj < " << out_tiles << "; oj++) {\n"
        << "            SIMD_MAT acc, a_tile, w_tile;\n"
        << "            simdgroup_load(a_tile, &" << src << "[0][0], HIDDEN_DIM);\n"
        << "            simdgroup_load(w_tile, mlp + w" << layer << "_off + oj * 8, HIDDEN_DIM);\n"
        << "            simdgroup_multiply(acc, a_tile, w_tile);\n"
        << "            for (int ik = 1; ik < " << in_tiles << "; ik++) {\n"
        << "                simdgroup_load(a_tile, &" << src << "[0][ik * 8], HIDDEN_DIM);\n"
        << "                simdgroup_load(w_tile, mlp + w" << layer << "_off + ik * 8 * HIDDEN_DIM + oj * 8, HIDDEN_DIM);\n"
        << "                simdgroup_multiply_accumulate(acc, a_tile, w_tile, acc);\n"
        << "            }\n"
        << "            simdgroup_store(acc, &" << dst << "[0][oj * 8], HIDDEN_DIM);\n"
        << "        }\n"
        << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        << "        for (int j = int(lid); j < " << 8 * hd << "; j += SIMD_TG_SIZE) {\n"
        << "            int row = j / HIDDEN_DIM;\n"
        << "            int col = j % HIDDEN_DIM;\n"
        << "            " << dst << "[row][col] = max(0.0f, " << dst << "[row][col] + mlp[b" << layer << "_off + col]);\n"
        << "        }\n"
        << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n\n";
    };

    if (!hybrid_scalar_w0) {
      // Layer 0: act_a → act_b
      emitSimdFwdLayer(0, "act_a", "act_b", id / 8, hd / 8);
    }

    // Hidden layers 1..nhl-1 with ping-pong
    for (int layer = 1; layer < nhl; ++layer) {
      const char* src = (layer % 2 == 1) ? "act_b" : "act_a";
      const char* dst = (layer % 2 == 1) ? "act_a" : "act_b";
      emitSimdFwdLayer(layer, src, dst, hd / 8, hd / 8);
    }

    // After forward: last hidden output buffer
    // Layer 0→act_b, 1→act_a, 2→act_b, 3→act_a, ...
    // nhl=1→act_b(1), nhl=2→act_a(0), nhl=3→act_b(1), nhl=4→act_a(0)
    int last_h_idx = nhl % 2;  // 0=act_a, 1=act_b
    const char* last_h = buf_names[last_h_idx];

    // -- Output layer + Loss --
    o << "        float thread_loss = 0.0f;\n"
      << "        float d_sdf = 0.0f;\n\n"
      << "        if (lid < POINTS_PER_TG) {\n"
      << "            uint pt = base_point + lid;\n"
      << "            if (int(pt) < N) {\n"
      << "                float sdf = mlp[bO_off];\n"
      << "                for (int j = 0; j < HIDDEN_DIM; j++)\n"
      << "                    sdf += " << last_h << "[lid][j] * mlp[wO_off + j];\n\n"
      << "                float target = targets[pt];\n";

    if (spec.loss == KernelSpec::L2) {
      o << "                if (unsigned_mode != 0) {\n"
        << "                    float abs_sdf = abs(sdf);\n"
        << "                    float residual = abs_sdf - target;\n"
        << "                    thread_loss = residual * residual;\n"
        << "                    float sign_sdf = (sdf >= 0.0f) ? 1.0f : -1.0f;\n"
        << "                    d_sdf = 2.0f * residual * sign_sdf * inv_N;\n"
        << "                } else {\n"
        << "                    float residual = sdf - target;\n"
        << "                    thread_loss = residual * residual;\n"
        << "                    d_sdf = 2.0f * residual * inv_N;\n"
        << "                }\n";
    } else if (spec.loss == KernelSpec::L1) {
      o << "                {\n"
        << "                    float residual = sdf - target;\n"
        << "                    thread_loss = abs(residual);\n"
        << "                    d_sdf = ((residual > 0.0f) ? 1.0f : ((residual < 0.0f) ? -1.0f : 0.0f)) * inv_N;\n"
        << "                }\n";
    } else { // Huber
      o << "                {\n"
        << "                    float residual = sdf - target;\n"
        << "                    float abs_r = abs(residual);\n"
        << "                    float delta = " << std::fixed << std::setprecision(6) << spec.huber_delta << "f;\n"
        << "                    if (abs_r <= delta) {\n"
        << "                        thread_loss = 0.5f * residual * residual;\n"
        << "                        d_sdf = residual * inv_N;\n"
        << "                    } else {\n"
        << "                        thread_loss = delta * (abs_r - 0.5f * delta);\n"
        << "                        d_sdf = delta * ((residual > 0.0f) ? 1.0f : -1.0f) * inv_N;\n"
        << "                    }\n"
        << "                }\n";
    }

    emitFP16Scale(o, spec, "d_sdf", true, 16);

    o << "            }\n"
      << "        }\n"
      << "        total_tg_loss += simd_sum(thread_loss);\n\n";

    // -- Output layer backward --
    o << "        if (lid < POINTS_PER_TG) {\n"
      << "            uint pt = base_point + lid;\n"
      << "            if (int(pt) < N) {\n"
      << "                for (int j = 0; j < HIDDEN_DIM; j++) {\n"
      << "                    float h_val = " << last_h << "[lid][j];\n"
      << "                    mlp_tg[wO_off + j] += d_sdf * h_val;\n"
      << "                    float relu_mask = (h_val > 0.0f) ? 1.0f : 0.0f;\n"
      << "                    " << last_h << "[lid][j] = d_sdf * mlp[wO_off + j] * relu_mask;\n"
      << "                }\n"
      << "                mlp_tg[bO_off] += d_sdf;\n"
      << "            } else {\n"
      << "                for (int j = 0; j < HIDDEN_DIM; j++)\n"
      << "                    " << last_h << "[lid][j] = 0.0f;\n"
      << "            }\n"
      << "        }\n"
      << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n\n";

    // -- Hidden layer backward loop --
    // Buffer tracking: d_idx holds d_pre_hk, we compute d_pre_h_{k-1}
    int d_idx = last_h_idx;
    int inner_h = hd / 8;
    int outer_h = hd / 8;
    int rows_per_thread = hd / 4;

    for (int layer = nhl - 1; layer >= 1; --layer) {
      int h_idx;       // buffer holding h_{layer-1}
      int result_idx;  // buffer to write d_pre_h_{layer-1}

      if (layer == nhl - 1) {
        // h_{nhl-2} survived forward in the other ping-pong buffer
        h_idx = 1 - last_h_idx;
        result_idx = 2;
      } else {
        // Gradient checkpointing: recompute h_{layer-1} from saved features
        // Available buffers: the 2 not holding d_idx
        int free_a = -1, free_b = -1;
        for (int i = 0; i < 3; ++i) {
          if (i == d_idx) continue;
          if (free_a < 0) free_a = i; else free_b = i;
        }

        // Copy features_tg → buf[free_a], then forward layers 0..layer-1
        o << "        // Recompute h" << (layer-1) << " from saved features\n"
          << "        for (int j = int(lid); j < 8 * INPUT_DIM; j += SIMD_TG_SIZE) {\n"
          << "            int row = j / INPUT_DIM;\n"
          << "            int col = j % INPUT_DIM;\n"
          << "            " << buf_names[free_a] << "[row][col] = features_tg[j];\n"
          << "        }\n"
          << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n\n";

        for (int fwd = 0; fwd <= layer - 1; ++fwd) {
          int src_i = (fwd % 2 == 0) ? free_a : free_b;
          int dst_i = (fwd % 2 == 0) ? free_b : free_a;
          int in_tiles = (fwd == 0) ? id / 8 : hd / 8;
          emitSimdFwdLayer(fwd, buf_names[src_i], buf_names[dst_i],
                           in_tiles, hd / 8);
        }

        // h_{layer-1} is in the dst of the last forward step
        h_idx = ((layer - 1) % 2 == 0) ? free_b : free_a;
        result_idx = 3 - d_idx - h_idx;  // the remaining third buffer
      }

      const char* d_name = buf_names[d_idx];
      const char* h_name = buf_names[h_idx];
      const char* r_name = buf_names[result_idx];

      // SIMD: d_pre_hk × Wk^T → result_buf, then ReLU mask
      o << "        // Layer " << layer << " backward: SIMD " << d_name
        << " × W" << layer << "^T → " << r_name << "\n"
        << "        {\n"
        << "            for (int oj = 0; oj < " << outer_h << "; oj++) {\n"
        << "                simdgroup_float8x8 acc_d, a_tile, w_tile;\n"
        << "                simdgroup_load(a_tile, &" << d_name << "[0][0], HIDDEN_DIM);\n"
        << "                simdgroup_load(w_tile, mlp + w" << layer << "_off + oj * 8 * HIDDEN_DIM, HIDDEN_DIM,\n"
        << "                               ulong2(0, 0), true);\n"
        << "                simdgroup_multiply(acc_d, a_tile, w_tile);\n"
        << "                for (int ik = 1; ik < " << inner_h << "; ik++) {\n"
        << "                    simdgroup_load(a_tile, &" << d_name << "[0][ik * 8], HIDDEN_DIM);\n"
        << "                    simdgroup_load(w_tile, mlp + w" << layer << "_off + oj * 8 * HIDDEN_DIM + ik * 8,\n"
        << "                                   HIDDEN_DIM, ulong2(0, 0), true);\n"
        << "                    simdgroup_multiply_accumulate(acc_d, a_tile, w_tile, acc_d);\n"
        << "                }\n"
        << "                simdgroup_store(acc_d, &" << r_name << "[0][oj * 8], HIDDEN_DIM);\n"
        << "            }\n"
        << "            threadgroup_barrier(mem_flags::mem_threadgroup);\n\n"
        << "            for (int j = int(lid); j < " << 8 * hd << "; j += SIMD_TG_SIZE) {\n"
        << "                int row = j / HIDDEN_DIM;\n"
        << "                int col = j % HIDDEN_DIM;\n"
        << "                " << r_name << "[row][col] *= (" << h_name << "[row][col] > 0.0f) ? 1.0f : 0.0f;\n"
        << "            }\n"
        << "            threadgroup_barrier(mem_flags::mem_threadgroup);\n\n";

      // SIMD Wk grads: outer product h^T × delta
      o << "            // SIMD W" << layer << " grads: outer product h^T × delta\n"
        << "            for (int oi = 0; oi < " << hd / 8 << "; oi++) {\n"
        << "                for (int oj = 0; oj < " << hd / 8 << "; oj++) {\n"
        << "                    SIMD_MAT dW_tile, h_T, d_tile;\n"
        << "                    simdgroup_load(dW_tile, &mlp_tg[w" << layer << "_off + oi * 8 * HIDDEN_DIM + oj * 8], HIDDEN_DIM);\n"
        << "                    simdgroup_load(h_T, &" << h_name << "[0][oi * 8], HIDDEN_DIM, ulong2(0, 0), true);\n"
        << "                    simdgroup_load(d_tile, &" << d_name << "[0][oj * 8], HIDDEN_DIM);\n"
        << "                    simdgroup_multiply_accumulate(dW_tile, h_T, d_tile, dW_tile);\n"
        << "                    simdgroup_store(dW_tile, &mlp_tg[w" << layer << "_off + oi * 8 * HIDDEN_DIM + oj * 8], HIDDEN_DIM);\n"
        << "                }\n"
        << "            }\n"
        << "            threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        << "        }\n\n";

      // Bias grads
      o << "        for (int col = int(lid); col < HIDDEN_DIM; col += SIMD_TG_SIZE) {\n"
        << "            float bias_grad = 0.0f;\n"
        << "            for (int row = 0; row < 8; ++row)\n"
        << "                bias_grad += " << d_name << "[row][col];\n"
        << "            mlp_tg[b" << layer << "_off + col] += bias_grad;\n"
        << "        }\n"
        << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n\n";

      d_idx = result_idx;
    }

    // d_pre_h0 is now in buf[d_idx]
    const char* d_h0_buf = buf_names[d_idx];

    // -- SIMD W0 grads: outer product features^T × d_h0 --
    o << "        // SIMD W0 grads: outer product features^T × d_h0\n"
      << "        for (int oi = 0; oi < " << id / 8 << "; oi++) {\n"
      << "            for (int oj = 0; oj < " << hd / 8 << "; oj++) {\n"
      << "                SIMD_MAT dW_tile, feat_T, d_tile;\n"
      << "                simdgroup_load(dW_tile, &mlp_tg[w0_off + oi * 8 * HIDDEN_DIM + oj * 8], HIDDEN_DIM);\n"
      << "                simdgroup_load(feat_T, features_tg + oi * 8, INPUT_DIM, ulong2(0, 0), true);\n"
      << "                simdgroup_load(d_tile, &" << d_h0_buf << "[0][oj * 8], HIDDEN_DIM);\n"
      << "                simdgroup_multiply_accumulate(dW_tile, feat_T, d_tile, dW_tile);\n"
      << "                simdgroup_store(dW_tile, &mlp_tg[w0_off + oi * 8 * HIDDEN_DIM + oj * 8], HIDDEN_DIM);\n"
      << "            }\n"
      << "        }\n"
      << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n\n";

    // -- Hash scatter (d_feat + scatter, W0 grads done via SIMD above) --
    o << "        // Hash scatter: d_feat = W0^T × d_h0, scatter to hash grid\n"
      << "        {\n"
      << "            uint local_pt = lid / 4;\n"
      << "            uint level_group = lid % 4;\n"
      << "            uint point_idx = base_point + local_pt;\n\n"
      << "            if (int(point_idx) < N) {\n";
    if (fpl != 2 || hybrid_scalar_w0) {
      o << "                float3 pos = float3(positions[point_idx*3],\n"
        << "                                    positions[point_idx*3+1],\n"
        << "                                    positions[point_idx*3+2]);\n";
    }
    o << "                for (int l_off = 0; l_off < " << levels_per_thread << "; l_off++) {\n"
      << "                    int l = int(level_group) * " << levels_per_thread << " + l_off;\n"
      << "                    if (l >= num_active_levels) continue;\n";
    if (fpl != 2 || hybrid_scalar_w0) {
      o << "                    float resolution = BASE_RESOLUTION * pow(PER_LEVEL_SCALE, float(l));\n"
        << "                    float3 scaled = pos * resolution;\n"
        << "                    float3 scaled_floor = floor(scaled);\n"
        << "                    int3 bc = int3(scaled_floor);\n"
        << "                    float3 frac = scaled - scaled_floor;\n";
    }

    if (fpl == 2) {
      o << "                    {\n"
        << "                        int fb = l * 2;\n"
        << "                        int w_row0 = fb * HIDDEN_DIM;\n"
        << "                        int w_row1 = (fb + 1) * HIDDEN_DIM;\n"
        << "                        threadgroup const float* d_row = &" << d_h0_buf << "[local_pt][0];\n"
        << "                        device const float* w_row0_ptr = mlp + w_row0;\n"
        << "                        device const float* w_row1_ptr = mlp + w_row1;\n"
        << "                        device const float4* w_row0_ptr4 = (device const float4*)w_row0_ptr;\n"
        << "                        device const float4* w_row1_ptr4 = (device const float4*)w_row1_ptr;\n"
        << "                        float2 d_feat2 = float2(0.0f);\n"
        << "                        int j = 0;\n"
        << "                        int j4 = 0;\n"
        << "                        for (; j + 3 < HIDDEN_DIM; j += 4, ++j4) {\n"
        << "                            float4 d4 = float4(d_row[j + 0], d_row[j + 1], d_row[j + 2], d_row[j + 3]);\n"
        << "                            d_feat2.x += dot(d4, w_row0_ptr4[j4]);\n"
        << "                            d_feat2.y += dot(d4, w_row1_ptr4[j4]);\n"
        << "                        }\n"
        << "                        for (; j < HIDDEN_DIM; ++j) {\n"
        << "                            float dj = d_row[j];\n"
        << "                            d_feat2.x += dj * w_row0_ptr[j];\n"
        << "                            d_feat2.y += dj * w_row1_ptr[j];\n"
        << "                        }\n";

    if (spec.use_fp16) {
      o << "                        d_feat2 *= inv_loss_scale;\n";
    }

      if (hybrid_scalar_w0) {
        o << "                        for (int dz = 0; dz < 2; ++dz)\n"
          << "                            for (int dy = 0; dy < 2; ++dy)\n"
          << "                                for (int dx = 0; dx < 2; ++dx) {\n"
          << "                                    uint h = hash_coords(bc + int3(dx, dy, dz), table_size);\n"
          << "                                    uint grid_off = uint(l) * table_size * 2u + h * 2u;\n"
          << "                                    float wx = (dx == 0) ? (1.0f - frac.x) : frac.x;\n"
          << "                                    float wy = (dy == 0) ? (1.0f - frac.y) : frac.y;\n"
          << "                                    float wz = (dz == 0) ? (1.0f - frac.z) : frac.z;\n"
          << "                                    float w = wx * wy * wz;\n";
      } else {
        o << "                        int corner_base = l_off * 8;\n"
          << "                        for (int corner_idx = 0; corner_idx < 8; corner_idx++) {\n"
          << "                                    uint grid_off = scatter_corner_grid_off[corner_base + corner_idx];\n"
          << "                                    float w = scatter_corner_weight[corner_base + corner_idx];\n";
      }

      if (spec.use_int_atomics) {
        o << "                                    atomic_fetch_add_explicit(&grad_hash[grid_off],\n"
          << "                                        int(d_feat2.x * w * GRAD_SCALE), memory_order_relaxed);\n"
          << "                                    atomic_fetch_add_explicit(&grad_hash[grid_off + 1],\n"
          << "                                        int(d_feat2.y * w * GRAD_SCALE), memory_order_relaxed);\n";
      } else {
        o << "                                    atomic_fetch_add_explicit(&grad_hash[grid_off],\n"
          << "                                        d_feat2.x * w, memory_order_relaxed);\n"
          << "                                    atomic_fetch_add_explicit(&grad_hash[grid_off + 1],\n"
          << "                                        d_feat2.y * w, memory_order_relaxed);\n";
      }
      if (spec.emit_active_hash_mask) {
        emit_active_hash_tracking(o, "grid_off", true,
                                  "                                    ");
      }

      o << "                                }\n"
        << "                    }\n";
    } else {
      o << "                    for (int f = 0; f < FEATURES_PER_LEVEL; f++) {\n"
        << "                        int feat_idx = l * FEATURES_PER_LEVEL + f;\n"
        << "                        int w_row = feat_idx * HIDDEN_DIM;\n"
        << "                        threadgroup const float* d_row = &" << d_h0_buf << "[local_pt][0];\n"
        << "                        device const float* w_row_ptr = mlp + w_row;\n"
        << "                        device const float4* w_row_ptr4 = (device const float4*)w_row_ptr;\n"
        << "                        float d_feat = 0.0f;\n"
        << "                        int j = 0;\n"
        << "                        int j4 = 0;\n"
        << "                        for (; j + 3 < HIDDEN_DIM; j += 4, ++j4)\n"
        << "                            d_feat += dot(float4(d_row[j + 0], d_row[j + 1], d_row[j + 2], d_row[j + 3]), w_row_ptr4[j4]);\n"
        << "                        for (; j < HIDDEN_DIM; ++j)\n"
        << "                            d_feat += d_row[j] * w_row_ptr[j];\n";

      emitFP16Scale(o, spec, "d_feat", false, 24);

      o << "                        for (int dz = 0; dz < 2; dz++)\n"
        << "                            for (int dy = 0; dy < 2; dy++)\n"
        << "                                for (int dx = 0; dx < 2; dx++) {\n"
        << "                                    uint h = hash_coords(bc + int3(dx, dy, dz), table_size);\n"
        << "                                    uint grid_off = uint(l) * table_size * FEATURES_PER_LEVEL + uint(h) * FEATURES_PER_LEVEL + f;\n"
        << "                                    float wx = (dx == 0) ? (1.0f - frac.x) : frac.x;\n"
        << "                                    float wy = (dy == 0) ? (1.0f - frac.y) : frac.y;\n"
        << "                                    float wz = (dz == 0) ? (1.0f - frac.z) : frac.z;\n";

      if (spec.use_int_atomics) {
        o << "                                    atomic_fetch_add_explicit(&grad_hash[grid_off],\n"
          << "                                        int(d_feat * wx * wy * wz * GRAD_SCALE), memory_order_relaxed);\n";
      } else {
        o << "                                    atomic_fetch_add_explicit(&grad_hash[grid_off],\n"
          << "                                        d_feat * wx * wy * wz, memory_order_relaxed);\n";
      }
      if (spec.emit_active_hash_mask) {
        if (fpl == 2) {
          emit_active_hash_tracking(o, "grid_off", true,
                                    "                                    ");
        } else {
          emit_active_hash_tracking(o, "grid_off", false,
                                    "                                    ");
        }
      }

      o << "                                }\n"
        << "                    }\n";
    }
    o
      << "                }\n"
      << "            }\n"
      << "        }\n"
      << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n\n";

    // Bias0 grads
    o << "        for (int col = int(lid); col < HIDDEN_DIM; col += SIMD_TG_SIZE) {\n"
      << "            float bias_grad = 0.0f;\n"
      << "            for (int row = 0; row < 8; ++row)\n"
      << "                bias_grad += " << d_h0_buf << "[row][col];\n"
      << "            mlp_tg[b0_off + col] += bias_grad;\n"
      << "        }\n"
      << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n\n"
      << "    } // end batch loop\n\n";

    // Flush TG grads
    o << "    for (int i = int(lid); i < MLP_WEIGHT_COUNT; i += SIMD_TG_SIZE) {\n"
      << "        float val = mlp_tg[i];\n"
      << "        if (val != 0.0f)\n";
    if (spec.use_fp16) {
      o << "            atomic_fetch_add_explicit(&grad_mlp[i], val * inv_loss_scale, memory_order_relaxed);\n";
    } else {
      o << "            atomic_fetch_add_explicit(&grad_mlp[i], val, memory_order_relaxed);\n";
    }
    o << "    }\n\n"
      << "    if (lid == 0) {\n"
      << "#ifdef REDUCTION_TERMS\n"
      << "        loss_partials[tgid * REDUCTION_TERMS] = total_tg_loss;\n"
      << "#else\n"
      << "        loss_partials[tgid] = total_tg_loss;\n"
      << "#endif\n"
      << "    }\n"
      << "}\n";

    return o.str();
  }

  // =========================================================================
  // Scalar training path — delegate to shared implementation.
  return emitScalarTrainKernelImpl(spec, LossMode::Internal,
                                    "neural_sdf_train_forward_backward");
}

std::string MLPKernelEmitter::emitScalarTrainKernelImpl(
    const KernelSpec& spec, LossMode mode, const char* entry_point) {
  std::ostringstream o;
  bool is4d = (spec.encoding == KernelSpec::FourD || spec.spatial_dims == 4);
  bool tg_cache = spec.use_tg_weight_cache && spec.canUseTGCache();

  {
    auto scalar_spec = spec;
    scalar_spec.use_simd = false;
    scalar_spec.use_fp16_simd = false;
    emitPreamble(o, scalar_spec);
  }

  // Grad scale for int atomics
  if (spec.use_int_atomics) {
    o << "#define GRAD_SCALE 65536.0f\n"
      << "#define INV_GRAD_SCALE (1.0f / 65536.0f)\n\n";
  }

  emitHashFunctions(o, spec);
  emit_train_param_macros(o);

  o << "#define TG_SIZE 128\n\n";

  // FP16 activation type macros
  if (spec.use_fp16) {
    o << "#define ACT_T half\n"
      << "#define ACT_ZERO half(0.0h)\n\n";
  } else {
    o << "#define ACT_T float\n"
      << "#define ACT_ZERO 0.0f\n\n";
  }


  // Kernel entry
  o << "kernel void " << entry_point << "(\n"
    << "    device const float*       positions      [[buffer(0)]],\n"
    << "    device const float*       targets        [[buffer(1)]],\n"
    << "    device const float*       config_weights [[buffer(2)]],\n"
    << "    device const HASH_T*       hash_grid      [[buffer(3)]],\n";

  if (spec.use_int_atomics) {
    o << "    device atomic_int*        grad_hash      [[buffer(4)]],\n";
  } else {
    o << "    device atomic_float*      grad_hash      [[buffer(4)]],\n";
  }
  o << "    device atomic_float*      grad_mlp       [[buffer(5)]],\n";

  o << "    device float*             loss_partials  [[buffer(6)]],\n"
    << "    constant float*           train_params   [[buffer(7)]],\n";

  // Buffer 8+: mode-dependent, with active-hash mask bound for backward-capable
  // kernels so both fused and split training can share the sparse update path.
  if (mode == LossMode::Internal) {
    if (spec.emit_active_hash_mask) {
      o << "    device atomic_uint*       active_hash_mask [[buffer(8)]],\n";
      o << "    device atomic_uint*       active_hash_summary_mask [[buffer(9)]],\n";
    }
  } else if (mode == LossMode::ExternalGrad) {
    o << "    device const float*       d_output_external [[buffer(8)]],\n";
    if (spec.emit_active_hash_mask) {
      o << "    device atomic_uint*       active_hash_mask [[buffer(9)]],\n";
      o << "    device atomic_uint*       active_hash_summary_mask [[buffer(10)]],\n";
    }
  } else if (mode == LossMode::ForwardOnly) {
    o << "    device float*             forward_output [[buffer(8)]],\n";
  }
  if (spec.emit_probes) {
    uint32_t probe_slot = 9u;
    if (mode == LossMode::Internal && spec.emit_active_hash_mask) {
      probe_slot = 10u;
    } else if (mode == LossMode::ExternalGrad && spec.emit_active_hash_mask) {
      probe_slot = 12u;
    }
    o << "    device float*             probe_partials [[buffer(" << probe_slot
      << ")]],\n";
  }
  o << "    device const float*       mlp_weights    [[buffer(11)]],\n";

  o << "    uint tid     [[thread_position_in_grid]],\n"
    << "    uint tgid    [[threadgroup_position_in_grid]],\n"
    << "    uint lid     [[thread_position_in_threadgroup]])\n"
    << "{\n"
    << "    int N = int(train_params[TMNN_TRAIN_PARAMS_IDX_N]);\n"
      << "    int unsigned_mode = int(train_params[TMNN_TRAIN_PARAMS_IDX_UNSIGNED_MODE]);\n"
      << "    int num_active_levels = int(train_params[TMNN_TRAIN_PARAMS_IDX_NUM_ACTIVE_LEVELS]);\n"
      << "    if (num_active_levels <= 0) num_active_levels = NUM_LEVELS;\n\n"
      << "    device const float* mlp = mlp_weights;\n";
  o << "\n";

  if (spec.encoding == KernelSpec::RMHE) {
    o << "    device const float* rotations = mlp + MLP_WEIGHT_COUNT;\n";
  }

  emitWeightOffsets(o, spec);

  o << "    uint table_size = 1u << uint(LOG2_HASHMAP_SIZE);\n"
    << "    float inv_N = 1.0f / float(N);\n\n";

  // Scalar kernels process one point per thread, so many threads update the
  // same MLP parameters concurrently. Accumulating into shared threadgroup
  // floats with plain "+=" is a data race; keep TG only for read-only weight
  // caching and write MLP gradients directly through device atomics.
  if (mode != LossMode::ForwardOnly) {
    o << "    threadgroup float tg_loss[TG_SIZE];\n";
    if (spec.use_fp16) {
      o << "#define MLP_GRAD_ADD(idx, val) atomic_fetch_add_explicit(&grad_mlp[idx], (val) * inv_loss_scale, memory_order_relaxed)\n";
    } else {
      o << "#define MLP_GRAD_ADD(idx, val) atomic_fetch_add_explicit(&grad_mlp[idx], (val), memory_order_relaxed)\n";
    }
    o << "#define LOSS_BUF tg_loss\n\n";
  }
  if (tg_cache) {
    o << "    threadgroup float tg_mlp_cache[MLP_WEIGHT_COUNT];\n"
      << "    // Phase 1: Load MLP weights into TG for forward/backward reuse\n"
      << "    for (int i = int(lid); i < MLP_WEIGHT_COUNT; i += TG_SIZE)\n"
      << "        tg_mlp_cache[i] = mlp[i];\n"
      << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      << "#define W_SRC(idx) tg_mlp_cache[idx]\n\n";
  } else {
    // No TG buffer for MLP — read weights from constant address space.
    o << "#define W_SRC(idx) mlp[idx]\n\n";
  }

  o << "    float thread_loss = 0.0f;\n"
    << "#ifdef BC_DIM_COUNT\n"
    << "    float thread_loss_bc = 0.0f;\n"
    << "    float thread_loss_piezo = 0.0f;\n"
    << "#endif\n\n";

  // Declare variables outside if-scope for TG weight cache phase transition
  if (is4d) {
    o << "    float px=0,py=0,pz=0,pw=0;\n";
  } else {
    o << "    float px=0,py=0,pz=0;\n";
  }
  o << "    ACT_T h0[HIDDEN_DIM]";
  if (spec.num_hidden_layers > 1) o << ", h1[HIDDEN_DIM]";
  o << ";\n";

  if (spec.num_outputs == 1) {
    o << "    float d_sdf = 0.0f;\n";
  } else {
    o << "    float d_out[NUM_OUTPUTS];\n"
      << "    for (int m = 0; m < NUM_OUTPUTS; m++) d_out[m] = 0.0f;\n";
  }

  // Hoist inv_loss_scale (survives TG cache phase transition; used by FP16 backward)
  o << "    float inv_loss_scale = 1.0f;\n";

  if (spec.emit_probes) {
    o << "\n    // Probe accumulators\n"
      << "    #define PROBE_STRIDE (" << (2 * spec.num_hidden_layers + 5) << ")\n"
      << "    float probe_fwd_nan = 0.0f;\n"
      << "    float probe_bwd_nan = 0.0f;\n"
      << "    float probe_hash_grad_l2 = 0.0f;\n"
      << "    float probe_output_abs_max = 0.0f;\n"
      << "    float probe_output_neg_min = 0.0f;\n";
    for (int l = 0; l < spec.num_hidden_layers; ++l) {
      o << "    float probe_act_max_" << l << " = 0.0f;\n"
        << "    float probe_grad_l2_" << l << " = 0.0f;\n";
    }
  }

  o << "\n    if (int(tid) < N) {\n";

  // Read position
  if (is4d) {
    o << "        px = positions[tid*4]; py = positions[tid*4+1]; pz = positions[tid*4+2]; pw = positions[tid*4+3];\n"
      << "        float4 pos = float4(px,py,pz,pw);\n\n";
  } else {
    o << "        px = positions[tid*3]; py = positions[tid*3+1]; pz = positions[tid*3+2];\n"
      << "        float3 pos = float3(px,py,pz);\n\n";
  }

  // Forward pass: hash encode fused with W0 → h0
  o << "        // === FORWARD PASS ===\n";
  o << "        for (int j = 0; j < HIDDEN_DIM; j++) h0[j] = ACT_ZERO;\n\n";

  // Hash encode (inline, using the pos variable)
  o << "        for (int l = 0; l < NUM_LEVELS; l++) {\n"
    << "            float resolution = BASE_RESOLUTION * pow(PER_LEVEL_SCALE, float(l));\n";

  if (is4d) {
    o << "            float sx=px*resolution, sy=py*resolution, sz=pz*resolution, sw=pw*resolution;\n"
      << "            int4 bc = int4(int(floor(sx)),int(floor(sy)),int(floor(sz)),int(floor(sw)));\n"
      << "            float4 frac = float4(sx-floor(sx),sy-floor(sy),sz-floor(sz),sw-floor(sw));\n\n"
      << "            for (int f = 0; f < FEATURES_PER_LEVEL; f++) {\n"
      << "                float feat = 0.0f;\n"
      << "                for (int dw=0;dw<2;dw++) for (int dz=0;dz<2;dz++) for (int dy=0;dy<2;dy++) for (int dx=0;dx<2;dx++) {\n"
      << "                    uint h = hash_coords_4d(bc+int4(dx,dy,dz,dw), table_size);\n"
      << "                    uint grid_off = uint(l)*table_size*FEATURES_PER_LEVEL+uint(h)*FEATURES_PER_LEVEL+f;\n"
      << "                    float wx=(dx==0)?(1.0f-frac.x):frac.x;\n"
      << "                    float wy=(dy==0)?(1.0f-frac.y):frac.y;\n"
      << "                    float wz=(dz==0)?(1.0f-frac.z):frac.z;\n"
      << "                    float wt=(dw==0)?(1.0f-frac.w):frac.w;\n"
      << "                    feat += wx*wy*wz*wt*hash_grid[grid_off];\n"
      << "                }\n";
  } else if (spec.encoding == KernelSpec::RMHE) {
    // RMHE: apply per-level rotation before hash lookup
    o << "            constant float* R = rotations + l * 9;\n"
      << "            float sx = (R[0]*px + R[1]*py + R[2]*pz) * resolution;\n"
      << "            float sy = (R[3]*px + R[4]*py + R[5]*pz) * resolution;\n"
      << "            float sz = (R[6]*px + R[7]*py + R[8]*pz) * resolution;\n"
      << "            int3 bc = int3(int(floor(sx)),int(floor(sy)),int(floor(sz)));\n"
      << "            float3 frac = float3(sx-floor(sx),sy-floor(sy),sz-floor(sz));\n\n"
      << "            for (int f = 0; f < FEATURES_PER_LEVEL; f++) {\n"
      << "                float feat = 0.0f;\n"
      << "                for (int dz=0;dz<2;dz++) for (int dy=0;dy<2;dy++) for (int dx=0;dx<2;dx++) {\n"
      << "                    uint h = hash_coords(bc+int3(dx,dy,dz), table_size);\n"
      << "                    uint grid_off = uint(l)*table_size*FEATURES_PER_LEVEL+uint(h)*FEATURES_PER_LEVEL+f;\n"
      << "                    float wx=(dx==0)?(1.0f-frac.x):frac.x;\n"
      << "                    float wy=(dy==0)?(1.0f-frac.y):frac.y;\n"
      << "                    float wz=(dz==0)?(1.0f-frac.z):frac.z;\n"
      << "                    feat += wx*wy*wz*hash_grid[grid_off];\n"
      << "                }\n";
  } else if (spec.features_per_level == 2) {
    // Vectorized float2 path for standard 3D with fpl==2
    o << "            float sx=px*resolution, sy=py*resolution, sz=pz*resolution;\n"
      << "            int3 bc = int3(int(floor(sx)),int(floor(sy)),int(floor(sz)));\n"
      << "            float3 frac = float3(sx-floor(sx),sy-floor(sy),sz-floor(sz));\n\n"
      << "            {\n"
      << "                float2 feat2 = float2(0.0f);\n"
      << "                for (int dz=0;dz<2;dz++) for (int dy=0;dy<2;dy++) for (int dx=0;dx<2;dx++) {\n"
      << "                    uint h = hash_coords(bc+int3(dx,dy,dz), table_size);\n"
      << "                    uint grid_off = uint(l)*table_size*2u+h*2u;\n"
      << "                    float wx=(dx==0)?(1.0f-frac.x):frac.x;\n"
      << "                    float wy=(dy==0)?(1.0f-frac.y):frac.y;\n"
      << "                    float wz=(dz==0)?(1.0f-frac.z):frac.z;\n"
      << "                    float w=wx*wy*wz;\n"
      << "                    feat2 += w * float2(*((device const HASH2_T*)(hash_grid+grid_off)));\n"
      << "                }\n"
      << "                int fb = l*2;\n"
      << "                for (int j = 0; j < HIDDEN_DIM; j++)\n"
      << "                    h0[j] += feat2.x * W_SRC(w0_off+fb*HIDDEN_DIM+j)\n"
      << "                           + feat2.y * W_SRC(w0_off+(fb+1)*HIDDEN_DIM+j);\n"
      << "            }\n"
      << "        }\n\n";
  } else {
    o << "            float sx=px*resolution, sy=py*resolution, sz=pz*resolution;\n"
      << "            int3 bc = int3(int(floor(sx)),int(floor(sy)),int(floor(sz)));\n"
      << "            float3 frac = float3(sx-floor(sx),sy-floor(sy),sz-floor(sz));\n\n"
      << "            for (int f = 0; f < FEATURES_PER_LEVEL; f++) {\n"
      << "                float feat = 0.0f;\n"
      << "                for (int dz=0;dz<2;dz++) for (int dy=0;dy<2;dy++) for (int dx=0;dx<2;dx++) {\n"
      << "                    uint h = hash_coords(bc+int3(dx,dy,dz), table_size);\n"
      << "                    uint grid_off = uint(l)*table_size*FEATURES_PER_LEVEL+uint(h)*FEATURES_PER_LEVEL+f;\n"
      << "                    float wx=(dx==0)?(1.0f-frac.x):frac.x;\n"
      << "                    float wy=(dy==0)?(1.0f-frac.y):frac.y;\n"
      << "                    float wz=(dz==0)?(1.0f-frac.z):frac.z;\n"
      << "                    feat += wx*wy*wz*hash_grid[grid_off];\n"
      << "                }\n";
  }

  // Close the per-feature loop (only for non-vec2 paths — vec2 path already closed)
  if (spec.features_per_level != 2 || is4d || spec.encoding == KernelSpec::RMHE) {
    o << "                int feat_idx = l*FEATURES_PER_LEVEL+f;\n"
      << "                int w_row = w0_off + feat_idx*HIDDEN_DIM;\n"
      << "                for (int j = 0; j < HIDDEN_DIM; j++)\n"
      << "                    h0[j] += feat * W_SRC(w_row+j);\n"
      << "            }\n"
      << "        }\n\n";
  }

  o << "        for (int j = 0; j < HIDDEN_DIM; j++)\n"
    << "            h0[j] = max(h0[j]+W_SRC(b0_off+j), 0.0f);\n\n";

  // Probe: layer 0 activation stats
  if (spec.emit_probes) {
    o << "        for (int j = 0; j < HIDDEN_DIM; j++) {\n"
      << "            probe_act_max_0 = max(probe_act_max_0, abs(float(h0[j])));\n"
      << "            if (isnan(float(h0[j]))) probe_fwd_nan = 1.0f;\n"
      << "        }\n\n";
  }

  // Hidden layers forward
  for (int layer = 1; layer < spec.num_hidden_layers; ++layer) {
    const char* src = ((layer - 1) % 2 == 0) ? "h0" : "h1";
    const char* dst = (layer % 2 == 0) ? "h0" : "h1";
    o << "        for (int j = 0; j < HIDDEN_DIM; j++) {\n"
      << "            ACT_T acc = ACT_ZERO;\n"
      << "            for (int i = 0; i < HIDDEN_DIM; i++)\n"
      << "                acc += " << src << "[i] * W_SRC(w" << layer << "_off + i*HIDDEN_DIM+j);\n"
      << "            " << dst << "[j] = max(float(acc)+W_SRC(b" << layer << "_off+j), 0.0f);\n"
      << "        }\n\n";

    // Probe: layer N activation stats
    if (spec.emit_probes) {
      o << "        for (int j = 0; j < HIDDEN_DIM; j++) {\n"
        << "            probe_act_max_" << layer << " = max(probe_act_max_" << layer
        << ", abs(float(" << dst << "[j])));\n"
        << "            if (isnan(float(" << dst << "[j]))) probe_fwd_nan = 1.0f;\n"
        << "        }\n\n";
    }
  }

  // Output layer
  const char* last_h = ((spec.num_hidden_layers - 1) % 2 == 0) ? "h0" : "h1";
  if (spec.num_outputs == 1) {
    o << "        float sdf = W_SRC(bO_off);\n"
      << "        for (int j = 0; j < HIDDEN_DIM; j++)\n"
      << "            sdf += " << last_h << "[j] * W_SRC(wO_off+j);\n\n";
  } else {
    o << "        float outputs[NUM_OUTPUTS];\n"
      << "        for (int m = 0; m < NUM_OUTPUTS; m++) {\n"
      << "            float acc = W_SRC(bO_off+m);\n"
      << "            for (int j = 0; j < HIDDEN_DIM; j++)\n"
      << "                acc += " << last_h << "[j] * W_SRC(wO_off + j*NUM_OUTPUTS+m);\n"
      << "            outputs[m] = acc;\n"
      << "        }\n\n";
  }

  // Probe: output layer stats
  if (spec.emit_probes) {
    if (spec.num_outputs == 1) {
      o << "        probe_output_abs_max = abs(sdf);\n"
        << "        probe_output_neg_min = -sdf;\n"
        << "        if (isnan(sdf)) probe_fwd_nan = 1.0f;\n\n";
    } else {
      o << "        for (int m = 0; m < NUM_OUTPUTS; m++) {\n"
        << "            probe_output_abs_max = max(probe_output_abs_max, abs(outputs[m]));\n"
        << "            if (isnan(outputs[m])) probe_fwd_nan = 1.0f;\n"
        << "        }\n\n";
    }
  }

  // Loss
  // Loss section — mode-dependent.
  if (mode == LossMode::Internal) {
    emitLoss(o, spec);
  } else if (mode == LossMode::ExternalGrad) {
    // Read external gradient directly — no loss computation.
    o << "        // External gradient (no internal loss)\n"
      << "        float loss_scale = train_params[TMNN_TRAIN_PARAMS_IDX_LOSS_SCALE];\n"
      << "        inv_loss_scale = 1.0f / loss_scale;\n";
    if (spec.num_outputs == 1) {
      o << "        d_sdf = d_output_external[tid];\n";
    } else {
      o << "        for (int m = 0; m < NUM_OUTPUTS; m++)\n"
        << "            d_out[m] = d_output_external[tid * NUM_OUTPUTS + m];\n";
    }
    o << "        thread_loss = 0.0f;\n";
  } else { // ForwardOnly
    // Write output and return — no backward pass.
    if (spec.num_outputs == 1) {
      o << "        forward_output[tid] = sdf;\n";
    } else {
      o << "        for (int k = 0; k < " << spec.num_outputs << "; ++k)\n"
        << "            forward_output[tid * " << spec.num_outputs << " + k] = outputs[k];\n";
    }
    o << "    }\n"  // close if (tid < N)
      << "}\n";
    return o.str();
  }

  o << "        // === BACKWARD PASS ===\n";

  // Output layer backward
  if (spec.num_outputs == 1) {
    o << "        for (int j = 0; j < HIDDEN_DIM; j++) {\n"
      << "            float h_val = " << last_h << "[j];\n"
      << "            MLP_GRAD_ADD(wO_off+j, d_sdf * h_val);\n"
      << "            float relu_mask = (h_val > 0.0f) ? 1.0f : 0.0f;\n"
      << "            " << last_h << "[j] = d_sdf * W_SRC(wO_off+j) * relu_mask;\n"
      << "        }\n"
      << "        MLP_GRAD_ADD(bO_off, d_sdf);\n\n";
  } else {
    o << "        for (int j = 0; j < HIDDEN_DIM; j++) {\n"
      << "            float h_val = " << last_h << "[j];\n"
      << "            float d_pre = 0.0f;\n"
      << "            for (int m = 0; m < NUM_OUTPUTS; m++) {\n"
      << "                MLP_GRAD_ADD(wO_off + j*NUM_OUTPUTS+m, d_out[m] * h_val);\n"
      << "                d_pre += d_out[m] * W_SRC(wO_off + j*NUM_OUTPUTS+m);\n"
      << "            }\n"
      << "            float relu_mask = (h_val > 0.0f) ? 1.0f : 0.0f;\n"
      << "            " << last_h << "[j] = d_pre * relu_mask;\n"
      << "        }\n"
      << "        for (int m = 0; m < NUM_OUTPUTS; m++)\n"
      << "            MLP_GRAD_ADD(bO_off+m, d_out[m]);\n\n";
  }

  // Probe: output layer gradient stats (d_sdf or d_out is the output grad)
  if (spec.emit_probes) {
    if (spec.num_outputs == 1) {
      o << "        if (isnan(d_sdf)) probe_bwd_nan = 1.0f;\n\n";
    } else {
      o << "        for (int m = 0; m < NUM_OUTPUTS; m++)\n"
        << "            if (isnan(d_out[m])) probe_bwd_nan = 1.0f;\n\n";
    }
  }

  // Hidden layers backward
  for (int layer = spec.num_hidden_layers - 1; layer >= 1; --layer) {
    const char* d_arr = (layer % 2 == 0) ? "h0" : "h1";
    const char* h_arr = ((layer - 1) % 2 == 0) ? "h0" : "h1";

    o << "        // Layer " << layer << " backward\n"
      << "        for (int i = 0; i < HIDDEN_DIM; i++) {\n"
      << "            float h_val = " << h_arr << "[i];\n"
      << "            for (int j = 0; j < HIDDEN_DIM; j++)\n"
      << "                MLP_GRAD_ADD(w" << layer << "_off + i*HIDDEN_DIM+j, " << d_arr << "[j] * h_val);\n"
      << "            float acc = 0.0f;\n"
      << "            for (int j = 0; j < HIDDEN_DIM; j++)\n"
      << "                acc += " << d_arr << "[j] * W_SRC(w" << layer << "_off + i*HIDDEN_DIM+j);\n"
      << "            " << h_arr << "[i] = acc * ((h_val > 0.0f) ? 1.0f : 0.0f);\n"
      << "        }\n"
      << "        for (int j = 0; j < HIDDEN_DIM; j++)\n"
      << "            MLP_GRAD_ADD(b" << layer << "_off+j, " << d_arr << "[j]);\n\n";

    // Probe: per-layer gradient L2
    if (spec.emit_probes) {
      o << "        for (int j = 0; j < HIDDEN_DIM; j++) {\n"
        << "            float g = float(" << d_arr << "[j]);\n"
        << "            probe_grad_l2_" << layer << " += g * g;\n"
        << "            if (isnan(g)) probe_bwd_nan = 1.0f;\n"
        << "        }\n\n";
    }
  }

  // Layer 0 backward + hash scatter
  emitHashScatter(o, spec);

  // Probe: layer 0 gradient L2 (h0 holds d_pre after layer 0 backward)
  if (spec.emit_probes) {
    o << "\n        for (int j = 0; j < HIDDEN_DIM; j++) {\n"
      << "            float g = float(h0[j]);\n"
      << "            probe_grad_l2_0 += g * g;\n"
      << "            if (isnan(g)) probe_bwd_nan = 1.0f;\n"
      << "        }\n";
  }

  // Close the if (tid < N)
  o << "    }\n\n";

  emitLossReduction(o, spec);

  // Probe TG reduction: per-field sequential reduce via LOSS_BUF, then write to probe_partials.
  if (spec.emit_probes) {
    emitProbeReduction(o, spec);
  }

  o << "}\n";

  return o.str();
}

std::string MLPKernelEmitter::emitBackwardExternalGradKernel(
    const KernelSpec& spec) {
  return emitScalarTrainKernelImpl(spec, LossMode::ExternalGrad,
                                    "neural_sdf_train_external_grad");
}

std::string MLPKernelEmitter::emitForwardForTrainingKernel(
    const KernelSpec& spec) {
  return emitScalarTrainKernelImpl(spec, LossMode::ForwardOnly,
                                    "neural_sdf_forward_for_training");
}

} // namespace tmnn

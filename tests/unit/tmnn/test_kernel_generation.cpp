/**
 * @file test_kernel_generation.cpp
 * @brief Tests for tmnn kernel generation: KernelSpec, MLPKernelEmitter,
 *        and KernelCompiler.
 */

#include <gtest/gtest.h>

#include "tiny-metal-nn/kernels/kernel_spec.h"
#include "tiny-metal-nn/kernels/mlp_kernel_emitter.h"
#include "tiny-metal-nn/kernels/kernel_compiler.h"

using namespace tmnn;

namespace {

KernelSpec make_standard_spec() {
  KernelSpec s;
  s.input_dim = 32;
  s.hidden_dim = 64;
  s.num_hidden_layers = 2;
  s.num_outputs = 1;
  s.num_levels = 16;
  s.features_per_level = 2;
  s.log2_hashmap_size = 19;
  s.base_resolution = 16.0f;
  s.per_level_scale = 1.447f;
  s.spatial_dims = 3;
  return s;
}

} // namespace

TEST(KernelSpec, DefaultConstructValid) {
  KernelSpec s;
  EXPECT_NO_THROW(s.validate());
}

TEST(KernelSpec, MlpWeightCountMatchesFormula) {
  KernelSpec s = make_standard_spec();
  // Layer 0: 32*64 + 64 = 2112
  // Layer 1: 64*64 + 64 = 4160
  // Output:  64*1 + 1 = 65
  EXPECT_EQ(s.mlpWeightCount(), 2112 + 4160 + 65);
}

TEST(KernelSpec, HashIsDeterministic) {
  auto a = make_standard_spec();
  auto b = make_standard_spec();
  EXPECT_EQ(a.hash(), b.hash());
}

TEST(KernelSpec, HashChangesWithHiddenDim) {
  auto a = make_standard_spec();
  auto b = make_standard_spec();
  b.hidden_dim = 128;
  EXPECT_NE(a.hash(), b.hash());
}

TEST(KernelSpec, ValidateRejectsBadHiddenDim) {
  KernelSpec s = make_standard_spec();
  s.hidden_dim = 0;
  EXPECT_THROW(s.validate(), std::invalid_argument);
}

TEST(KernelSpec, ValidateRejectsSIMDWithoutAlignment) {
  KernelSpec s = make_standard_spec();
  s.hidden_dim = 63;
  s.use_simd = true;
  EXPECT_THROW(s.validate(), std::invalid_argument);
}

TEST(KernelSpec, FromConfigHeaderRoundTrips) {
  float header[8] = {16, 2, 19, 16.0f, 1.447f, 64, 2, 0};
  auto s = KernelSpec::fromConfigHeader(header, 1, 3);
  EXPECT_EQ(s.num_levels, 16);
  EXPECT_EQ(s.hidden_dim, 64);
  EXPECT_EQ(s.num_outputs, 1);
  EXPECT_EQ(s.spatial_dims, 3);
}

TEST(KernelSpec, CanUseSIMD) {
  KernelSpec s = make_standard_spec();
  EXPECT_TRUE(s.canUseSIMD());  // 64 % 8 == 0
  s.hidden_dim = 63;
  EXPECT_FALSE(s.canUseSIMD());
}

TEST(MLPKernelEmitter, SIMDEvalKernelDiffersFromScalar) {
  MLPKernelEmitter emitter;
  auto spec = make_standard_spec();
  auto scalar = emitter.emitEvalKernel(spec);
  spec.use_simd = true;
  auto simd = emitter.emitEvalKernel(spec);
  EXPECT_NE(scalar, simd);
}

TEST(MLPKernelEmitter, FP16TrainKernelDiffersFromFP32) {
  MLPKernelEmitter emitter;
  auto spec = make_standard_spec();
  auto fp32 = emitter.emitTrainKernel(spec);
  spec.use_fp16 = true;
  auto fp16 = emitter.emitTrainKernel(spec);
  EXPECT_NE(fp32, fp16);
}

TEST(KernelCompiler, ExternalGradScalarTGCacheUsesReadOnlyTGCacheAndAtomicGrad) {
  auto spec = make_standard_spec();
  auto schema = KernelCompiler::makeDefaultSchema(spec);
  extension::KernelCompileSpec compile_spec;
  compile_spec.allow_simd = false;
  compile_spec.allow_fp16 = false;
  compile_spec.allow_tg_weight_cache = true;

  KernelCompileRequest request;
  request.role = KernelRole::BackwardFromExternalGrad;
  request.spec = spec;
  request.schema = schema;
  request.compile_spec = compile_spec;

  auto result = KernelCompiler::compile(request);
  ASSERT_TRUE(result.resolved_spec.use_tg_weight_cache);
  EXPECT_NE(result.source.find("threadgroup float tg_mlp_cache[MLP_WEIGHT_COUNT];"),
            std::string::npos);
  EXPECT_NE(result.source.find("atomic_fetch_add_explicit(&grad_mlp[idx], (val), memory_order_relaxed)"),
            std::string::npos);
  EXPECT_EQ(result.source.find("threadgroup float tg_grad_mlp[MLP_WEIGHT_COUNT];"),
            std::string::npos);
  EXPECT_EQ(result.source.find("tg_grad_mlp[idx] += (val)"),
            std::string::npos);
  EXPECT_EQ(result.source.find("d_feat2.x += h0[j] * mlp[w_row0 + j];"),
            std::string::npos);
  EXPECT_EQ(result.source.find("d_feat2.y += h0[j] * mlp[w_row1 + j];"),
            std::string::npos);
  EXPECT_EQ(result.source.find("d_feat += h0[j] * mlp[w_row + j];"),
            std::string::npos);
  EXPECT_EQ(result.source.find("acc += h1[j] * mlp[w1_off + i*HIDDEN_DIM+j];"),
            std::string::npos);
}

TEST(KernelCompiler, ExternalGradMultiOutputLoadsFlatExternalGradient) {
  auto spec = make_standard_spec();
  spec.num_outputs = 32;
  auto schema = KernelCompiler::makeDefaultSchema(spec);
  extension::KernelCompileSpec compile_spec;
  compile_spec.allow_simd = false;
  compile_spec.allow_fp16 = false;

  KernelCompileRequest request;
  request.role = KernelRole::BackwardFromExternalGrad;
  request.spec = spec;
  request.schema = schema;
  request.compile_spec = compile_spec;

  auto result = KernelCompiler::compile(request);
  EXPECT_NE(result.source.find("float d_out[NUM_OUTPUTS];"),
            std::string::npos);
  EXPECT_NE(result.source.find("d_out[m] = d_output_external[tid * NUM_OUTPUTS + m];"),
            std::string::npos);
  EXPECT_EQ(result.source.find("d_sdf = d_output_external[tid];"),
            std::string::npos);
}

TEST(KernelCompiler, ExternalGradProbeKernelUsesSparseSummaryBufferSlots) {
  auto spec = make_standard_spec();
  auto schema = KernelCompiler::makeDefaultSchema(spec);
  extension::KernelCompileSpec compile_spec;
  compile_spec.allow_simd = false;
  compile_spec.allow_fp16 = false;
  compile_spec.enable_probes = true;

  KernelCompileRequest request;
  request.role = KernelRole::BackwardFromExternalGrad;
  request.spec = spec;
  request.schema = schema;
  request.compile_spec = compile_spec;

  auto result = KernelCompiler::compile(request);
  EXPECT_NE(result.source.find("device atomic_uint*       active_hash_mask [[buffer(9)]]"),
            std::string::npos);
  EXPECT_NE(result.source.find("device atomic_uint*       active_hash_summary_mask [[buffer(10)]]"),
            std::string::npos);
  EXPECT_NE(result.source.find("device const float*       mlp_weights    [[buffer(11)]]"),
            std::string::npos);
  EXPECT_NE(result.source.find("device float*             probe_partials [[buffer(12)]]"),
            std::string::npos);
}

TEST(KernelCompiler, ForwardOnlyScalarTGCacheAvoidsUnusedLossScratch) {
  auto spec = make_standard_spec();
  auto schema = KernelCompiler::makeDefaultSchema(spec);
  extension::KernelCompileSpec compile_spec;
  compile_spec.allow_simd = false;
  compile_spec.allow_fp16 = false;
  compile_spec.allow_tg_weight_cache = true;

  KernelCompileRequest request;
  request.role = KernelRole::ForwardForTraining;
  request.spec = spec;
  request.schema = schema;
  request.compile_spec = compile_spec;

  auto result = KernelCompiler::compile(request);
  ASSERT_TRUE(result.resolved_spec.use_tg_weight_cache);
  EXPECT_NE(result.source.find("threadgroup float tg_mlp_cache[MLP_WEIGHT_COUNT];"),
            std::string::npos);
  EXPECT_EQ(result.source.find("threadgroup float tg_loss[TG_SIZE];"),
            std::string::npos);
}

TEST(KernelCompiler, EvalScalarUsesDirectMlpBuffer) {
  auto spec = make_standard_spec();
  auto schema = KernelCompiler::makeDefaultSchema(spec);
  extension::KernelCompileSpec compile_spec;
  compile_spec.allow_simd = false;
  compile_spec.allow_fp16 = false;

  KernelCompileRequest request;
  request.role = KernelRole::Eval;
  request.spec = spec;
  request.schema = schema;
  request.compile_spec = compile_spec;

  auto result = KernelCompiler::compile(request);
  EXPECT_NE(result.source.find("mlp_direct"), std::string::npos);
  EXPECT_EQ(result.source.find("config_weights + 8"), std::string::npos);
}

TEST(KernelCompiler, TrainForwardBackwardUsesDirectMlpWeightsBuffer) {
  auto spec = make_standard_spec();
  auto schema = KernelCompiler::makeDefaultSchema(spec);
  extension::KernelCompileSpec compile_spec;
  compile_spec.allow_simd = false;
  compile_spec.allow_fp16 = false;

  KernelCompileRequest request;
  request.role = KernelRole::TrainForwardBackward;
  request.spec = spec;
  request.schema = schema;
  request.compile_spec = compile_spec;

  auto result = KernelCompiler::compile(request);
  EXPECT_NE(result.source.find("mlp_weights"), std::string::npos);
  EXPECT_NE(result.source.find("[[buffer(11)]]"), std::string::npos);
  EXPECT_EQ(result.source.find("config_weights + 8"), std::string::npos);
}

TEST(KernelCompiler, TrainForwardBackwardSimdCooperativeEncodeHoistsFloor) {
  auto spec = make_standard_spec();
  auto schema = KernelCompiler::makeDefaultSchema(spec);
  extension::KernelCompileSpec compile_spec;
  compile_spec.allow_simd = true;
  compile_spec.allow_fp16 = false;

  KernelCompileRequest request;
  request.role = KernelRole::TrainForwardBackward;
  request.spec = spec;
  request.schema = schema;
  request.compile_spec = compile_spec;

  auto result = KernelCompiler::compile(request);
  ASSERT_TRUE(result.resolved_spec.use_simd);
  EXPECT_NE(result.source.find("float3 scaled_floor = floor(scaled);"),
            std::string::npos);
  EXPECT_NE(result.source.find("int3 bc = int3(scaled_floor);"),
            std::string::npos);
  EXPECT_NE(result.source.find("float3 frac = scaled - scaled_floor;"),
            std::string::npos);
}

TEST(KernelCompiler, TrainForwardBackwardSimdUsesHybridScalarW0HotPath) {
  auto spec = make_standard_spec();
  auto schema = KernelCompiler::makeDefaultSchema(spec);
  extension::KernelCompileSpec compile_spec;
  compile_spec.allow_simd = true;
  compile_spec.allow_fp16 = false;

  KernelCompileRequest request;
  request.role = KernelRole::TrainForwardBackward;
  request.spec = spec;
  request.schema = schema;
  request.compile_spec = compile_spec;

  auto result = KernelCompiler::compile(request);
  ASSERT_TRUE(result.resolved_spec.use_simd);
  EXPECT_NE(result.source.find("inline void encode_hash_grid_scalar_w0_train("),
            std::string::npos);
  EXPECT_NE(result.source.find("encode_hash_grid_scalar_w0_train("),
            std::string::npos);
  EXPECT_NE(result.source.find("device atomic_uint*       active_hash_summary_mask [[buffer(9)]]"),
            std::string::npos);
  EXPECT_NE(result.source.find("active_hash_summary_mask[child_word_idx >> 5u]"),
            std::string::npos);
  EXPECT_EQ(result.source.find("features_tg[j] = act_a[j / INPUT_DIM][j % INPUT_DIM];"),
            std::string::npos);
  EXPECT_EQ(result.source.find("uint scatter_corner_grid_off["),
            std::string::npos);
}

TEST(KernelCompiler, ForwardForTrainingIgnoresProbePreference) {
  auto spec = make_standard_spec();
  auto schema = KernelCompiler::makeDefaultSchema(spec);
  extension::KernelCompileSpec compile_spec;
  compile_spec.allow_simd = false;
  compile_spec.allow_fp16 = false;
  compile_spec.enable_probes = true;

  KernelCompileRequest request;
  request.role = KernelRole::ForwardForTraining;
  request.spec = spec;
  request.schema = schema;
  request.compile_spec = compile_spec;

  auto result = KernelCompiler::compile(request);
  EXPECT_FALSE(result.resolved_spec.emit_probes);
  EXPECT_EQ(result.source.find("probe_partials"), std::string::npos);
  EXPECT_EQ(result.source.find("PROBE_STRIDE"), std::string::npos);
}

TEST(KernelCompiler, CacheProducesSameResult) {
  auto spec = make_standard_spec();
  auto schema = KernelCompiler::makeDefaultSchema(spec);
  extension::KernelCompileSpec compile_spec;

  KernelCompileRequest request;
  request.role = KernelRole::Eval;
  request.spec = spec;
  request.schema = schema;
  request.compile_spec = compile_spec;

  auto r1 = KernelCompiler::compile(request);
  auto r2 = KernelCompiler::compile(request);
  EXPECT_EQ(r1.key.hash(), r2.key.hash());
  EXPECT_EQ(r1.entry_point, r2.entry_point);
}

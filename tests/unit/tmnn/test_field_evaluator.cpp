/**
 * @file test_field_evaluator.cpp
 * @brief Pure CPU tests for the FieldEvaluator public API.
 *
 * Uses only the installed SDK header — no runtime internals.
 */

#include <gtest/gtest.h>

#include "tiny-metal-nn/evaluator.h"

#include <cmath>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Mock evaluator helpers
// ---------------------------------------------------------------------------

namespace {

/// Generic mock evaluator with configurable dimensionality.
class MockEvaluator : public tmnn::FieldEvaluator {
public:
  MockEvaluator(uint32_t in_dims, uint32_t out_dims)
      : in_dims_(in_dims), out_dims_(out_dims) {}

  uint32_t n_input_dims() const override { return in_dims_; }
  uint32_t n_output_dims() const override { return out_dims_; }

  bool evaluate(const float *positions, float *output, int N) override {
    // Mock: output[i * out_dims + j] = sum(positions[i * in_dims + k]) for all k
    for (int i = 0; i < N; ++i) {
      float sum = 0.0f;
      for (uint32_t k = 0; k < in_dims_; ++k)
        sum += positions[i * in_dims_ + k];
      for (uint32_t j = 0; j < out_dims_; ++j)
        output[i * out_dims_ + j] = sum + static_cast<float>(j);
    }
    return true;
  }

  bool evaluate_with_gradient(const float *positions, float *output,
                              float *gradients, int N) override {
    evaluate(positions, output, N);
    // Jacobian: d(output_j)/d(input_k) = 1.0 for all j,k
    const size_t grad_count = static_cast<size_t>(N) * out_dims_ * in_dims_;
    for (size_t g = 0; g < grad_count; ++g)
      gradients[g] = 1.0f;
    return true;
  }

private:
  uint32_t in_dims_;
  uint32_t out_dims_;
};

} // namespace

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(FieldEvaluator, MockEvaluator3D) {
  MockEvaluator eval(3, 1);
  EXPECT_EQ(eval.n_input_dims(), 3u);
  EXPECT_EQ(eval.n_output_dims(), 1u);

  const int N = 2;
  float positions[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float output[2] = {};
  float gradients[6] = {};

  EXPECT_TRUE(eval.evaluate(positions, output, N));
  EXPECT_FLOAT_EQ(output[0], 6.0f);  // 1+2+3
  EXPECT_FLOAT_EQ(output[1], 15.0f); // 4+5+6

  EXPECT_TRUE(eval.evaluate_with_gradient(positions, output, gradients, N));
  for (int i = 0; i < 6; ++i)
    EXPECT_FLOAT_EQ(gradients[i], 1.0f);
}

TEST(FieldEvaluator, MockEvaluator4D) {
  MockEvaluator eval(4, 1);
  EXPECT_EQ(eval.n_input_dims(), 4u);
  EXPECT_EQ(eval.n_output_dims(), 1u);

  float positions[4] = {1.0f, 2.0f, 3.0f, 0.5f};
  float output[1] = {};
  EXPECT_TRUE(eval.evaluate(positions, output, 1));
  EXPECT_FLOAT_EQ(output[0], 6.5f); // 1+2+3+0.5
}

TEST(FieldEvaluator, DimensionCheckRejects4D) {
  MockEvaluator eval(4, 1);

  // Consumer code that expects 3D input should reject 4D evaluators.
  auto consumer = [](tmnn::FieldEvaluator &e) -> bool {
    return e.n_input_dims() == 3;
  };
  EXPECT_FALSE(consumer(eval));

  // A 3D evaluator should pass.
  MockEvaluator eval3d(3, 1);
  EXPECT_TRUE(consumer(eval3d));
}

TEST(FieldEvaluator, MultiOutputEvaluator) {
  MockEvaluator eval(3, 3);
  EXPECT_EQ(eval.n_output_dims(), 3u);

  const int N = 2;
  float positions[6] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
  float output[6] = {};
  float gradients[18] = {}; // N(2) * out(3) * in(3)

  EXPECT_TRUE(eval.evaluate_with_gradient(positions, output, gradients, N));

  // Sample 0: sum=1.0, output = {1.0, 2.0, 3.0}
  EXPECT_FLOAT_EQ(output[0], 1.0f);
  EXPECT_FLOAT_EQ(output[1], 2.0f);
  EXPECT_FLOAT_EQ(output[2], 3.0f);

  // Jacobian layout: gradients[(s * out + o) * in + i]
  // All Jacobian entries should be 1.0 for this mock.
  for (int g = 0; g < 18; ++g)
    EXPECT_FLOAT_EQ(gradients[g], 1.0f);
}

TEST(FieldEvaluator, PolymorphicDispatch) {
  MockEvaluator concrete(3, 1);
  tmnn::FieldEvaluator *ptr = &concrete;

  EXPECT_EQ(ptr->n_input_dims(), 3u);
  EXPECT_EQ(ptr->n_output_dims(), 1u);

  float pos[3] = {1.0f, 1.0f, 1.0f};
  float out[1] = {};
  EXPECT_TRUE(ptr->evaluate(pos, out, 1));
  EXPECT_FLOAT_EQ(out[0], 3.0f);

  float grad[3] = {};
  EXPECT_TRUE(ptr->evaluate_with_gradient(pos, out, grad, 1));
  EXPECT_FLOAT_EQ(out[0], 3.0f);
  for (int i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(grad[i], 1.0f);
  EXPECT_FALSE(ptr->last_diagnostic().has_value());
}

TEST(FieldEvaluator, FormatDiagnosticIncludesCodeOperationAndMessage) {
  const tmnn::DiagnosticInfo info{
      .code = tmnn::DiagnosticCode::KernelCompilationFailed,
      .operation = "FieldEvaluator::evaluate",
      .message = "empty source",
  };
  EXPECT_EQ(tmnn::format_diagnostic(info),
            "KernelCompilationFailed during FieldEvaluator::evaluate: empty source");
}

/**
 * @file test_command_batch_error.mm
 * @brief GPU-conditional C2 tests for authoritative async error completion.
 */

#include "tiny-metal-nn/metal_context.h"

#include "tiny_metal_nn/runtime/metal_context_internal.h"
#include "tiny_metal_nn/runtime/command_batch_test_access.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

using namespace tmnn;

namespace {

std::string inject_async_error(void *) { return "injected async batch failure"; }

} // namespace

TEST(CommandBatch, CompletePropagatesAsyncGpuError) {
  auto ctx = MetalContext::create();
  if (!ctx->is_gpu_available())
    GTEST_SKIP() << "No GPU";

  auto &pool = detail::context_batch_pool(*ctx);
  auto batch = pool.begin_batch();
  ASSERT_NE(batch.generation, 0u);
  ASSERT_NE(pool.current_command_buffer(batch), nullptr);

  auto fence = pool.submit(batch, SubmitMode::Async);
  EXPECT_FALSE(pool.is_complete(fence));

  detail::set_command_batch_error_hook_for_testing(&inject_async_error);
  std::string message;
  try {
    pool.complete(fence);
  } catch (const std::runtime_error &e) {
    message = e.what();
  }
  detail::set_command_batch_error_hook_for_testing(nullptr);

  EXPECT_FALSE(message.empty());
  EXPECT_NE(message.find("Metal async batch failed"), std::string::npos);
  EXPECT_NE(message.find("injected async batch failure"), std::string::npos);
  EXPECT_TRUE(pool.is_complete(fence));
}

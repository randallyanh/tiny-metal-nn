#pragma once

/**
 * @file cpp_api.h
 * @brief tcnn-inspired C++ entry point for tiny-metal-nn.
 *
 * For the common config-driven workflow, tcnn users can often replace
 *   #include <tiny-cuda-nn/cpp_api.h>
 * with
 *   #include <tiny-metal-nn/cpp_api.h>
 *
 * This is not a full drop-in replacement for every tcnn module-level surface;
 * it focuses on the high-value C++ path of JSON configuration, trainer
 * creation, training, inference, and capability queries.
 *
 * This header re-exports the normal tmnn umbrella surface plus JSON/capability
 * helpers such as:
 *   - tmnn::json
 *   - tmnn::Result<T>
 *   - tmnn::create_encoding(..., json)
 *   - tmnn::create_network(..., json)
 *   - tmnn::create_loss(json)
 *   - tmnn::create_optimizer(json)
 *   - tmnn::create_from_config(...)
 *   - tmnn::try_create_from_config(...)
 *   - tmnn::try_create_trainer(...)
 *   - tmnn::Trainer::try_create_evaluator()
 *   - tmnn::set_logger_hook(...)
 *   - tmnn::supports_fp16(...)
 *   - tmnn::preferred_precision(...)
 */

#include "tiny-metal-nn/tiny-metal-nn.h"
#include "tiny-metal-nn/factory_json.h"

namespace tmnn {

/// Whether the current or provided Metal device can run fp16 kernels.
[[nodiscard]] inline bool supports_fp16(const MetalContext &ctx) {
  return ctx.capabilities().supports_fp16;
}

[[nodiscard]] inline bool
supports_fp16(const std::shared_ptr<MetalContext> &ctx) {
  return ctx && supports_fp16(*ctx);
}

[[nodiscard]] inline bool supports_fp16() {
  return supports_fp16(MetalContext::create());
}

/// Preferred precision for the detected Metal device.
[[nodiscard]] inline Precision preferred_precision(const MetalContext &ctx) {
  return supports_fp16(ctx) ? Precision::F16 : Precision::F32;
}

[[nodiscard]] inline Precision
preferred_precision(const std::shared_ptr<MetalContext> &ctx) {
  return supports_fp16(ctx) ? Precision::F16 : Precision::F32;
}

[[nodiscard]] inline Precision preferred_precision() {
  return preferred_precision(MetalContext::create());
}

} // namespace tmnn

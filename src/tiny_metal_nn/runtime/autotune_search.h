#pragma once

/**
 * @file autotune_search.h
 * @brief Internal bounded-autotune search orchestration for tmnn runtimes.
 */

#include "tiny-metal-nn/metal_context.h"
#include "tiny-metal-nn/detail/network_planning.h"
#include "tiny-metal-nn/network_with_input_encoding.h"
#include "tiny-metal-nn/trainer.h"

#include <functional>
#include <memory>

namespace tmnn::detail {

using AutotuneSearchRuntimeFactory =
    std::function<std::unique_ptr<ITrainerRuntime>(
        const NetworkPlan &, const std::shared_ptr<MetalContext> &)>;
using AutotuneSearchExtraBuildCostFn =
    std::function<uint64_t(const NetworkPlan &,
                           const std::shared_ptr<MetalContext> &,
                           const std::shared_ptr<const RuntimeAuthority> &)>;

[[nodiscard]] NetworkPlan run_bounded_autotune_search(
    const NetworkWithInputEncoding &model, const NetworkPlan &baseline_plan,
    const NetworkFactoryOptions &factory_options,
    const std::shared_ptr<MetalContext> &ctx, int configured_batch_size,
    const char *surface,
    const AutotuneSearchRuntimeFactory &make_runtime_for_plan,
    const AutotuneSearchExtraBuildCostFn &measure_extra_build_cost = {});

} // namespace tmnn::detail

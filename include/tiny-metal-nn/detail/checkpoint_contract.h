#pragma once

/**
 * @file checkpoint_contract.h
 * @brief Public optimizer-state blob contract constants for tmnn runtimes.
 */

#include <cstdint>

namespace tmnn {

/// Canonical product transport version for OptimizerStateBlob.
/// Readers require an exact version match.
inline constexpr uint32_t kOptimizerStateBlobVersion = 1;

} // namespace tmnn

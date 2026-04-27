#pragma once

/**
 * @file extension/schema.h
 * @brief TrainParamsLayout + ExtensionSchema — SDK surface for extension
 *        authors describing their training parameter layout and I/O schema.
 *
 * Promoted from src/tiny_metal_nn/runtime/parameter_store.h so that
 * extension code can depend on installed SDK headers only.
 *
 * Fully header-only: validate() and fill_train_params() are inline so
 * extensions never need to link tiny_metal_nn_runtime.
 *
 * Dependencies: <cstdint>, <cstring>, <stdexcept>, <string>.
 */

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace tmnn {

/// Frozen constant: train_params buffer size (legacy default).
static constexpr uint32_t kTrainParamFloats = 8;

/// Layout schema for the train_params buffer.
/// Allows DNL (5 floats), RMHE (4 floats), or custom layouts.
struct TrainParamsLayout {
  uint32_t float_count = kTrainParamFloats; ///< Total floats in buffer.
  uint32_t idx_n = 0;                       ///< Index of batch-N field.
  uint32_t idx_unsigned_mode = 1;           ///< Index of unsigned_mode flag.
  uint32_t idx_loss_scale = 2;              ///< Index of loss_scale field.
  uint32_t idx_num_active_levels = 3;       ///< Index of active levels count.

  /// Validate layout consistency. Throws on error.
  inline void validate() const {
    if (float_count < 1)
      throw std::invalid_argument("TrainParamsLayout: float_count must be >= 1");

    const uint32_t indices[] = {idx_n, idx_unsigned_mode, idx_loss_scale,
                                idx_num_active_levels};
    for (auto idx : indices) {
      if (idx >= float_count)
        throw std::out_of_range(
            "TrainParamsLayout: index " + std::to_string(idx) +
            " >= float_count " + std::to_string(float_count));
    }

    // Check uniqueness without <set> to keep includes minimal.
    for (int i = 0; i < 4; ++i)
      for (int j = i + 1; j < 4; ++j)
        if (indices[i] == indices[j])
          throw std::invalid_argument(
              "TrainParamsLayout: all 4 indices must be distinct");
  }
};

/// Fill a train_params buffer according to a TrainParamsLayout.
inline void fill_train_params(float *dst, const TrainParamsLayout &layout,
                              uint32_t N, bool unsigned_mode, float loss_scale,
                              uint32_t num_active_levels) {
#ifndef NDEBUG
  layout.validate();
#endif
  std::memset(dst, 0, layout.float_count * sizeof(float));
  dst[layout.idx_n] = static_cast<float>(N);
  dst[layout.idx_unsigned_mode] = unsigned_mode ? 1.0f : 0.0f;
  dst[layout.idx_loss_scale] = loss_scale;
  dst[layout.idx_num_active_levels] = static_cast<float>(num_active_levels);
}

namespace extension {

/// Extension I/O schema describing dimensionality and parameter layout.
struct ExtensionSchema {
  uint32_t input_dims = 3;          ///< Input dimensionality (e.g. 3 for spatial).
  uint32_t target_dims = 1;         ///< Target/output dimensionality (e.g. 1 for SDF).
  uint32_t reduction_terms = 1;     ///< Number of loss reduction terms per sample.
  TrainParamsLayout train_params_layout; ///< Layout for the train_params buffer.
  uint32_t config_tail_floats = 0;  ///< Extra config floats appended after standard header.

  /// Validate schema consistency. Throws on error.
  inline void validate() const {
    if (input_dims != 3 && input_dims != 4)
      throw std::invalid_argument(
          "ExtensionSchema: input_dims must be 3 or 4");
    if (target_dims < 1)
      throw std::invalid_argument(
          "ExtensionSchema: target_dims must be >= 1");
    if (reduction_terms < 1)
      throw std::invalid_argument(
          "ExtensionSchema: reduction_terms must be >= 1");
    train_params_layout.validate();
  }
};

} // namespace extension

} // namespace tmnn

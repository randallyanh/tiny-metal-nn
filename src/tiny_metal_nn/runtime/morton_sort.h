#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace tmnn::detail {

inline uint32_t expand_bits_3d(uint32_t v) {
  v = (v | (v << 16)) & 0x030000FFu;
  v = (v | (v << 8)) & 0x0300F00Fu;
  v = (v | (v << 4)) & 0x030C30C3u;
  v = (v | (v << 2)) & 0x09249249u;
  return v;
}

inline uint32_t morton_code_3d(float x, float y, float z) {
  const auto clamp = [](float v) -> uint32_t {
    const float mapped = (v * 0.5f + 0.5f) * 1024.0f;
    return static_cast<uint32_t>(
        std::fmin(std::fmax(mapped, 0.0f), 1023.0f));
  };

  const uint32_t ix = clamp(x);
  const uint32_t iy = clamp(y);
  const uint32_t iz = clamp(z);
  return (expand_bits_3d(iz) << 2) | (expand_bits_3d(iy) << 1) |
         expand_bits_3d(ix);
}

inline uint32_t expand_bits_4d(uint32_t v) {
  v &= 0xFFu;
  v = (v | (v << 16)) & 0x000F000Fu;
  v = (v | (v << 8)) & 0x00C30C03u;
  v = (v | (v << 4)) & 0x09090909u;
  v = (v | (v << 2)) & 0x11111111u;
  return v;
}

inline uint32_t morton_code_4d(float x, float y, float z, float w) {
  const auto clamp = [](float v) -> uint32_t {
    const float mapped = (v * 0.5f + 0.5f) * 256.0f;
    return static_cast<uint32_t>(
        std::fmin(std::fmax(mapped, 0.0f), 255.0f));
  };

  const uint32_t ix = clamp(x);
  const uint32_t iy = clamp(y);
  const uint32_t iz = clamp(z);
  const uint32_t iw = clamp(w);
  return (expand_bits_4d(iw) << 3) | (expand_bits_4d(iz) << 2) |
         (expand_bits_4d(iy) << 1) | expand_bits_4d(ix);
}

inline void morton_sort_batch(const float *input, const float *target, int N,
                              uint32_t input_dims, uint32_t target_dims,
                              std::vector<uint32_t> &indices,
                              std::vector<uint32_t> &codes,
                              std::vector<float> &sorted_positions,
                              std::vector<float> &sorted_targets) {
  if (N < 0) {
    throw std::invalid_argument(
        "morton_sort_batch: batch size must not be negative");
  }
  if (input_dims != 3u && input_dims != 4u) {
    throw std::invalid_argument("morton_sort_batch: unsupported input_dims="
                                + std::to_string(input_dims)
                                + " (expected 3 or 4)");
  }

  const size_t n = static_cast<size_t>(N);
  indices.resize(n);
  codes.resize(n);
  sorted_positions.resize(n * input_dims);
  sorted_targets.resize(n * target_dims);

  if (n == 0)
    return;

  std::iota(indices.begin(), indices.end(), 0u);
  if (input_dims == 4u) {
    for (size_t i = 0; i < n; ++i) {
      codes[i] = morton_code_4d(input[i * 4], input[i * 4 + 1],
                                input[i * 4 + 2], input[i * 4 + 3]);
    }
  } else {
    for (size_t i = 0; i < n; ++i) {
      codes[i] =
          morton_code_3d(input[i * 3], input[i * 3 + 1], input[i * 3 + 2]);
    }
  }

  if (n > 1) {
    std::sort(indices.begin(), indices.end(),
              [&codes](uint32_t a, uint32_t b) { return codes[a] < codes[b]; });
  }

  for (size_t i = 0; i < n; ++i) {
    const size_t src = indices[i];
    std::copy_n(input + src * input_dims, input_dims,
                sorted_positions.data() + i * input_dims);
    std::copy_n(target + src * target_dims, target_dims,
                sorted_targets.data() + i * target_dims);
  }
}

} // namespace tmnn::detail

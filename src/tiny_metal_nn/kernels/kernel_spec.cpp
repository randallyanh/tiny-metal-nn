/**
 * @file kernel_spec.cpp
 */

#include "tiny-metal-nn/kernels/kernel_spec.h"

#include <cstring>
#include <stdexcept>

namespace tmnn {

void KernelSpec::validate() const {
  if (hidden_dim <= 0 || hidden_dim > 512)
    throw std::invalid_argument("KernelSpec: hidden_dim must be in (0, 512]");
  if (num_hidden_layers < 1 || num_hidden_layers > 8)
    throw std::invalid_argument("KernelSpec: num_hidden_layers must be in [1, 8]");
  if (num_levels < 1 || num_levels > 32)
    throw std::invalid_argument("KernelSpec: num_levels must be in [1, 32]");
  if (log2_hashmap_size < 10 || log2_hashmap_size > 24)
    throw std::invalid_argument("KernelSpec: log2_hashmap_size must be in [10, 24]");
  if (features_per_level != 1 && features_per_level != 2 &&
      features_per_level != 4)
    throw std::invalid_argument("KernelSpec: features_per_level must be 1, 2, or 4");
  if (spatial_dims != 3 && spatial_dims != 4)
    throw std::invalid_argument("KernelSpec: spatial_dims must be 3 or 4");
  if (encoding == RMHE && spatial_dims != 3)
    throw std::invalid_argument("KernelSpec: RMHE requires spatial_dims == 3");
  if (encoding == FourD && spatial_dims != 4)
    throw std::invalid_argument("KernelSpec: FourD encoding requires spatial_dims == 4");
  if (use_simd && hidden_dim % 8 != 0)
    throw std::invalid_argument("KernelSpec: SIMD requires hidden_dim % 8 == 0");
  if (use_fp16_simd && !use_simd)
    throw std::invalid_argument("KernelSpec: use_fp16_simd requires use_simd");
  if (use_fp16_simd && !use_fp16)
    throw std::invalid_argument("KernelSpec: use_fp16_simd requires use_fp16");
  if (loss < L2 || loss > Cosine)
    throw std::invalid_argument("KernelSpec: invalid loss type");
  if (loss == Huber && huber_delta <= 0.0f)
    throw std::invalid_argument(
        "KernelSpec: Huber loss requires huber_delta > 0");
  if (loss == Cosine && num_outputs < 2)
    throw std::invalid_argument(
        "KernelSpec: Cosine loss requires num_outputs >= 2");
}

KernelSpec KernelSpec::fromConfigHeader(const float header[8],
                                       uint32_t target_dims,
                                       uint32_t input_dims) {
  KernelSpec s;
  s.num_levels = static_cast<int>(header[0]);
  s.features_per_level = static_cast<int>(header[1]);
  s.log2_hashmap_size = static_cast<int>(header[2]);
  s.base_resolution = header[3];
  s.per_level_scale = header[4];
  s.hidden_dim = static_cast<int>(header[5]);
  s.num_hidden_layers = static_cast<int>(header[6]);
  s.input_dim = s.num_levels * s.features_per_level;
  s.num_outputs = static_cast<int>(target_dims);
  s.spatial_dims = static_cast<int>(input_dims);
  s.encoding = (input_dims == 4) ? FourD : Standard;
  s.validate();
  return s;
}

int KernelSpec::mlpWeightCount() const {
  int count = input_dim * hidden_dim + hidden_dim;
  for (int i = 1; i < num_hidden_layers; ++i)
    count += hidden_dim * hidden_dim + hidden_dim;
  count += hidden_dim * num_outputs + num_outputs;
  return count;
}

uint64_t KernelSpec::hash() const {
  constexpr uint64_t kFnvBasis = 14695981039346656037ULL;
  constexpr uint64_t kFnvPrime = 1099511628211ULL;

  uint64_t h = kFnvBasis;
  auto mix = [&](uint64_t val) {
    h ^= val;
    h *= kFnvPrime;
  };

  mix(static_cast<uint64_t>(input_dim));
  mix(static_cast<uint64_t>(hidden_dim));
  mix(static_cast<uint64_t>(num_hidden_layers));
  mix(static_cast<uint64_t>(num_outputs));
  mix(static_cast<uint64_t>(num_levels));
  mix(static_cast<uint64_t>(features_per_level));
  mix(static_cast<uint64_t>(log2_hashmap_size));
  mix(static_cast<uint64_t>(spatial_dims));

  uint32_t br_bits, pls_bits;
  std::memcpy(&br_bits, &base_resolution, sizeof(uint32_t));
  std::memcpy(&pls_bits, &per_level_scale, sizeof(uint32_t));
  mix(static_cast<uint64_t>(br_bits));
  mix(static_cast<uint64_t>(pls_bits));

  mix(use_fp16 ? 1ULL : 0ULL);
  mix(use_int_atomics ? 1ULL : 0ULL);
  mix(use_tg_weight_cache ? 1ULL : 0ULL);
  mix(use_simd ? 1ULL : 0ULL);
  mix(use_fp16_hash_grid ? 1ULL : 0ULL);
  mix(use_fp16_simd ? 1ULL : 0ULL);
  mix(static_cast<uint64_t>(encoding));
  mix(static_cast<uint64_t>(loss));
  if (loss == Huber) {
    uint32_t hd_bits;
    std::memcpy(&hd_bits, &huber_delta, sizeof(uint32_t));
    mix(static_cast<uint64_t>(hd_bits));
  }
  mix(static_cast<uint64_t>(activation));
  mix(emit_probes ? 1ULL : 0ULL);
  mix(emit_active_hash_mask ? 1ULL : 0ULL);

  return h;
}

} // namespace tmnn

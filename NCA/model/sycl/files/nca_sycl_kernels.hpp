#pragma once

#include <sycl/sycl.hpp>

#include <cstdlib>
#include <cstring>
#include <cstdint>

namespace nca_sycl {

constexpr std::int64_t kMetadataVersion = 4;
constexpr std::int64_t kIdFlag = 1 << 0;
constexpr std::int64_t kDiffFlag = 1 << 1;
constexpr std::int64_t kGradFlag = 1 << 2;
constexpr std::int64_t kAverageFlag = 1 << 3;
constexpr std::int64_t kLaplacianFlag = 1 << 4;
constexpr std::int64_t kIntermediateRegulariserFlag = 1 << 0;
constexpr std::int64_t kBoundaryRegulariserFlag = 1 << 1;
constexpr std::int64_t kDenseTile = 16;
constexpr std::int64_t kSpatialTileY = 8;
constexpr std::int64_t kSpatialTileX = 16;
constexpr float kGradNormEpsilon = 1.0e-12F;

inline bool SynchronizeCustomCallsEnabled() {
  const char* value = std::getenv("NCA_SYCL_SYNCHRONIZE_CUSTOM_CALLS");
  return value != nullptr && value[0] != '\0' &&
         std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 &&
         std::strcmp(value, "False") != 0;
}

inline void SynchronizeCustomCall(sycl::queue& queue) {
  if (SynchronizeCustomCallsEnabled()) queue.wait_and_throw();
}

inline float StableGradNormDenominator(float gx, float gy) {
  return sycl::sqrt(gx * gx + gy * gy + kGradNormEpsilon);
}

inline float StableGradNorm(float gx, float gy) {
  return StableGradNormDenominator(gx, gy) - sycl::sqrt(kGradNormEpsilon);
}

enum class Padding : std::int64_t {
  kZeros = 0,
  kReflect = 1,
  kReplicate = 2,
  kCircular = 3,
};

struct Shape {
  std::int64_t batch;
  std::int64_t channels;
  std::int64_t height;
  std::int64_t width;
  std::int64_t features;
  std::int64_t kernel_size;
  std::int64_t kernel_flags;
  Padding padding;
};

inline std::int64_t RoundUp(std::int64_t value, std::int64_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

inline std::int64_t MapCoordinate(std::int64_t coordinate,
                                  std::int64_t extent, Padding padding) {
  if (coordinate >= 0 && coordinate < extent) return coordinate;
  if (padding == Padding::kReplicate) {
    return coordinate < 0 ? std::int64_t{0} : extent - 1;
  }
  if (padding == Padding::kCircular) {
    return ((coordinate % extent) + extent) % extent;
  }
  if (padding == Padding::kReflect) {
    if (extent == 1) return 0;
    const std::int64_t period = 2 * (extent - 1);
    const std::int64_t folded =
        ((coordinate % period) + period) % period;
    return folded < extent ? folded : period - folded;
  }
  return -1;
}

inline std::int64_t TensorIndex(std::int64_t batch, std::int64_t channel,
                                std::int64_t y, std::int64_t x,
                                std::int64_t channels, const Shape& shape) {
  return ((batch * channels + channel) * shape.height + y) * shape.width + x;
}

inline std::int64_t FeatureCellIndex(std::int64_t cell,
                                     std::int64_t feature,
                                     std::int64_t feature_count,
                                     const Shape& shape) {
  const std::int64_t spatial_size = shape.height * shape.width;
  const std::int64_t batch = cell / spatial_size;
  const std::int64_t spatial = cell % spatial_size;
  return (batch * feature_count + feature) * spatial_size + spatial;
}

inline float StateAt(const float* state, std::int64_t batch,
                     std::int64_t channel, std::int64_t y, std::int64_t x,
                     const Shape& shape) {
  y = MapCoordinate(y, shape.height, shape.padding);
  x = MapCoordinate(x, shape.width, shape.padding);
  if (y < 0 || x < 0) return 0.0F;
  return state[TensorIndex(batch, channel, y, x, shape.channels, shape)];
}

inline float FilterAt(const float* state, const float* kernels,
                      std::int64_t batch, std::int64_t channel,
                      std::int64_t y, std::int64_t x,
                      std::int64_t kernel_index, const Shape& shape) {
  float value = 0.0F;
  if (shape.kernel_size == 3) {
#pragma unroll
    for (std::int64_t ky = 0; ky < 3; ++ky) {
#pragma unroll
      for (std::int64_t kx = 0; kx < 3; ++kx) {
        value += kernels[(kernel_index * 3 + ky) * 3 + kx] *
                 StateAt(state, batch, channel, y + ky - 1, x + kx - 1,
                         shape);
      }
    }
    return value;
  }
  const std::int64_t radius = shape.kernel_size / 2;
  for (std::int64_t ky = 0; ky < shape.kernel_size; ++ky) {
    for (std::int64_t kx = 0; kx < shape.kernel_size; ++kx) {
      value += kernels[(kernel_index * shape.kernel_size + ky) *
                           shape.kernel_size +
                       kx] *
               StateAt(state, batch, channel, y + ky - radius,
                       x + kx - radius, shape);
    }
  }
  return value;
}

template <bool Kernel3>
inline void SubmitPerceptionImpl(sycl::queue& queue, const float* state,
                                 const float* kernels, float* perception,
                                 const Shape& shape) {
  const std::int64_t radius = Kernel3 ? 1 : shape.kernel_size / 2;
  const std::int64_t local_height = kSpatialTileY + 2 * radius;
  const std::int64_t local_width = kSpatialTileX + 2 * radius;
  const std::int64_t tile_rows =
      (shape.height + kSpatialTileY - 1) / kSpatialTileY;
  const std::int64_t tile_cols =
      (shape.width + kSpatialTileX - 1) / kSpatialTileX;
  const std::int64_t group_count =
      shape.batch * shape.channels * tile_rows * tile_cols;

  queue.submit([&](sycl::handler& handler) {
    sycl::local_accessor<float, 1> state_tile(
        sycl::range<1>(local_height * local_width), handler);
    handler.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(group_count * kSpatialTileY * kSpatialTileX),
            sycl::range<1>(kSpatialTileY * kSpatialTileX)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
          const std::int64_t local = item.get_local_linear_id();
          std::int64_t group = item.get_group_linear_id();
          const std::int64_t tile_x = group % tile_cols;
          group /= tile_cols;
          const std::int64_t tile_y = group % tile_rows;
          group /= tile_rows;
          const std::int64_t channel = group % shape.channels;
          const std::int64_t batch = group / shape.channels;
          const std::int64_t origin_y = tile_y * kSpatialTileY;
          const std::int64_t origin_x = tile_x * kSpatialTileX;

          for (std::int64_t index = local;
               index < local_height * local_width;
               index += kSpatialTileY * kSpatialTileX) {
            const std::int64_t local_y = index / local_width;
            const std::int64_t local_x = index % local_width;
            state_tile[index] =
                StateAt(state, batch, channel,
                        origin_y + local_y - radius,
                        origin_x + local_x - radius, shape);
          }
          item.barrier(sycl::access::fence_space::local_space);

          const std::int64_t thread_y = local / kSpatialTileX;
          const std::int64_t thread_x = local % kSpatialTileX;
          const std::int64_t y = origin_y + thread_y;
          const std::int64_t x = origin_x + thread_x;
          if (y >= shape.height || x >= shape.width) return;

          auto local_filter = [&](std::int64_t kernel_index) {
            float value = 0.0F;
            if constexpr (Kernel3) {
#pragma unroll
              for (std::int64_t ky = 0; ky < 3; ++ky) {
#pragma unroll
                for (std::int64_t kx = 0; kx < 3; ++kx) {
                  value += kernels[(kernel_index * 3 + ky) * 3 + kx] *
                           state_tile[(thread_y + ky) * local_width +
                                      thread_x + kx];
                }
              }
            } else {
              for (std::int64_t ky = 0; ky < shape.kernel_size; ++ky) {
                for (std::int64_t kx = 0; kx < shape.kernel_size; ++kx) {
                  value += kernels[(kernel_index * shape.kernel_size + ky) *
                                       shape.kernel_size +
                                   kx] *
                           state_tile[(thread_y + ky) * local_width +
                                      thread_x + kx];
                }
              }
            }
            return value;
          };

          const std::int64_t spatial = y * shape.width + x;
          const std::int64_t spatial_size = shape.height * shape.width;
          const std::int64_t cell = batch * spatial_size + spatial;
          std::int64_t feature = 0;
          auto store = [&](float value) {
            perception[cell * shape.features + feature + channel] = value;
            feature += shape.channels;
          };
          if (shape.kernel_flags & kIdFlag) {
            store(state_tile[(thread_y + radius) * local_width + thread_x +
                             radius]);
          }
          if (shape.kernel_flags & kDiffFlag) {
            const float gx = local_filter(0);
            const float gy = local_filter(1);
            store(StableGradNorm(gx, gy));
          }
          if (shape.kernel_flags & kGradFlag) {
            store(local_filter(0));
            store(local_filter(1));
          }
          if (shape.kernel_flags & kAverageFlag) store(local_filter(2));
          if (shape.kernel_flags & kLaplacianFlag) store(local_filter(3));
        });
  });
}

inline void SubmitPerception(sycl::queue& queue, const float* state,
                             const float* kernels, float* perception,
                             const Shape& shape) {
  if (shape.kernel_size == 3) {
    SubmitPerceptionImpl<true>(queue, state, kernels, perception, shape);
  } else {
    SubmitPerceptionImpl<false>(queue, state, kernels, perception, shape);
  }
}

#if 0  // Superseded by oneMKL XMX GEMMs; retained temporarily for comparison.
template <bool FeatureAligned>
inline void SubmitHiddenImpl(sycl::queue& queue, const float* perception,
                             const float* weight_hidden, float* hidden,
                             const Shape& shape) {
  const std::int64_t cells = shape.batch * shape.height * shape.width;
  queue.submit([&](sycl::handler& handler) {
    sycl::local_accessor<float, 2> weights(
        sycl::range<2>(kDenseTile, kDenseTile), handler);
    sycl::local_accessor<float, 2> inputs(
        sycl::range<2>(kDenseTile, kDenseTile), handler);
    handler.parallel_for(
        sycl::nd_range<2>(
            sycl::range<2>(RoundUp(shape.features, kDenseTile),
                           RoundUp(cells, kDenseTile)),
            sycl::range<2>(kDenseTile, kDenseTile)),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
          const std::int64_t row = item.get_global_id(0);
          const std::int64_t cell = item.get_global_id(1);
          const std::int64_t local_row = item.get_local_id(0);
          const std::int64_t local_col = item.get_local_id(1);
          float value = 0.0F;
          for (std::int64_t start = 0; start < shape.features;
               start += kDenseTile) {
            const std::int64_t weight_input = start + local_col;
            weights[local_row][local_col] =
                (FeatureAligned || row < shape.features) &&
                        (FeatureAligned || weight_input < shape.features)
                    ? weight_hidden[row * shape.features + weight_input]
                    : 0.0F;
            const std::int64_t input_feature = start + local_row;
            inputs[local_row][local_col] =
                (FeatureAligned || input_feature < shape.features) &&
                        cell < cells
                    ? perception[FeatureCellIndex(cell, input_feature,
                                                  shape.features, shape)]
                    : 0.0F;
            item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
            for (std::int64_t k = 0; k < kDenseTile; ++k) {
              value += weights[local_row][k] * inputs[k][local_col];
            }
            item.barrier(sycl::access::fence_space::local_space);
          }
          if ((FeatureAligned || row < shape.features) && cell < cells) {
            hidden[FeatureCellIndex(cell, row, shape.features, shape)] =
                value > 0.0F ? value : 0.0F;
          }
        });
  });
}

inline void SubmitHidden(sycl::queue& queue, const float* perception,
                         const float* weight_hidden, float* hidden,
                         const Shape& shape) {
  if (shape.features % kDenseTile == 0) {
    SubmitHiddenImpl<true>(queue, perception, weight_hidden, hidden, shape);
  } else {
    SubmitHiddenImpl<false>(queue, perception, weight_hidden, hidden, shape);
  }
}

template <bool ChannelAligned, bool FeatureAligned>
inline void SubmitOutputImpl(sycl::queue& queue, const float* state,
                             const float* hidden,
                             const float* weight_output,
                             const float* bias_output,
                             const float* update_mask, float* output,
                             const Shape& shape) {
  const std::int64_t cells = shape.batch * shape.height * shape.width;
  queue.submit([&](sycl::handler& handler) {
    sycl::local_accessor<float, 2> weights(
        sycl::range<2>(kDenseTile, kDenseTile), handler);
    sycl::local_accessor<float, 2> inputs(
        sycl::range<2>(kDenseTile, kDenseTile), handler);
    handler.parallel_for(
        sycl::nd_range<2>(
            sycl::range<2>(RoundUp(shape.channels, kDenseTile),
                           RoundUp(cells, kDenseTile)),
            sycl::range<2>(kDenseTile, kDenseTile)),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
          const std::int64_t channel = item.get_global_id(0);
          const std::int64_t cell = item.get_global_id(1);
          const std::int64_t local_row = item.get_local_id(0);
          const std::int64_t local_col = item.get_local_id(1);
          float value = (ChannelAligned || channel < shape.channels)
                            ? bias_output[channel]
                            : 0.0F;
          for (std::int64_t start = 0; start < shape.features;
               start += kDenseTile) {
            const std::int64_t input_feature = start + local_col;
            weights[local_row][local_col] =
                (ChannelAligned || channel < shape.channels) &&
                        (FeatureAligned || input_feature < shape.features)
                    ? weight_output[channel * shape.features + input_feature]
                    : 0.0F;
            const std::int64_t feature = start + local_row;
            inputs[local_row][local_col] =
                (FeatureAligned || feature < shape.features) && cell < cells
                    ? hidden[FeatureCellIndex(cell, feature, shape.features,
                                              shape)]
                    : 0.0F;
            item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
            for (std::int64_t k = 0; k < kDenseTile; ++k) {
              value += weights[local_row][k] * inputs[k][local_col];
            }
            item.barrier(sycl::access::fence_space::local_space);
          }
          if ((ChannelAligned || channel < shape.channels) && cell < cells) {
            const std::int64_t index =
                FeatureCellIndex(cell, channel, shape.channels, shape);
            output[index] = state[index] + update_mask[index] * value;
          }
        });
  });
}

inline void SubmitOutput(sycl::queue& queue, const float* state,
                         const float* hidden, const float* weight_output,
                         const float* bias_output, const float* update_mask,
                         float* output, const Shape& shape) {
  const bool channels_aligned = shape.channels % kDenseTile == 0;
  const bool features_aligned = shape.features % kDenseTile == 0;
  if (channels_aligned && features_aligned) {
    SubmitOutputImpl<true, true>(queue, state, hidden, weight_output,
                                 bias_output, update_mask, output, shape);
  } else if (channels_aligned) {
    SubmitOutputImpl<true, false>(queue, state, hidden, weight_output,
                                  bias_output, update_mask, output, shape);
  } else if (features_aligned) {
    SubmitOutputImpl<false, true>(queue, state, hidden, weight_output,
                                  bias_output, update_mask, output, shape);
  } else {
    SubmitOutputImpl<false, false>(queue, state, hidden, weight_output,
                                   bias_output, update_mask, output, shape);
  }
}
#endif

}  // namespace nca_sycl

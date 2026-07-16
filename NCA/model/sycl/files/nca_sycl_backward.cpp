#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {

constexpr std::int64_t kMetadataVersion = 1;
constexpr std::int64_t kIdFlag = 1 << 0;
constexpr std::int64_t kDiffFlag = 1 << 1;
constexpr std::int64_t kGradFlag = 1 << 2;
constexpr std::int64_t kAverageFlag = 1 << 3;
constexpr std::int64_t kLaplacianFlag = 1 << 4;

enum class Padding : std::int64_t {
  kZeros = 0,
  kReflect = 1,
  kReplicate = 2,
  kCircular = 3,
};

struct Metadata {
  std::int64_t version;
  std::int64_t batch;
  std::int64_t channels;
  std::int64_t height;
  std::int64_t width;
  std::int64_t features;
  std::int64_t kernel_size;
  std::int64_t kernel_flags;
  std::int64_t padding;
  std::int64_t workgroup_size;
};

static_assert(sizeof(Metadata) == 10 * sizeof(std::int64_t));

inline std::int64_t MapCoordinate(std::int64_t coordinate,
                                  std::int64_t extent,
                                  Padding padding) {
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
                                std::int64_t channels,
                                const Metadata& metadata) {
  return ((batch * channels + channel) * metadata.height + y) *
             metadata.width +
         x;
}

inline float StateAt(const float* state, std::int64_t batch,
                     std::int64_t channel, std::int64_t y, std::int64_t x,
                     const Metadata& metadata) {
  const Padding padding = static_cast<Padding>(metadata.padding);
  y = MapCoordinate(y, metadata.height, padding);
  x = MapCoordinate(x, metadata.width, padding);
  if (y < 0 || x < 0) return 0.0F;
  return state[TensorIndex(batch, channel, y, x, metadata.channels,
                           metadata)];
}

inline float FilterAt(const float* state, const float* kernels,
                      std::int64_t batch, std::int64_t channel,
                      std::int64_t y, std::int64_t x,
                      std::int64_t kernel_index,
                      const Metadata& metadata) {
  const std::int64_t radius = metadata.kernel_size / 2;
  float value = 0.0F;
  for (std::int64_t ky = 0; ky < metadata.kernel_size; ++ky) {
    for (std::int64_t kx = 0; kx < metadata.kernel_size; ++kx) {
      const std::int64_t kernel_offset =
          (kernel_index * metadata.kernel_size + ky) * metadata.kernel_size +
          kx;
      value += kernels[kernel_offset] *
               StateAt(state, batch, channel, y + ky - radius,
                       x + kx - radius, metadata);
    }
  }
  return value;
}

inline float PerceptionAt(const float* state, const float* kernels,
                          std::int64_t batch, std::int64_t feature,
                          std::int64_t y, std::int64_t x,
                          const Metadata& metadata) {
  if (metadata.kernel_flags & kIdFlag) {
    if (feature < metadata.channels) {
      return StateAt(state, batch, feature, y, x, metadata);
    }
    feature -= metadata.channels;
  }
  if (metadata.kernel_flags & kDiffFlag) {
    if (feature < metadata.channels) {
      const float gx = FilterAt(state, kernels, batch, feature, y, x, 0,
                                metadata);
      const float gy = FilterAt(state, kernels, batch, feature, y, x, 1,
                                metadata);
      return sycl::sqrt(gx * gx + gy * gy);
    }
    feature -= metadata.channels;
  }
  if (metadata.kernel_flags & kGradFlag) {
    if (feature < metadata.channels) {
      return FilterAt(state, kernels, batch, feature, y, x, 0, metadata);
    }
    feature -= metadata.channels;
    if (feature < metadata.channels) {
      return FilterAt(state, kernels, batch, feature, y, x, 1, metadata);
    }
    feature -= metadata.channels;
  }
  if (metadata.kernel_flags & kAverageFlag) {
    if (feature < metadata.channels) {
      return FilterAt(state, kernels, batch, feature, y, x, 2, metadata);
    }
    feature -= metadata.channels;
  }
  if (metadata.kernel_flags & kLaplacianFlag) {
    if (feature < metadata.channels) {
      return FilterAt(state, kernels, batch, feature, y, x, 3, metadata);
    }
  }
  return 0.0F;
}

inline void AtomicAdd(float* address, float value) {
  sycl::atomic_ref<float, sycl::memory_order::relaxed,
                   sycl::memory_scope::device,
                   sycl::access::address_space::global_space>(*address)
      .fetch_add(value);
}

inline void ScatterFilterGradient(float* state_gradient, const float* kernels,
                                  std::int64_t batch, std::int64_t channel,
                                  std::int64_t y, std::int64_t x,
                                  std::int64_t kernel_index, float gradient,
                                  const Metadata& metadata) {
  const Padding padding = static_cast<Padding>(metadata.padding);
  const std::int64_t radius = metadata.kernel_size / 2;
  for (std::int64_t ky = 0; ky < metadata.kernel_size; ++ky) {
    for (std::int64_t kx = 0; kx < metadata.kernel_size; ++kx) {
      std::int64_t input_y =
          MapCoordinate(y + ky - radius, metadata.height, padding);
      std::int64_t input_x =
          MapCoordinate(x + kx - radius, metadata.width, padding);
      if (input_y < 0 || input_x < 0) continue;
      const std::int64_t kernel_offset =
          (kernel_index * metadata.kernel_size + ky) * metadata.kernel_size +
          kx;
      const std::int64_t input_index =
          TensorIndex(batch, channel, input_y, input_x, metadata.channels,
                      metadata);
      AtomicAdd(state_gradient + input_index,
                gradient * kernels[kernel_offset]);
    }
  }
}

bool ValidMetadata(const Metadata& metadata) {
  return metadata.version == kMetadataVersion && metadata.batch > 0 &&
         metadata.channels > 0 && metadata.height > 0 && metadata.width > 0 &&
         metadata.features > 0 && metadata.features <= 256 &&
         metadata.kernel_size > 0 && metadata.kernel_size % 2 == 1 &&
         metadata.workgroup_size >=
             std::max(metadata.features, metadata.channels);
}

}  // namespace

// Operands: state, kernels, W0, W1, bias, mask, output cotangent.
// Results: d_state, d_W0, d_W1, d_bias, perception scratch, hidden scratch.
extern "C" void nca_sycl_backward(sycl::queue* queue, void** buffers,
                                  const char* opaque,
                                  std::size_t opaque_len) {
  if (queue == nullptr || buffers == nullptr || opaque == nullptr ||
      opaque_len != sizeof(Metadata)) {
    return;
  }

  Metadata metadata{};
  std::memcpy(&metadata, opaque, sizeof(metadata));
  if (!ValidMetadata(metadata)) return;

  const auto* state = static_cast<const float*>(buffers[0]);
  const auto* kernels = static_cast<const float*>(buffers[1]);
  const auto* weight_hidden = static_cast<const float*>(buffers[2]);
  const auto* weight_output = static_cast<const float*>(buffers[3]);
  const auto* update_mask = static_cast<const float*>(buffers[5]);
  const auto* output_cotangent = static_cast<const float*>(buffers[6]);
  auto* state_gradient = static_cast<float*>(buffers[7]);
  auto* hidden_weight_gradient = static_cast<float*>(buffers[8]);
  auto* output_weight_gradient = static_cast<float*>(buffers[9]);
  auto* bias_gradient = static_cast<float*>(buffers[10]);
  auto* perception = static_cast<float*>(buffers[11]);
  auto* hidden = static_cast<float*>(buffers[12]);

  const std::int64_t spatial_size = metadata.height * metadata.width;
  const std::int64_t cell_count = metadata.batch * spatial_size;
  const std::size_t local_size =
      static_cast<std::size_t>(metadata.workgroup_size);

  // Recompute perception and activated hidden values. These are stored in
  // XLA-allocated result buffers and overwritten by their cotangents later.
  queue->submit([&](sycl::handler& handler) {
    sycl::local_accessor<float, 1> local_perception(
        sycl::range<1>(metadata.features), handler);
    handler.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(cell_count * local_size),
                          sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item) {
          const std::int64_t feature = item.get_local_linear_id();
          const std::int64_t cell = item.get_group_linear_id();
          const std::int64_t batch = cell / spatial_size;
          const std::int64_t spatial = cell % spatial_size;
          const std::int64_t y = spatial / metadata.width;
          const std::int64_t x = spatial % metadata.width;
          if (feature < metadata.features) {
            local_perception[feature] = PerceptionAt(
                state, kernels, batch, feature, y, x, metadata);
          }
          item.barrier(sycl::access::fence_space::local_space);
          if (feature < metadata.features) {
            const std::int64_t index = TensorIndex(
                batch, feature, y, x, metadata.features, metadata);
            const float feature_value = local_perception[feature];
            perception[index] = feature_value;
            float preactivation = 0.0F;
            for (std::int64_t input = 0; input < metadata.features; ++input) {
              preactivation +=
                  weight_hidden[feature * metadata.features + input] *
                  local_perception[input];
            }
            hidden[index] = preactivation > 0.0F ? preactivation : 0.0F;
          }
        });
  });

  // Reduce output-layer weight and bias gradients across all cells.
  const std::int64_t output_reduction_groups =
      metadata.channels * metadata.features + metadata.channels;
  queue->submit([&](sycl::handler& handler) {
    sycl::local_accessor<float, 1> reduction(sycl::range<1>(local_size),
                                              handler);
    handler.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(output_reduction_groups * local_size),
            sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item) {
          const std::int64_t group = item.get_group_linear_id();
          const std::int64_t local = item.get_local_linear_id();
          float sum = 0.0F;
          if (group < metadata.channels * metadata.features) {
            const std::int64_t channel = group / metadata.features;
            const std::int64_t feature = group % metadata.features;
            for (std::int64_t cell = local; cell < cell_count;
                 cell += local_size) {
              const std::int64_t batch = cell / spatial_size;
              const std::int64_t spatial = cell % spatial_size;
              const std::int64_t y = spatial / metadata.width;
              const std::int64_t x = spatial % metadata.width;
              const std::int64_t state_index = TensorIndex(
                  batch, channel, y, x, metadata.channels, metadata);
              const std::int64_t hidden_index = TensorIndex(
                  batch, feature, y, x, metadata.features, metadata);
              sum += output_cotangent[state_index] * update_mask[state_index] *
                     hidden[hidden_index];
            }
          } else {
            const std::int64_t channel =
                group - metadata.channels * metadata.features;
            for (std::int64_t cell = local; cell < cell_count;
                 cell += local_size) {
              const std::int64_t batch = cell / spatial_size;
              const std::int64_t spatial = cell % spatial_size;
              const std::int64_t y = spatial / metadata.width;
              const std::int64_t x = spatial % metadata.width;
              const std::int64_t state_index = TensorIndex(
                  batch, channel, y, x, metadata.channels, metadata);
              sum += output_cotangent[state_index] * update_mask[state_index];
            }
          }
          reduction[local] = sum;
          item.barrier(sycl::access::fence_space::local_space);
          for (std::int64_t stride = local_size / 2; stride > 0; stride /= 2) {
            if (local < stride) reduction[local] += reduction[local + stride];
            item.barrier(sycl::access::fence_space::local_space);
          }
          if (local == 0) {
            if (group < metadata.channels * metadata.features) {
              output_weight_gradient[group] = reduction[0];
            } else {
              bias_gradient[group - metadata.channels * metadata.features] =
                  reduction[0];
            }
          }
        });
  });

  // Hidden cotangents overwrite the activated-hidden scratch buffer.
  queue->parallel_for(
      sycl::range<1>(cell_count * metadata.features), [=](sycl::id<1> id) {
        const std::int64_t linear = id[0];
        const std::int64_t feature = (linear / spatial_size) % metadata.features;
        const std::int64_t batch = linear / (metadata.features * spatial_size);
        const std::int64_t spatial = linear % spatial_size;
        const std::int64_t y = spatial / metadata.width;
        const std::int64_t x = spatial % metadata.width;
        float gradient = 0.0F;
        for (std::int64_t channel = 0; channel < metadata.channels; ++channel) {
          const std::int64_t state_index = TensorIndex(
              batch, channel, y, x, metadata.channels, metadata);
          gradient += output_cotangent[state_index] * update_mask[state_index] *
                      weight_output[channel * metadata.features + feature];
        }
        const std::int64_t hidden_index = TensorIndex(
            batch, feature, y, x, metadata.features, metadata);
        hidden[hidden_index] = hidden[hidden_index] > 0.0F ? gradient : 0.0F;
      });

  // Reduce hidden-layer weight gradients across all cells.
  const std::int64_t hidden_weight_count =
      metadata.features * metadata.features;
  queue->submit([&](sycl::handler& handler) {
    sycl::local_accessor<float, 1> reduction(sycl::range<1>(local_size),
                                              handler);
    handler.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(hidden_weight_count * local_size),
                          sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item) {
          const std::int64_t weight = item.get_group_linear_id();
          const std::int64_t local = item.get_local_linear_id();
          const std::int64_t output_feature = weight / metadata.features;
          const std::int64_t input_feature = weight % metadata.features;
          float sum = 0.0F;
          for (std::int64_t cell = local; cell < cell_count;
               cell += local_size) {
            const std::int64_t batch = cell / spatial_size;
            const std::int64_t spatial = cell % spatial_size;
            const std::int64_t y = spatial / metadata.width;
            const std::int64_t x = spatial % metadata.width;
            sum += hidden[TensorIndex(batch, output_feature, y, x,
                                      metadata.features, metadata)] *
                   perception[TensorIndex(batch, input_feature, y, x,
                                          metadata.features, metadata)];
          }
          reduction[local] = sum;
          item.barrier(sycl::access::fence_space::local_space);
          for (std::int64_t stride = local_size / 2; stride > 0; stride /= 2) {
            if (local < stride) reduction[local] += reduction[local + stride];
            item.barrier(sycl::access::fence_space::local_space);
          }
          if (local == 0) hidden_weight_gradient[weight] = reduction[0];
        });
  });

  // Perception cotangents overwrite the perception scratch buffer.
  queue->parallel_for(
      sycl::range<1>(cell_count * metadata.features), [=](sycl::id<1> id) {
        const std::int64_t linear = id[0];
        const std::int64_t input_feature =
            (linear / spatial_size) % metadata.features;
        const std::int64_t batch = linear / (metadata.features * spatial_size);
        const std::int64_t spatial = linear % spatial_size;
        const std::int64_t y = spatial / metadata.width;
        const std::int64_t x = spatial % metadata.width;
        float gradient = 0.0F;
        for (std::int64_t output_feature = 0;
             output_feature < metadata.features; ++output_feature) {
          gradient +=
              weight_hidden[output_feature * metadata.features + input_feature] *
              hidden[TensorIndex(batch, output_feature, y, x,
                                 metadata.features, metadata)];
        }
        perception[TensorIndex(batch, input_feature, y, x, metadata.features,
                               metadata)] = gradient;
      });

  // Residual connection contributes the output cotangent directly.
  const std::int64_t state_element_count =
      metadata.batch * metadata.channels * spatial_size;
  queue->parallel_for(sycl::range<1>(state_element_count),
                      [=](sycl::id<1> id) {
                        state_gradient[id[0]] = output_cotangent[id[0]];
                      });

  // Transpose the perception operation. Atomics correctly accumulate padding
  // aliases and contributions from overlapping stencil positions.
  queue->parallel_for(
      sycl::range<1>(cell_count * metadata.channels), [=](sycl::id<1> id) {
        const std::int64_t linear = id[0];
        const std::int64_t channel = (linear / spatial_size) % metadata.channels;
        const std::int64_t batch = linear / (metadata.channels * spatial_size);
        const std::int64_t spatial = linear % spatial_size;
        const std::int64_t y = spatial / metadata.width;
        const std::int64_t x = spatial % metadata.width;
        std::int64_t feature_offset = 0;

        auto perception_gradient = [=](std::int64_t feature) {
          return perception[TensorIndex(batch, feature, y, x,
                                        metadata.features, metadata)];
        };

        if (metadata.kernel_flags & kIdFlag) {
          const std::int64_t state_index = TensorIndex(
              batch, channel, y, x, metadata.channels, metadata);
          AtomicAdd(state_gradient + state_index,
                    perception_gradient(feature_offset + channel));
          feature_offset += metadata.channels;
        }
        if (metadata.kernel_flags & kDiffFlag) {
          const float gx = FilterAt(state, kernels, batch, channel, y, x, 0,
                                    metadata);
          const float gy = FilterAt(state, kernels, batch, channel, y, x, 1,
                                    metadata);
          const float norm = sycl::sqrt(gx * gx + gy * gy);
          const float norm_gradient =
              perception_gradient(feature_offset + channel);
          if (norm > 0.0F) {
            ScatterFilterGradient(state_gradient, kernels, batch, channel, y,
                                  x, 0, norm_gradient * gx / norm, metadata);
            ScatterFilterGradient(state_gradient, kernels, batch, channel, y,
                                  x, 1, norm_gradient * gy / norm, metadata);
          }
          feature_offset += metadata.channels;
        }
        if (metadata.kernel_flags & kGradFlag) {
          ScatterFilterGradient(
              state_gradient, kernels, batch, channel, y, x, 0,
              perception_gradient(feature_offset + channel), metadata);
          feature_offset += metadata.channels;
          ScatterFilterGradient(
              state_gradient, kernels, batch, channel, y, x, 1,
              perception_gradient(feature_offset + channel), metadata);
          feature_offset += metadata.channels;
        }
        if (metadata.kernel_flags & kAverageFlag) {
          ScatterFilterGradient(
              state_gradient, kernels, batch, channel, y, x, 2,
              perception_gradient(feature_offset + channel), metadata);
          feature_offset += metadata.channels;
        }
        if (metadata.kernel_flags & kLaplacianFlag) {
          ScatterFilterGradient(
              state_gradient, kernels, batch, channel, y, x, 3,
              perception_gradient(feature_offset + channel), metadata);
        }
      });
}

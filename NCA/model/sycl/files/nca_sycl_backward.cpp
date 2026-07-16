#include "nca_sycl_kernels.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {

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
  std::int64_t per_example_weights;
};

static_assert(sizeof(Metadata) == 11 * sizeof(std::int64_t));

bool ValidMetadata(const Metadata& metadata) {
  return metadata.version == nca_sycl::kMetadataVersion &&
         metadata.batch > 0 && metadata.channels > 0 &&
         metadata.height > 0 && metadata.width > 0 && metadata.features > 0 &&
         metadata.features <= 256 && metadata.kernel_size > 0 &&
         metadata.kernel_size % 2 == 1 &&
         (metadata.per_example_weights == 0 ||
          metadata.per_example_weights == 1) &&
         metadata.workgroup_size >=
             std::max(metadata.features, metadata.channels);
}

inline float DeltaGradient(const float* output_cotangent,
                           const float* update_mask, std::int64_t cell,
                           std::int64_t channel,
                           const nca_sycl::Shape& shape) {
  const std::int64_t index =
      nca_sycl::FeatureCellIndex(cell, channel, shape.channels, shape);
  return output_cotangent[index] * update_mask[index];
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
                                  const nca_sycl::Shape& shape) {
  const std::int64_t radius = shape.kernel_size / 2;
  for (std::int64_t ky = 0; ky < shape.kernel_size; ++ky) {
    for (std::int64_t kx = 0; kx < shape.kernel_size; ++kx) {
      const std::int64_t input_y = nca_sycl::MapCoordinate(
          y + ky - radius, shape.height, shape.padding);
      const std::int64_t input_x = nca_sycl::MapCoordinate(
          x + kx - radius, shape.width, shape.padding);
      if (input_y < 0 || input_x < 0) continue;
      const std::int64_t kernel_offset =
          (kernel_index * shape.kernel_size + ky) * shape.kernel_size + kx;
      const std::int64_t input_index = nca_sycl::TensorIndex(
          batch, channel, input_y, input_x, shape.channels, shape);
      AtomicAdd(state_gradient + input_index,
                gradient * kernels[kernel_offset]);
    }
  }
}

void SubmitOutputWeightAndBiasGradients(
    sycl::queue& queue, const float* hidden, const float* update_mask,
    const float* output_cotangent, float* output_weight_gradient,
    float* bias_gradient, const nca_sycl::Shape& shape,
    bool per_example_weights, std::size_t reduction_local_size) {
  const std::int64_t spatial_size = shape.height * shape.width;
  const std::int64_t cells = shape.batch * spatial_size;
  const std::int64_t gradient_batches =
      per_example_weights ? shape.batch : 1;

  // dW1 = dDelta @ hidden^T, with every 16x16 output tile sharing both
  // operands. This replaces one full workgroup and one spatial scan per weight.
  queue.submit([&](sycl::handler& handler) {
    sycl::local_accessor<float, 2> left(
        sycl::range<2>(nca_sycl::kDenseTile, nca_sycl::kDenseTile), handler);
    sycl::local_accessor<float, 2> right(
        sycl::range<2>(nca_sycl::kDenseTile, nca_sycl::kDenseTile), handler);
    handler.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(
                gradient_batches,
                nca_sycl::RoundUp(shape.channels, nca_sycl::kDenseTile),
                nca_sycl::RoundUp(shape.features, nca_sycl::kDenseTile)),
            sycl::range<3>(1, nca_sycl::kDenseTile,
                           nca_sycl::kDenseTile)),
        [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(16)]] {
          const std::int64_t gradient_batch = item.get_global_id(0);
          const std::int64_t channel = item.get_global_id(1);
          const std::int64_t feature = item.get_global_id(2);
          const std::int64_t local_row = item.get_local_id(1);
          const std::int64_t local_col = item.get_local_id(2);
          const std::int64_t reduction_cells =
              per_example_weights ? spatial_size : cells;
          const std::int64_t cell_offset =
              per_example_weights ? gradient_batch * spatial_size : 0;
          float value = 0.0F;
          for (std::int64_t start = 0; start < reduction_cells;
               start += nca_sycl::kDenseTile) {
            const std::int64_t left_cell = start + local_col;
            const std::int64_t right_cell = start + local_row;
            left[local_row][local_col] =
                channel < shape.channels && left_cell < reduction_cells
                    ? DeltaGradient(output_cotangent, update_mask,
                                    cell_offset + left_cell, channel, shape)
                    : 0.0F;
            right[local_row][local_col] =
                feature < shape.features && right_cell < reduction_cells
                    ? hidden[nca_sycl::FeatureCellIndex(
                          cell_offset + right_cell, feature, shape.features,
                          shape)]
                    : 0.0F;
            item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
            for (std::int64_t k = 0; k < nca_sycl::kDenseTile; ++k) {
              value += left[local_row][k] * right[k][local_col];
            }
            item.barrier(sycl::access::fence_space::local_space);
          }
          if (channel < shape.channels && feature < shape.features) {
            const std::int64_t offset =
                per_example_weights
                    ? gradient_batch * shape.channels * shape.features
                    : 0;
            output_weight_gradient[offset + channel * shape.features +
                                   feature] = value;
          }
        });
  });

  const std::int64_t bias_groups = gradient_batches * shape.channels;
  queue.submit([&](sycl::handler& handler) {
    sycl::local_accessor<float, 1> reduction(
        sycl::range<1>(reduction_local_size), handler);
    handler.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(bias_groups * reduction_local_size),
                          sycl::range<1>(reduction_local_size)),
        [=](sycl::nd_item<1> item) {
          const std::int64_t group = item.get_group_linear_id();
          const std::int64_t local = item.get_local_linear_id();
          const std::int64_t gradient_batch = group / shape.channels;
          const std::int64_t channel = group % shape.channels;
          const std::int64_t reduction_cells =
              per_example_weights ? spatial_size : cells;
          const std::int64_t cell_offset =
              per_example_weights ? gradient_batch * spatial_size : 0;
          float value = 0.0F;
          for (std::int64_t cell = local; cell < reduction_cells;
               cell += reduction_local_size) {
            value += DeltaGradient(output_cotangent, update_mask,
                                   cell_offset + cell, channel, shape);
          }
          reduction[local] = value;
          item.barrier(sycl::access::fence_space::local_space);
          for (std::int64_t stride = reduction_local_size / 2; stride > 0;
               stride /= 2) {
            if (local < stride) reduction[local] += reduction[local + stride];
            item.barrier(sycl::access::fence_space::local_space);
          }
          if (local == 0) {
            const std::int64_t offset =
                per_example_weights ? gradient_batch * shape.channels : 0;
            bias_gradient[offset + channel] = reduction[0];
          }
        });
  });
}

void SubmitHiddenGradient(sycl::queue& queue, const float* weight_output,
                          const float* update_mask,
                          const float* output_cotangent, float* hidden,
                          const nca_sycl::Shape& shape) {
  const std::int64_t cells = shape.batch * shape.height * shape.width;
  queue.submit([&](sycl::handler& handler) {
    sycl::local_accessor<float, 2> weights(
        sycl::range<2>(nca_sycl::kDenseTile, nca_sycl::kDenseTile), handler);
    sycl::local_accessor<float, 2> deltas(
        sycl::range<2>(nca_sycl::kDenseTile, nca_sycl::kDenseTile), handler);
    handler.parallel_for(
        sycl::nd_range<2>(
            sycl::range<2>(
                nca_sycl::RoundUp(shape.features, nca_sycl::kDenseTile),
                nca_sycl::RoundUp(cells, nca_sycl::kDenseTile)),
            sycl::range<2>(nca_sycl::kDenseTile,
                           nca_sycl::kDenseTile)),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
          const std::int64_t feature = item.get_global_id(0);
          const std::int64_t cell = item.get_global_id(1);
          const std::int64_t local_row = item.get_local_id(0);
          const std::int64_t local_col = item.get_local_id(1);
          float value = 0.0F;
          for (std::int64_t start = 0; start < shape.channels;
               start += nca_sycl::kDenseTile) {
            const std::int64_t channel_for_weight = start + local_col;
            weights[local_row][local_col] =
                feature < shape.features &&
                        channel_for_weight < shape.channels
                    ? weight_output[channel_for_weight * shape.features +
                                    feature]
                    : 0.0F;
            const std::int64_t channel_for_delta = start + local_row;
            deltas[local_row][local_col] =
                channel_for_delta < shape.channels && cell < cells
                    ? DeltaGradient(output_cotangent, update_mask, cell,
                                    channel_for_delta, shape)
                    : 0.0F;
            item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
            for (std::int64_t k = 0; k < nca_sycl::kDenseTile; ++k) {
              value += weights[local_row][k] * deltas[k][local_col];
            }
            item.barrier(sycl::access::fence_space::local_space);
          }
          if (feature < shape.features && cell < cells) {
            const std::int64_t index = nca_sycl::FeatureCellIndex(
                cell, feature, shape.features, shape);
            hidden[index] = hidden[index] > 0.0F ? value : 0.0F;
          }
        });
  });
}

void SubmitHiddenWeightGradients(
    sycl::queue& queue, const float* hidden_gradient, const float* perception,
    float* hidden_weight_gradient, const nca_sycl::Shape& shape,
    bool per_example_weights) {
  const std::int64_t spatial_size = shape.height * shape.width;
  const std::int64_t cells = shape.batch * spatial_size;
  const std::int64_t gradient_batches =
      per_example_weights ? shape.batch : 1;
  queue.submit([&](sycl::handler& handler) {
    sycl::local_accessor<float, 2> left(
        sycl::range<2>(nca_sycl::kDenseTile, nca_sycl::kDenseTile), handler);
    sycl::local_accessor<float, 2> right(
        sycl::range<2>(nca_sycl::kDenseTile, nca_sycl::kDenseTile), handler);
    handler.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(
                gradient_batches,
                nca_sycl::RoundUp(shape.features, nca_sycl::kDenseTile),
                nca_sycl::RoundUp(shape.features, nca_sycl::kDenseTile)),
            sycl::range<3>(1, nca_sycl::kDenseTile,
                           nca_sycl::kDenseTile)),
        [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(16)]] {
          const std::int64_t gradient_batch = item.get_global_id(0);
          const std::int64_t output_feature = item.get_global_id(1);
          const std::int64_t input_feature = item.get_global_id(2);
          const std::int64_t local_row = item.get_local_id(1);
          const std::int64_t local_col = item.get_local_id(2);
          const std::int64_t reduction_cells =
              per_example_weights ? spatial_size : cells;
          const std::int64_t cell_offset =
              per_example_weights ? gradient_batch * spatial_size : 0;
          float value = 0.0F;
          for (std::int64_t start = 0; start < reduction_cells;
               start += nca_sycl::kDenseTile) {
            const std::int64_t left_cell = start + local_col;
            const std::int64_t right_cell = start + local_row;
            left[local_row][local_col] =
                output_feature < shape.features &&
                        left_cell < reduction_cells
                    ? hidden_gradient[nca_sycl::FeatureCellIndex(
                          cell_offset + left_cell, output_feature,
                          shape.features, shape)]
                    : 0.0F;
            right[local_row][local_col] =
                input_feature < shape.features &&
                        right_cell < reduction_cells
                    ? perception[nca_sycl::FeatureCellIndex(
                          cell_offset + right_cell, input_feature,
                          shape.features, shape)]
                    : 0.0F;
            item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
            for (std::int64_t k = 0; k < nca_sycl::kDenseTile; ++k) {
              value += left[local_row][k] * right[k][local_col];
            }
            item.barrier(sycl::access::fence_space::local_space);
          }
          if (output_feature < shape.features &&
              input_feature < shape.features) {
            const std::int64_t offset =
                per_example_weights
                    ? gradient_batch * shape.features * shape.features
                    : 0;
            hidden_weight_gradient[offset + output_feature * shape.features +
                                   input_feature] = value;
          }
        });
  });
}

void SubmitPerceptionGradient(sycl::queue& queue, const float* weight_hidden,
                              const float* hidden_gradient,
                              float* perception_gradient,
                              const nca_sycl::Shape& shape) {
  const std::int64_t cells = shape.batch * shape.height * shape.width;
  queue.submit([&](sycl::handler& handler) {
    sycl::local_accessor<float, 2> weights(
        sycl::range<2>(nca_sycl::kDenseTile, nca_sycl::kDenseTile), handler);
    sycl::local_accessor<float, 2> gradients(
        sycl::range<2>(nca_sycl::kDenseTile, nca_sycl::kDenseTile), handler);
    handler.parallel_for(
        sycl::nd_range<2>(
            sycl::range<2>(
                nca_sycl::RoundUp(shape.features, nca_sycl::kDenseTile),
                nca_sycl::RoundUp(cells, nca_sycl::kDenseTile)),
            sycl::range<2>(nca_sycl::kDenseTile,
                           nca_sycl::kDenseTile)),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
          const std::int64_t input_feature = item.get_global_id(0);
          const std::int64_t cell = item.get_global_id(1);
          const std::int64_t local_row = item.get_local_id(0);
          const std::int64_t local_col = item.get_local_id(1);
          float value = 0.0F;
          for (std::int64_t start = 0; start < shape.features;
               start += nca_sycl::kDenseTile) {
            const std::int64_t output_for_weight = start + local_col;
            weights[local_row][local_col] =
                input_feature < shape.features &&
                        output_for_weight < shape.features
                    ? weight_hidden[output_for_weight * shape.features +
                                    input_feature]
                    : 0.0F;
            const std::int64_t output_for_gradient = start + local_row;
            gradients[local_row][local_col] =
                output_for_gradient < shape.features && cell < cells
                    ? hidden_gradient[nca_sycl::FeatureCellIndex(
                          cell, output_for_gradient, shape.features, shape)]
                    : 0.0F;
            item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
            for (std::int64_t k = 0; k < nca_sycl::kDenseTile; ++k) {
              value += weights[local_row][k] * gradients[k][local_col];
            }
            item.barrier(sycl::access::fence_space::local_space);
          }
          if (input_feature < shape.features && cell < cells) {
            perception_gradient[nca_sycl::FeatureCellIndex(
                cell, input_feature, shape.features, shape)] = value;
          }
        });
  });
}

void SubmitGatheredStateGradient(
    sycl::queue& queue, const float* state, const float* kernels,
    const float* perception_gradient, const float* output_cotangent,
    float* state_gradient, const nca_sycl::Shape& shape) {
  const std::int64_t spatial_size = shape.height * shape.width;
  const std::int64_t elements = shape.batch * shape.channels * spatial_size;
  queue.parallel_for(sycl::range<1>(elements), [=](sycl::id<1> id) {
    const std::int64_t linear = id[0];
    const std::int64_t spatial = linear % spatial_size;
    const std::int64_t channel =
        (linear / spatial_size) % shape.channels;
    const std::int64_t batch = linear / (shape.channels * spatial_size);
    const std::int64_t y = spatial / shape.width;
    const std::int64_t x = spatial % shape.width;
    float value = output_cotangent[linear];
    std::int64_t feature_offset = 0;

    auto gradient_at = [&](std::int64_t feature, std::int64_t yy,
                           std::int64_t xx) {
      return perception_gradient[nca_sycl::TensorIndex(
          batch, feature, yy, xx, shape.features, shape)];
    };
    if (shape.kernel_flags & nca_sycl::kIdFlag) {
      value += gradient_at(feature_offset + channel, y, x);
      feature_offset += shape.channels;
    }

    const std::int64_t radius = shape.kernel_size / 2;
    // For circular and zero padding, every stencil transpose contribution can
    // be gathered by its sole output state element. No global atomics or
    // nondeterministic accumulation are required.
    for (std::int64_t ky = 0; ky < shape.kernel_size; ++ky) {
      for (std::int64_t kx = 0; kx < shape.kernel_size; ++kx) {
        std::int64_t source_y = y - (ky - radius);
        std::int64_t source_x = x - (kx - radius);
        source_y = nca_sycl::MapCoordinate(source_y, shape.height,
                                           shape.padding);
        source_x = nca_sycl::MapCoordinate(source_x, shape.width,
                                           shape.padding);
        if (source_y < 0 || source_x < 0) continue;
        std::int64_t block = feature_offset;
        if (shape.kernel_flags & nca_sycl::kDiffFlag) {
          const float gx = nca_sycl::FilterAt(
              state, kernels, batch, channel, source_y, source_x, 0, shape);
          const float gy = nca_sycl::FilterAt(
              state, kernels, batch, channel, source_y, source_x, 1, shape);
          const float norm = sycl::sqrt(gx * gx + gy * gy);
          if (norm > 0.0F) {
            const float norm_gradient =
                gradient_at(block + channel, source_y, source_x);
            value += norm_gradient *
                     (gx * kernels[(0 * shape.kernel_size + ky) *
                                       shape.kernel_size +
                                   kx] +
                      gy * kernels[(1 * shape.kernel_size + ky) *
                                       shape.kernel_size +
                                   kx]) /
                     norm;
          }
          block += shape.channels;
        }
        if (shape.kernel_flags & nca_sycl::kGradFlag) {
          value += gradient_at(block + channel, source_y, source_x) *
                   kernels[(0 * shape.kernel_size + ky) * shape.kernel_size +
                           kx];
          block += shape.channels;
          value += gradient_at(block + channel, source_y, source_x) *
                   kernels[(1 * shape.kernel_size + ky) * shape.kernel_size +
                           kx];
          block += shape.channels;
        }
        if (shape.kernel_flags & nca_sycl::kAverageFlag) {
          value += gradient_at(block + channel, source_y, source_x) *
                   kernels[(2 * shape.kernel_size + ky) * shape.kernel_size +
                           kx];
          block += shape.channels;
        }
        if (shape.kernel_flags & nca_sycl::kLaplacianFlag) {
          value += gradient_at(block + channel, source_y, source_x) *
                   kernels[(3 * shape.kernel_size + ky) * shape.kernel_size +
                           kx];
        }
      }
    }
    state_gradient[linear] = value;
  });
}

void SubmitAtomicStateGradientFallback(
    sycl::queue& queue, const float* state, const float* kernels,
    const float* perception_gradient, const float* output_cotangent,
    float* state_gradient, const nca_sycl::Shape& shape) {
  const std::int64_t spatial_size = shape.height * shape.width;
  const std::int64_t elements = shape.batch * shape.channels * spatial_size;
  queue.parallel_for(sycl::range<1>(elements), [=](sycl::id<1> id) {
    state_gradient[id[0]] = output_cotangent[id[0]];
  });
  queue.parallel_for(sycl::range<1>(elements), [=](sycl::id<1> id) {
    const std::int64_t linear = id[0];
    const std::int64_t spatial = linear % spatial_size;
    const std::int64_t channel =
        (linear / spatial_size) % shape.channels;
    const std::int64_t batch = linear / (shape.channels * spatial_size);
    const std::int64_t y = spatial / shape.width;
    const std::int64_t x = spatial % shape.width;
    std::int64_t feature = 0;
    auto gradient = [&](std::int64_t index) {
      return perception_gradient[nca_sycl::TensorIndex(
          batch, index, y, x, shape.features, shape)];
    };
    if (shape.kernel_flags & nca_sycl::kIdFlag) {
      AtomicAdd(state_gradient + linear, gradient(feature + channel));
      feature += shape.channels;
    }
    if (shape.kernel_flags & nca_sycl::kDiffFlag) {
      const float gx = nca_sycl::FilterAt(state, kernels, batch, channel, y, x,
                                          0, shape);
      const float gy = nca_sycl::FilterAt(state, kernels, batch, channel, y, x,
                                          1, shape);
      const float norm = sycl::sqrt(gx * gx + gy * gy);
      if (norm > 0.0F) {
        const float g = gradient(feature + channel) / norm;
        ScatterFilterGradient(state_gradient, kernels, batch, channel, y, x, 0,
                              g * gx, shape);
        ScatterFilterGradient(state_gradient, kernels, batch, channel, y, x, 1,
                              g * gy, shape);
      }
      feature += shape.channels;
    }
    if (shape.kernel_flags & nca_sycl::kGradFlag) {
      ScatterFilterGradient(state_gradient, kernels, batch, channel, y, x, 0,
                            gradient(feature + channel), shape);
      feature += shape.channels;
      ScatterFilterGradient(state_gradient, kernels, batch, channel, y, x, 1,
                            gradient(feature + channel), shape);
      feature += shape.channels;
    }
    if (shape.kernel_flags & nca_sycl::kAverageFlag) {
      ScatterFilterGradient(state_gradient, kernels, batch, channel, y, x, 2,
                            gradient(feature + channel), shape);
      feature += shape.channels;
    }
    if (shape.kernel_flags & nca_sycl::kLaplacianFlag) {
      ScatterFilterGradient(state_gradient, kernels, batch, channel, y, x, 3,
                            gradient(feature + channel), shape);
    }
  });
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

  const nca_sycl::Shape shape{
      metadata.batch,       metadata.channels, metadata.height,
      metadata.width,       metadata.features, metadata.kernel_size,
      metadata.kernel_flags,
      static_cast<nca_sycl::Padding>(metadata.padding)};
  const bool per_example_weights = metadata.per_example_weights != 0;
  const std::size_t reduction_local_size =
      static_cast<std::size_t>(metadata.workgroup_size);

  nca_sycl::SubmitPerception(*queue, state, kernels, perception, shape);
  nca_sycl::SubmitHidden(*queue, perception, weight_hidden, hidden, shape);
  SubmitOutputWeightAndBiasGradients(
      *queue, hidden, update_mask, output_cotangent, output_weight_gradient,
      bias_gradient, shape, per_example_weights, reduction_local_size);
  SubmitHiddenGradient(*queue, weight_output, update_mask, output_cotangent,
                       hidden, shape);
  SubmitHiddenWeightGradients(*queue, hidden, perception,
                              hidden_weight_gradient, shape,
                              per_example_weights);
  SubmitPerceptionGradient(*queue, weight_hidden, hidden, perception, shape);

  if (shape.padding == nca_sycl::Padding::kCircular ||
      shape.padding == nca_sycl::Padding::kZeros) {
    SubmitGatheredStateGradient(*queue, state, kernels, perception,
                                output_cotangent, state_gradient, shape);
  } else {
    SubmitAtomicStateGradientFallback(*queue, state, kernels, perception,
                                      output_cotangent, state_gradient, shape);
  }
}

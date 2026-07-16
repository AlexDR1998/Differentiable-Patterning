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

}  // namespace

// Intel Extension for OpenXLA 0.7.0 legacy custom-call ABI. XLA passes its
// active in-order queue and buffers in operand-then-result order.
extern "C" void nca_sycl_forward(sycl::queue* queue, void** buffers,
                                 const char* opaque,
                                 std::size_t opaque_len) {
  if (queue == nullptr || buffers == nullptr || opaque == nullptr ||
      opaque_len != sizeof(Metadata)) {
    return;
  }

  Metadata metadata{};
  std::memcpy(&metadata, opaque, sizeof(metadata));
  if (metadata.version != kMetadataVersion || metadata.batch <= 0 ||
      metadata.channels <= 0 || metadata.height <= 0 || metadata.width <= 0 ||
      metadata.features <= 0 || metadata.features > 256 ||
      metadata.kernel_size <= 0 || metadata.kernel_size % 2 == 0 ||
      metadata.workgroup_size <
          std::max(metadata.features, metadata.channels)) {
    return;
  }

  const auto* state = static_cast<const float*>(buffers[0]);
  const auto* kernels = static_cast<const float*>(buffers[1]);
  const auto* weight_hidden = static_cast<const float*>(buffers[2]);
  const auto* weight_output = static_cast<const float*>(buffers[3]);
  const auto* bias_output = static_cast<const float*>(buffers[4]);
  const auto* update_mask = static_cast<const float*>(buffers[5]);
  auto* output = static_cast<float*>(buffers[6]);

  const std::size_t cell_count = static_cast<std::size_t>(
      metadata.batch * metadata.height * metadata.width);
  const std::size_t local_size =
      static_cast<std::size_t>(metadata.workgroup_size);

  queue->submit([&](sycl::handler& handler) {
    sycl::local_accessor<float, 1> scratch(
        sycl::range<1>(2 * metadata.features), handler);

    handler.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(cell_count * local_size),
                          sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item) {
          const std::int64_t local = item.get_local_linear_id();
          const std::int64_t cell = item.get_group_linear_id();
          const std::int64_t spatial_size = metadata.height * metadata.width;
          const std::int64_t batch_index = cell / spatial_size;
          const std::int64_t spatial_index = cell % spatial_size;
          const std::int64_t y = spatial_index / metadata.width;
          const std::int64_t x = spatial_index % metadata.width;
          const std::int64_t kernel_radius = metadata.kernel_size / 2;
          const Padding padding = static_cast<Padding>(metadata.padding);

          auto map_coordinate = [=](std::int64_t coordinate,
                                    std::int64_t extent) {
            if (coordinate >= 0 && coordinate < extent) return coordinate;
            if (padding == Padding::kReplicate) {
              return sycl::clamp<std::int64_t>(coordinate, 0, extent - 1);
            }
            if (padding == Padding::kCircular) {
              return ((coordinate % extent) + extent) % extent;
            }
            if (padding == Padding::kReflect) {
              if (extent == 1) return std::int64_t{0};
              const std::int64_t period = 2 * (extent - 1);
              std::int64_t folded = ((coordinate % period) + period) % period;
              return folded < extent ? folded : period - folded;
            }
            return std::int64_t{-1};
          };

          auto state_at = [=](std::int64_t channel, std::int64_t yy,
                              std::int64_t xx) {
            yy = map_coordinate(yy, metadata.height);
            xx = map_coordinate(xx, metadata.width);
            if (yy < 0 || xx < 0) return 0.0F;
            const std::int64_t index =
                ((batch_index * metadata.channels + channel) * metadata.height +
                 yy) *
                    metadata.width +
                xx;
            return state[index];
          };

          auto filtered = [=](std::int64_t channel,
                              std::int64_t kernel_index) {
            float value = 0.0F;
            for (std::int64_t ky = 0; ky < metadata.kernel_size; ++ky) {
              for (std::int64_t kx = 0; kx < metadata.kernel_size; ++kx) {
                const std::int64_t kernel_offset =
                    (kernel_index * metadata.kernel_size + ky) *
                        metadata.kernel_size +
                    kx;
                value += kernels[kernel_offset] *
                         state_at(channel, y + ky - kernel_radius,
                                  x + kx - kernel_radius);
              }
            }
            return value;
          };

          if (local < metadata.features) {
            std::int64_t feature = local;
            float value = 0.0F;
            bool assigned = false;

            auto take_channel_block = [&](std::int64_t flag,
                                          std::int64_t kernel_index) {
              if ((metadata.kernel_flags & flag) == 0 || assigned) return;
              if (feature < metadata.channels) {
                value = kernel_index < 0
                            ? state_at(feature, y, x)
                            : filtered(feature, kernel_index);
                assigned = true;
              } else {
                feature -= metadata.channels;
              }
            };

            take_channel_block(kIdFlag, -1);
            if ((metadata.kernel_flags & kDiffFlag) != 0 && !assigned) {
              if (feature < metadata.channels) {
                const float gx = filtered(feature, 0);
                const float gy = filtered(feature, 1);
                value = sycl::sqrt(gx * gx + gy * gy);
                assigned = true;
              } else {
                feature -= metadata.channels;
              }
            }
            take_channel_block(kGradFlag, 0);
            take_channel_block(kGradFlag, 1);
            take_channel_block(kAverageFlag, 2);
            take_channel_block(kLaplacianFlag, 3);
            scratch[local] = value;
          }
          item.barrier(sycl::access::fence_space::local_space);

          if (local < metadata.features) {
            float value = 0.0F;
            for (std::int64_t input = 0; input < metadata.features; ++input) {
              value += weight_hidden[local * metadata.features + input] *
                       scratch[input];
            }
            scratch[metadata.features + local] =
                value > 0.0F ? value : 0.0F;
          }
          item.barrier(sycl::access::fence_space::local_space);

          if (local < metadata.channels) {
            float update = bias_output[local];
            for (std::int64_t hidden = 0; hidden < metadata.features; ++hidden) {
              update += weight_output[local * metadata.features + hidden] *
                        scratch[metadata.features + hidden];
            }
            const std::int64_t index =
                ((batch_index * metadata.channels + local) * metadata.height +
                 y) *
                    metadata.width +
                x;
            output[index] = state[index] + update_mask[index] * update;
          }
        });
  });
}

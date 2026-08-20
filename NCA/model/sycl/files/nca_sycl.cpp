#include "nca_sycl_kernels.hpp"
#include "nca_sycl_onemkl.hpp"

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
  std::int64_t output_channels;
  std::int64_t kernel_size;
  std::int64_t kernel_flags;
  std::int64_t padding;
  std::int64_t workgroup_size;
  std::int64_t xmx_mode;
};

static_assert(sizeof(Metadata) == 12 * sizeof(std::int64_t));

bool ValidMetadata(const Metadata& metadata) {
  return metadata.version == nca_sycl::kMetadataVersion &&
         metadata.batch > 0 && metadata.channels > 0 &&
         metadata.height > 0 && metadata.width > 0 && metadata.features > 0 &&
         (metadata.output_channels == metadata.channels ||
          metadata.output_channels == 2 * metadata.channels) &&
         metadata.features <= 256 && metadata.kernel_size > 0 &&
         metadata.kernel_size % 2 == 1 && metadata.workgroup_size >=
             std::max(metadata.features, metadata.channels);
}

}  // namespace

// Intel Extension for OpenXLA 0.7.0 legacy custom-call ABI. Buffers are the
// six operands followed by output, perception, hidden, and delta scratch.
extern "C" void nca_sycl_forward(sycl::queue* queue, void** buffers,
                                 const char* opaque,
                                 std::size_t opaque_len) {
  if (queue == nullptr || buffers == nullptr || opaque == nullptr ||
      opaque_len != sizeof(Metadata)) {
    return;
  }

  Metadata metadata{};
  std::memcpy(&metadata, opaque, sizeof(metadata));
  if (!ValidMetadata(metadata)) return;
  nca_sycl::SerializedCustomCall serialized(*queue, "forward/serialized");

  const auto* state = static_cast<const float*>(buffers[0]);
  const auto* kernels = static_cast<const float*>(buffers[1]);
  const auto* weight_hidden = static_cast<const float*>(buffers[2]);
  const auto* weight_output = static_cast<const float*>(buffers[3]);
  const auto* bias_output = static_cast<const float*>(buffers[4]);
  const auto* update_mask = static_cast<const float*>(buffers[5]);
  auto* output = static_cast<float*>(buffers[6]);
  auto* perception = static_cast<float*>(buffers[7]);
  auto* hidden = static_cast<float*>(buffers[8]);
  auto* delta = static_cast<float*>(buffers[9]);

  const nca_sycl::Shape shape{
      metadata.batch,       metadata.channels, metadata.height,
      metadata.width,       metadata.features, metadata.kernel_size,
      metadata.kernel_flags,
      static_cast<nca_sycl::Padding>(metadata.padding)};

  const std::int64_t cells =
      metadata.batch * metadata.height * metadata.width;
  nca_sycl::SubmitPerception(*queue, state, kernels, perception, shape);
  nca_sycl::SynchronizeStage(*queue, "forward/perception");
  nca_sycl::Gemm(*queue, oneapi::mkl::transpose::nontrans,
                 oneapi::mkl::transpose::trans, cells, metadata.features,
                 metadata.features, perception, metadata.features,
                 weight_hidden, metadata.features, hidden, metadata.features,
                 metadata.xmx_mode);
  nca_sycl::SynchronizeStage(*queue, "forward/hidden_gemm");
  queue->parallel_for(sycl::range<1>(cells * metadata.features),
                      [=](sycl::id<1> id) {
                        const std::int64_t index = id[0];
                        hidden[index] =
                            hidden[index] > 0.0F ? hidden[index] : 0.0F;
                      });
  nca_sycl::SynchronizeStage(*queue, "forward/relu");
  nca_sycl::Gemm(*queue, oneapi::mkl::transpose::nontrans,
                 oneapi::mkl::transpose::trans, cells, metadata.output_channels,
                 metadata.features, hidden, metadata.features, weight_output,
                 metadata.features, delta, metadata.output_channels,
                 metadata.xmx_mode);
  nca_sycl::SynchronizeStage(*queue, "forward/output_gemm");
  // Fuse bias, fire mask, residual update, and the cell-major to NCHW layout
  // conversion into one epilogue.
  queue->parallel_for(
      sycl::range<1>(cells * metadata.channels), [=](sycl::id<1> id) {
        const std::int64_t linear = id[0];
        const std::int64_t channel = linear % metadata.channels;
        const std::int64_t cell = linear / metadata.channels;
        const std::int64_t spatial_size = metadata.height * metadata.width;
        const std::int64_t batch = cell / spatial_size;
        const std::int64_t spatial = cell % spatial_size;
        const std::int64_t state_index =
            (batch * metadata.channels + channel) * spatial_size + spatial;
        const std::int64_t value_index =
            cell * metadata.output_channels + channel;
        float update = delta[value_index] + bias_output[channel];
        if (metadata.output_channels == 2 * metadata.channels) {
          const float gate_logit =
              delta[value_index + metadata.channels] +
              bias_output[channel + metadata.channels];
          update *= 1.0F / (1.0F + sycl::exp(-gate_logit));
        }
        output[state_index] = state[state_index] +
                              update_mask[state_index] * update;
      });
  nca_sycl::SynchronizeStage(*queue, "forward/epilogue");
}

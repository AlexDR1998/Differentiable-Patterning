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
};

static_assert(sizeof(Metadata) == 10 * sizeof(std::int64_t));

bool ValidMetadata(const Metadata& metadata) {
  return metadata.version == nca_sycl::kMetadataVersion &&
         metadata.batch > 0 && metadata.channels > 0 &&
         metadata.height > 0 && metadata.width > 0 && metadata.features > 0 &&
         metadata.features <= 256 && metadata.kernel_size > 0 &&
         metadata.kernel_size % 2 == 1 && metadata.workgroup_size >=
             std::max(metadata.features, metadata.channels);
}

}  // namespace

// Intel Extension for OpenXLA 0.7.0 legacy custom-call ABI. Buffers are the
// six operands followed by output, perception scratch, and hidden scratch.
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

  const auto* state = static_cast<const float*>(buffers[0]);
  const auto* kernels = static_cast<const float*>(buffers[1]);
  const auto* weight_hidden = static_cast<const float*>(buffers[2]);
  const auto* weight_output = static_cast<const float*>(buffers[3]);
  const auto* bias_output = static_cast<const float*>(buffers[4]);
  const auto* update_mask = static_cast<const float*>(buffers[5]);
  auto* output = static_cast<float*>(buffers[6]);
  auto* perception = static_cast<float*>(buffers[7]);
  auto* hidden = static_cast<float*>(buffers[8]);

  const nca_sycl::Shape shape{
      metadata.batch,       metadata.channels, metadata.height,
      metadata.width,       metadata.features, metadata.kernel_size,
      metadata.kernel_flags,
      static_cast<nca_sycl::Padding>(metadata.padding)};

  // The state is spatially tiled through SLM. The two pointwise layers are
  // exact-FP32 16x16 tiled matrix products; dimensions that are multiples of
  // 16 avoid all edge lanes and are the intended fast path.
  nca_sycl::SubmitPerception(*queue, state, kernels, perception, shape);
  nca_sycl::SubmitHidden(*queue, perception, weight_hidden, hidden, shape);
  nca_sycl::SubmitOutput(*queue, state, hidden, weight_output, bias_output,
                         update_mask, output, shape);
}

#include "nca_sycl_kernels.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>

extern "C" void nca_sycl_forward(sycl::queue*, void**, const char*,
                                  std::size_t);

namespace {

struct RolloutMetadata {
  std::int64_t version, batch, channels, height, width, features;
  std::int64_t kernel_size, kernel_flags, padding, workgroup_size, xmx_mode;
  std::int64_t steps, boundary_code, boundary_channels;
};

struct ForwardMetadata {
  std::int64_t version, batch, channels, height, width, features;
  std::int64_t kernel_size, kernel_flags, padding, workgroup_size, xmx_mode;
};

static_assert(sizeof(RolloutMetadata) == 14 * sizeof(std::int64_t));
static_assert(sizeof(ForwardMetadata) == 11 * sizeof(std::int64_t));

bool ValidMetadata(const RolloutMetadata& m) {
  return m.version == nca_sycl::kMetadataVersion && m.batch > 0 &&
         m.channels > 0 && m.height > 0 && m.width > 0 && m.features > 0 &&
         m.steps > 0 && m.kernel_size > 0 && m.kernel_size % 2 == 1 &&
         m.boundary_code >= 0 && m.boundary_code <= 2 &&
         m.boundary_channels >= 0 && m.boundary_channels <= m.channels;
}

void ApplyBoundary(sycl::queue& queue, float* state, const float* mask,
                   const RolloutMetadata& m) {
  if (m.boundary_code == 0) return;
  const std::int64_t spatial_size = m.height * m.width;
  const std::int64_t elements = m.batch * m.channels * spatial_size;
  queue.parallel_for(sycl::range<1>(elements), [=](sycl::id<1> id) {
    const std::int64_t linear = id[0];
    const std::int64_t spatial = linear % spatial_size;
    const std::int64_t channel = (linear / spatial_size) % m.channels;
    if (m.boundary_code == 1) {
      const std::int64_t first_fixed = m.channels - m.boundary_channels;
      if (channel >= first_fixed) {
        state[linear] =
            mask[(channel - first_fixed) * spatial_size + spatial];
      }
    } else {
      state[linear] *= mask[spatial];
    }
  });
}

}  // namespace

// Operands: initial state, kernels, W0, W1, bias, [K,B,C,H,W] masks,
// boundary mask. Results: final state, [K,B,C,H,W] trajectory, perception,
// hidden, and delta scratch.
extern "C" void nca_sycl_rollout_forward(sycl::queue* queue, void** buffers,
                                          const char* opaque,
                                          std::size_t opaque_len) {
  if (queue == nullptr || buffers == nullptr || opaque == nullptr ||
      opaque_len != sizeof(RolloutMetadata)) return;
  RolloutMetadata m{};
  std::memcpy(&m, opaque, sizeof(m));
  if (!ValidMetadata(m)) return;

  auto* initial_state = static_cast<float*>(buffers[0]);
  auto* kernels = static_cast<float*>(buffers[1]);
  auto* weight_hidden = static_cast<float*>(buffers[2]);
  auto* weight_output = static_cast<float*>(buffers[3]);
  auto* bias_output = static_cast<float*>(buffers[4]);
  auto* masks = static_cast<float*>(buffers[5]);
  auto* boundary_mask = static_cast<float*>(buffers[6]);
  auto* output = static_cast<float*>(buffers[7]);
  auto* trajectory = static_cast<float*>(buffers[8]);
  auto* perception = static_cast<float*>(buffers[9]);
  auto* hidden = static_cast<float*>(buffers[10]);
  auto* delta = static_cast<float*>(buffers[11]);

  const std::int64_t state_elements =
      m.batch * m.channels * m.height * m.width;
  const ForwardMetadata forward_metadata{
      m.version, m.batch, m.channels, m.height, m.width, m.features,
      m.kernel_size, m.kernel_flags, m.padding, m.workgroup_size, m.xmx_mode};
  for (std::int64_t step = 0; step < m.steps; ++step) {
    float* step_input =
        step == 0 ? initial_state : trajectory + (step - 1) * state_elements;
    float* step_output = trajectory + step * state_elements;
    void* step_buffers[] = {
        step_input, kernels, weight_hidden, weight_output, bias_output,
        masks + step * state_elements, step_output, perception, hidden, delta};
    nca_sycl_forward(queue, step_buffers,
                     reinterpret_cast<const char*>(&forward_metadata),
                     sizeof(forward_metadata));
    ApplyBoundary(*queue, step_output, boundary_mask, m);
  }
  queue->memcpy(output, trajectory + (m.steps - 1) * state_elements,
                static_cast<std::size_t>(state_elements) * sizeof(float));
}

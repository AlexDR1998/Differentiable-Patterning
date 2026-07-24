#include "nca_sycl_kernels.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>

extern "C" void nca_sycl_backward(sycl::queue*, void**, const char*,
                                   std::size_t);

namespace {

struct RolloutMetadata {
  std::int64_t version, batch, channels, height, width, features;
  std::int64_t kernel_size, kernel_flags, padding, workgroup_size, xmx_mode;
  std::int64_t steps, boundary_code, boundary_channels, regulariser_flags;
};

struct BackwardMetadata {
  std::int64_t version, batch, channels, height, width, features;
  std::int64_t kernel_size, kernel_flags, padding, workgroup_size;
  std::int64_t per_example_weights, xmx_mode;
};

static_assert(sizeof(RolloutMetadata) == 15 * sizeof(std::int64_t));
static_assert(sizeof(BackwardMetadata) == 12 * sizeof(std::int64_t));

void ApplyBoundaryCotangent(sycl::queue& queue, const float* input,
                            const float* direct, float* output,
                            const float* state, const float* mask,
                            const float* regulariser_cotangent,
                            const RolloutMetadata& m) {
  const std::int64_t spatial_size = m.height * m.width;
  const std::int64_t elements = m.batch * m.channels * spatial_size;
  const std::int64_t boundary_channel_count =
      m.boundary_code == 1 ? m.channels - m.boundary_channels : m.channels;
  const float intermediate_scale = 1.0F / static_cast<float>(elements);
  const float boundary_scale =
      boundary_channel_count > 0
          ? 1.0F / static_cast<float>(
                         m.batch * boundary_channel_count * spatial_size)
          : 0.0F;
  queue.parallel_for(sycl::range<1>(elements), [=](sycl::id<1> id) {
    const std::int64_t linear = id[0];
    const std::int64_t spatial = linear % spatial_size;
    const std::int64_t channel = (linear / spatial_size) % m.channels;
    float value = input[linear] + direct[linear];
    const float state_value = state[linear];
    if (m.regulariser_flags & nca_sycl::kIntermediateRegulariserFlag) {
      const float derivative =
          state_value < 0.0F ? -2.0F : (state_value >= 1.0F ? 2.0F : 0.0F);
      value += regulariser_cotangent[0] * derivative * intermediate_scale;
    }
    if ((m.regulariser_flags & nca_sycl::kBoundaryRegulariserFlag) &&
        m.boundary_code != 0 && channel < boundary_channel_count) {
      float spatial_mask = mask[spatial];
      if (m.boundary_code == 1) {
        spatial_mask = 0.0F;
        for (std::int64_t mask_channel = 0;
             mask_channel < m.boundary_channels; ++mask_channel) {
          spatial_mask = sycl::fmax(
              spatial_mask,
              mask[mask_channel * spatial_size + spatial]);
        }
      }
      const float absolute_derivative = state_value < 0.0F ? -1.0F : 1.0F;
      value += regulariser_cotangent[1] * absolute_derivative *
               (1.0F - spatial_mask) * boundary_scale;
    }
    if (m.boundary_code == 1 &&
        channel >= m.channels - m.boundary_channels) {
      value = 0.0F;
    } else if (m.boundary_code == 2) {
      value *= mask[spatial];
    }
    output[linear] = value;
  });
}

void AddInPlace(sycl::queue& queue, float* destination, const float* source,
                std::int64_t elements) {
  queue.parallel_for(sycl::range<1>(elements), [=](sycl::id<1> id) {
    destination[id[0]] += source[id[0]];
  });
}

}  // namespace

// Operands: initial state, kernels, W0, W1, bias, masks, boundary mask,
// trajectory, final cotangent, trajectory cotangents, and two regulariser
// cotangents. Results: dState, accumulated parameter gradients,
// boundary/dState scratch, per-step parameter scratch, and three reusable
// backward activation buffers.
extern "C" void nca_sycl_rollout_backward(sycl::queue* queue, void** buffers,
                                           const char* opaque,
                                           std::size_t opaque_len) {
  if (queue == nullptr || buffers == nullptr || opaque == nullptr ||
      opaque_len != sizeof(RolloutMetadata)) return;
  RolloutMetadata m{};
  std::memcpy(&m, opaque, sizeof(m));
  if (m.version != nca_sycl::kMetadataVersion || m.steps <= 0) return;

  auto* initial_state = static_cast<float*>(buffers[0]);
  auto* kernels = static_cast<float*>(buffers[1]);
  auto* weight_hidden = static_cast<float*>(buffers[2]);
  auto* weight_output = static_cast<float*>(buffers[3]);
  auto* bias_output = static_cast<float*>(buffers[4]);
  auto* masks = static_cast<float*>(buffers[5]);
  auto* boundary_mask = static_cast<float*>(buffers[6]);
  auto* trajectory = static_cast<float*>(buffers[7]);
  auto* output_cotangent = static_cast<float*>(buffers[8]);
  auto* trajectory_cotangent = static_cast<float*>(buffers[9]);
  const bool compute_regularisers = m.regulariser_flags != 0;
  auto* regulariser_cotangent = compute_regularisers
                                    ? static_cast<float*>(buffers[10])
                                    : nullptr;
  const std::int64_t result_offset = compute_regularisers ? 1 : 0;
  auto* state_gradient = static_cast<float*>(buffers[10 + result_offset]);
  auto* hidden_weight_gradient =
      static_cast<float*>(buffers[11 + result_offset]);
  auto* output_weight_gradient =
      static_cast<float*>(buffers[12 + result_offset]);
  auto* bias_gradient = static_cast<float*>(buffers[13 + result_offset]);
  auto* boundary_cotangent =
      static_cast<float*>(buffers[14 + result_offset]);
  auto* rolling_state_gradient =
      static_cast<float*>(buffers[15 + result_offset]);
  auto* step_hidden_weight_gradient =
      static_cast<float*>(buffers[16 + result_offset]);
  auto* step_output_weight_gradient =
      static_cast<float*>(buffers[17 + result_offset]);
  auto* step_bias_gradient =
      static_cast<float*>(buffers[18 + result_offset]);
  auto* perception = static_cast<float*>(buffers[19 + result_offset]);
  auto* hidden = static_cast<float*>(buffers[20 + result_offset]);
  auto* hidden_gradient = static_cast<float*>(buffers[21 + result_offset]);

  const std::int64_t spatial_size = m.height * m.width;
  const std::int64_t state_elements = m.batch * m.channels * spatial_size;
  const std::int64_t hidden_weight_elements = m.features * m.features;
  const std::int64_t output_weight_elements = m.channels * m.features;
  const BackwardMetadata backward_metadata{
      m.version, m.batch, m.channels, m.height, m.width, m.features,
      m.kernel_size, m.kernel_flags, m.padding, m.workgroup_size, 0,
      m.xmx_mode};

  queue->fill(hidden_weight_gradient, 0.0F,
              static_cast<std::size_t>(hidden_weight_elements));
  queue->fill(output_weight_gradient, 0.0F,
              static_cast<std::size_t>(output_weight_elements));
  queue->fill(bias_gradient, 0.0F,
              static_cast<std::size_t>(m.channels));

  const float* current_cotangent = output_cotangent;
  for (std::int64_t step = m.steps; step-- > 0;) {
    const float* step_state =
        step == 0 ? initial_state : trajectory + (step - 1) * state_elements;
    float* next_state_gradient =
        step == 0 ? state_gradient : rolling_state_gradient;
    ApplyBoundaryCotangent(
        *queue, current_cotangent,
        trajectory_cotangent + step * state_elements, boundary_cotangent,
        trajectory + step * state_elements, boundary_mask,
        regulariser_cotangent, m);
    nca_sycl::SynchronizeStage(*queue, "rollout_backward/boundary_cotangent");
    void* step_buffers[] = {
        const_cast<float*>(step_state), kernels, weight_hidden, weight_output,
        bias_output, masks + step * state_elements, boundary_cotangent,
        next_state_gradient, step_hidden_weight_gradient,
        step_output_weight_gradient, step_bias_gradient, perception, hidden,
        hidden_gradient};
    nca_sycl_backward(queue, step_buffers,
                      reinterpret_cast<const char*>(&backward_metadata),
                      sizeof(backward_metadata));
    AddInPlace(*queue, hidden_weight_gradient, step_hidden_weight_gradient,
               hidden_weight_elements);
    AddInPlace(*queue, output_weight_gradient, step_output_weight_gradient,
               output_weight_elements);
    AddInPlace(*queue, bias_gradient, step_bias_gradient, m.channels);
    nca_sycl::SynchronizeStage(*queue, "rollout_backward/accumulate_parameters");
    current_cotangent = next_state_gradient;
  }
  nca_sycl::SynchronizeCustomCall(*queue, "rollout/backward_complete");
}

#include "nca_sycl_kernels.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>

extern "C" void nca_sycl_forward(sycl::queue*, void**, const char*,
                                  std::size_t);

namespace {

struct RolloutMetadata {
  std::int64_t version, batch, channels, height, width, features;
  std::int64_t kernel_size, kernel_flags, padding, workgroup_size, xmx_mode;
  std::int64_t steps, boundary_code, boundary_channels, regulariser_flags;
};

struct ForwardMetadata {
  std::int64_t version, batch, channels, height, width, features;
  std::int64_t kernel_size, kernel_flags, padding, workgroup_size, xmx_mode;
};

static_assert(sizeof(RolloutMetadata) == 15 * sizeof(std::int64_t));
static_assert(sizeof(ForwardMetadata) == 11 * sizeof(std::int64_t));

bool ValidMetadata(const RolloutMetadata& m) {
  return m.version == nca_sycl::kMetadataVersion && m.batch > 0 &&
         m.channels > 0 && m.height > 0 && m.width > 0 && m.features > 0 &&
         m.steps > 0 && m.kernel_size > 0 && m.kernel_size % 2 == 1 &&
         m.boundary_code >= 0 && m.boundary_code <= 2 &&
         m.boundary_channels >= 0 && m.boundary_channels <= m.channels &&
         m.regulariser_flags >= 0 && m.regulariser_flags <= 3;
}

void ReportQueueOrderingOnce(const sycl::queue& queue) {
  if (std::getenv("NCA_SYCL_REPORT_QUEUE_ORDERING") == nullptr) return;
  static const bool reported = [&queue]() {
    std::cout << "NCA_SYCL_CUSTOM_CALL_QUEUE_IN_ORDER="
              << (queue.is_in_order() ? 1 : 0) << std::endl;
    return true;
  }();
  (void)reported;
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

float SpatialMask(const float* mask, std::int64_t spatial,
                  const RolloutMetadata& m) {
  if (m.boundary_code == 2) return mask[spatial];
  float value = 0.0F;
  const std::int64_t spatial_size = m.height * m.width;
  for (std::int64_t channel = 0; channel < m.boundary_channels; ++channel) {
    value = sycl::fmax(value, mask[channel * spatial_size + spatial]);
  }
  return value;
}

template <bool ComputeIntermediate, bool ComputeBoundary>
void SubmitBoundaryAndRegularisersAtomic(sycl::queue& queue, float* state,
                                         const float* mask,
                                         float* regularisers,
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
  constexpr std::int64_t local_size = 256;
  const std::int64_t global_size =
      nca_sycl::RoundUp(elements, local_size);

  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(global_size),
                        sycl::range<1>(local_size)),
      [=](sycl::nd_item<1> item) {
        using Atomic = sycl::atomic_ref<
            float, sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space>;
        const std::int64_t linear = item.get_global_linear_id();
        float intermediate_value = 0.0F;
        float boundary_value = 0.0F;
        if (linear < elements) {
          const std::int64_t spatial = linear % spatial_size;
          const std::int64_t channel =
              (linear / spatial_size) % m.channels;
          float value = state[linear];
          if (m.boundary_code == 1 &&
              channel >= m.channels - m.boundary_channels) {
            value = mask[(channel - (m.channels - m.boundary_channels)) *
                         spatial_size + spatial];
            state[linear] = value;
          } else if (m.boundary_code == 2) {
            value *= mask[spatial];
            state[linear] = value;
          }

          if constexpr (ComputeIntermediate) {
            intermediate_value =
                (sycl::fabs(value) + sycl::fabs(value - 1.0F) - 1.0F) *
                intermediate_scale;
          }
          if constexpr (ComputeBoundary) {
            if (m.boundary_code != 0 && channel < boundary_channel_count) {
              const float outside = 1.0F - SpatialMask(mask, spatial, m);
              boundary_value = sycl::fabs(value) * outside * boundary_scale;
            }
          }
        }

        const auto group = item.get_group();
        if constexpr (ComputeIntermediate) {
          const float intermediate_sum = sycl::reduce_over_group(
              group, intermediate_value, sycl::plus<>());
          if (item.get_local_linear_id() == 0) {
            Atomic intermediate_atomic(regularisers[0]);
            intermediate_atomic.fetch_add(intermediate_sum);
          }
        }
        if constexpr (ComputeBoundary) {
          const float boundary_sum = sycl::reduce_over_group(
              group, boundary_value, sycl::plus<>());
          if (item.get_local_linear_id() == 0) {
            Atomic boundary_atomic(regularisers[1]);
            boundary_atomic.fetch_add(boundary_sum);
          }
        }
      });
}

template <bool ComputeIntermediate, bool ComputeBoundary>
void SubmitBoundaryAndRegularisersTwoStage(
    sycl::queue& queue, float* state, const float* mask, float* regularisers,
    float* partials, const RolloutMetadata& m) {
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
  constexpr std::int64_t local_size = 256;
  const std::int64_t global_size =
      nca_sycl::RoundUp(elements, local_size);
  const std::int64_t group_count = global_size / local_size;

  queue.submit([&](sycl::handler& handler) {
    sycl::local_accessor<float, 1> intermediate_values(
        sycl::range<1>(local_size), handler);
    sycl::local_accessor<float, 1> boundary_values(
        sycl::range<1>(local_size), handler);
    handler.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global_size),
                          sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item) {
          const std::int64_t linear = item.get_global_linear_id();
          const std::int64_t local = item.get_local_linear_id();
          float intermediate_value = 0.0F;
          float boundary_value = 0.0F;
          if (linear < elements) {
            const std::int64_t spatial = linear % spatial_size;
            const std::int64_t channel =
                (linear / spatial_size) % m.channels;
            float value = state[linear];
            if (m.boundary_code == 1 &&
                channel >= m.channels - m.boundary_channels) {
              value = mask[(channel -
                            (m.channels - m.boundary_channels)) *
                               spatial_size +
                           spatial];
              state[linear] = value;
            } else if (m.boundary_code == 2) {
              value *= mask[spatial];
              state[linear] = value;
            }
            if constexpr (ComputeIntermediate) {
              intermediate_value =
                  (sycl::fabs(value) + sycl::fabs(value - 1.0F) - 1.0F) *
                  intermediate_scale;
            }
            if constexpr (ComputeBoundary) {
              if (m.boundary_code != 0 && channel < boundary_channel_count) {
                const float outside = 1.0F - SpatialMask(mask, spatial, m);
                boundary_value =
                    sycl::fabs(value) * outside * boundary_scale;
              }
            }
          }
          intermediate_values[local] = intermediate_value;
          boundary_values[local] = boundary_value;
          item.barrier(sycl::access::fence_space::local_space);
          for (std::int64_t stride = local_size / 2; stride > 0;
               stride /= 2) {
            if (local < stride) {
              intermediate_values[local] +=
                  intermediate_values[local + stride];
              boundary_values[local] += boundary_values[local + stride];
            }
            item.barrier(sycl::access::fence_space::local_space);
          }
          if (local == 0) {
            const std::int64_t group = item.get_group_linear_id();
            partials[2 * group] = intermediate_values[0];
            partials[2 * group + 1] = boundary_values[0];
          }
        });
  });

  queue.submit([&](sycl::handler& handler) {
    sycl::local_accessor<float, 1> intermediate_values(
        sycl::range<1>(local_size), handler);
    sycl::local_accessor<float, 1> boundary_values(
        sycl::range<1>(local_size), handler);
    handler.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(local_size),
                          sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item) {
          const std::int64_t local = item.get_local_linear_id();
          float intermediate_value = 0.0F;
          float boundary_value = 0.0F;
          for (std::int64_t group = local; group < group_count;
               group += local_size) {
            intermediate_value += partials[2 * group];
            boundary_value += partials[2 * group + 1];
          }
          intermediate_values[local] = intermediate_value;
          boundary_values[local] = boundary_value;
          item.barrier(sycl::access::fence_space::local_space);
          for (std::int64_t stride = local_size / 2; stride > 0;
               stride /= 2) {
            if (local < stride) {
              intermediate_values[local] +=
                  intermediate_values[local + stride];
              boundary_values[local] += boundary_values[local + stride];
            }
            item.barrier(sycl::access::fence_space::local_space);
          }
          if (local == 0) {
            if constexpr (ComputeIntermediate) {
              regularisers[0] += intermediate_values[0];
            }
            if constexpr (ComputeBoundary) {
              regularisers[1] += boundary_values[0];
            }
          }
        });
  });
}

template <bool ComputeIntermediate, bool ComputeBoundary>
void SubmitBoundaryAndRegularisers(sycl::queue& queue, float* state,
                                   const float* mask, float* regularisers,
                                   float* partials,
                                   const RolloutMetadata& m) {
  constexpr std::int64_t local_size = 256;
  const std::int64_t elements =
      m.batch * m.channels * m.height * m.width;
  const std::int64_t group_count =
      nca_sycl::RoundUp(elements, local_size) / local_size;
  if (nca_sycl::TwoStageRegulariserReductionEnabled() &&
      2 * group_count <= elements) {
    SubmitBoundaryAndRegularisersTwoStage<ComputeIntermediate,
                                          ComputeBoundary>(
        queue, state, mask, regularisers, partials, m);
  } else {
    SubmitBoundaryAndRegularisersAtomic<ComputeIntermediate, ComputeBoundary>(
        queue, state, mask, regularisers, m);
  }
}

void ApplyBoundaryAndRegularisers(sycl::queue& queue, float* state,
                                  const float* mask, float* regularisers,
                                  float* partials, const RolloutMetadata& m) {
  switch (m.regulariser_flags) {
    case nca_sycl::kIntermediateRegulariserFlag:
      SubmitBoundaryAndRegularisers<true, false>(
          queue, state, mask, regularisers, partials, m);
      break;
    case nca_sycl::kBoundaryRegulariserFlag:
      SubmitBoundaryAndRegularisers<false, true>(
          queue, state, mask, regularisers, partials, m);
      break;
    case nca_sycl::kIntermediateRegulariserFlag |
         nca_sycl::kBoundaryRegulariserFlag:
      SubmitBoundaryAndRegularisers<true, true>(
          queue, state, mask, regularisers, partials, m);
      break;
    default:
      ApplyBoundary(queue, state, mask, m);
  }
}

}  // namespace

// Operands: initial state, kernels, W0, W1, bias, [K,B,C,H,W] masks,
// boundary mask. Results: final state, [K,B,C,H,W] trajectory, two
// regulariser sums, perception, hidden, and delta scratch.
extern "C" void nca_sycl_rollout_forward(sycl::queue* queue, void** buffers,
                                          const char* opaque,
                                          std::size_t opaque_len) {
  if (queue == nullptr || buffers == nullptr || opaque == nullptr ||
      opaque_len != sizeof(RolloutMetadata)) return;
  RolloutMetadata m{};
  std::memcpy(&m, opaque, sizeof(m));
  if (!ValidMetadata(m)) return;
  ReportQueueOrderingOnce(*queue);

  auto* initial_state = static_cast<float*>(buffers[0]);
  auto* kernels = static_cast<float*>(buffers[1]);
  auto* weight_hidden = static_cast<float*>(buffers[2]);
  auto* weight_output = static_cast<float*>(buffers[3]);
  auto* bias_output = static_cast<float*>(buffers[4]);
  auto* masks = static_cast<float*>(buffers[5]);
  auto* boundary_mask = static_cast<float*>(buffers[6]);
  auto* output = static_cast<float*>(buffers[7]);
  auto* trajectory = static_cast<float*>(buffers[8]);
  const bool compute_regularisers = m.regulariser_flags != 0;
  auto* regularisers = compute_regularisers
                           ? static_cast<float*>(buffers[9])
                           : nullptr;
  const std::int64_t scratch_offset = compute_regularisers ? 1 : 0;
  auto* perception = static_cast<float*>(buffers[9 + scratch_offset]);
  auto* hidden = static_cast<float*>(buffers[10 + scratch_offset]);
  auto* delta = static_cast<float*>(buffers[11 + scratch_offset]);

  const std::int64_t state_elements =
      m.batch * m.channels * m.height * m.width;
  const ForwardMetadata forward_metadata{
      m.version, m.batch, m.channels, m.height, m.width, m.features,
      m.kernel_size, m.kernel_flags, m.padding, m.workgroup_size, m.xmx_mode};
  if (compute_regularisers) queue->fill(regularisers, 0.0F, 2);
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
    ApplyBoundaryAndRegularisers(*queue, step_output, boundary_mask,
                                 regularisers, delta, m);
    nca_sycl::SynchronizeStage(*queue, "rollout/boundary_regularisers");
  }
  queue->memcpy(output, trajectory + (m.steps - 1) * state_elements,
                static_cast<std::size_t>(state_elements) * sizeof(float));
  nca_sycl::SynchronizeCustomCall(*queue, "rollout/forward_complete");
}

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>

// Legacy XLA GPU custom-call ABI used by Intel Extension for OpenXLA 0.7.0.
// XLA supplies its active SYCL queue, followed by input and output buffers.
extern "C" void jax_sycl_axpy(sycl::queue* queue, void** buffers,
                              const char* opaque, std::size_t opaque_len) {
  if (queue == nullptr || buffers == nullptr || opaque == nullptr ||
      opaque_len != sizeof(std::uint64_t)) {
    return;
  }

  std::uint64_t element_count = 0;
  std::memcpy(&element_count, opaque, sizeof(element_count));

  const auto* x = static_cast<const float*>(buffers[0]);
  const auto* y = static_cast<const float*>(buffers[1]);
  auto* output = static_cast<float*>(buffers[2]);

  // Do not wait here. Enqueuing on JAX/XLA's own queue preserves dependency
  // ordering while allowing execution to remain asynchronous.
  queue->parallel_for(sycl::range<1>(element_count), [=](sycl::id<1> index) {
    const std::size_t i = index[0];
    output[i] = 1.5F * x[i] + y[i];
  });
}

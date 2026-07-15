#include <sycl/sycl.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kElementCount = 1U << 20;

std::string JoinSubgroupSizes(const std::vector<std::size_t>& sizes) {
  std::string result;
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    if (i != 0) {
      result += ',';
    }
    result += std::to_string(sizes[i]);
  }
  return result;
}

}  // namespace

int main() {
  std::atomic<bool> saw_async_error = false;
  auto async_handler = [&saw_async_error](sycl::exception_list exceptions) {
    for (const std::exception_ptr& exception : exceptions) {
      try {
        std::rethrow_exception(exception);
      } catch (const sycl::exception& error) {
        saw_async_error.store(true);
        std::cerr << "SYCL_ASYNC_ERROR=" << error.what() << '\n';
      }
    }
  };

  try {
    sycl::queue queue(
        sycl::gpu_selector_v,
        async_handler,
        sycl::property_list{sycl::property::queue::enable_profiling{}});
    const sycl::device device = queue.get_device();
    const sycl::platform platform = device.get_platform();

    std::cout << "SYCL_SMOKE_VERSION=1\n";
    std::cout << "DEVICE_NAME="
              << device.get_info<sycl::info::device::name>() << '\n';
    std::cout << "DEVICE_VENDOR="
              << device.get_info<sycl::info::device::vendor>() << '\n';
    std::cout << "DRIVER_VERSION="
              << device.get_info<sycl::info::device::driver_version>() << '\n';
    std::cout << "PLATFORM_NAME="
              << platform.get_info<sycl::info::platform::name>() << '\n';
    std::cout << "PLATFORM_VERSION="
              << platform.get_info<sycl::info::platform::version>() << '\n';
    std::cout << "MAX_COMPUTE_UNITS="
              << device.get_info<sycl::info::device::max_compute_units>() << '\n';
    std::cout << "GLOBAL_MEMORY_BYTES="
              << device.get_info<sycl::info::device::global_mem_size>() << '\n';
    std::cout << "LOCAL_MEMORY_BYTES="
              << device.get_info<sycl::info::device::local_mem_size>() << '\n';
    std::cout << "SUBGROUP_SIZES="
              << JoinSubgroupSizes(
                     device.get_info<sycl::info::device::sub_group_sizes>())
              << '\n';
    std::cout << "HAS_FP16=" << device.has(sycl::aspect::fp16) << '\n';
    std::cout << "HAS_FP64=" << device.has(sycl::aspect::fp64) << '\n';
    std::cout << "HAS_USM_DEVICE_ALLOCATIONS="
              << device.has(sycl::aspect::usm_device_allocations) << '\n';

    if (!device.has(sycl::aspect::usm_device_allocations)) {
      throw std::runtime_error(
          "Selected GPU does not support device USM allocations");
    }

    std::vector<float> host_a(kElementCount);
    std::vector<float> host_b(kElementCount);
    std::vector<float> host_output(kElementCount, 0.0F);
    for (std::size_t i = 0; i < kElementCount; ++i) {
      host_a[i] = static_cast<float>(i % 251) * 0.125F;
      host_b[i] = static_cast<float>(i % 127) * -0.25F;
    }

    float* device_a = sycl::malloc_device<float>(kElementCount, queue);
    float* device_b = sycl::malloc_device<float>(kElementCount, queue);
    float* device_output = sycl::malloc_device<float>(kElementCount, queue);
    if (device_a == nullptr || device_b == nullptr || device_output == nullptr) {
      if (device_a != nullptr) sycl::free(device_a, queue);
      if (device_b != nullptr) sycl::free(device_b, queue);
      if (device_output != nullptr) sycl::free(device_output, queue);
      throw std::runtime_error("A device USM allocation returned null");
    }

    queue.memcpy(
        device_a, host_a.data(), kElementCount * sizeof(float));
    queue.memcpy(
        device_b, host_b.data(), kElementCount * sizeof(float));
    queue.wait_and_throw();

    // Warm up runtime and kernel-module initialization before profiling.
    queue.parallel_for(sycl::range<1>(kElementCount), [=](sycl::id<1> index) {
      const std::size_t i = index[0];
      device_output[i] = 1.5F * device_a[i] + device_b[i];
    });
    queue.wait_and_throw();

    const sycl::event event = queue.parallel_for(
        sycl::range<1>(kElementCount), [=](sycl::id<1> index) {
          const std::size_t i = index[0];
          device_output[i] = 1.5F * device_a[i] + device_b[i];
        });
    event.wait_and_throw();

    queue.memcpy(
        host_output.data(), device_output, kElementCount * sizeof(float));
    queue.wait_and_throw();

    double max_absolute_error = 0.0;
    for (std::size_t i = 0; i < kElementCount; ++i) {
      const float expected = 1.5F * host_a[i] + host_b[i];
      max_absolute_error = std::max(
          max_absolute_error,
          std::abs(static_cast<double>(host_output[i] - expected)));
    }

    const std::uint64_t start_ns = event.get_profiling_info<
        sycl::info::event_profiling::command_start>();
    const std::uint64_t end_ns = event.get_profiling_info<
        sycl::info::event_profiling::command_end>();
    const double kernel_ms = static_cast<double>(end_ns - start_ns) * 1.0e-6;

    sycl::free(device_a, queue);
    sycl::free(device_b, queue);
    sycl::free(device_output, queue);

    std::cout << std::setprecision(9);
    std::cout << "ELEMENT_COUNT=" << kElementCount << '\n';
    std::cout << "KERNEL_TIME_MS=" << kernel_ms << '\n';
    std::cout << "MAX_ABSOLUTE_ERROR=" << max_absolute_error << '\n';

    if (saw_async_error.load()) {
      std::cerr << "SYCL_SMOKE_RESULT=FAIL_ASYNC_ERROR\n";
      return 3;
    }
    if (max_absolute_error != 0.0) {
      std::cerr << "SYCL_SMOKE_RESULT=FAIL_NUMERICAL_ERROR\n";
      return 4;
    }

    std::cout << "SYCL_SMOKE_RESULT=PASS\n";
    return 0;
  } catch (const sycl::exception& error) {
    std::cerr << "SYCL_EXCEPTION=" << error.what() << '\n';
    std::cerr << "SYCL_SMOKE_RESULT=FAIL_SYCL_EXCEPTION\n";
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "ERROR=" << error.what() << '\n';
    std::cerr << "SYCL_SMOKE_RESULT=FAIL_HOST_EXCEPTION\n";
    return 1;
  }
}

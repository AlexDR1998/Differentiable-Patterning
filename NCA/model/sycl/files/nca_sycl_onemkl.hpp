#pragma once

#include <oneapi/mkl/blas.hpp>
#include <sycl/sycl.hpp>

#include <cstdint>

namespace nca_sycl {

inline oneapi::mkl::blas::compute_mode ComputeMode(std::int64_t mode) {
  using oneapi::mkl::blas::compute_mode;
  switch (mode) {
    case 1:
      return compute_mode::float_to_tf32 | compute_mode::prefer_alternate;
    case 2:
      return compute_mode::float_to_bf16 | compute_mode::prefer_alternate;
    case 3:
      return compute_mode::float_to_bf16x2 | compute_mode::prefer_alternate;
    case 4:
      return compute_mode::float_to_bf16x3 | compute_mode::prefer_alternate;
    default:
      return compute_mode::standard;
  }
}

inline sycl::event Gemm(sycl::queue& queue,
                        oneapi::mkl::transpose transpose_a,
                        oneapi::mkl::transpose transpose_b,
                        std::int64_t rows, std::int64_t columns,
                        std::int64_t reduction, const float* a,
                        std::int64_t leading_a, const float* b,
                        std::int64_t leading_b, float* output,
                        std::int64_t leading_output, std::int64_t mode) {
  return oneapi::mkl::blas::row_major::gemm(
      queue, transpose_a, transpose_b, rows, columns, reduction, 1.0F, a,
      leading_a, b, leading_b, 0.0F, output, leading_output,
      ComputeMode(mode));
}

}  // namespace nca_sycl

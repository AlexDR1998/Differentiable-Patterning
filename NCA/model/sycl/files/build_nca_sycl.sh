#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="${SCRIPT_DIR}/nca_sycl.cpp"
BACKWARD_SOURCE="${SCRIPT_DIR}/nca_sycl_backward.cpp"
ROLLOUT_SOURCE="${SCRIPT_DIR}/nca_sycl_rollout.cpp"
ROLLOUT_BACKWARD_SOURCE="${SCRIPT_DIR}/nca_sycl_rollout_backward.cpp"
OUTPUT="${1:-${SCRIPT_DIR}/libnca_sycl.so}"
SYCL_CXX="${CXX:-icpx}"

OPTIMISATION_FLAGS=(-O3)
SANITIZER_FLAGS=()
if [[ "${NCA_SYCL_DEVICE_ASAN:-0}" != "0" ]]; then
    OPTIMISATION_FLAGS=(-O0 -g)
    SANITIZER_FLAGS=(-Xarch_device -fsanitize=address)
fi

if ! command -v "${SYCL_CXX}" >/dev/null 2>&1; then
    echo "Compiler not found: ${SYCL_CXX}" >&2
    echo "Source the cluster JAX/oneAPI setup before running this script." >&2
    exit 1
fi

mkdir -p "$(dirname -- "${OUTPUT}")"
set -x
"${SYCL_CXX}" \
    -fsycl \
    "${OPTIMISATION_FLAGS[@]}" \
    "${SANITIZER_FLAGS[@]}" \
    -qmkl=sequential \
    -std=c++17 \
    -fPIC \
    -shared \
    "${SOURCE}" \
    "${BACKWARD_SOURCE}" \
    "${ROLLOUT_SOURCE}" \
    "${ROLLOUT_BACKWARD_SOURCE}" \
    -o "${OUTPUT}"
set +x

echo "NCA_SYCL_LIBRARY=${OUTPUT}"

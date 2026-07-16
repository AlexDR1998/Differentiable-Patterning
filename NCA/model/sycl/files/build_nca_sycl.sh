#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="${SCRIPT_DIR}/nca_sycl.cpp"
OUTPUT="${1:-${SCRIPT_DIR}/libnca_sycl.so}"
SYCL_CXX="${CXX:-icpx}"

if ! command -v "${SYCL_CXX}" >/dev/null 2>&1; then
    echo "Compiler not found: ${SYCL_CXX}" >&2
    echo "Source the cluster JAX/oneAPI setup before running this script." >&2
    exit 1
fi

mkdir -p "$(dirname -- "${OUTPUT}")"
set -x
"${SYCL_CXX}" \
    -fsycl \
    -O3 \
    -std=c++17 \
    -fPIC \
    -shared \
    "${SOURCE}" \
    -o "${OUTPUT}"
set +x

echo "NCA_SYCL_LIBRARY=${OUTPUT}"

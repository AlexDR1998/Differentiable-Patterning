#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_FILE="${SCRIPT_DIR}/intel_sycl_smoke.cpp"
BUILD_ROOT="${SYCL_SMOKE_BUILD_DIR:-${TMPDIR:-/tmp}/differentiable_patterning_sycl_smoke}"
BUILD_DIR="${BUILD_ROOT}/${SLURM_JOB_ID:-local}_${SLURM_PROCID:-0}"
BINARY="${BUILD_DIR}/intel_sycl_smoke"

if [[ -n "${CXX:-}" ]]; then
    SYCL_CXX="${CXX}"
elif command -v icpx >/dev/null 2>&1; then
    SYCL_CXX="icpx"
elif command -v dpcpp >/dev/null 2>&1; then
    SYCL_CXX="dpcpp"
else
    echo "SYCL_SMOKE_RESULT=FAIL_NO_COMPILER" >&2
    echo "Neither icpx nor dpcpp is on PATH. Source oneAPI setvars.sh or load the cluster oneAPI module." >&2
    exit 10
fi

mkdir -p "${BUILD_DIR}"

echo "SYCL_CXX=$(command -v "${SYCL_CXX}" || true)"
"${SYCL_CXX}" --version | sed -n '1,3p'
echo "BUILD_DIR=${BUILD_DIR}"
echo "ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-<unset>}"
echo "SYCL_DEVICE_FILTER=${SYCL_DEVICE_FILTER:-<unset>}"

if command -v sycl-ls >/dev/null 2>&1; then
    echo "SYCL_LS_BEGIN"
    sycl-ls
    echo "SYCL_LS_END"
else
    echo "SYCL_LS=not_found"
fi

EXTRA_FLAGS=()
if [[ -n "${SYCL_CXX_FLAGS:-}" ]]; then
    # Intentional shell-style splitting so callers can supply several compiler flags.
    read -r -a EXTRA_FLAGS <<< "${SYCL_CXX_FLAGS}"
fi

set -x
"${SYCL_CXX}" \
    -fsycl \
    -O3 \
    -std=c++17 \
    "${EXTRA_FLAGS[@]}" \
    "${SOURCE_FILE}" \
    -o "${BINARY}"
set +x

echo "SYCL_COMPILE_RESULT=PASS"
"${BINARY}"

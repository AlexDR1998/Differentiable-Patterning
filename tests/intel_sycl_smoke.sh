#!/usr/bin/env bash
#SBATCH --job-name=intel-sycl-smoke
#SBATCH --account=AIRR-P100-DAWN-GPU
#SBATCH --partition=pvc9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --output=intel-sycl-smoke-%j.out
#SBATCH --error=intel-sycl-smoke-%j.err

set -eo pipefail

ORIGINAL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "SYCL_SMOKE_RESULT=FAIL_NO_SBATCH" >&2
        exit 11
    fi
    JOB_ID="$(sbatch --parsable \
        --export=ALL,SYCL_SMOKE_SOURCE_DIR="${ORIGINAL_SCRIPT_DIR}" \
        "$(realpath "${BASH_SOURCE[0]}")")"
    echo "Submitted Intel SYCL smoke job ${JOB_ID}"
    echo "Expected logs in the submission directory:"
    echo "  intel-sycl-smoke-${JOB_ID}.out"
    echo "  intel-sycl-smoke-${JOB_ID}.err"
    exit 0
fi

if [[ -n "${SYCL_SMOKE_SOURCE_DIR:-}" ]]; then
    SOURCE_DIR="${SYCL_SMOKE_SOURCE_DIR}"
elif [[ -f "${SLURM_SUBMIT_DIR:-}/tests/intel_sycl_smoke.cpp" ]]; then
    SOURCE_DIR="${SLURM_SUBMIT_DIR}/tests"
else
    SOURCE_DIR="${ORIGINAL_SCRIPT_DIR}"
fi

SOURCE_FILE="${SOURCE_DIR}/intel_sycl_smoke.cpp"
if [[ ! -f "${SOURCE_FILE}" ]]; then
    echo "SYCL_SMOKE_RESULT=FAIL_SOURCE_NOT_FOUND" >&2
    echo "Expected source file: ${SOURCE_FILE}" >&2
    exit 12
fi

SYCL_SETUP_SCRIPT="${SYCL_SETUP_SCRIPT:-${HOME}/dawn-jax/envs/jaxeqx-setup.sh}"
echo "SYCL_SETUP_SCRIPT=${SYCL_SETUP_SCRIPT}"
if [[ -f "${SYCL_SETUP_SCRIPT}" ]]; then
    # Keep oneAPI 2025.2 compiler/runtime exports in this job shell. This also
    # matches the ABI used by intel_extension_for_openxla 0.7.0.
    # shellcheck disable=SC1090
    source "${SYCL_SETUP_SCRIPT}"
else
    echo "SYCL_SETUP_SCRIPT_STATUS=not_found"
fi

# oneAPI/Conda activation may inspect unset internal variables.
set -u

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

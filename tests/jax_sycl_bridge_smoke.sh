#!/usr/bin/env bash
#SBATCH --job-name=jax-sycl-bridge
#SBATCH --account=AIRR-P100-DAWN-GPU
#SBATCH --partition=pvc9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --output=jax-sycl-bridge-%j.out
#SBATCH --error=jax-sycl-bridge-%j.err

set -eo pipefail

ORIGINAL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "JAX_SYCL_BRIDGE_RESULT=FAIL_NO_SBATCH" >&2
        exit 11
    fi
    JOB_ID="$(sbatch --parsable \
        --export=ALL,JAX_SYCL_BRIDGE_SOURCE_DIR="${ORIGINAL_SCRIPT_DIR}" \
        "$(realpath "${BASH_SOURCE[0]}")")"
    echo "Submitted JAX-SYCL bridge job ${JOB_ID}"
    echo "Expected logs in the submission directory:"
    echo "  jax-sycl-bridge-${JOB_ID}.out"
    echo "  jax-sycl-bridge-${JOB_ID}.err"
    exit 0
fi

SOURCE_DIR="${JAX_SYCL_BRIDGE_SOURCE_DIR:-${ORIGINAL_SCRIPT_DIR}}"
CPP_SOURCE="${SOURCE_DIR}/jax_sycl_bridge_smoke.cpp"
PYTHON_SOURCE="${SOURCE_DIR}/jax_sycl_bridge_smoke.py"
if [[ ! -f "${CPP_SOURCE}" || ! -f "${PYTHON_SOURCE}" ]]; then
    echo "JAX_SYCL_BRIDGE_RESULT=FAIL_SOURCE_NOT_FOUND" >&2
    echo "Expected sources under: ${SOURCE_DIR}" >&2
    exit 12
fi

SYCL_SETUP_SCRIPT="${SYCL_SETUP_SCRIPT:-${HOME}/dawn-jax/envs/jaxeqx-setup.sh}"
echo "SYCL_SETUP_SCRIPT=${SYCL_SETUP_SCRIPT}"
if [[ ! -f "${SYCL_SETUP_SCRIPT}" ]]; then
    echo "JAX_SYCL_BRIDGE_RESULT=FAIL_SETUP_NOT_FOUND" >&2
    exit 13
fi
# The extension must load against the same oneAPI runtime as the Intel PJRT
# plugin, so compile and execute after sourcing the training environment.
# shellcheck disable=SC1090
source "${SYCL_SETUP_SCRIPT}"
set -u

if ! command -v icpx >/dev/null 2>&1; then
    echo "JAX_SYCL_BRIDGE_RESULT=FAIL_NO_ICPX" >&2
    exit 14
fi

BUILD_ROOT="${JAX_SYCL_BRIDGE_BUILD_DIR:-${TMPDIR:-/tmp}/differentiable_patterning_jax_sycl_bridge}"
BUILD_DIR="${BUILD_ROOT}/${SLURM_JOB_ID}_${SLURM_PROCID:-0}"
LIBRARY="${BUILD_DIR}/libjax_sycl_bridge_smoke.so"
mkdir -p "${BUILD_DIR}"

echo "SYCL_CXX=$(command -v icpx)"
icpx --version | sed -n '1,3p'
echo "PYTHON=$(command -v python)"
python --version
echo "BUILD_DIR=${BUILD_DIR}"

set -x
icpx \
    -fsycl \
    -O3 \
    -std=c++17 \
    -fPIC \
    -shared \
    "${CPP_SOURCE}" \
    -o "${LIBRARY}"
set +x

echo "JAX_SYCL_BRIDGE_COMPILE_RESULT=PASS"
python "${PYTHON_SOURCE}" --library "${LIBRARY}"

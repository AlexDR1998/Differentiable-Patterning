#!/usr/bin/env bash
#SBATCH --job-name=nca-sycl-smoke
#SBATCH --account=AIRR-P100-DAWN-GPU
#SBATCH --partition=pvc9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --output=nca-sycl-smoke-%j.out
#SBATCH --error=nca-sycl-smoke-%j.err

set -eo pipefail

ORIGINAL_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "NCA_SYCL_SMOKE_RESULT=FAIL_NO_SBATCH" >&2
        exit 11
    fi
    JOB_ID="$(sbatch --parsable \
        --export=ALL,NCA_SYCL_SMOKE_REPO_ROOT="${ORIGINAL_REPO_ROOT}" \
        "$(realpath "${BASH_SOURCE[0]}")")"
    echo "Submitted NCA SYCL smoke job ${JOB_ID}"
    echo "Expected logs in the submission directory:"
    echo "  nca-sycl-smoke-${JOB_ID}.out"
    echo "  nca-sycl-smoke-${JOB_ID}.err"
    exit 0
fi

REPO_ROOT="${NCA_SYCL_SMOKE_REPO_ROOT:-${ORIGINAL_REPO_ROOT}}"
PYTHON_TEST="${REPO_ROOT}/tests/nca_sycl_smoke.py"
BUILD_SCRIPT="${REPO_ROOT}/NCA/model/sycl/files/build_nca_sycl.sh"
if [[ ! -f "${PYTHON_TEST}" || ! -f "${BUILD_SCRIPT}" ]]; then
    echo "NCA_SYCL_SMOKE_RESULT=FAIL_SOURCE_NOT_FOUND" >&2
    echo "Expected repository root: ${REPO_ROOT}" >&2
    exit 12
fi

SYCL_SETUP_SCRIPT="${SYCL_SETUP_SCRIPT:-${HOME}/dawn-jax/envs/jaxeqx-setup.sh}"
echo "SYCL_SETUP_SCRIPT=${SYCL_SETUP_SCRIPT}"
if [[ ! -f "${SYCL_SETUP_SCRIPT}" ]]; then
    echo "NCA_SYCL_SMOKE_RESULT=FAIL_SETUP_NOT_FOUND" >&2
    exit 13
fi
# shellcheck disable=SC1090
source "${SYCL_SETUP_SCRIPT}"
set -u

BUILD_ROOT="${NCA_SYCL_SMOKE_BUILD_DIR:-${TMPDIR:-/tmp}/differentiable_patterning_nca_sycl_smoke}"
BUILD_DIR="${BUILD_ROOT}/${SLURM_JOB_ID}_${SLURM_PROCID:-0}"
export NCA_SYCL_LIBRARY="${BUILD_DIR}/libnca_sycl.so"
mkdir -p "${BUILD_DIR}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "BUILD_DIR=${BUILD_DIR}"
echo "NCA_SYCL_LIBRARY=${NCA_SYCL_LIBRARY}"
echo "SYCL_CXX=$(command -v icpx || true)"
echo "PYTHON=$(command -v python || true)"

"${BUILD_SCRIPT}" "${NCA_SYCL_LIBRARY}"
echo "NCA_SYCL_COMPILE_RESULT=PASS"

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    python -u "${PYTHON_TEST}"

#!/usr/bin/env bash
#SBATCH --job-name=nca-sycl-two-tile-smoke
#SBATCH --account=AIRR-P100-DAWN-GPU
#SBATCH --partition=pvc9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --output=nca-sycl-two-tile-smoke-%j.out
#SBATCH --error=nca-sycl-two-tile-smoke-%j.err

set -eo pipefail

ORIGINAL_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    JOB_ID="$(sbatch --parsable \
        --export=ALL,NCA_SYCL_SMOKE_REPO_ROOT="${ORIGINAL_REPO_ROOT}" \
        "$(realpath "${BASH_SOURCE[0]}")")"
    echo "Submitted two-tile NCA SYCL smoke job ${JOB_ID}"
    exit 0
fi

REPO_ROOT="${NCA_SYCL_SMOKE_REPO_ROOT:-${ORIGINAL_REPO_ROOT}}"
SYCL_SETUP_SCRIPT="${SYCL_SETUP_SCRIPT:-${HOME}/dawn-jax/envs/jaxeqx-setup.sh}"
source "${SYCL_SETUP_SCRIPT}"
set -u

BUILD_DIR="${TMPDIR:-/tmp}/differentiable_patterning_nca_sycl_two_tile/${SLURM_JOB_ID}_${SLURM_PROCID:-0}"
export NCA_SYCL_LIBRARY="${BUILD_DIR}/libnca_sycl.so"
export NCA_SYCL_REPORT_QUEUE_ORDERING=1
mkdir -p "${BUILD_DIR}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "BUILD_DIR=${BUILD_DIR}"
echo "NCA_SYCL_LIBRARY=${NCA_SYCL_LIBRARY}"
"${REPO_ROOT}/NCA/model/sycl/files/build_nca_sycl.sh" "${NCA_SYCL_LIBRARY}"
echo "NCA_SYCL_COMPILE_RESULT=PASS"

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    python -u tests/hardware/nca_sycl_two_tile_smoke.py

#!/usr/bin/env bash
#SBATCH --job-name=nca-sycl-scratch-guard
#SBATCH --account=AIRR-P100-DAWN-GPU
#SBATCH --partition=pvc9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G

set -eo pipefail

REPO_ROOT="${NCA_SYCL_SCRATCH_REPO_ROOT:?NCA_SYCL_SCRATCH_REPO_ROOT is required}"
DEFAULT_MODES="reuse per_step"
read -r -a MODES <<< "${NCA_SYCL_SCRATCH_MODES:-${DEFAULT_MODES}}"
if [[ "${#MODES[@]}" -eq 0 ]]; then
    echo "NCA_SYCL_SCRATCH_MODES must select at least one mode" >&2
    exit 2
fi

MODE_INDEX=$((SLURM_ARRAY_TASK_ID % ${#MODES[@]}))
REPEAT_INDEX=$((SLURM_ARRAY_TASK_ID / ${#MODES[@]}))
MODE="${MODES[$MODE_INDEX]}"

source "${SYCL_SETUP_SCRIPT:-${HOME}/dawn-jax/envs/jaxeqx-setup.sh}"
set -u

echo "SCRATCH_MODE=${MODE}"
echo "REPEAT_INDEX=${REPEAT_INDEX}"
echo "DEVICE_ASAN=${NCA_SYCL_SCRATCH_ASAN:-0}"
echo "TILES=${NCA_SYCL_SCRATCH_TILES:-2}"
echo "HOSTNAME=$(hostname)"
echo "SLURMD_NODENAME=${SLURMD_NODENAME:-<unset>}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<unset>}"

BUILD_DIR="${SLURM_TMPDIR:-/tmp}/nca-sycl-scratch-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${BUILD_DIR}"
export NCA_SYCL_LIBRARY="${BUILD_DIR}/libnca_sycl.so"
export NCA_SYCL_DEVICE_ASAN="${NCA_SYCL_SCRATCH_ASAN:-0}"
"${REPO_ROOT}/NCA/model/sycl/files/build_nca_sycl.sh" "${NCA_SYCL_LIBRARY}"

python -m pip show jax jaxlib intel-extension-for-openxla 2>&1 || true
icpx --version 2>&1 | sed -n '1,3p' || true
sycl-ls --verbose 2>&1 || true
ldd "${NCA_SYCL_LIBRARY}" 2>&1 || true

ulimit -c 0
cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    python -X faulthandler -u \
    tests/hardware/nca_sycl_rollout_scratch_guard.py \
    --mode "${MODE}" \
    --tiles "${NCA_SYCL_SCRATCH_TILES:-2}"

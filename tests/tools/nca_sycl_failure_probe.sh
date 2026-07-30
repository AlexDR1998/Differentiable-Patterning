#!/usr/bin/env bash
#SBATCH --job-name=nca-sycl-failure-probes
#SBATCH --account=AIRR-P100-DAWN-GPU
#SBATCH --partition=pvc9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --output=nca-sycl-probes-%A-%a.out
#SBATCH --error=nca-sycl-probes-%A-%a.err

set -eo pipefail

REPO_ROOT="${NCA_SYCL_PROBE_REPO_ROOT:?NCA_SYCL_PROBE_REPO_ROOT is required}"
PROBE_REPEATS="${NCA_SYCL_PROBE_REPEATS:-20}"
PROBES=(
    baseline
    strict_stages
    serialize_onemkl
    serialize_backward
    bf16_compute
)
# Interleave probe types so scheduler timing and node allocation are not
# correlated with one contiguous block of probe variants.
PROBE_INDEX=$((SLURM_ARRAY_TASK_ID % ${#PROBES[@]}))
REPEAT_INDEX=$((SLURM_ARRAY_TASK_ID / ${#PROBES[@]}))
PROBE="${PROBES[$PROBE_INDEX]}"

source "${SYCL_SETUP_SCRIPT:-${HOME}/dawn-jax/envs/jaxeqx-setup.sh}"
set -u

echo "PROBE=${PROBE}"
echo "REPEAT_INDEX=${REPEAT_INDEX}"
echo "HOSTNAME=$(hostname)"
echo "SLURMD_NODENAME=${SLURMD_NODENAME:-<unset>}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<unset>}"

BUILD_DIR="${SLURM_TMPDIR:-/tmp}/nca-sycl-probe-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${BUILD_DIR}"
export NCA_SYCL_LIBRARY="${BUILD_DIR}/libnca_sycl.so"
export NCA_SYCL_REPORT_QUEUE_ORDERING=1
"${REPO_ROOT}/NCA/model/sycl/files/build_nca_sycl.sh" "${NCA_SYCL_LIBRARY}"

python -m pip show jax jaxlib intel-extension-for-openxla 2>&1 || true
icpx --version 2>&1 | sed -n '1,3p' || true
sycl-ls --verbose 2>&1 || true
ldd "${NCA_SYCL_LIBRARY}" 2>&1 || true
if command -v dpkg-query >/dev/null 2>&1; then
    dpkg-query -W 2>/dev/null | grep -Ei 'level-zero|igc|intel.*(mkl|compute|opencl)|oneapi' || true
fi

if [[ "${NCA_SYCL_TRACE:-0}" == "1" ]]; then
    export OCL_ICD_ENABLE_TRACE=1
    export ZE_LOADER_DEBUG_TRACE=1
    export NEOReadDebugKeys=1
    export PrintDebugMessages=1
fi

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    python -X faulthandler -u tests/tools/nca_sycl_failure_probe.py --probe "${PROBE}"

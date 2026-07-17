#!/usr/bin/env bash
set -eo pipefail
#SBATCH --account=AIRR-P100-DAWN-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p pvc9

# Intel's conda hooks can read unset internal variables, so enable nounset only
# after the module/conda environment is ready.
# module purge
# module load rhel9/default-dawn
# module load intelpython-conda
# module load intel-oneapi-mkl
# conda activate jax_intel_gpu

# bash ~/dawn-jax/envs/jax-setup.sh
source ~/dawn-jax/envs/jaxeqx-setup.sh

ONEAPI_ROOT=/usr/local/dawn/software/external/intel-oneapi/2025.2.1

# UNITRACE="$(command -v unitrace || true)"

# if [[ -z "$UNITRACE" ]]; then
#     UNITRACE="$(
#         find "$ONEAPI_ROOT" -type f -name unitrace -executable \
#             2>/dev/null | head -n 1
#     )"
# fi

# if [[ -z "$UNITRACE" ]]; then
#     echo "unitrace was not found."
#     echo "Available PTI-related modules:"
#     module avail 2>&1 | grep -Ei 'pti|profil' || true
#     echo "VTune location: $(command -v vtune || echo unavailable)"
#     exit 20
# fi

# echo "UNITRACE=$UNITRACE"
# "$UNITRACE" --version


SYCL_BUILD_DIR="${SLURM_TMPDIR:-/tmp}/nca-sycl-${SLURM_JOB_ID}"
mkdir -p "${SYCL_BUILD_DIR}"

NCA/model/sycl/files/build_nca_sycl.sh \
    "${SYCL_BUILD_DIR}/libnca_sycl.so"

export NCA_SYCL_LIBRARY="${SYCL_BUILD_DIR}/libnca_sycl.so"

# module purge
# module load rhel9/default-dawn
# source /usr/local/dawn/software/external/intel-oneapi/2025.2.1/setvars.sh
# if [[ -z "${ZE_FLAT_DEVICE_HIERARCHY}" ]]; then
#     export ZE_FLAT_DEVICE_HIERARCHY="FLAT"
# fi 
# source /home/rc-rich1/miniforge3/bin/activate
# conda activate jax


python - <<'PY'
from importlib import metadata
import sys
import jax
print("jax.devices(): ", jax.devices())
print("jax.local_devices(): ", jax.local_devices())
PY

set -u

: "${PY_SCRIPT:?PY_SCRIPT is not set}"
: "${MANIFEST:?MANIFEST is not set}"
: "${N_JOBS:?N_JOBS is not set}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}"
: "${PROFILE_GPU:=0}"

if [[ "$PROFILE_GPU" != "0" && "$PROFILE_GPU" != "1" ]]; then
    echo "PROFILE_GPU must be 0 or 1, got: $PROFILE_GPU"
    exit 1
fi

ulimit -c 0

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= N_JOBS )); then
    echo "SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID is outside manifest range 0-$((N_JOBS - 1))"
    exit 1
fi

export JOB_WORKER_INDEX=0
export JOB_WORKER_COUNT=1
export JOB_COMPLETION_INDEX="$SLURM_ARRAY_TASK_ID"

IO_ROOT="${SLURM_IO_ROOT:-/home/rc-rich1/rds/rds-airr-p100-NQDJLHPwRqs}"
IO_ROOT="${IO_ROOT%/}"
CODE_ROOT="${SLURM_CODE_ROOT:-$(cd "$(dirname "$PY_SCRIPT")/.." && pwd)}"
CODE_ROOT="${CODE_ROOT%/}"
ARRAY_LOG_ROOT="${SLURM_LOG_DIR:-$IO_ROOT/slurm_logs}"
ARRAY_LOG_ROOT="${ARRAY_LOG_ROOT%/}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
ARRAY_LOG_DIR="$ARRAY_LOG_ROOT/$ARRAY_JOB_ID"

export SLURM_ARRAY_LOG_DIR="$ARRAY_LOG_DIR"
export RUN_CONFIG_PROFILE_DIR="${RUN_CONFIG_PROFILE_DIR:-$ARRAY_LOG_DIR/${SLURM_ARRAY_TASK_ID}.profile}"
export PROFILE_GPU_DIR="${PROFILE_GPU_DIR:-$RUN_CONFIG_PROFILE_DIR}"

PVC_PATH="${PVC_PATH:-$CODE_ROOT/}"
[[ "$PVC_PATH" == */ ]] || PVC_PATH="$PVC_PATH/"
export PVC_PATH
export DATA_PATH_BASE="${DATA_PATH_BASE:-$IO_ROOT/Data/}"
export MODEL_SAVE_PATH="${MODEL_SAVE_PATH:-$IO_ROOT/Models/}"

export RUN_CONFIG_PROFILE="$PROFILE_GPU"
export RUN_CONFIG_PROFILE_TRACE=0
export RUN_CONFIG_PROFILE_MEMORY=1

WANDB_TASK_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}_${SLURM_ARRAY_TASK_ID}"
WANDB_SCRATCH_ROOT="${WANDB_SCRATCH_ROOT:-$IO_ROOT/wandb-fast}"

export WANDB_DIR="${WANDB_DIR:-$WANDB_SCRATCH_ROOT/$WANDB_TASK_ID}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WANDB_DIR/cache}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$WANDB_DIR/data}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$WANDB_DIR/artifacts}"
export WANDB_FLUSH_INTERVAL="${WANDB_FLUSH_INTERVAL:-60}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"

mkdir -p "$MODEL_SAVE_PATH" "$IO_ROOT/output" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR" "$RUN_CONFIG_PROFILE_DIR"

echo "Running manifest index $SLURM_ARRAY_TASK_ID/$((N_JOBS - 1)): $MANIFEST"
echo "Using code root: $PVC_PATH"
echo "Using job IO root: $IO_ROOT/"
echo "Writing wandb local files to: $WANDB_DIR"
echo "GPU profiling: $PROFILE_GPU"
echo "JAX profiles: $RUN_CONFIG_PROFILE_DIR"

echo "GPU view:"
echo "  Host: $(hostname)"
echo "  SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"
echo "  SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-unset}"
command -v sycl-ls >/dev/null 2>&1 && sycl-ls || echo "  sycl-ls: not found"

# python -X faulthandler "$PY_SCRIPT" --manifest "$MANIFEST" --index "$SLURM_ARRAY_TASK_ID"

VTUNE=/usr/local/dawn/software/external/intel-oneapi/2025.2.1/vtune/2025.4/bin64/vtune

PROFILE_PARENT="$SLURM_ARRAY_LOG_DIR"
VTUNE_RESULT="$PROFILE_PARENT/${SLURM_ARRAY_TASK_ID}-vtune-offload"

export MKL_VERBOSE=0
export ZE_ENABLE_TRACING_LAYER=1
export UseCyclesPerSecondTimer=1


"$VTUNE" \
    -collect gpu-offload \
    -knob gpu-counters-mode=none \
    -knob collect-programming-api=true \
    -result-dir "$VTUNE_RESULT" \
    -- \
    python -X faulthandler "$PY_SCRIPT" \
        --manifest "$MANIFEST" \
        --index "$SLURM_ARRAY_TASK_ID"



#!/usr/bin/env bash
set -eo pipefail
#SBATCH --account=AIRR-P100-DAWN-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p pvc9

# Intel's conda hooks can read unset internal variables, so enable nounset only
# after the module/conda environment is ready.
module purge
module load rhel9/default-dawn
module load intelpython-conda
module load intel-oneapi-mkl
conda activate jax_intel_gpu

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

export INTEL_MAX_GPU_VRAM_GB="${INTEL_MAX_GPU_VRAM_GB:-128}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.98}"
export ZE_FLAT_DEVICE_HIERARCHY="${ZE_FLAT_DEVICE_HIERARCHY:-COMPOSITE}"
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
export SYCL_DEVICE_FILTER="${SYCL_DEVICE_FILTER:-level_zero:gpu}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-sycl}"
export ZE_ENABLE_TRACING_LAYER="${ZE_ENABLE_TRACING_LAYER:-1}"
export UseCyclesPerSecondTimer="${UseCyclesPerSecondTimer:-1}"
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
echo "Intel GPU target: ${INTEL_MAX_GPU_VRAM_GB} GB, XLA fraction $XLA_PYTHON_CLIENT_MEM_FRACTION"
echo "GPU profiling: $PROFILE_GPU"
echo "JAX profiles: $RUN_CONFIG_PROFILE_DIR"
python - <<'PY'
import os

vram_gb = float(os.environ["INTEL_MAX_GPU_VRAM_GB"])
fraction = float(os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"])
print(f"Expected XLA preallocation target: {vram_gb * fraction:.1f} GB")
PY

echo "GPU view:"
echo "  Host: $(hostname)"
echo "  SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"
echo "  SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-unset}"
echo "  ZE_AFFINITY_MASK: $ZE_AFFINITY_MASK"
echo "  ZE_FLAT_DEVICE_HIERARCHY: $ZE_FLAT_DEVICE_HIERARCHY"
command -v sycl-ls >/dev/null 2>&1 && sycl-ls || echo "  sycl-ls: not found"

python -X faulthandler "$PY_SCRIPT" --manifest "$MANIFEST" --index "$SLURM_ARRAY_TASK_ID"

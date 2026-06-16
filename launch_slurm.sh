#!/usr/bin/env bash
set -eo pipefail
#SBATCH --account=AIRR-P100-DAWN-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p pvc9

# Intel's conda activation hooks can read unset internal variables, so do not
# enable nounset until after the module/conda environment is ready.
module purge
module load rhel9/default-dawn
module load intelpython-conda
module load intel-oneapi-mkl
module load intel-oneapi-ccl
conda activate jax_intel_gpu
python -m pip list | grep -E "jax|jaxlib|intel-extension-for-openxla"

set -u

: "${PY_SCRIPT:?PY_SCRIPT is not set}"
: "${MANIFEST:?MANIFEST is not set}"
: "${N_JOBS:?N_JOBS is not set}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}"

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= N_JOBS )); then
    echo "SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID is outside manifest range 0-$((N_JOBS - 1))"
    exit 1
fi

# -------------------------
# Kubernetes-style env vars
# -------------------------

export JOB_WORKER_INDEX=0
export JOB_WORKER_COUNT=1
export JOB_COMPLETION_INDEX="$SLURM_ARRAY_TASK_ID"

IO_ROOT="${SLURM_IO_ROOT:-/home/rc-rich1/rds/rds-airr-p100-NQDJLHPwRqs}"
IO_ROOT="${IO_ROOT%/}"
CODE_ROOT="${SLURM_CODE_ROOT:-$(cd "$(dirname "$PY_SCRIPT")/.." && pwd)}"
CODE_ROOT="${CODE_ROOT%/}"

PVC_PATH="${PVC_PATH:-$CODE_ROOT/}"
[[ "$PVC_PATH" == */ ]] || PVC_PATH="$PVC_PATH/"
export PVC_PATH
export DATA_PATH_BASE="${DATA_PATH_BASE:-$IO_ROOT/Data/}"
export MODEL_SAVE_PATH="${MODEL_SAVE_PATH:-$IO_ROOT/Models/}"

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

WANDB_TASK_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}_${SLURM_ARRAY_TASK_ID}"
WANDB_SCRATCH_ROOT="${WANDB_SCRATCH_ROOT:-$IO_ROOT/wandb-fast}"

export WANDB_DIR="${WANDB_DIR:-$WANDB_SCRATCH_ROOT/$WANDB_TASK_ID}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WANDB_DIR/cache}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$WANDB_DIR/data}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$WANDB_DIR/artifacts}"
export WANDB_FLUSH_INTERVAL="${WANDB_FLUSH_INTERVAL:-60}"

mkdir -p "$MODEL_SAVE_PATH" "$IO_ROOT/output" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR"

echo "Running manifest index $SLURM_ARRAY_TASK_ID/$((N_JOBS - 1)): $MANIFEST"
echo "Using code root: $PVC_PATH"
echo "Using job IO root: $IO_ROOT/"
echo "Writing wandb local files to: $WANDB_DIR"
srun python "$PY_SCRIPT" --manifest "$MANIFEST" --index "$SLURM_ARRAY_TASK_ID"

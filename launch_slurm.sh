#!/usr/bin/env bash
set -euo pipefail
#SBATCH --account=AIRR-P100-DAWN-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p pvc9

module purge
module load rhel9/default-dawn
module load intelpython-conda
module load intel-oneapi-mkl
module load intel-oneapi-ccl
conda activate jax_intel_gpu
python -m pip list | grep -E "jax|jaxlib|intel-extension-for-openxla"

: "${PY_SCRIPT:?PY_SCRIPT is not set}"
: "${MANIFEST:?MANIFEST is not set}"
: "${N_JOBS:?N_JOBS is not set}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}"

# -------------------------
# Kubernetes-style env vars
# -------------------------

export JOB_WORKER_INDEX="$SLURM_ARRAY_TASK_ID"
export JOB_WORKER_COUNT="$N_JOBS"

export PVC_PATH="${PVC_PATH:-$PWD}"
export DATA_PATH_BASE="${DATA_PATH_BASE:-$PVC_PATH/Data/}"
export MODEL_SAVE_PATH="${MODEL_SAVE_PATH:-$PVC_PATH/Models/}"

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

export WANDB_DIR="${WANDB_DIR:-$PVC_PATH/wandb-fast}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WANDB_DIR/cache}"
export WANDB_FLUSH_INTERVAL="${WANDB_FLUSH_INTERVAL:-60}"

# mkdir -p "$DATA_PATH_BASE" "$MODEL_SAVE_PATH" "$WANDB_CACHE_DIR"

# -------------------------
# Pick this task's config
# -------------------------

CONFIG_LINE="$(
    awk -v target="$SLURM_ARRAY_TASK_ID" '
        /^[[:space:]]*$/ { next }
        /^[[:space:]]*#/ { next }
        count++ == target { print; exit }
    ' "$MANIFEST"
)"

[[ -n "$CONFIG_LINE" ]] || {
    echo "No config found for task $SLURM_ARRAY_TASK_ID"
    exit 1
}

# Allows quoted Hydra args in manifest lines.
# Only use trusted manifests.
eval "set -- $CONFIG_LINE"
srun echo "Running task $SLURM_ARRAY_TASK_ID with config: $CONFIG_LINE"
# python "$PY_SCRIPT" "$@"

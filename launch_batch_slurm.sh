#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 PATH_TO_PYTHON_SCRIPT PATH_TO_MANIFEST"
    exit 1
fi

PY_SCRIPT="$(realpath "$1")"
MANIFEST="$(realpath "$2")"

[[ -f "$PY_SCRIPT" ]] || { echo "Python script not found: $PY_SCRIPT"; exit 1; }
[[ -f "$MANIFEST" ]] || { echo "Manifest not found: $MANIFEST"; exit 1; }

EXPERIMENT_NAME="$(head -1 "$MANIFEST" | sed 's/^[^:]*: //')"
[[ -n "$EXPERIMENT_NAME" ]] || { echo "Manifest first line is empty"; exit 1; }

# Count occurrences of "- index: N" (N is an integer) in the manifest
N_JOBS="$(grep -E '^[[:space:]]*-+[[:space:]]*index:[[:space:]]*[0-9][0-9]*' "$MANIFEST" | wc -l | tr -d ' ')"
[[ "$N_JOBS" -gt 0 ]] || { echo "Manifest contains no runnable lines"; exit 1; }

# -------------------------
# Slurm settings
# -------------------------

JOB_NAME="${USER}-job"
PARTITION="pvc9"
TIME="08:00:00"

MEM="64G"
GPUS=1
ARRAY_PARALLELISM=4
LOG_DIR="slurm_logs/$EXPERIMENT_NAME"

mkdir -p "$LOG_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARRAY_SCRIPT="$SCRIPT_DIR/launch_slurm.sh"

[[ -f "$ARRAY_SCRIPT" ]] || { echo "Missing array script: $ARRAY_SCRIPT"; exit 1; }

sbatch \
    --job-name="$JOB_NAME" \
    --partition="$PARTITION" \
    --account=AIRR-P100-DAWN-GPU \
    --array="0-$((N_JOBS - 1))%$ARRAY_PARALLELISM" \
    --time="$TIME" \
    --mem="$MEM" \
    --nodes=1 \
    --ntasks=1 \
    --gres="gpu:$GPUS" \
    --output="$LOG_DIR/%A_%a.out" \
    --error="$LOG_DIR/%A_%a.err" \
    --export=ALL,PY_SCRIPT="$PY_SCRIPT",MANIFEST="$MANIFEST",N_JOBS="$N_JOBS" \
    "$ARRAY_SCRIPT"

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

extract_yaml_scalar() {
    local key="$1"
    local path="$2"
    local line value

    line="$(grep -m1 -E "^[[:space:]]*$key[[:space:]]*:" "$path" || true)"
    [[ -n "$line" ]] || return 1

    value="${line#*:}"
    value="$(printf '%s' "$value" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")"
    [[ -n "$value" ]] || return 1
    printf '%s\n' "$value"
}

EXPERIMENT_NAME="$(extract_yaml_scalar experiment_name "$MANIFEST" || true)"
[[ -n "$EXPERIMENT_NAME" ]] || EXPERIMENT_NAME="$(basename "$(dirname "$MANIFEST")")"
[[ -n "$EXPERIMENT_NAME" ]] || { echo "Could not derive experiment name from manifest: $MANIFEST"; exit 1; }

N_JOBS="$(extract_yaml_scalar count "$MANIFEST" || true)"
if [[ -z "$N_JOBS" ]]; then
    N_JOBS="$(grep -E '^[[:space:]]*-+[[:space:]]*index:[[:space:]]*[0-9][0-9]*' "$MANIFEST" | wc -l | tr -d ' ')"
fi
[[ "$N_JOBS" =~ ^[0-9]+$ ]] || { echo "Manifest count is not an integer: $N_JOBS"; exit 1; }
[[ "$N_JOBS" -gt 0 ]] || { echo "Manifest contains no runnable lines"; exit 1; }

# -------------------------
# Slurm settings
# -------------------------

IO_ROOT="${SLURM_IO_ROOT:-/home/rc-rich1/rds/rds-airr-p100-NQDJLHPwRqs}"
IO_ROOT="${IO_ROOT%/}"
CODE_ROOT="${SLURM_CODE_ROOT:-$(cd "$(dirname "$PY_SCRIPT")/.." && pwd)}"
CODE_ROOT="${CODE_ROOT%/}"

# Default the Slurm array name to the manifest/YAML experiment name while
# retaining an explicit environment override for one-off submissions.
JOB_NAME="${SLURM_JOB_NAME:-$EXPERIMENT_NAME}"
TIME="${SLURM_TIME:-08:00:00}"

MEM="${SLURM_MEM:-64G}"
ARRAY_PARALLELISM="${SLURM_ARRAY_PARALLELISM:-4}"
LOG_DIR="${SLURM_LOG_DIR:-$IO_ROOT/slurm_logs/$EXPERIMENT_NAME}"
PROFILE_GPU="${PROFILE_GPU:-0}"
NCA_SYCL_DIAGNOSTICS="${NCA_SYCL_DIAGNOSTICS:-0}"
NCA_SYCL_TRACE="${NCA_SYCL_TRACE:-0}"

[[ "$ARRAY_PARALLELISM" =~ ^[0-9]+$ ]] || { echo "SLURM_ARRAY_PARALLELISM must be an integer: $ARRAY_PARALLELISM"; exit 1; }
[[ "$ARRAY_PARALLELISM" -gt 0 ]] || { echo "SLURM_ARRAY_PARALLELISM must be greater than zero"; exit 1; }
[[ "$PROFILE_GPU" == "0" || "$PROFILE_GPU" == "1" ]] || { echo "PROFILE_GPU must be 0 or 1: $PROFILE_GPU"; exit 1; }
[[ "$NCA_SYCL_DIAGNOSTICS" == "0" || "$NCA_SYCL_DIAGNOSTICS" == "1" ]] || { echo "NCA_SYCL_DIAGNOSTICS must be 0 or 1: $NCA_SYCL_DIAGNOSTICS"; exit 1; }
[[ "$NCA_SYCL_TRACE" == "0" || "$NCA_SYCL_TRACE" == "1" ]] || { echo "NCA_SYCL_TRACE must be 0 or 1: $NCA_SYCL_TRACE"; exit 1; }

mkdir -p "$LOG_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARRAY_SCRIPT="$SCRIPT_DIR/launch_slurm.sh"

[[ -f "$ARRAY_SCRIPT" ]] || { echo "Missing array script: $ARRAY_SCRIPT"; exit 1; }

echo "Submitting Slurm array with launch script: $ARRAY_SCRIPT"
echo "Experiment: $EXPERIMENT_NAME"
echo "Array: 0-$((N_JOBS - 1))%$ARRAY_PARALLELISM"
echo "Logs: $LOG_DIR"
echo "GPU profiling: $PROFILE_GPU"
echo "SYCL diagnostics: $NCA_SYCL_DIAGNOSTICS"
echo "SYCL runtime tracing: $NCA_SYCL_TRACE"

sbatch \
    --job-name="$JOB_NAME" \
    --partition="pvc9" \
    --account="AIRR-P100-DAWN-GPU" \
    --array="0-$((N_JOBS - 1))%$ARRAY_PARALLELISM" \
    --time="$TIME" \
    --mem="$MEM" \
    --nodes=1 \
    --ntasks=1 \
    --gres="gpu:1" \
    --output="$LOG_DIR/%A/%a.out" \
    --error="$LOG_DIR/%A/%a.err" \
    --export=ALL,PY_SCRIPT="$PY_SCRIPT",MANIFEST="$MANIFEST",N_JOBS="$N_JOBS",SLURM_IO_ROOT="$IO_ROOT",SLURM_CODE_ROOT="$CODE_ROOT",SLURM_LOG_DIR="$LOG_DIR",PROFILE_GPU="$PROFILE_GPU",NCA_SYCL_DIAGNOSTICS="$NCA_SYCL_DIAGNOSTICS",NCA_SYCL_TRACE="$NCA_SYCL_TRACE" \
    "$ARRAY_SCRIPT"

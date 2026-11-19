#!/bin/bash

# Runs one manifest entry inside a Kubernetes pod (see run.tpl.yml).
source "$(dirname "${BASH_SOURCE[0]}")/manifest.sh"

normalize_workspace_path() { # make a path absolute under /workspace (accepts absolute or workspace-relative paths)
  local input_path="$1"
  case "$input_path" in
    /*)
      printf '%s\n' "$input_path"
      ;;
    workspace/*)
      printf '/%s\n' "$input_path"
      ;;
    *)
      printf '/workspace/%s\n' "$input_path"
      ;;
  esac

}

if [ $# -lt 2 ]; then
  echo "Usage: $0 <path_to_python_script> <path_to_experiment_config>"
  exit 1
fi


SCRIPT_PATH="$(normalize_workspace_path "$1")"
MANIFEST_PATH="$(normalize_workspace_path "$2")"

if [ -f "$MANIFEST_PATH" ]; then
  WANDB_DIR_NAME="$(manifest_experiment_name "$MANIFEST_PATH")"
else
  WANDB_DIR_NAME="default"
fi

mkdir -p /workspace/writeable/data/logs/wandb/"$WANDB_DIR_NAME"/"$JOB_WORKER_INDEX"_"$JOB_COMPLETION_INDEX"

# Keep the job's printed output (and any error) on the persistent volume, since
# pods and their logs can be deleted before anyone reads them
LOG_DIR=/workspace/writeable/data/logs/jobs/"$WANDB_DIR_NAME"
mkdir -p "$LOG_DIR"
set -o pipefail
python -X faulthandler "$SCRIPT_PATH" --manifest "$MANIFEST_PATH" 2>&1 \
  | tee "$LOG_DIR"/worker_"$JOB_WORKER_INDEX"_index_"$JOB_COMPLETION_INDEX".log
STATUS=$?

# The W&B cache (WANDB_CACHE_DIR) stays in the pod: only the run logs are kept
rsync -a --inplace --exclude cache/ /workspace/writeable/wandb-fast/ /workspace/writeable/data/logs/wandb/"$WANDB_DIR_NAME"/"$JOB_WORKER_INDEX"_"$JOB_COMPLETION_INDEX"/
exit $STATUS

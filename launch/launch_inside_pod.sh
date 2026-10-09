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

python "$SCRIPT_PATH" --manifest "$MANIFEST_PATH"

rsync -a --inplace /workspace/writeable/wandb-fast/ /workspace/writeable/data/logs/wandb/"$WANDB_DIR_NAME"/"$JOB_WORKER_INDEX"_"$JOB_COMPLETION_INDEX"/
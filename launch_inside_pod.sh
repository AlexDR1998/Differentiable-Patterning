#!/bin/bash

extract_experiment_name() { # Finds the experiment name field from the manifest yaml, if it exists.
  local manifest="$1"
  # Look for a line starting with experiment_name:, capture value after colon
  if [ -f "$manifest" ]; then
    local line
    line=$(grep -m1 -E '^\s*experiment_name\s*:' "$manifest" || true)
    if [ -n "$line" ]; then
      # remove key and colon
      local val=${line#*:}
      # trim leading/trailing whitespace and surrounding quotes
      val=$(echo "$val" | sed -e 's/^\s*//' -e 's/\s*$//' -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
      printf '%s' "$val"
      return 0
    fi
  fi
  return 1
}

normalize_workspace_path() { # ensures the path is absolute and rooted at /workspace; handles absolute paths and workspace-relative paths
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

# try to extract experiment_name from manifest
EXPERIMENT_NAME=$(extract_experiment_name "$MANIFEST_PATH" || true)
if [ -n "$EXPERIMENT_NAME" ]; then
  WANDB_DIR_NAME="$EXPERIMENT_NAME"
else
  WANDB_DIR_NAME="default"
fi

mkdir -p /workspace/writeable/logs/wandb/"$WANDB_DIR_NAME"/"$JOB_WORKER_INDEX"_"$JOB_COMPLETION_INDEX"

python "$SCRIPT_PATH" --manifest "$MANIFEST_PATH"

rsync -a --inplace /workspace/writeable/wandb-fast/ /workspace/writeable/data/logs/wandb/"$WANDB_DIR_NAME"/"$JOB_WORKER_INDEX"_"$JOB_COMPLETION_INDEX"/
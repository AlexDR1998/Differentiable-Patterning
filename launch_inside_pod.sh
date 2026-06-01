#!/bin/bash

if [ $# -lt 2 ]; then
  echo "Usage: $0 <path_to_python_script> <path_to_experiment_config>"
  exit 1
fi

normalize_workspace_path() {
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

SCRIPT_PATH="$(normalize_workspace_path "$1")"
MANIFEST_PATH="$(normalize_workspace_path "$2")"

python "$SCRIPT_PATH" --manifest "$MANIFEST_PATH"
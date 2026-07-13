#!/bin/bash

if [ $# -lt 4 ]; then
  echo "Usage: $0 <path_to_python_script> <path_to_experiment_config> <number_of_workers> <gpu_type> [start_index]"
  exit 1
fi

FILE="$1"
CONFIGS="$2"
JOB_WORKER_COUNT="$3"
GPU_TYPE="${4:-H200}"
START_INDEX="${5:-0}"

if ! [[ "$START_INDEX" =~ ^[0-9]+$ ]]; then
  echo "Start index must be a non-negative integer: $START_INDEX"
  exit 1
fi

if [ "$GPU_TYPE" == "H200" ]; then
  GPU_MEMORY=128
  GPU_FULL_NAME="NVIDIA-H200"
elif [ "$GPU_TYPE" == "A100" ]; then
  GPU_MEMORY=80
  GPU_FULL_NAME="NVIDIA-A100-SXM4-80GB"
elif [ "$GPU_TYPE" == "H100" ]; then
  GPU_MEMORY=80
  GPU_FULL_NAME="NVIDIA-H100-80GB-HBM3"
else
  echo "Unknown GPU type: $GPU_TYPE"
  exit 1
fi

TOTAL_CONFIGS=$(grep -c '^-[[:space:]]*index:' "$CONFIGS")
if (( START_INDEX >= TOTAL_CONFIGS )); then
  echo "Start index $START_INDEX is outside manifest range 0-$((TOTAL_CONFIGS - 1))"
  exit 1
fi
REMAINING_CONFIGS=$((TOTAL_CONFIGS - START_INDEX))
BASE_PODS=$((REMAINING_CONFIGS / JOB_WORKER_COUNT))
REMAINDER=$((REMAINING_CONFIGS % JOB_WORKER_COUNT))



export WORKING_DIR=/user/$USER/Differentiable-Patterning/
export PATH_TO_PYTHON_SCRIPT=$FILE
# export COMPLETION_INDEX=$CONFIGS
export PATH_TO_EXPERIMENT_CONFIG=$CONFIGS
export JOB_WORKER_COUNT=$JOB_WORKER_COUNT
export MANIFEST_START_INDEX=$START_INDEX
export GPU_TYPE=$GPU_FULL_NAME
export GPU_MEM_LIMIT="${GPU_MEMORY}Gi"

for ((i=0; i<JOB_WORKER_COUNT; i++)); do
  export JOB_WORKER_INDEX=$i
  if (( i < REMAINDER )); then
    export N_PODS=$((BASE_PODS + 1))
  else
    export N_PODS=$BASE_PODS
  fi
  if (( N_PODS == 0 )); then
    echo "Skipping worker $i because there are no configs left for it"
    continue
  fi
  envsubst < run.tpl.yml > run_$i.yml
  kubectl create -f run_$i.yml
done

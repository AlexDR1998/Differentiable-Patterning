#!/bin/bash

if [ $# -lt 3 ]; then
  echo "Usage: $0 <path_to_python_script> <path_to_experiment_config> <number_of_workers>"
  exit 1
fi

FILE="$1"
CONFIGS="$2"
JOB_WORKER_COUNT="$3"

TOTAL_CONFIGS=$(grep -c '^-[[:space:]]*index:' "$CONFIGS")
BASE_PODS=$((TOTAL_CONFIGS / JOB_WORKER_COUNT))
REMAINDER=$((TOTAL_CONFIGS % JOB_WORKER_COUNT))



export WORKING_DIR=/user/$USER/Differentiable-Patterning/
export PATH_TO_PYTHON_SCRIPT=$FILE
# export COMPLETION_INDEX=$CONFIGS
export PATH_TO_EXPERIMENT_CONFIG=$CONFIGS
export JOB_WORKER_COUNT=$JOB_WORKER_COUNT

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


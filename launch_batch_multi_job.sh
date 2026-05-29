#!/bin/bash

if [ $# -lt 3 ]; then
  echo "Usage: $0 <path_to_python_script> <path_to_experiment_config> <number_of_workers>"
  exit 1
fi

FILE="$1"
CONFIGS="$2"
JOB_WORKER_COUNT="$3"


export WORKING_DIR=/user/$USER/Differentiable-Patterning/
export PATH_TO_PYTHON_SCRIPT=$FILE
# export COMPLETION_INDEX=$CONFIGS
export PATH_TO_EXPERIMENT_CONFIG=$CONFIGS
export JOB_WORKER_COUNT=$JOB_WORKER_COUNT

for ((i=0; i<JOB_WORKER_COUNT; i++)); do
  export JOB_WORKER_INDEX=$i
  envsubst < run.tpl.yml > run_$i.yml
  kubectl create -f run_$i.yml
done


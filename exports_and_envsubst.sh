#!/bin/bash

if [ $# -lt 2 ]; then
  echo "Usage: $0 <arg1> <arg2>"
  exit 1
fi

ARG1="$1"
ARG2="$2"

export WORKING_DIR=/user/$USER/Differentiable-Patterning/
export JOB_PATH_TO_PYTHON_SCRIPT=$ARG1
export JOB_COMPLETION_INDEX=$ARG2
envsubst < run.tpl.yml > run_test.yml
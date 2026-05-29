#!/bin/bash

if [ $# -lt 2 ]; then
  echo "Usage: $0 <path_to_python_script> <path_to_experiment_config>"
  exit 1
fi

# FILE="$1"
# CONFIGS="$2"
# python $FILE $CONFIGS
python $1 --manifest $2
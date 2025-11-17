#!/bin/bash
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <path-to-python-script> <num-gpus>"
  exit 1
fi
for i in $(seq 0 $(($2 - 1)));
do
    echo "Starting job $i on GPU $((i))"
    CUDA_VISIBLE_DEVICES=$((i)) python $1 $i &
done
wait
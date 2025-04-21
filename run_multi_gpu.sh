#!/bin/bash
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <path-to-python-script> <num-gpus>"
  exit 1
fi
for i in $(seq 0 $2);
do
    CUDA_VISIBLE_DEVICES=$i python $1 $i &
done
wait
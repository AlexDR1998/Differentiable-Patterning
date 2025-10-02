#!/bin/bash

#SBATCH --gpus=1
#SBATCH --time=24:00:00         # Hours:Mins:Secs



source ~/miniforge3/bin/activate

conda activate jax_gpu
export HF_HOME=/projects/u5be/.cache/huggingface


python micropattern_individual_train.py \
    --downsample $1 \
    --channels $2
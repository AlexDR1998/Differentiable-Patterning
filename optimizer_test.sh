#!/bin/bash

#SBATCH --gpus=1
#SBATCH --time=24:00:00         # Hours:Mins:Secs



source ~/miniforge3/bin/activate

conda activate jax_gpu
export HF_HOME=/projects/u5be/.cache/huggingface
export WANDB_CACHE_DIR=/scratch/u5be/alexdr.u5be/.wandb_cache
export WANDB_DATA_DIR=/scratch/u5be/alexdr.u5be/.wandb_data

python optimizer_test.py \
    --optimizer $1 \
    --block_norm $2 \
    --multistep $3 \
    --task $4 \
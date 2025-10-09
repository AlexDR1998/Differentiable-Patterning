#!/bin/bash

#SBATCH --gpus=1
#SBATCH --time=6:00:00         # Hours:Mins:Secs



source ~/miniforge3/bin/activate

conda activate jax_gpu
export HF_HOME=/projects/u5be/.cache/huggingface


python multi_species_gnca_train.py \
    --contiguous_regulariser $1
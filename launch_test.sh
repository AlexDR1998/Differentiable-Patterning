#!/bin/bash

#SBATCH --job-name=test_run
#SBATCH --output=logs/test_run.out
#SBATCH --gpus=1
#SBATCH --time=12:00:00         # Hours:Mins:Secs



source ~/miniforge3/bin/activate

conda activate jax_gpu
export HF_HOME=/projects/u5be/.cache/huggingface

# python src/record_run.py \
#     --job_id sdxl_record_esd_vangogh_test_3 \
#     --gradient_token_number 8 \
#     --base_model sdxl \
#     --token_number 64 \
#     --run_length 1 \
#     --opt_steps 5 \
#     --encoder_blend 0.5


# python src/string_to_token.py

# python src/generate_training_data.py \
#     --base_model flux \
#     --num_images 100 \
#     --save_latents \
#     --prompt "A painting of starry night by Vincent van Gogh" \
#     --output_dir training_images_lowres/flux/starrynight/
#     # --save_images \


python micropattern_individual_train.py
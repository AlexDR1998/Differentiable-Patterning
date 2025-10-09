import subprocess
import numpy as np
import os
from helper_scripts.utils import generate_hyperparameter_combinations
from pprint import pprint
from einops import rearrange
import time


"""
This code launches a series of SLURM jobs (micropatterns.sh) to train NCA models on individual micropattern data
The point of this is to evaluate how the downsampling factor and number of channels affect performance

"""

DOWNSAMPLES = [2,4,8,16]
# DOWNSAMPLES = [16]
CHANNELS = [8,16,24,32,64]
job_id = "micropattern_individual_hyperparameters_texture_long"
sbatch_log_dir = "logs"
hparams = generate_hyperparameter_combinations(
    {
        "downsample": DOWNSAMPLES,
        "channels": CHANNELS
    }
)

pprint(hparams)
for h in hparams:
    downsample = h["downsample"]
    channels = h["channels"]
    print(f"Launching job with downsample={downsample}, channels={channels}")
    # Create log directory if it doesn't exist
    os.makedirs(sbatch_log_dir, exist_ok=True)
    command = [
        "sbatch",
        f"--job-name={job_id}_d{downsample}_c{channels}",
        f"--output={sbatch_log_dir}/{job_id}_d{downsample}_c{channels}.out",
        f"--error={sbatch_log_dir}/{job_id}_d{downsample}_c{channels}.err",
        "micropatterns.sh",
        str(downsample),
        str(channels)
    ]
    subprocess.Popen(command)
    time.sleep(5)  # slight delay to avoid overwhelming the scheduler
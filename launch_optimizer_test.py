import subprocess
import numpy as np
import os
from helper_scripts.utils import generate_hyperparameter_combinations
from pprint import pprint
from einops import rearrange
import time


OPTIMIZERS = ["optimistic_adam","nadam","muon","sam"]
BLOCK_NORM = [True,False]
MULTISTEP = [1,4,8]
TASKS = ["micropattern","emoji"]


job_id_base = "optimizer_test"
sbatch_log_dir = "logs"
hparams = generate_hyperparameter_combinations(
    {
        "optimizer": OPTIMIZERS,
        "block_norm": BLOCK_NORM,
        "multistep": MULTISTEP,
        "task": TASKS
    }
)
pprint(hparams)

for h in hparams:
    optimizer = h["optimizer"]
    block_norm = h["block_norm"]
    multistep = h["multistep"]
    task = h["task"]
    print(f"Launching job with optimizer={optimizer}, block_norm={block_norm}, multistep={multistep}, task={task}")
    # Create log directory if it doesn't exist
    # os.makedirs(sbatch_log_dir, exist_ok=True)
    _idstr = f"{optimizer}_bn{int(block_norm)}_ms{multistep}_{task}"
    command = [
        "sbatch",
        f"--job-name={job_id_base}_{_idstr}",
        f"--output={sbatch_log_dir}/{job_id_base}_{_idstr}.out",
        f"--error={sbatch_log_dir}/{job_id_base}_{_idstr}.err",
        "optimizer_test.sh",
        optimizer,
        str(block_norm),
        str(multistep),
        task
    ]
    subprocess.Popen(command)
    time.sleep(5)  # slight delay to avoid overwhelming the scheduler
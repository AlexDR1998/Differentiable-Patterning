import subprocess
import numpy as np
import os
from itertools import product
from collections.abc import Iterable
from typing import Any, Dict, List
from pprint import pprint
from einops import rearrange
import time

def generate_hyperparameter_combinations(param_grid: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Given a dict mapping hyperparameter names to iterables of values, return a list of
    dicts representing every combination (cartesian product) of the provided values.
    Example:
        generate_hyperparameter_combinations({
            "lr": [1e-3, 1e-4],
            "batch": [16, 32]
        })
    returns:
        [{"lr": 1e-3, "batch": 16}, {"lr": 1e-3, "batch": 32}, {"lr": 1e-4, "batch": 16}, ...]
    """
    keys = list(param_grid.keys())
    value_lists = []
    for k in keys:
        v = param_grid[k]
        # Treat strings and non-iterables as singletons
        if isinstance(v, str) or not isinstance(v, Iterable):
            value_lists.append([v])
        else:
            value_lists.append(list(v))

    return [dict(zip(keys, combo)) for combo in product(*value_lists)]




DOWNSAMPLES = [1,2,4,8,16]
# DOWNSAMPLES = [16]
CHANNELS = [8,16,32,64]
job_id = "micropattern_individual_hyperparameters"
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
        "micropatterns.sh",
        str(downsample),
        str(channels)
    ]
    subprocess.Popen(command)
    time.sleep(5)  # slight delay to avoid overwhelming the scheduler
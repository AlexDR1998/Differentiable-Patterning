"""Optional JAX profiling and timing, isolated from training semantics."""

import os
import time
from pathlib import Path

import jax


def compile_and_time(jitted_function, *args):
    start = time.perf_counter()
    compiled = jitted_function.lower(*args).compile()
    return compiled, time.perf_counter() - start


def call_and_time(compiled_function, *args):
    start = time.perf_counter()
    outputs = compiled_function(*args)
    jax.block_until_ready(outputs)
    return outputs, time.perf_counter() - start


def maybe_save_device_memory_profile(step):
    if os.getenv("PROFILE_GPU", "0") != "1":
        return
    if step != int(os.getenv("PROFILE_GPU_STEP", "0")):
        return
    task_id = os.getenv("SLURM_ARRAY_TASK_ID", "0")
    configured_directory = (
        os.getenv("PROFILE_GPU_DIR") or os.getenv("RUN_CONFIG_PROFILE_DIR")
    )
    if configured_directory is None:
        job_id = os.getenv("SLURM_JOB_ID", "manual")
        directory = Path(os.getenv("SLURM_IO_ROOT", "output")) / "profiles"
        directory /= f"{job_id}_{task_id}"
    else:
        directory = Path(configured_directory)
    directory.mkdir(parents=True, exist_ok=True)
    profile = directory / f"train_step_{step}_device_memory.prof"
    try:
        jax.block_until_ready(jax.device_put(0))
        jax.profiler.save_device_memory_profile(str(profile))
    except Exception as exc:
        profile.with_suffix(".error.txt").write_text(f"{exc!r}\n")


def start_trace(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    options_class = getattr(jax.profiler, "ProfileOptions", None)
    kwargs = {}
    if options_class is not None:
        options = options_class()
        options.python_tracer_level = 0
        options.host_tracer_level = 2
        kwargs["profiler_options"] = options
    jax.block_until_ready(jax.device_put(0))
    try:
        jax.profiler.start_trace(str(directory), **kwargs)
    except TypeError:
        jax.profiler.start_trace(str(directory))


def stop_trace(outputs):
    jax.block_until_ready(outputs)
    jax.profiler.stop_trace()


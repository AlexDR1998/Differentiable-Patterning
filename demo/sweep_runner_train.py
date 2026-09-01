# sweep_runner.py ----------------------------------------------------
## cd ~; cd Differentiable-Patterning/; source venv/bin/activate; export WANDB_MODE=online; export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_ALLOCATOR=platform
import itertools, subprocess, pathlib, yaml, datetime as dt

cfg = yaml.safe_load(pathlib.Path("demo/sweep_config.yaml").read_text())

# build Cartesian product of all hyper-parameter values
keys, values = zip(*cfg["grid"].items())
for combo in itertools.product(*values):
    kwargs = dict(zip(keys, combo))

    stamp = dt.datetime.now().strftime("%Y-%m-%dT%H-%M")
    tag   = "_".join(f"{k[:2]}{v}" for k, v in kwargs.items())
    model_dir = pathlib.Path(cfg["output_root"], f"{stamp}_{tag}")
    model_dir.mkdir(parents=True, exist_ok=True)
    print(str(model_dir))

    args = [
        "python", cfg["script"],
        "--batches", str(kwargs["BATCHES"]),
        "--time_sampling", str(kwargs["TIME_SAMPLING"]),
        "--channels", str(kwargs["CHANNELS"]),
        "--pde", kwargs["PDE"],
        "--learn_rate",    str(kwargs["LEARN_RATE"]),
        "--loss",          kwargs["LOSS"],
        "--model_filename", str(model_dir),
        "--fire_rate", str(kwargs["FIRE_RATE"]),
        "--state_reg", str(kwargs["STATE_REGULARISER"]),
        "--target_sparsity",   str(kwargs["TARGET_SPARSITY"]),
        "--sparse_pruning",   str(kwargs["SPARSE_PRUNING"]),
        "--kernel_scale", str(kwargs["KERNEL_SCALE"]),
    ]
    print("Launching:", " ".join(args))
    subprocess.run(args, check=True)

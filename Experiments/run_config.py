from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, cast

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

workflow = importlib.import_module("Experiments.hydra_template.workflow")
import_callable = workflow.import_callable
load_config_from_entry = workflow.load_config_from_entry
resolve_manifest_index = workflow.resolve_manifest_index


def env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


def default_profile_dir() -> Path:
    task_id = os.getenv("SLURM_ARRAY_TASK_ID", "0")
    array_log_dir = os.getenv("SLURM_ARRAY_LOG_DIR")
    if array_log_dir:
        return Path(array_log_dir) / f"{task_id}.profile"

    job_id = os.getenv("SLURM_JOB_ID", "manual")
    root = Path(os.getenv("SLURM_IO_ROOT", REPO_ROOT / "output"))
    return root / "profiles" / f"{job_id}_{task_id}"


def synchronize_jax(jax_module: Any) -> None:
    if hasattr(jax_module, "effects_barrier"):
        jax_module.effects_barrier()

    jax_module.block_until_ready(jax_module.device_put(0))


def optional_cfg_value(cfg: Any, path: str, default: Any = None) -> Any:
    value = OmegaConf.select(cfg, path, default=default)
    return value


def configure_xla_flags(cfg: Any) -> None:
    xla_flag_map = {
        "triton_gemm": "--xla_gpu_triton_gemm_any=True",
        "latency_hiding_scheduler": "--xla_gpu_enable_latency_hiding_scheduler=true",
        "command_buffer": "--xla_gpu_enable_command_buffer=FUSION",
    }

    xla_flags = optional_cfg_value(cfg, "system.xla_flags")
    if xla_flags:
        flag_parts: list[str] = []
        for flag in xla_flags:
            flag_parts.append(xla_flag_map.get(str(flag), str(flag)))
        os.environ["XLA_FLAGS"] = " ".join(flag_parts)
        print(f"XLA_FLAGS={os.environ['XLA_FLAGS']}")
    else:
        os.environ.pop("XLA_FLAGS", None)
        print("XLA_FLAGS=<unset>")


def configure_jax_runtime(cfg: Any) -> Any:
    configure_xla_flags(cfg)

    import jax

    precision = optional_cfg_value(cfg, "system.precision", default="highest")
    jax.config.update("jax_default_matmul_precision", str(precision))
    print(f"jax_default_matmul_precision={precision}")

    if env_flag("RUN_CONFIG_INITIALISE_JAX_BACKEND"):
        jax.devices()

    return jax


def run_entrypoint(entrypoint_spec: str, cfg: Any) -> None:
    jax = configure_jax_runtime(cfg)
    entrypoint: Callable[[Any], Any] = import_callable(entrypoint_spec)
    if not env_flag("RUN_CONFIG_PROFILE"):
        entrypoint(cfg)
        return

    profile_dir = Path(os.getenv("RUN_CONFIG_PROFILE_DIR", default_profile_dir()))
    profile_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing JAX profile to: {profile_dir}")
    (profile_dir / "profile_started.txt").write_text(f"{time.time()}\n")

    if env_flag("RUN_CONFIG_PROFILE_TRACE"):
        create_perfetto_link = env_flag("RUN_CONFIG_PROFILE_PERFETTO_LINK")
        with jax.profiler.trace(str(profile_dir), create_perfetto_link=create_perfetto_link):
            entrypoint(cfg)
    else:
        entrypoint(cfg)

    try:
        synchronize_jax(jax)
        time.sleep(float(os.getenv("RUN_CONFIG_PROFILE_FLUSH_SECONDS", "2")))
    except Exception as exc:
        print(f"Warning: JAX profiling synchronization failed: {exc!r}", file=sys.stderr)

    if env_flag("RUN_CONFIG_PROFILE_MEMORY", "1"):
        memory_profile = profile_dir / "device_memory.prof"
        try:
            jax.profiler.save_device_memory_profile(str(memory_profile))
            print(f"Writing JAX device memory profile to: {memory_profile}")
        except Exception as exc:
            (profile_dir / "device_memory_error.txt").write_text(f"{exc!r}\n")
            print(f"Warning: JAX device memory profile failed: {exc!r}", file=sys.stderr)
    (profile_dir / "profile_finished.txt").write_text(f"{time.time()}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one config from a manifest or a standalone YAML file")
    parser.add_argument("--config-file", required=False, help="Run one concrete YAML config directly")
    parser.add_argument("--manifest", required=False, help="Inline manifest YAML produced by the generator")
    parser.add_argument("--index", type=int, required=False, help="Index to select from the manifest")
    parser.add_argument(
        "--worker-index",
        type=int,
        default=None,
        help="Worker id in the range [0, worker-count). Defaults to JOB_WORKER_INDEX.",
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=None,
        help="Total number of workers. Defaults to JOB_WORKER_COUNT.",
    )
    parser.add_argument(
        "--entrypoint",
        required=False,
        help="Python callable in the form module.path:function_name. Overrides the manifest entrypoint.",
    )
    args = parser.parse_args()

    if args.config_file:
        config_path = Path(args.config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        cfg = OmegaConf.load(config_path)
        entrypoint_spec = args.entrypoint
        if not entrypoint_spec:
            raise ValueError("--entrypoint is required when running a standalone config file")
        run_entrypoint(entrypoint_spec, cfg)
        return

    if not args.manifest:
        raise ValueError("Either --config-file or --manifest must be provided")

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest_cfg_container = OmegaConf.to_container(OmegaConf.load(manifest_path), resolve=True)
    if not isinstance(manifest_cfg_container, dict):
        raise TypeError("Manifest must resolve to a dictionary")
    manifest_cfg = cast(dict[str, Any], manifest_cfg_container)
    worker_index = args.worker_index
    if worker_index is None:
        worker_index = int(os.getenv("JOB_WORKER_INDEX", 0))
    worker_count = args.worker_count
    if worker_count is None:
        worker_count = int(os.getenv("JOB_WORKER_COUNT", 1))
    _, entry = resolve_manifest_index(manifest_cfg, index=args.index, worker_index=worker_index, worker_count=worker_count)
    cfg = load_config_from_entry(entry, manifest_dir=manifest_path.parent)

    entrypoint_spec = args.entrypoint or manifest_cfg.get("entrypoint")
    if not entrypoint_spec:
        raise ValueError("No entrypoint specified. Set it in the manifest or pass --entrypoint")

    run_entrypoint(entrypoint_spec, cfg)


if __name__ == "__main__":
    main()

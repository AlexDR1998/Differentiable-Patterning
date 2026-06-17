from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

workflow = importlib.import_module("Experiments.hydra_template.workflow")
import_callable = workflow.import_callable
load_config_from_entry = workflow.load_config_from_entry
resolve_manifest_index = workflow.resolve_manifest_index


def initialise_jax_backend() -> None:
    if os.getenv("RUN_CONFIG_INITIALISE_JAX_BACKEND", "1") != "1":
        return

    import jax

    jax.devices()


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
    initialise_jax_backend()

    if args.config_file:
        config_path = Path(args.config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        cfg = OmegaConf.load(config_path)
        entrypoint_spec = args.entrypoint
        if not entrypoint_spec:
            raise ValueError("--entrypoint is required when running a standalone config file")
        import_callable(entrypoint_spec)(cfg)
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

    import_callable(entrypoint_spec)(cfg)


if __name__ == "__main__":
    main()

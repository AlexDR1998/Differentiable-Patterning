#!/usr/bin/env python3
"""Run the smoke-test sweeps to check that training still works end to end.

Each smoke sweep (``conf/experiments/smoke_*.yaml``) contains a few very short
runs. A run passes when it exits cleanly and publishes a complete model bundle.
The runs go one after another on a single GPU, each in a fresh process through
``Experiments.run_config``, exactly as the cluster launchers run them.

The Nodal knockout fine-tuning sweep starts from an existing trained model,
set by hand in ``smoke_micropatterns_ko_finetune.yaml``. It is skipped until
that placeholder is replaced.

Usage (from the repository root, on a GPU machine):

    python Experiments/pipeline_smoke_test.py
    python Experiments/pipeline_smoke_test.py --only emoji snowmelt
    python Experiments/pipeline_smoke_test.py --dry-run    # manifests + Kubernetes commands

Manifests and logs go into one timestamped folder, by default
``logs/smoke/<time>/``. Bundles are published to the normal model store
(``MODEL_STORE_ROOT``, e.g. from ``.env``) under the ``pipeline-smoke``
collection, which is also where the fine-tuning parent is looked up.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Experiments.config_workflow import generate_manifest, load_yaml

# (domain, sweep name), in the order they run.
STAGES = [
    ("emoji", "smoke_emoji"),
    ("micropatterns", "smoke_micropatterns"),
    ("micropatterns", "smoke_micropatterns_ko_finetune"),
    ("snowmelt", "smoke_snowmelt"),
]
PARENT_PLACEHOLDER = "REPLACE_WITH_PARENT_MODEL_ID"
SMOKE_COLLECTION = "pipeline-smoke"


def smoke_bundles(model_root: Path) -> set[Path]:
    return set(model_root.glob(f"bundles/{SMOKE_COLLECTION}/*/*/manifest.yaml"))


def load_sweep(domain: str, sweep_name: str) -> tuple[dict, dict]:
    conf = REPO_ROOT / "Experiments" / domain / "conf"
    return load_yaml(conf / "base_config.yaml"), load_yaml(conf / "experiments" / f"{sweep_name}.yaml")


def run_entry(manifest_path: Path, index: int, env: dict[str, str], log_dir: Path) -> int:
    command = [
        sys.executable, "-m", "Experiments.run_config",
        "--manifest", str(manifest_path), "--index", str(index),
    ]
    stdout_path = log_dir / f"{index:02d}.out.log"
    stderr_path = log_dir / f"{index:02d}.err.log"
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        return subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=stdout, stderr=stderr).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only", nargs="+", choices=["emoji", "micropatterns", "snowmelt"],
        help="Run only these domains (default: all)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Folder for manifests and logs (default: logs/smoke/<UTC time>)",
    )
    parser.add_argument(
        "--model-store-root", type=Path, default=None,
        help="Model store (default: $MODEL_STORE_ROOT, else ./models)",
    )
    parser.add_argument(
        "--wandb-mode", choices=["online", "offline", "disabled"], default="online",
        help=f"W&B mode for every run (default: online, project '{SMOKE_COLLECTION}')",
    )
    parser.add_argument(
        "--gpu", default="H100",
        help="GPU type written into the printed Kubernetes commands (--dry-run only)",
    )
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed run")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Write manifests and print Kubernetes launch commands, without training",
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (args.out or REPO_ROOT / "logs" / "smoke" / stamp).resolve()
    model_root = (
        args.model_store_root or Path(os.environ.get("MODEL_STORE_ROOT", REPO_ROOT / "models"))
    ).resolve()
    stages = [stage for stage in STAGES if args.only is None or stage[0] in args.only]
    print(f"Output: {out_dir}")
    print(f"Model store: {model_root} (collection '{SMOKE_COLLECTION}')")

    env = os.environ.copy()
    env["MODEL_STORE_ROOT"] = str(model_root)
    env["WANDB_MODE"] = args.wandb_mode
    env.setdefault("PVC_PATH", f"{REPO_ROOT}/")

    results: list[tuple[str, int, str, float]] = []
    launch_commands: list[str] = []
    for domain, sweep_name in stages:
        base_cfg, sweep_cfg = load_sweep(domain, sweep_name)
        if PARENT_PLACEHOLDER in sweep_cfg["grid"].get("initialization.model_id", []):
            print(f"{sweep_name}: skipped, set initialization.model_id in its YAML file first")
            results.append((sweep_name, -1, "skipped: parent model ID not set", 0.0))
            continue

        manifest = generate_manifest(base_cfg, sweep_cfg, out_dir / sweep_name)
        manifest_path = out_dir / sweep_name / "manifest.yaml"
        count = int(manifest["count"])
        print(f"{sweep_name}: {count} runs")
        if args.dry_run:
            try:
                relative = manifest_path.relative_to(REPO_ROOT)
            except ValueError:
                relative = manifest_path  # outside the repo, so not visible to pods
            launch_commands.append(
                f"bash launch_batch_multi_job.sh Experiments/run_config.py {relative} {count} {args.gpu}"
            )
            continue

        log_dir = manifest_path.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        env["WANDB_DIR"] = str(log_dir)
        for index in range(count):
            before = smoke_bundles(model_root)
            start = time.time()
            returncode = run_entry(manifest_path, index, env, log_dir)
            minutes = (time.time() - start) / 60
            new_bundles = smoke_bundles(model_root) - before

            if returncode != 0:
                status = f"FAILED (exit {returncode}), see {log_dir / f'{index:02d}.err.log'}"
            elif len(new_bundles) != 1:
                status = f"FAILED: expected 1 new bundle, found {len(new_bundles)}"
            elif OmegaConf.load(new_bundles.pop()).status != "complete":
                status = "FAILED: bundle status is not 'complete'"
            else:
                status = "ok"
            results.append((sweep_name, index, status, minutes))
            print(f"  [{index}] {status} ({minutes:.1f} min)")
            if status != "ok" and not args.keep_going:
                break
        if results and results[-1][2].startswith("FAILED") and not args.keep_going:
            break

    if args.dry_run:
        print("\nDry run: manifests written, nothing trained. To run every smoke run as its own")
        print("pod on Kubernetes (from the repository root on the NFS checkout):")
        for command in launch_commands:
            print(f"  {command}")
        return 0

    print("\nSummary")
    for sweep_name, index, status, minutes in results:
        print(f"  {sweep_name:36s} {index:>3}  {minutes:5.1f} min  {status}")
    failed = [result for result in results if result[2].startswith("FAILED")]
    print(f"{len(results) - len(failed)}/{len(results)} runs passed or skipped. Output: {out_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

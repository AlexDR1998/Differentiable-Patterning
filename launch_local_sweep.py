#!/usr/bin/env python3
"""Run entries from an experiment manifest sequentially on one local GPU.

This is intentionally a scheduler-free companion to ``launch_batch_slurm.sh``:
the manifest remains the experiment contract, while each entry runs in a fresh
Python process through ``Experiments.run_config``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_manifest_metadata(path: Path) -> tuple[str, int]:
    """Read the two top-level fields needed by this launcher.

    The actual YAML is deliberately left to ``Experiments.run_config``, which
    uses OmegaConf. This keeps command discovery/dry-runs usable before the
    project environment is activated.
    """
    source = path.read_text(encoding="utf-8")
    name_match = re.search(r"^experiment_name:\s*([^#\n]+)", source, re.MULTILINE)
    count_match = re.search(r"^count:\s*(\d+)\s*$", source, re.MULTILINE)
    if not count_match:
        raise ValueError("Manifest has no top-level integer 'count' field")
    count = int(count_match.group(1))
    if count <= 0:
        raise ValueError("Manifest contains no runnable configs")
    name = name_match.group(1).strip().strip("\\\"'") if name_match else path.parent.name
    return name or path.parent.name, count


def successful_indices(status_path: Path) -> set[int]:
    """Return indices whose latest recorded terminal status was successful."""
    completed: dict[int, bool] = {}
    if not status_path.exists():
        return set()
    for line in status_path.read_text().splitlines():
        try:
            item = json.loads(line)
            index = int(item["index"])
            if item.get("event") == "finished":
                completed[index] = item.get("returncode") == 0
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return {index for index, succeeded in completed.items() if succeeded}


def append_status(status_path: Path, **record: Any) -> None:
    record["timestamp"] = utc_now()
    with status_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a manifest sequentially, using one local GPU."
    )
    parser.add_argument("manifest", type=Path, help="Path to manifest.yaml")
    parser.add_argument("--start", type=int, default=0, help="First index to run (inclusive)")
    parser.add_argument(
        "--stop", type=int, default=None, help="First index not to run (exclusive)"
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=REPO_ROOT / "logs" / "local_sweeps",
        help="Root directory for per-run stdout, stderr, and status records",
    )
    parser.add_argument(
        "--model-store-root",
        type=Path,
        default=REPO_ROOT / "models" / "local",
        help="Root directory for locally produced model bundles",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("offline", "online", "disabled"),
        default="offline",
        help="W&B mode passed to each run (default: offline)",
    )
    parser.add_argument(
        "--disable-preallocation",
        action="store_true",
        help="Set XLA_PYTHON_CLIENT_PREALLOCATE=false for each child process",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip indices whose latest status record completed successfully",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue with later entries after a failed run",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    if not manifest_path.is_file():
        parser.error(f"Manifest not found: {manifest_path}")
    experiment_name, count = load_manifest_metadata(manifest_path)
    stop = count if args.stop is None else args.stop
    if not 0 <= args.start <= stop <= count:
        parser.error(f"Require 0 <= start <= stop <= {count}; got {args.start}, {stop}")

    run_root = args.log_root.expanduser().resolve() / experiment_name
    model_root = args.model_store_root.expanduser().resolve() / experiment_name
    status_path = run_root / "status.jsonl"
    already_succeeded = successful_indices(status_path) if args.resume else set()
    indices = range(args.start, stop)

    print(f"Manifest: {manifest_path}")
    print(f"Experiment: {experiment_name}")
    print(f"Indices: {args.start}..{stop - 1} ({count} total in manifest)")
    print(f"Logs: {run_root}")
    print(f"Models: {model_root}")
    print(f"W&B mode: {args.wandb_mode}")

    if not args.dry_run:
        run_root.mkdir(parents=True, exist_ok=True)
        model_root.mkdir(parents=True, exist_ok=True)

    failed = False
    for index in indices:
        if index in already_succeeded:
            print(f"[{index}] skipping completed run (--resume)")
            continue

        command = [
            sys.executable,
            "-m",
            "Experiments.run_config",
            "--manifest",
            str(manifest_path),
            "--index",
            str(index),
        ]
        print(f"[{index}] {' '.join(command)}")
        if args.dry_run:
            continue

        env = os.environ.copy()
        env["MODEL_STORE_ROOT"] = str(model_root)
        env["WANDB_MODE"] = args.wandb_mode
        env.setdefault("PVC_PATH", f"{REPO_ROOT}/")
        env["WANDB_DIR"] = str(run_root / "wandb" / f"{index:04d}")
        if args.disable_preallocation:
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

        stdout_path = run_root / f"{index:04d}.out.log"
        stderr_path = run_root / f"{index:04d}.err.log"
        append_status(status_path, event="started", index=index, command=command)
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            result = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=stdout, stderr=stderr)
        append_status(status_path, event="finished", index=index, returncode=result.returncode)

        if result.returncode == 0:
            print(f"[{index}] completed")
            continue
        failed = True
        print(f"[{index}] failed (exit {result.returncode}); see {stderr_path}", file=sys.stderr)
        if not args.keep_going:
            return result.returncode

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run a heavy section of a notebook as remote jobs.

A job file is an ordinary sweep manifest (as read by ``launch/manifest.sh``
and ``Experiments.config_workflow.resolve_manifest_index``), so the
Kubernetes and Slurm launchers in ``launch/`` run it unchanged. Each entry
of ``configs`` is one part of the work, and each part writes one results
file ``part_<index>.npz`` into a shared folder. See
``docs/remote_notebook_jobs.md``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from omegaconf import OmegaConf

from Experiments.config_workflow import resolve_manifest_index

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def write_job_manifest(
    path: str | Path, experiment_name: str, configs: Sequence[dict[str, Any]]
) -> Path:
    """Write ``configs`` (one plain dict per part) as a manifest at ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_name": experiment_name,
        "count": len(configs),
        "configs": [{"index": index, "config": config} for index, config in enumerate(configs)],
    }
    OmegaConf.save(OmegaConf.create(manifest), path)
    return path


def read_job_entry(path: str | Path, index: int | None = None) -> tuple[int, dict[str, Any]]:
    """``(index, config)`` of one part of a manifest.

    Without ``index`` the part comes from ``JOB_COMPLETION_INDEX``,
    ``JOB_WORKER_*`` and ``MANIFEST_START_INDEX``, as for training runs.
    """
    manifest = OmegaConf.to_container(OmegaConf.load(Path(path)), resolve=True)
    if not isinstance(manifest, dict):
        raise TypeError(f"{path} is not a manifest")
    index, entry = resolve_manifest_index(manifest, index=index)
    return index, entry["config"]


def part_path(output_dir: str | Path, index: int) -> Path:
    """Results file of part ``index``."""
    return Path(output_dir) / f"part_{index:03d}.npz"


def part_paths(output_dir: str | Path) -> list[Path]:
    """Results files in ``output_dir``, in part order."""
    return sorted(Path(output_dir).glob("part_*.npz"))


def repository_path(path: str | Path) -> Path:
    """``path``, read relative to the repository root unless absolute."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def save_results_npz(path: str | Path, results: Sequence[dict[str, Any]], array_keys: Sequence[str]) -> Path:
    """Write result dicts to one ``.npz`` file.

    The entries named in ``array_keys`` are stored as arrays
    ``<result index>.<name>`` (None is left out); everything else (labels,
    groups, settings) as JSON under ``metadata``.
    """
    arrays, metadata = {}, []
    for index, result in enumerate(results):
        metadata.append({key: value for key, value in result.items() if key not in array_keys})
        for key in array_keys:
            if result.get(key) is not None:
                arrays[f"{index}.{key}"] = np.asarray(result[key])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written under a temporary name and then renamed, so a job killed while
    # writing leaves no half-written part (which would be skipped on a rerun)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "wb") as file:
        np.savez_compressed(
            file,
            metadata=np.asarray(json.dumps(
                {"version": 1, "results": metadata},
                default=lambda value: value.item() if hasattr(value, "item") else list(value),
            )),
            **arrays,
        )
    os.replace(temporary, path)
    return path


def load_results_npz(path: str | Path, array_keys: Sequence[str]) -> tuple[dict[str, Any], ...]:
    """Results written by ``save_results_npz``: one file, or a folder of
    ``part_*.npz`` files of a remote job (joined in part order). Arrays
    that were not saved come back as None."""
    path = Path(path)
    if path.is_dir():
        files = part_paths(path)
        if not files:
            raise FileNotFoundError(f"No part_*.npz files in {path}")
    else:
        files = [path]
    results = []
    for file in files:
        with np.load(file, allow_pickle=False) as stored:
            for index, result in enumerate(json.loads(str(stored["metadata"]))["results"]):
                for key in array_keys:
                    name = f"{index}.{key}"
                    result[key] = stored[name] if name in stored.files else None
                results.append(result)
    return tuple(results)

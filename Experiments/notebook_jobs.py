"""Run a heavy section of a notebook as remote jobs.

A job file is an ordinary sweep manifest (as read by ``launch/manifest.sh``
and ``Experiments.config_workflow.resolve_manifest_index``), so the
Kubernetes and Slurm launchers in ``launch/`` run it unchanged. Each entry
of ``configs`` is one part of the work, and each part writes one results
file ``part_<index>.npz`` into a shared folder. See
``docs/remote_notebook_jobs.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

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

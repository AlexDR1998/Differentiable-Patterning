from __future__ import annotations

import importlib
import itertools
import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def load_yaml(path: str | Path) -> dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # type: ignore[return-value]


def flatten_grid(node: dict[str, Any], prefix: str = "") -> list[tuple[str, list[Any]]]:
    items: list[tuple[str, list[Any]]] = []
    for key, value in node.items():
        dotted_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            items.extend(flatten_grid(value, dotted_key))
            continue
        if not isinstance(value, list):
            raise TypeError(f"Sweep value for '{dotted_key}' must be a list")
        items.append((dotted_key, value))
    return items


def build_nested_override(dotlist_items: dict[str, Any]) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for dotted_key, value in dotlist_items.items():
        cursor = root
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return root


def generate_manifest(base_cfg: dict[str, Any], sweep_cfg: dict[str, Any], output_dir: Path, emit_files: bool = False) -> dict[str, Any]:
    flat_grid = flatten_grid(sweep_cfg["grid"])
    keys = [key for key, _ in flat_grid]
    value_lists = [values for _, values in flat_grid]
    combos = list(itertools.product(*value_lists))

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_entries: list[dict[str, Any]] = []

    for index, combo in enumerate(combos):
        override_values = dict(zip(keys, combo))
        override_values["seed"] = index if sweep_cfg.get("seed_mode", "index") == "index" else base_cfg.get("seed", 0)
        generated_cfg = OmegaConf.merge(base_cfg, OmegaConf.create(build_nested_override(override_values)))

        entry: dict[str, Any] = {
            "index": index,
            "overrides": override_values,
            "config": OmegaConf.to_container(generated_cfg, resolve=True),
        }

        if emit_files:
            config_path = output_dir / f"config_{index:04d}.yaml"
            OmegaConf.save(generated_cfg, config_path)
            entry["config_path"] = str(config_path)

        generated_entries.append(entry)

    manifest = {
        "experiment_name": sweep_cfg.get("experiment_name"),
        "entrypoint": sweep_cfg.get("entrypoint"),
        "base_config": sweep_cfg.get("base_config"),
        "sweep_file": sweep_cfg.get("sweep_file"),
        "output_dir": str(output_dir),
        "count": len(generated_entries),
        "configs": generated_entries,
        "emit_files": bool(emit_files),
    }
    OmegaConf.save(OmegaConf.create(manifest), output_dir / "manifest.yaml")
    return manifest


def resolve_manifest_index(
    manifest_cfg: dict[str, Any],
    index: int | None = None,
    worker_index: int | None = None,
    worker_count: int | None = None,
) -> tuple[int, dict[str, Any]]:
    local_index = int(index if index is not None else os.getenv("JOB_COMPLETION_INDEX", 0))
    resolved_worker_index = int(worker_index if worker_index is not None else os.getenv("JOB_WORKER_INDEX", 0))
    resolved_worker_count = int(worker_count if worker_count is not None else os.getenv("JOB_WORKER_COUNT", 1))
    if resolved_worker_count <= 0:
        raise ValueError("worker_count must be greater than zero")
    if not 0 <= resolved_worker_index < resolved_worker_count:
        raise ValueError(f"worker_index must be in [0, {resolved_worker_count - 1}]")

    resolved_index = local_index * resolved_worker_count + resolved_worker_index
    configs = manifest_cfg.get("configs", [])
    if not configs:
        raise ValueError("Manifest contains no configs")

    entry = next((config for config in configs if int(config.get("index", -1)) == resolved_index), None)
    if entry is None:
        if 0 <= resolved_index < len(configs):
            entry = configs[resolved_index]
        else:
            raise IndexError(f"Index {resolved_index} out of range for manifest with {len(configs)} entries")

    return resolved_index, entry


def load_config_from_entry(entry: dict[str, Any], manifest_dir: Path | None = None):
    if "config" in entry:
        return OmegaConf.create(entry["config"])

    if "config_path" in entry:
        config_path = Path(entry["config_path"])
        if not config_path.is_absolute() and manifest_dir is not None:
            config_path = manifest_dir / config_path
        return OmegaConf.load(config_path)

    raise ValueError("Manifest entry contains neither 'config' nor 'config_path'")


def import_callable(spec: str):
    module_name, function_name = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Could not import entrypoint module '{module_name}'. "
            "If you run scripts by path, prefer local entrypoints like 'example_experiment:run'. "
            "Package-style entrypoints (e.g. 'Experiments...') require running with '-m' from repo root."
        ) from exc
    return getattr(module, function_name)
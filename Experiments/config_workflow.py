from __future__ import annotations

import importlib
import itertools
import os
from pathlib import Path
from typing import Any, cast

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


def _expand_cartesian_grid(grid: dict[str, Any]) -> list[dict[str, Any]]:
    flat_grid = flatten_grid(grid)
    if not flat_grid:
        return [{}]

    keys = [key for key, _ in flat_grid]
    value_lists = [values for _, values in flat_grid]
    return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]


def _expand_group_block(group: dict[str, Any]) -> list[dict[str, Any]]:
    flat_group = flatten_grid(group)
    if not flat_group:
        return [{}]

    lengths = {len(values) for _, values in flat_group}
    if len(lengths) != 1:
        raise ValueError("All values in a grouped sweep block must have the same length")

    length = lengths.pop()
    return [{key: values[index] for key, values in flat_group} for index in range(length)]


def _expand_section(section: dict[str, Any]) -> list[dict[str, Any]]:
    grid_combos = _expand_cartesian_grid(cast(dict[str, Any], section.get("grid", {})))
    group_blocks = cast(list[dict[str, Any]], section.get("groups", []))
    if not group_blocks:
        return grid_combos

    grouped_combos = [_expand_group_block(group) for group in group_blocks]
    expanded: list[dict[str, Any]] = []
    for combo in itertools.product(grid_combos, *grouped_combos):
        merged: dict[str, Any] = {}
        for values in combo:
            overlap = set(merged).intersection(values)
            if overlap:
                raise ValueError(f"Duplicate sweep keys across grouped sections: {sorted(overlap)}")
            merged.update(values)
        expanded.append(merged)
    return expanded


def build_nested_override(dotlist_items: dict[str, Any]) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for dotted_key, value in dotlist_items.items():
        cursor = root
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return root


def _matches_condition(values: dict[str, Any], condition: dict[str, Any]) -> bool:
    return all(values.get(key) == expected_value for key, expected_value in condition.items())


def _branch_keys(branches: list[dict[str, Any]]) -> set[str]:
    branch_keys: set[str] = set()
    for branch in branches:
        branch_keys.update(key for key, _ in flatten_grid(cast(dict[str, Any], branch.get("grid", {}))))
        for group in cast(list[dict[str, Any]], branch.get("groups", [])):
            branch_keys.update(key for key, _ in flatten_grid(group))
    return branch_keys


def generate_manifest(base_cfg: dict[str, Any], sweep_cfg: dict[str, Any], output_dir: Path, emit_files: bool = False) -> dict[str, Any]:
    combos = _expand_section(sweep_cfg)
    branches = cast(list[dict[str, Any]], sweep_cfg.get("branches", []))
    branch_value_keys = _branch_keys(branches) if branches else set()

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_entries: list[dict[str, Any]] = []

    for combo in combos:
        base_overrides = dict(combo)
        matching_branches = [branch for branch in branches if _matches_condition(base_overrides, cast(dict[str, Any], branch.get("when", {})))]

        if len(matching_branches) > 1:
            raise ValueError(f"Multiple branches matched overrides {base_overrides}: {matching_branches}")

        branch_options = matching_branches or [None]
        for branch in branch_options:
            branch_combos = _expand_section(branch) if branch else [{}]

            for branch_combo in branch_combos:
                override_values = dict(base_overrides)
                override_values.update(branch_combo)
                for branch_key in branch_value_keys:
                    override_values.setdefault(branch_key, None)
                generated_index = len(generated_entries)
                override_values["seed"] = (
                    generated_index if sweep_cfg.get("seed_mode", "index") == "index" else base_cfg.get("seed", 0)
                )

                generated_cfg = OmegaConf.merge(base_cfg, OmegaConf.create(build_nested_override(override_values)))

                entry: dict[str, Any] = {
                    "index": generated_index,
                    "overrides": override_values,
                    "config": OmegaConf.to_container(generated_cfg, resolve=True),
                }

                if emit_files:
                    config_path = output_dir / f"config_{entry['index']:04d}.yaml"
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
    start_index = int(os.getenv("MANIFEST_START_INDEX", 0))
    if resolved_worker_count <= 0:
        raise ValueError("worker_count must be greater than zero")
    if not 0 <= resolved_worker_index < resolved_worker_count:
        raise ValueError(f"worker_index must be in [0, {resolved_worker_count - 1}]")

    resolved_index = start_index + local_index * resolved_worker_count + resolved_worker_index
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

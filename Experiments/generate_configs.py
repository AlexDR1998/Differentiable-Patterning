from __future__ import annotations
import random
import argparse
import os
import sys
from pathlib import Path
from typing import Any
from omegaconf import OmegaConf  

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from Experiments.config_workflow import generate_manifest, load_yaml


def _list_yaml_files(folder: Path) -> list[Path]:
    return sorted([*folder.glob("*.yaml"), *folder.glob("*.yml")])


def _sweep_stem_candidates(sweep_file: Path) -> list[str]:
    stem = sweep_file.stem
    candidates = [stem]
    for suffix in ("_sweep", "-sweep", "sweep"):
        if stem.endswith(suffix):
            candidates.append(stem[: -len(suffix)].rstrip("_-") or stem)
    return candidates


def _resolve_base_config(
    sweep_file: Path,
    sweep_cfg: dict[str, Any],
    baselines_path: Path,
    experiments_dir: Path,
) -> Path:
    configured_base = sweep_cfg.get("base_config")
    if isinstance(configured_base, str) and configured_base:
        configured_path = Path(configured_base)
        if configured_path.is_absolute() and configured_path.exists():
            return configured_path

        for parent in (sweep_file.parent, experiments_dir, baselines_path if baselines_path.is_dir() else baselines_path.parent):
            candidate = (parent / configured_path).resolve()
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"base_config '{configured_base}' from {sweep_file} was not found in expected locations"
        )

    if baselines_path.is_file():
        return baselines_path

    if not baselines_path.is_dir():
        raise FileNotFoundError(f"PathToBaselines does not exist: {baselines_path}")

    stem_candidates = _sweep_stem_candidates(sweep_file)
    filename_candidates: list[str] = []
    for stem in stem_candidates:
        filename_candidates.extend([f"{stem}.yaml", f"{stem}.yml"])
    filename_candidates.extend(["config.yaml", "config.yml", "baseline.yaml", "baseline.yml", "base.yaml", "base.yml"])

    for filename in filename_candidates:
        candidate = baselines_path / filename
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find a baseline config. Checked "
        f"{baselines_path} using sweep-derived names {filename_candidates}."
    )


def _resolve_output_dir(output_root: Path, sweep_cfg: dict[str, Any], sweep_file: Path, sweep_count: int) -> Path:
    output_subdir = sweep_cfg.get("output_subdir")
    if isinstance(output_subdir, str) and output_subdir:
        leaf = Path(output_subdir).name
    else:
        experiment_name = sweep_cfg.get("experiment_name")
        leaf = str(experiment_name) if experiment_name else sweep_file.stem

    # For one sweep, allow writing directly to the requested destination.
    if sweep_count == 1 and not output_subdir:
        return output_root

    return output_root / leaf


def _discover_sweep_files(experiments_path: Path) -> tuple[Path, list[Path]]:
    if experiments_path.is_file():
        return experiments_path.parent.parent, [experiments_path]

    experiments_dir = experiments_path
    conf_experiments = experiments_dir / "conf" / "experiments"
    if not conf_experiments.exists():
        raise FileNotFoundError(f"Expected sweep directory at {conf_experiments}")

    sweep_files = _list_yaml_files(conf_experiments)
    if not sweep_files:
        raise FileNotFoundError(f"No sweep YAML files found in {conf_experiments}")

    return experiments_dir, sweep_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Hydra manifests for one experiment folder using the shared template workflow"
    )
    parser.add_argument("path_to_experiments", help="Path to experiment folder (e.g. Experiments/micropatterns)")
    parser.add_argument("path_to_baselines", help="Path to baseline YAML file or directory of baseline YAML files")
    parser.add_argument("path_to_output", help="Path to directory where generated manifests/config files are written")
    parser.add_argument("--emit-files", action="store_true", help="Also write one YAML file per generated config")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for deterministic generation of per-config random seeds in the manifest",
    )
    parser.add_argument(
        "--shuffle-indices",
        action="store_true",
        help="Randomly permute the per-entry 'index' values in the manifest (for better load balancing)",
    )
    args = parser.parse_args()
    rng = random.Random(args.seed)
    experiments_path = Path(args.path_to_experiments).resolve()
    baselines_path = Path(args.path_to_baselines).resolve()
    output_root = Path(args.path_to_output).resolve()

    experiments_dir, sweep_files = _discover_sweep_files(experiments_path)

    wrote: list[tuple[Path, int]] = []
    for sweep_file in sweep_files:
        sweep_cfg = load_yaml(sweep_file)
        base_config = _resolve_base_config(sweep_file, sweep_cfg, baselines_path, experiments_dir)
        base_cfg = load_yaml(base_config)

        sweep_cfg["base_config"] = str(base_config)
        sweep_cfg["sweep_file"] = str(sweep_file)
        sweep_cfg["entrypoint"] = sweep_cfg.get(
            "entrypoint", f"Experiments.{experiments_dir.name}.example_experiment:run"
        )

        output_dir = _resolve_output_dir(output_root, sweep_cfg, sweep_file, len(sweep_files))
        manifest = generate_manifest(base_cfg, sweep_cfg, output_dir, emit_files=bool(args.emit_files))
        should_resave_manifest = False

        if args.seed is not None:
            for item in manifest.get("configs", []):
                new_seed = rng.randint(0, 2**31 - 1)

                if isinstance(item.get("overrides"), dict) and "seed" in item["overrides"]:
                    item["overrides"]["seed"] = new_seed
                if isinstance(item.get("config"), dict) and "seed" in item["config"]:
                    item["config"]["seed"] = new_seed

            should_resave_manifest = True

        # If requested, shuffle the 'index' values (and update any config_path that depends on it).
        if args.shuffle_indices:
            configs = manifest.get("configs", [])
            old_indices = [item.get("index") for item in configs]
            new_indices = list(old_indices)
            rng.shuffle(new_indices)

            for item, new_idx in zip(configs, new_indices, strict=False):
                item["index"] = int(new_idx)
                if "config_path" in item and isinstance(item["config_path"], str):
                    item["config_path"] = str(output_dir / f"config_{int(new_idx):04d}.yaml")

            should_resave_manifest = True

        # generate_manifest writes manifest.yaml internally; resave only if we changed it.
        if should_resave_manifest:
            from omegaconf import OmegaConf  # local import to keep changes minimal

            OmegaConf.save(OmegaConf.create(manifest), output_dir / "manifest.yaml")


        wrote.append((output_dir / "manifest.yaml", int(manifest["count"])))

    for manifest_path, count in wrote:
        print(f"Wrote {count} configs to {manifest_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

workflow = importlib.import_module("Experiments.hydra_template.workflow")
generate_manifest = workflow.generate_manifest
load_yaml = workflow.load_yaml


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_CONFIG = BASE_DIR / "conf" / "config.yaml"
DEFAULT_SWEEP_FILE = BASE_DIR / "conf" / "experiments" / "example_sweep.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an inline manifest for a generic Hydra experiment")
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG), help="Base YAML config")
    parser.add_argument("--sweep-file", default=str(DEFAULT_SWEEP_FILE), help="Sweep YAML file")
    parser.add_argument("--output-dir", default=None, help="Directory for the manifest and optional configs")
    parser.add_argument("--emit-files", action="store_true", help="Also write one YAML file per generated config")
    args = parser.parse_args()

    base_config_path = Path(args.base_config)
    sweep_file_path = Path(args.sweep_file)
    output_dir = Path(args.output_dir) if args.output_dir else BASE_DIR / "generated"

    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")
    if not sweep_file_path.exists():
        raise FileNotFoundError(f"Sweep file not found: {sweep_file_path}")

    base_cfg = load_yaml(base_config_path)
    sweep_cfg = load_yaml(sweep_file_path)
    sweep_cfg["base_config"] = str(base_config_path)
    sweep_cfg["sweep_file"] = str(sweep_file_path)

    manifest = generate_manifest(base_cfg, sweep_cfg, output_dir, emit_files=bool(args.emit_files))
    print(f"Wrote {manifest['count']} configs to {output_dir / 'manifest.yaml'}")


if __name__ == "__main__":
    main()
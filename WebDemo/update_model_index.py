#!/usr/bin/env python3
"""Refresh the static model list used by the WebDemo model selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def update_model_index(models_dir: Path) -> Path:
    models = []
    for model_dir in sorted(models_dir.iterdir()):
        if model_dir.is_dir() and (model_dir / "manifest.json").exists():
            models.append({"id": model_dir.name, "label": model_dir.name})
    out_path = models_dir / "index.json"
    out_path.write_text(json.dumps({"models": models}, indent=2) + "\n")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("WebDemo/public/models"),
        help="Directory containing exported model subdirectories.",
    )
    return parser.parse_args()


def main() -> None:
    out_path = update_model_index(parse_args().models_dir)
    print(f"Updated {out_path}")


if __name__ == "__main__":
    main()

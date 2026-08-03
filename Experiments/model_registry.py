"""Inspect and rebuild the local model catalogue without running experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from Common.model_registry import ModelRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        help="Model store; defaults to MODEL_STORE_ROOT, then ./models",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("reindex", help="Rebuild registry.sqlite from manifests")
    subparsers.add_parser("list", help="Print the indexed model table")
    show = subparsers.add_parser("show", help="Print one bundle manifest")
    show.add_argument("model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root or os.environ.get("MODEL_STORE_ROOT") or Path("models")
    registry = ModelRegistry(root)
    if args.command == "reindex":
        print(registry.reindex())
    elif args.command == "list":
        print(registry.models_df().to_string(index=False))
    elif args.command == "show":
        bundle = registry.get(args.model)
        manifest = OmegaConf.to_container(bundle.manifest, resolve=True)
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

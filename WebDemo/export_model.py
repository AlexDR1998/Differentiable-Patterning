#!/usr/bin/env python3
"""Export a plain NCA model to static assets for the WebGL demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from NCA.model.NCA_model import NCA


SUPPORTED_KERNEL_ORDER = ["ID", "GRAD", "LAP"]


def _activation(name: str):
    if name == "relu":
        return jax.nn.relu
    raise ValueError(f"Unsupported activation {name!r}; MVP supports only 'relu'.")


def _expanded_kernel_count(kernels: Iterable[str]) -> int:
    count = 0
    for kernel in kernels:
        if kernel == "GRAD":
            count += 2
        else:
            count += 1
    return count


def _write_array(
    path,
    arrays: dict[str, np.ndarray],
    name: str,
    value,
    dtype=np.float32,
) -> None:
    arr = np.asarray(value, dtype=dtype)
    arrays[name] = arr
    arr.tofile(path)


def _make_seed_state(channels: int, height: int, width: int) -> np.ndarray:
    x0 = np.zeros((channels, height, width), dtype=np.float32)
    x0[3:, height // 2, width // 2] = 1.0
    return x0


def _rollout_reference(model: NCA, x0: np.ndarray, steps: int) -> np.ndarray:
    deterministic = eqx.tree_at(lambda m: m.FIRE_RATE, model, 1.0)
    trajectory = deterministic.run(
        iters=steps,
        x=jnp.asarray(x0),
        key=jax.random.PRNGKey(0),
    )
    return np.asarray(trajectory[-1], dtype=np.float32)


def _tensor_entry(name: str, arr: np.ndarray, offset: int) -> tuple[dict, int]:
    nbytes = int(arr.nbytes)
    return (
        {
            "name": name,
            "dtype": "float32",
            "shape": list(arr.shape),
            "byteOffset": offset,
            "byteLength": nbytes,
        },
        offset + nbytes,
    )


def _refresh_model_index(output_dir: Path) -> None:
    models = []
    for model_dir in sorted(output_dir.iterdir()):
        if model_dir.is_dir() and (model_dir / "manifest.json").exists():
            models.append({"id": model_dir.name, "label": model_dir.name})
    (output_dir / "index.json").write_text(json.dumps({"models": models}, indent=2) + "\n")


def export(args: argparse.Namespace) -> Path:
    kernels = list(args.kernels)
    if kernels != SUPPORTED_KERNEL_ORDER:
        raise ValueError(
            "MVP exporter/runtime supports only kernels "
            f"{SUPPORTED_KERNEL_ORDER}; got {kernels}."
        )
    if args.family != "NCA":
        raise ValueError("MVP exporter supports only family=NCA.")
    if args.padding != "CIRCULAR":
        raise ValueError("MVP WebGL runtime supports only CIRCULAR padding.")

    height, width = args.grid_size
    model = NCA(
        N_CHANNELS=args.channels,
        KERNEL_STR=kernels,
        ACTIVATION=_activation(args.activation),
        PADDING=args.padding,
        FIRE_RATE=args.fire_rate,
        key=jax.random.PRNGKey(0),
    )
    model = model.load(args.model_path)

    expected_features = args.channels * _expanded_kernel_count(kernels)
    if model.N_FEATURES != expected_features:
        raise ValueError(
            f"Feature mismatch: model has {model.N_FEATURES}, "
            f"expected {expected_features} from channels/kernels."
        )

    w0 = np.asarray(jnp.squeeze(model.layers[0].weight), dtype=np.float32)
    w1 = np.asarray(jnp.squeeze(model.layers[2].weight), dtype=np.float32)
    b1 = np.asarray(jnp.squeeze(model.layers[2].bias), dtype=np.float32)
    grad_x = np.asarray(jnp.squeeze(model.op.grad_x.weight), dtype=np.float32)
    grad_y = np.asarray(jnp.squeeze(model.op.grad_y.weight), dtype=np.float32)
    lap = np.asarray(jnp.squeeze(model.op.laplacian.weight), dtype=np.float32)
    average = np.asarray(jnp.squeeze(model.op.average.weight), dtype=np.float32)

    if w0.shape != (expected_features, expected_features):
        raise ValueError(f"Unexpected w0 shape {w0.shape}")
    if w1.shape != (args.channels, expected_features):
        raise ValueError(f"Unexpected w1 shape {w1.shape}")
    if b1.shape != (args.channels,):
        raise ValueError(f"Unexpected b1 shape {b1.shape}")
    for name, kernel in {
        "grad_x": grad_x,
        "grad_y": grad_y,
        "lap": lap,
        "average": average,
    }.items():
        if kernel.shape != (3, 3):
            raise ValueError(f"{name} has shape {kernel.shape}; MVP supports 3x3 kernels.")

    if args.x0_npy is None:
        x0 = _make_seed_state(args.channels, height, width)
    else:
        x0 = np.asarray(np.load(args.x0_npy), dtype=np.float32)
        if x0.shape != (args.channels, height, width):
            raise ValueError(f"x0 has shape {x0.shape}; expected {(args.channels, height, width)}")

    out_dir = args.output_dir / args.model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = {
        "w0": w0,
        "w1": w1,
        "b1": b1,
        "grad_x": grad_x,
        "grad_y": grad_y,
        "lap": lap,
        "average": average,
    }
    weight_path = out_dir / "weights.bin"
    offset = 0
    tensor_entries = {}
    with weight_path.open("wb") as f:
        for name, arr in arrays.items():
            arr.tofile(f)
            tensor_entries[name], offset = _tensor_entry(name, arr, offset)

    x0.tofile(out_dir / "x0.bin")
    reference = _rollout_reference(model, x0, args.reference_steps)
    reference.tofile(out_dir / "reference.bin")

    manifest = {
        "modelId": args.model_id,
        "family": args.family,
        "channels": args.channels,
        "kernels": kernels,
        "activation": args.activation,
        "padding": args.padding,
        "fireRate": args.fire_rate,
        "gridSize": [width, height],
        "featureChannels": expected_features,
        "hiddenChannels": expected_features,
        "weights": {
            "path": "weights.bin",
            "tensors": tensor_entries,
        },
        "initialState": {
            "path": "x0.bin",
            "dtype": "float32",
            "shape": [args.channels, height, width],
        },
        "display": {
            "channels": [0, 1, 2],
            "range": [0.0, 1.0],
        },
        "validation": {
            "referenceSteps": args.reference_steps,
            "referenceFireRate": 1.0,
            "reference": {
                "path": "reference.bin",
                "dtype": "float32",
                "shape": [args.channels, height, width],
            },
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _refresh_model_index(args.output_dir)
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--family", default="NCA")
    parser.add_argument("--channels", type=int, required=True)
    parser.add_argument("--kernels", nargs="+", required=True)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--padding", default="CIRCULAR")
    parser.add_argument("--fire-rate", type=float, default=0.5)
    parser.add_argument("--grid-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), required=True)
    parser.add_argument("--reference-steps", type=int, default=8)
    parser.add_argument("--x0-npy", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("WebDemo/public/models"))
    return parser.parse_args()


def main() -> None:
    out_dir = export(parse_args())
    print(f"Exported WebGL assets to {out_dir}")


if __name__ == "__main__":
    main()

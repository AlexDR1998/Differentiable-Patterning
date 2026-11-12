"""Segment the exported 0 h SOX2/LMBR images with Cellpose, once, and store the labels.

Runs in the separate segmentation environment (NumPy, PyTorch, Cellpose;
``requirements/requirements_segmentation.txt``), not the training one, so
it imports nothing from the repository. The inputs come from
``nucleus_segmentation.export_inputs``; see that module for the workflow.

    ~/.venvs/dp-segmentation/bin/python Experiments/micropatterns/segment_nuclei_cellpose.py \\
        --root <segmentation root> --method cpsam_sox2_lmbr_d13 --channels sox2 lmbr --diameter 13

Each method writes ``<root>/<method>/<image>.npz`` (``labels``) and a
``manifest.yaml`` with its settings. Images already done are skipped.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import yaml


def normalised(image, colony):
    """``image`` scaled between its 1st and 99.8th percentiles inside the colony."""

    low, high = np.percentile(image[colony], [1.0, 99.8])
    return np.clip((image - low) / max(high - low, 1e-6), 0.0, 1.0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, help="Segmentation root (with inputs/)")
    parser.add_argument("--method", required=True, help="Name of the stored segmentation")
    parser.add_argument("--channels", nargs="+", default=["sox2", "lmbr"], choices=["sox2", "lmbr"])
    parser.add_argument("--model", default="cpsam_v2")
    parser.add_argument("--diameter", type=float, default=None, help="Nucleus diameter in px (None: the model's own scale)")
    parser.add_argument("--flow-threshold", type=float, default=0.4)
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
    parser.add_argument("--min-size", type=int, default=15)
    parser.add_argument("--images", nargs="*", default=None, help="Only these input names (default: all)")
    parser.add_argument("--gpu", action="store_true", help="Run on the GPU (CUDA PyTorch)")
    parser.add_argument("--batch-size", type=int, default=8, help="Tiles per batch (lower it if GPU memory runs out)")
    args = parser.parse_args()

    from cellpose import models, version

    root = Path(args.root).expanduser()
    out = root / args.method
    out.mkdir(parents=True, exist_ok=True)
    settings = {
        "method": "cellpose", "cellpose_version": str(version), "model": args.model, "channels": args.channels,
        "diameter": args.diameter, "flow_threshold": args.flow_threshold,
        "cellprob_threshold": args.cellprob_threshold, "min_size": args.min_size,
    }
    model = models.CellposeModel(gpu=args.gpu, pretrained_model=args.model)
    manifest = out / "manifest.yaml"
    content = yaml.safe_load(manifest.read_text()) if manifest.exists() else {}
    content.update({"method": args.method, "settings": settings})
    content.setdefault("images", [])
    inputs = sorted((root / "inputs").glob("*.npz"))
    if args.images:
        inputs = [path for path in inputs if path.stem in args.images]
    for path in inputs:
        target = out / path.name
        if target.exists():
            continue
        with np.load(path) as stored:
            colony = stored["colony"]
            channels = [normalised(stored[name], colony) for name in args.channels]
            relative = str(stored["relative_path"])
        # Only the colony's bounding box is segmented
        rows, cols = np.flatnonzero(colony.any(axis=1)), np.flatnonzero(colony.any(axis=0))
        box = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
        image = np.stack([channel[box] for channel in channels], axis=-1)
        started = time.time()
        masks, _, _ = model.eval(
            image, channel_axis=-1, batch_size=args.batch_size, diameter=args.diameter, flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold, min_size=args.min_size, normalize=True,
        )
        labels = np.zeros(colony.shape, dtype=np.int32)
        labels[box] = masks
        np.savez_compressed(target, labels=labels)
        if relative not in content["images"]:
            content["images"].append(relative)
        manifest.write_text(yaml.safe_dump(content, sort_keys=False))
        print(f"{path.stem}: {int(masks.max())} nuclei in {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()

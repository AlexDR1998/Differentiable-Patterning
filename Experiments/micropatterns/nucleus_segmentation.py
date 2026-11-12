"""Nuclei of the 0 h SOX2/LMBR images, segmented once and stored.

The cell density and the real-cell pieces of the synthetic initial textures
come from the ``cell_fate_s1`` panel (SOX2 fills each nucleus at 0 h, LMBR
outlines it). Segmenting them well is slow, so it is done once per method
and the label images are stored:

1. ``export_inputs`` writes each image's raw SOX2 and LMBR and its colony
   mask to ``<root>/inputs`` (this environment).
2. ``segment_nuclei_cellpose.py`` segments them in a separate environment
   with PyTorch and Cellpose (``requirements/requirements_segmentation.txt``)
   and writes ``<root>/<method>/<image>.npz``; ``write_classical_labels``
   stores the classical method (``cell_counts.count_nuclei``) the same way.
3. ``nucleus_segmentation_explorer.py`` collects hand-clicked nuclei on a
   few crops (ground truth) and scores each method against them
   (``score_labels``).
4. With ``CellSettings.segmentation`` set to a method, the counts and the
   real-cell pieces use its stored labels (``stored_labels``); other images
   fall back to the classical method.

``<root>`` is ``NUCLEUS_SEGMENTATION_ROOT``, else ``260726_nca_segmentation``
next to the dataset folder.
"""

import os
from pathlib import Path

import numpy as np
import yaml

GROUP = "cell_fate_s1"
CHANNELS = ("SOX2", "LMBR")
# Ground truth clicked in the explorer (repository relative)
ANNOTATION_FILE = Path(__file__).resolve().parent / "conf/segmentation/260726_s1_0h_annotations.yaml"
# Side of an annotation crop (µm)
CROP_UM = 100.0


def segmentation_root(dataset_root):
    """Folder of the stored segmentations of the dataset at ``dataset_root``."""

    configured = os.environ.get("NUCLEUS_SEGMENTATION_ROOT")
    if configured:
        return Path(configured).expanduser()
    return Path(dataset_root).expanduser().resolve().parent / "260726_nca_segmentation"


def image_name(relative_path):
    """File name of an image's stored files: its dataset-relative path, flattened."""

    return Path(relative_path).as_posix().replace("/", "__").removesuffix(".tif").removesuffix(".ome")


def segmentation_images(dataset_root):
    """Dataset-relative paths of the 0 h ``cell_fate_s1`` images, every projection."""

    from Common.dataloader.micropattern_projections import PROJECTION_FOLDERS

    root = Path(dataset_root).expanduser().resolve()
    paths = []
    for folder in PROJECTION_FOLDERS.values():
        for path in sorted((root / "cell_fate_markers" / f"ctrl_s1_{folder}").glob("*_0h*.tif")):
            paths.append(path.relative_to(root).as_posix())
    return paths


def projection_of(relative_path):
    """``max``, ``sum`` or ``slice``, from an image's folder."""

    from Common.dataloader.micropattern_projections import PROJECTION_FOLDERS

    folder = Path(relative_path).parent.name
    for projection, suffix in PROJECTION_FOLDERS.items():
        if folder.endswith(f"_{suffix}"):
            return projection
    raise ValueError(f"No projection in {relative_path}")


def export_inputs(dataset_root, alignment=None):
    """Write ``<root>/inputs/<image>.npz`` (raw ``sox2``, ``lmbr``, ``colony`` mask, ``pixel_um``) for every image."""

    from Common.dataloader.micropattern_260726 import read_raw_channels
    from Experiments.micropatterns.cell_counts import colony_mask, default_alignment, um_per_pixel

    alignment = alignment or default_alignment()
    root = Path(dataset_root).expanduser().resolve()
    folder = segmentation_root(root) / "inputs"
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for relative in segmentation_images(root):
        channels, names, _ = read_raw_channels(root / relative, GROUP)
        markers = [name.rsplit("/", 1)[-1] for name in names]
        sox2, lmbr = (channels[..., markers.index(marker)] for marker in CHANNELS)
        colony = colony_mask(root / relative, root, alignment, sox2.shape)
        path = folder / f"{image_name(relative)}.npz"
        np.savez_compressed(
            path, sox2=sox2.astype(np.float32), lmbr=lmbr.astype(np.float32), colony=colony,
            pixel_um=um_per_pixel(alignment.pattern_radius), relative_path=relative,
        )
        written.append(path)
    return written


def write_labels(dataset_root, method, relative_path, labels, settings):
    """Store the labels ``[X, Y]`` of one image for ``method``, and ``settings`` in its manifest."""

    folder = segmentation_root(dataset_root) / method
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f"{image_name(relative_path)}.npz", labels=labels.astype(np.int32))
    manifest = folder / "manifest.yaml"
    content = yaml.safe_load(manifest.read_text()) if manifest.exists() else {}
    content.update({"method": method, "settings": settings})
    content.setdefault("images", [])
    if relative_path not in content["images"]:
        content["images"].append(relative_path)
    manifest.write_text(yaml.safe_dump(content, sort_keys=False))


def write_classical_labels(dataset_root, method, options_by_projection):
    """Store ``cell_counts.count_nuclei`` labels of every exported image as ``method``.

    ``options_by_projection`` maps a projection to its
    ``CellSettings.nucleus_options()`` (the classical settings tuned for it).
    """

    from Experiments.micropatterns.cell_counts import count_nuclei

    for path in sorted((segmentation_root(dataset_root) / "inputs").glob("*.npz")):
        with np.load(path) as stored:
            lmbr, colony = stored["lmbr"], stored["colony"]
            pixel_um, relative = float(stored["pixel_um"]), str(stored["relative_path"])
        options = {_k: _v for _k, _v in options_by_projection[projection_of(relative)].items() if _k != "segmentation"}
        diameter_um = options.pop("nucleus_diameter_um")
        _, labels, _ = count_nuclei(lmbr, colony, diameter_um / pixel_um, **options)
        settings = {"method": "classical", "by_projection": {
            _p: {_k: _v for _k, _v in _o.items() if _k != "segmentation"} for _p, _o in options_by_projection.items()
        }}
        write_labels(dataset_root, method, relative, labels, settings)


def stored_methods(dataset_root):
    """Names of the stored segmentation methods (folders with a manifest)."""

    root = segmentation_root(dataset_root)
    return sorted(path.parent.name for path in root.glob("*/manifest.yaml")) if root.exists() else []


def stored_labels(dataset_root, method, relative_path):
    """The stored labels ``[X, Y]`` of one image for ``method``, or None if it has none."""

    path = segmentation_root(dataset_root) / method / f"{image_name(relative_path)}.npz"
    if not path.exists():
        return None
    with np.load(path) as stored:
        return stored["labels"]


def relabel_inside(labels, mask):
    """``labels`` with nuclei whose centre lies outside ``mask`` removed, numbered 1 .. n."""

    from scipy import ndimage as ndi

    labels = np.asarray(labels)
    ids = np.unique(labels[labels > 0])
    if not len(ids):
        return np.zeros_like(labels, dtype=np.int32)
    centres = np.round(ndi.center_of_mass(np.ones_like(labels), labels, ids)).astype(int)
    keep = ids[np.asarray(mask)[tuple(centres.T)]]
    renumber = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    renumber[keep] = np.arange(1, len(keep) + 1)
    return renumber[labels]


# ---------------------------------------------------------------- ground truth


def annotation_crops(dataset_root, alignment=None):
    """The crops to click nuclei on: a few per projection, at the centre, half way and near the edge.

    Returns dicts with ``image`` (dataset-relative path), ``top``, ``left``
    and ``size`` (px), in a fixed order so annotations can be resumed.
    """

    from Experiments.micropatterns.cell_counts import default_alignment, um_per_pixel

    alignment = alignment or default_alignment()
    size = int(round(CROP_UM / um_per_pixel(alignment.pattern_radius)))
    images = segmentation_images(dataset_root)
    by_projection = {}
    for relative in images:
        by_projection.setdefault(projection_of(relative), []).append(relative)
    # (projection, image index, distance from the centre as a share of the radius, angle)
    plan = [
        ("slice", 0, 0.0, 0.0), ("slice", 1, 0.5, 45.0), ("slice", 2, 0.8, 200.0),
        ("max", 0, 0.0, 0.0), ("max", 3, 0.5, 120.0),
        ("sum", 1, 0.5, 300.0),
    ]
    crops = []
    for projection, index, reach, angle in plan:
        options = by_projection.get(projection, [])
        if not options:
            continue
        relative = options[index % len(options)]
        from Common.dataloader.micropattern_projections import image_key

        centre = np.asarray(alignment.centre(relative), dtype=float)
        # Keep the whole crop inside the colony
        distance = reach * max(alignment.pattern_radius - 0.75 * size, 0.0)
        offset = distance * np.array([np.sin(np.radians(angle)), np.cos(np.radians(angle))])
        top, left = np.round(centre + offset - size / 2).astype(int)
        crops.append({"image": relative, "key": image_key(relative), "top": int(top), "left": int(left), "size": size})
    return crops


def read_annotations(path=ANNOTATION_FILE):
    """``{crop id: [[row, column], ...]}`` of clicked nucleus centres (image pixels)."""

    path = Path(path)
    if not path.exists():
        return {}
    content = yaml.safe_load(path.read_text()) or {}
    return {crop["id"]: [list(map(float, point)) for point in crop.get("points", [])] for crop in content.get("crops", [])}


def crop_id(crop):
    return f"{crop['image']}@{crop['top']},{crop['left']},{crop['size']}"


def save_annotations(crops, points, path=ANNOTATION_FILE):
    """Write the clicked centres ``points`` (``{crop id: [[row, column], ...]}``) of ``crops``."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = {
        "note": "Nucleus centres clicked by hand on 0 h cell_fate_s1 images (SOX2 + LMBR), image pixels",
        "crops": [
            {"id": crop_id(crop), **crop, "points": [[round(r, 2), round(c, 2)] for r, c in points.get(crop_id(crop), [])]}
            for crop in crops
            if crop_id(crop) in points
        ],
    }
    path.write_text(yaml.safe_dump(content, sort_keys=False))
    return path


# ---------------------------------------------------------------- scoring


def _matches(found, clicked, radius):
    """Index pairs ``(found, clicked)`` of the one-to-one matches within ``radius`` (Hungarian method)."""

    from scipy.optimize import linear_sum_assignment

    if not len(found) or not len(clicked):
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    distance = np.hypot(*(found[:, None] - clicked[None]).transpose(2, 0, 1))
    rows, columns = linear_sum_assignment(np.where(distance <= radius, distance, 1e6))
    close = distance[rows, columns] <= radius
    return rows[close], columns[close]


def match_points(found, clicked, radius):
    """``(matched, extra, missed)``: one-to-one matches of ``found`` and ``clicked`` points within ``radius``.

    Points are ``[n, 2]``. Matches minimise the total distance; pairs further
    apart than ``radius`` do not match. ``extra`` are found points with no
    click (false positives), ``missed`` clicks with no found point (false
    negatives, e.g. from merged nuclei).
    """

    found = np.asarray(found, dtype=float).reshape(-1, 2)
    clicked = np.asarray(clicked, dtype=float).reshape(-1, 2)
    rows, _ = _matches(found, clicked, radius)
    return len(rows), len(found) - len(rows), len(clicked) - len(rows)


def label_centres(labels):
    """Centres ``[n, 2]`` (row, column) of the labelled nuclei."""

    from scipy import ndimage as ndi

    ids = np.unique(labels[labels > 0])
    if not len(ids):
        return np.zeros((0, 2))
    return np.asarray(ndi.center_of_mass(np.ones_like(labels), labels, ids)).reshape(-1, 2)


def score_labels(labels, crop, clicked, radius):
    """Scores of one crop: the found nuclei against the clicked centres (image pixels).

    Clicks are nuclei centred in the crop. Found nuclei centred up to
    ``radius`` outside the crop can match them, so a centre that falls just
    over the edge still counts. Unmatched found nuclei count as extra only
    when centred at least ``radius`` inside the edge, since near it a click
    and a centre can fall on either side.
    """

    top, left, size = crop["top"], crop["left"], crop["size"]
    margin = int(np.ceil(radius))
    row0, column0 = max(top - margin, 0), max(left - margin, 0)
    centres = label_centres(labels[row0 : top + size + margin, column0 : left + size + margin]) + [row0, column0]
    clicked = np.asarray(clicked, dtype=float).reshape(-1, 2)
    rows, _ = _matches(centres, clicked, radius)
    unmatched = np.setdiff1d(np.arange(len(centres)), rows)
    inner = (
        (centres[unmatched, 0] >= top + radius) & (centres[unmatched, 0] < top + size - radius)
        & (centres[unmatched, 1] >= left + radius) & (centres[unmatched, 1] < left + size - radius)
    )
    matched, extra = len(rows), int(np.count_nonzero(inner))
    missed = len(clicked) - matched
    return {
        "clicked": len(clicked),
        "found": matched + extra,
        "matched": matched,
        "extra": extra,
        "missed": missed,
        "precision": matched / max(matched + extra, 1),
        "recall": matched / max(len(clicked), 1),
        "f1": 2 * matched / max(2 * matched + extra + missed, 1),
        "count_ratio": (matched + extra) / max(len(clicked), 1),
    }

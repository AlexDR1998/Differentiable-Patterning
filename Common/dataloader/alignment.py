"""Centring circular micropattern colonies, and the file the centres are kept in.

Every micropattern has the same size, so all images share one pattern radius
and only the centre of each colony is fitted. The colony pixels of an image
(``colony_foreground`` in ``micropattern_260726.py``) are averaged into a
coarse coverage map and matched against a flat disk of the pattern radius,
with a penalty for colony just outside it. The best match puts the disk edge
at the outer edge of the cells, so a denser side of the colony, or a sparse
early timepoint, does not pull the centre or shrink the circle.

The centres are computed once, checked and corrected by hand in
``Experiments/micropatterns/data_cleaning.py``, and saved in an alignment
file that the training loader reads (see ``load_colony_alignment``).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import scipy.ndimage as ndi
import scipy.signal as signal
import yaml

ALIGNMENT_FORMAT = "colony_alignment_v1"


def block_centre(index, downsample):
    """Full-resolution position of a coordinate on a ``downsample`` grid.

    Block ``i`` covers full-resolution pixels ``d*i`` to ``d*i + d - 1``, so
    its centre is at ``d*i + (d - 1) / 2``.
    """
    return downsample * np.asarray(index, dtype=float) + (downsample - 1) / 2.0


def grid_position(position, downsample):
    """Inverse of ``block_centre``: a full-resolution position on a coarse grid."""
    return (np.asarray(position, dtype=float) - (downsample - 1) / 2.0) / downsample


def coverage_map(foreground, downsample=4, blur=0.0):
    """Fraction of colony pixels per ``downsample`` block, scaled so the densest part is 1.

    ``blur`` (full-resolution pixels) optionally smooths the map first. It is
    off by default: blur spreads the dense side of a colony outwards and
    pulls the fitted centre towards it.
    """
    foreground = np.asarray(foreground, dtype=np.float32)
    height = foreground.shape[0] // downsample
    width = foreground.shape[1] // downsample
    coverage = (
        foreground[: height * downsample, : width * downsample]
        .reshape(height, downsample, width, downsample)
        .mean(axis=(1, 3))
    )
    if blur:
        coverage = ndi.gaussian_filter(coverage, blur / downsample)
    top = np.percentile(coverage, 99)
    return np.clip(coverage / top, 0.0, 1.0) if top > 0 else coverage


def fit_colony_centre(coverage, radius, edge_width=0.08):
    """Fit the centre of a disk of ``radius`` (coverage-map pixels) to a coverage map.

    The template is 1 inside the disk and -1 in a ring of width
    ``edge_width * radius`` just outside it; the inside is flat, because a
    soft edge pulls the disk towards the denser side of the colony. Returns
    ``(centre, score)``: the ``(row, column)`` centre on the coverage map,
    refined to a fraction of a pixel, and a fit score. The score is 1 minus
    the coverage just outside the disk relative to the coverage inside it,
    so it does not depend on cell density: near 1 when no colony lies
    outside the fitted pattern.
    """
    coverage = np.asarray(coverage, dtype=np.float64)
    edge = max(edge_width * radius, 1.0)
    size = int(np.ceil(radius + edge)) + 1
    rows, columns = np.mgrid[-size : size + 1, -size : size + 1]
    distance = np.hypot(rows, columns)
    template = (distance <= radius).astype(np.float64)
    template[(distance > radius) & (distance <= radius + edge)] = -1.0
    scores = signal.fftconvolve(coverage, template, mode="same")
    peak = np.unravel_index(np.argmax(scores), scores.shape)

    centre = []
    for axis in range(2):
        low, high = list(peak), list(peak)
        low[axis] -= 1
        high[axis] += 1
        if low[axis] < 0 or high[axis] >= scores.shape[axis]:
            centre.append(float(peak[axis]))
            continue
        left, middle, right = scores[tuple(low)], scores[peak], scores[tuple(high)]
        curvature = left - 2 * middle + right
        step = 0.5 * (left - right) / curvature if curvature < 0 else 0.0
        centre.append(float(peak[axis]) + float(np.clip(step, -0.5, 0.5)))
    centre = tuple(centre)
    return centre, fit_score(coverage, centre, radius, edge_width)


def fit_score(coverage, centre, radius, edge_width=0.08):
    """1 minus the coverage just outside a disk relative to inside it (see ``fit_colony_centre``)."""
    edge = max(edge_width * radius, 1.0)
    rows, columns = np.ogrid[: coverage.shape[0], : coverage.shape[1]]
    distance = np.hypot(rows - centre[0], columns - centre[1])
    inside = coverage[distance <= radius].mean()
    ring = coverage[(distance > radius) & (distance <= radius + edge)].mean()
    return float(np.clip(1.0 - ring / inside, 0.0, 1.0)) if inside > 0 else 0.0


@dataclass(frozen=True)
class ColonyAlignment:
    """Colony centres for a fixed dataset, as saved in an alignment file.

    ``centres`` maps each image path, relative to the dataset root (e.g.
    ``cell_fate_markers/ctrl_s1/A2_F14_24h.ome.tif``), to the colony centre
    ``(row, column)`` in full-resolution pixels of the raw image. All colonies
    share ``pattern_radius`` (full-resolution pixels). ``details`` keeps, per
    image, the automatic fit, the manual correction and the fit score, and
    ``settings`` the fit settings; the loader only reads the centres.
    """

    pattern_radius: float
    centres: Mapping[str, tuple[float, float]]
    details: Mapping[str, Mapping] = field(default_factory=dict)
    settings: Mapping = field(default_factory=dict)

    def centre(self, relative_path):
        try:
            return self.centres[relative_path]
        except KeyError:
            raise KeyError(
                f"The alignment file has no centre for {relative_path}. Export the "
                "alignment again from Experiments/micropatterns/data_cleaning.py."
            ) from None


def save_colony_alignment(path, alignment):
    """Write an alignment file (YAML, readable and easy to compare in git)."""
    images = {}
    for name in sorted(alignment.centres):
        entry = {"centre": [round(float(value), 2) for value in alignment.centres[name]]}
        for key, value in alignment.details.get(name, {}).items():
            entry[key] = (
                [round(float(item), 2) for item in value]
                if isinstance(value, (tuple, list))
                else round(float(value), 3)
            )
        images[name] = entry
    content = {
        "format": ALIGNMENT_FORMAT,
        "pattern_radius": round(float(alignment.pattern_radius), 2),
        "settings": dict(alignment.settings),
        "images": images,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        file.write(
            "# Colony centres for the 260726 micropattern dataset, made with\n"
            "# Experiments/micropatterns/data_cleaning.py. Centres are (row, column)\n"
            "# in full-resolution pixels of the raw image: the automatic fit plus\n"
            "# any manual correction.\n"
        )
        yaml.safe_dump(content, file, sort_keys=False)


def load_colony_alignment(path):
    """Read an alignment file written by ``save_colony_alignment``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Alignment file {path} does not exist. Export it from "
            "Experiments/micropatterns/data_cleaning.py (section Export alignment)."
        )
    with open(path) as file:
        content = yaml.safe_load(file)
    if not isinstance(content, Mapping) or content.get("format") != ALIGNMENT_FORMAT:
        raise ValueError(f"{path} is not a {ALIGNMENT_FORMAT} alignment file")
    images = content.get("images") or {}
    return ColonyAlignment(
        pattern_radius=float(content["pattern_radius"]),
        centres={
            name: (float(entry["centre"][0]), float(entry["centre"][1]))
            for name, entry in images.items()
        },
        details={
            name: {key: value for key, value in entry.items() if key != "centre"}
            for name, entry in images.items()
        },
        settings=dict(content.get("settings") or {}),
    )

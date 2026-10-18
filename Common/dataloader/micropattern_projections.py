"""Projections of the 260726 confocal stacks, and projection-free image names.

The cell-fate and protein response stainings are thin 3D confocal stacks,
stored as three 2D versions in separate folders, e.g. for control stain 1:

- ``cell_fate_markers/ctrl_s1_max/A2_F14_24h.ome.tif`` (max projection),
- ``cell_fate_markers/ctrl_s1_sums/A2_F14_24h_sum.tif`` (sum projection),
- ``cell_fate_markers/ctrl_s1_slices/A2_F14_24h_slice.tif`` (one slice),

and likewise ``signalling/protein_response/ctrl_{max,sums,slices}``. The RNA
was imaged in 2D, so it has one version only.

All versions of an image show the same colony in the same field of view, so
the colony centres and the quality flags are shared between them. They are
kept under ``image_key``, which drops the projection from the path.
"""

from pathlib import PurePosixPath
import re

# Projection name (data.micropattern.projection) -> folder name suffix.
PROJECTION_FOLDERS = {"max": "max", "sum": "sums", "slice": "slices"}
PROJECTIONS = tuple(PROJECTION_FOLDERS)

_FOLDER_SUFFIX = re.compile(
    "_(" + "|".join(PROJECTION_FOLDERS.values()) + ")$"
)
_FILE_PREFIX = re.compile(r"^MAX_")
_FILE_SUFFIX = re.compile(r"(_sum|_slice)?(\.ome)?\.tif$", re.IGNORECASE)
# Condition written before the timestep in protein response names, spelled
# differently between projections (CTRL, sl0, S0, SL0, sl24, S24, ...). The
# folder already gives the condition, so it is dropped.
_CONDITION_TOKEN = re.compile(r"_(ctrl|sl?\d+)(?=_\d+h$)", re.IGNORECASE)


def check_projection(projection):
    if projection not in PROJECTION_FOLDERS:
        raise ValueError(
            f"Unknown projection {projection!r}; expected one of {PROJECTIONS}"
        )
    return projection


def image_stem(filename):
    """File name without the projection or condition.

    ``MAX_A2_F1_12h.ome.tif`` -> ``A2_F1_12h`` and
    ``B2_F1_S0_36h_sum.tif`` -> ``B2_F1_36h``.
    """
    stem = _FILE_SUFFIX.sub("", _FILE_PREFIX.sub("", filename))
    return _CONDITION_TOKEN.sub("", stem)


def image_key(relative_path):
    """Path relative to the dataset root, without the projection.

    ``cell_fate_markers/ctrl_s1_sums/A2_F14_24h_sum.tif`` and
    ``cell_fate_markers/ctrl_s1_max/A2_F14_24h.ome.tif`` both give
    ``cell_fate_markers/ctrl_s1/A2_F14_24h``. Paths from before the
    projections were added (``cell_fate_markers/ctrl_s1/A2_F14_24h.ome.tif``)
    give the same key, so older alignment and quality flag files still work.
    """
    path = PurePosixPath(relative_path)
    folder = _FOLDER_SUFFIX.sub("", path.parent.name)
    return (path.parent.parent / folder / image_stem(path.name)).as_posix()


def keyed_by_image(mapping, label):
    """``mapping`` with its path keys replaced by ``image_key``."""
    keyed = {}
    for path, value in mapping.items():
        key = image_key(path)
        if key in keyed:
            raise ValueError(f"{label} lists {key} more than once (under different projections)")
        keyed[key] = value
    return keyed

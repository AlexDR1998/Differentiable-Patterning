"""Settings for synthetic 0 h initial textures, one section per projection, in a YAML file.

Tuned and saved in ``synthetic_textured_ic_explorer.py``. Each projection
(``max``, ``sum``, ``slice``) has two settings blocks and a note:

    slice:
      texture:    # textures that keep the measured colonies' own gaps
        mode: patch
        patch_size: 11
        ...
      cells:      # whole measured cells placed at a chosen cell density
        placement: tiled
        pieces: nuclei
        ...
      tuned_on: {condition: ctrl, downsample: 8, saved: 2026-11-05T12:00:00Z}

Read by:

- inverse design (``inverse_design.texture.file``): ``texture``
- the density × KO sweep (``density_ko_sweep.py``, set up in
  ``nodal_ko_eval.py``): ``cells``
- ``boundary_mask_explorer.py`` and ``drawn_mask_explorer.py``: starting values

Patch sizes are in pixels at ``patch_downsample`` and are rescaled to the
resolution they are used at (``scaled_patch_geometry``). With
``patch_downsample: null`` they are used as given. Pixel sizes in ``cells``
are model pixels at the downsample noted in ``tuned_on``.
"""

from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import yaml

from Common.dataloader.micropattern_projections import PROJECTIONS
from Experiments.micropatterns.cell_counts import (
    NUCLEUS_DIAMETER_UM,
    NUCLEUS_LARGEST,
    NUCLEUS_SMALLEST,
    NUCLEUS_ROUNDNESS,
    NUCLEUS_SEED_SPACING,
    NUCLEUS_SPLIT_METHOD,
    NUCLEUS_SPLIT_METHODS,
    NUCLEUS_THRESHOLD,
)

DEFAULT_TEXTURE_SETTINGS_FILE = Path(__file__).resolve().parent / "conf/textures/260726_texture_settings.yaml"

# Must match Experiments/inverse_design/config.py:TextureConfig (apart from bank_size and file)
TEXTURE_SETTING_KEYS = (
    "mode",
    "patch_size",
    "patch_overlap",
    "patch_jitter",
    "patch_downsample",
    "interior_fraction",
    "candidates",
    "iaaft_iterations",
    "normalise",
    "normalise_iterations",
)
NOTE_KEY = "tuned_on"
SECTION_KEYS = ("texture", "cells", NOTE_KEY)


@dataclass(frozen=True)
class CellSettings:
    """How whole measured cells are placed at a chosen density (``synthetic_ic.py``)."""

    # tiled: cells grown from seed points, never overlapping, so every placed
    # cell is one cell in the image. layers: layers of cells that may overlap,
    # kept in a random order up to the density (inverse design's density model).
    placement: str = "tiled"
    # nuclei: real cells grown from the nuclei of the raw images; segments:
    # pieces cut out at model resolution (often several cells each)
    pieces: str = "nuclei"
    # Generic cell markers that tell cells from gaps
    markers: tuple = ("LMBR", "SOX2")
    # Segments split at marker maxima at least this far apart (px); 0: no split
    split_distance: int = 2
    # Smaller pieces are dropped (px)
    min_cell_area: int = 2
    # Pieces only come from within this fraction of the colony radius
    interior_fraction: float = 0.8
    # How far a cell reaches beyond its nucleus (µm; nuclei only)
    expansion_um: float = 2.0
    # Counting nuclei in the raw images (cell numbers, and pieces from nuclei):
    # the typical nucleus diameter (µm), the closest two nuclei are split at
    # (share of a diameter), and the stain threshold (multiple of its Otsu threshold)
    nucleus_diameter_um: float = NUCLEUS_DIAMETER_UM
    nucleus_seed_spacing: float = NUCLEUS_SEED_SPACING
    nucleus_threshold: float = NUCLEUS_THRESHOLD
    # How touching nuclei are told apart: distance (where blobs narrow), rim
    # (each nucleus's dark interior inside its LMBR rim) or intensity (stain maxima)
    nucleus_split_method: str = NUCLEUS_SPLIT_METHOD
    # Nuclei less round than this (solidity) are split again; 0: never
    nucleus_roundness: float = NUCLEUS_ROUNDNESS
    # Size limits, as multiples of the area of nucleus_diameter_um: smaller
    # pieces are dropped (debris); larger ones are split into about area /
    # nucleus area pieces (clumps; 0: never). Used for counts and pieces alike.
    nucleus_smallest: float = NUCLEUS_SMALLEST
    nucleus_largest: float = NUCLEUS_LARGEST
    # A stored nucleus segmentation (nucleus_segmentation.py) used where it has
    # the image (the 0 h SOX2/LMBR panel), instead of the settings above; "": none
    segmentation: str = ""
    # tiled: how far a cell reaches from its seed, times a measured cell's radius
    reach_scale: float = 1.0
    # tiled: closest two seeds, as a share of the mean seed spacing at the most cells
    seed_spacing: float = 0.6
    # tiled: cells are drawn on a grid this many times finer, then averaged
    tile_factor: int = 4
    # layers: layers placed at first (more are added when a density needs them)
    layers: int = 2
    # layers: random places tried per cell that would fit in the grid
    attempts: int = 10
    # layers: width of the density cut, in cells
    softness_cells: float = 0.5
    # layers: layer k draws cells with probability ~ area ** (-size_bias * k),
    # so the deeper layers that fill at high densities hold smaller cells
    size_bias: float = 0.0
    # layers: a placed cell counts as a cell of the initial state when at
    # least this share of it shows (layer 0 is drawn over the deeper layers)
    visible_fraction: float = 0.5
    # Staining panels with no cell structure at 0 h (low noise only): their
    # channels are pixels drawn at random from the measured colonies
    noise_panels: tuple = ("rna_expression",)
    # Staining panels whose nuclei give the measured cell density (and covered
    # share) of the real colonies; the SOX2/LMBR panel holds the 0 h cell markers
    density_panels: tuple = ("cell_fate_s1",)

    def __post_init__(self):
        object.__setattr__(self, "markers", tuple(self.markers))
        object.__setattr__(self, "noise_panels", tuple(self.noise_panels))
        object.__setattr__(self, "density_panels", tuple(self.density_panels))
        if self.placement not in ("tiled", "layers"):
            raise ValueError(f"Unknown cell placement {self.placement!r}; use tiled or layers")
        if self.pieces not in ("nuclei", "segments"):
            raise ValueError(f"Unknown cell pieces {self.pieces!r}; use nuclei or segments")
        if self.placement == "tiled" and self.pieces != "nuclei":
            raise ValueError("Tiled placement needs real cells (pieces: nuclei)")
        if not self.markers:
            raise ValueError("Give at least one cell marker")
        if not self.density_panels:
            raise ValueError("Give at least one staining panel for the measured density")
        if not 0.0 < self.visible_fraction <= 1.0:
            raise ValueError("visible_fraction must be in (0, 1]")
        if min(self.nucleus_diameter_um, self.nucleus_seed_spacing, self.nucleus_threshold) <= 0.0:
            raise ValueError("The nucleus settings must be above 0")
        if self.nucleus_split_method not in NUCLEUS_SPLIT_METHODS:
            raise ValueError(f"Unknown nucleus split method {self.nucleus_split_method!r}; use {NUCLEUS_SPLIT_METHODS}")
        if not 0.0 <= self.nucleus_roundness < 1.0:
            raise ValueError("nucleus_roundness must be in [0, 1)")
        if self.nucleus_smallest < 0.0 or (0.0 < self.nucleus_largest <= self.nucleus_smallest):
            raise ValueError("nucleus_largest must be 0 (off) or above nucleus_smallest, which must be 0 or more")

    def nucleus_options(self):
        """Keyword arguments of the nucleus counting functions (``cell_counts.py``)."""
        return {
            "nucleus_diameter_um": float(self.nucleus_diameter_um),
            "seed_spacing": float(self.nucleus_seed_spacing),
            "threshold_scale": float(self.nucleus_threshold),
            "split_method": str(self.nucleus_split_method),
            "roundness": float(self.nucleus_roundness),
            "smallest": float(self.nucleus_smallest),
            "largest": float(self.nucleus_largest),
            "segmentation": self.segmentation or None,
        }

    def to_dict(self):
        return {
            **asdict(self),
            "markers": list(self.markers),
            "noise_panels": list(self.noise_panels),
            "density_panels": list(self.density_panels),
        }


CELL_SETTING_KEYS = tuple(field.name for field in fields(CellSettings))


def scaled_patch_geometry(patch_size, overlap, jitter, patch_downsample, downsample):
    """Patch width (odd, at least 3), overlap and jitter in pixels at ``downsample``.

    The sizes are given at ``patch_downsample`` (None: already at ``downsample``).
    """

    scale = 1.0 if patch_downsample is None else float(patch_downsample) / float(downsample)
    width = max(3, 2 * int(round((float(patch_size) * scale - 1.0) / 2.0)) + 1)
    return (
        width,
        max(0, int(round(float(overlap) * scale))),
        max(0, int(round(float(jitter) * scale))),
    )


def read_texture_settings_file(path):
    """``{projection: section}`` of a settings file ({} if it does not exist)."""

    path = Path(path)
    if not path.exists():
        return {}
    content = yaml.safe_load(path.read_text()) or {}
    unknown = set(content) - set(PROJECTIONS)
    if unknown:
        raise ValueError(f"Unknown projections {sorted(unknown)} in {path}; use {list(PROJECTIONS)}")
    for projection, section in content.items():
        if set(section) - set(SECTION_KEYS):
            raise ValueError(
                f"The {projection!r} section of {path} has keys {sorted(section)}; "
                f"use {list(SECTION_KEYS)}"
            )
    return content


def _section(path, projection, key):
    content = read_texture_settings_file(path)
    if key not in content.get(projection, {}):
        raise KeyError(
            f"{path} has no {key!r} settings for the {projection!r} projection; "
            "save them from synthetic_textured_ic_explorer.py"
        )
    return dict(content[projection][key])


def load_texture_settings(path, projection):
    """The ``texture`` settings ``{key: value}`` saved for ``projection``."""

    section = _section(path, projection, "texture")
    unknown = set(section) - set(TEXTURE_SETTING_KEYS)
    missing = set(TEXTURE_SETTING_KEYS) - set(section)
    if unknown or missing:
        raise ValueError(
            f"Texture settings for {projection!r} in {path}: "
            f"unknown keys {sorted(unknown)}, missing keys {sorted(missing)}"
        )
    return section


def load_cell_settings(path, projection):
    """The ``cells`` settings saved for ``projection``, as ``CellSettings``."""

    section = _section(path, projection, "cells")
    missing = set(CELL_SETTING_KEYS) - set(section)
    if missing:
        raise ValueError(f"Cell settings for {projection!r} in {path}: missing keys {sorted(missing)}")
    return CellSettings(**section)


def save_texture_settings(path, projection, texture, cells, note=None):
    """Write the ``texture`` dict and ``cells`` (``CellSettings``) for ``projection``, keeping the other projections."""

    if projection not in PROJECTIONS:
        raise ValueError(f"Unknown projection {projection!r}; use {list(PROJECTIONS)}")
    if set(texture) != set(TEXTURE_SETTING_KEYS):
        raise ValueError(f"Texture settings need exactly the keys {list(TEXTURE_SETTING_KEYS)}")
    path = Path(path)
    content = read_texture_settings_file(path)
    content[projection] = {
        "texture": {key: texture[key] for key in TEXTURE_SETTING_KEYS},
        "cells": cells.to_dict(),
        NOTE_KEY: {
            **(note or {}),
            "saved": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(
        {name: content[name] for name in PROJECTIONS if name in content}, sort_keys=False
    ))
    return path

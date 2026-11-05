"""Texture synthesis settings for each projection, saved in a YAML file.

Tuned in ``boundary_mask_explorer.py`` and read by inverse design
(``inverse_design.texture.file``). The file has one section per projection
(``max``, ``sum``, ``slice``) with the keys in ``TEXTURE_SETTING_KEYS`` and an
optional ``tuned_on`` note (ignored when loading):

    slice:
      mode: patch
      patch_size: 11
      ...
      tuned_on: {condition: ctrl, downsample: 8, saved: 2026-11-05T12:00:00Z}

Patch sizes are in pixels at ``patch_downsample`` and are rescaled to the
resolution they are used at (``scaled_patch_geometry``). With
``patch_downsample: null`` they are used as given.
"""

from datetime import datetime, timezone
from pathlib import Path

import yaml

from Common.dataloader.micropattern_projections import PROJECTIONS

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
    return content


def load_texture_settings(path, projection):
    """The settings ``{key: value}`` saved for ``projection`` (without the note)."""

    content = read_texture_settings_file(path)
    if projection not in content:
        raise KeyError(
            f"{path} has no texture settings for the {projection!r} projection; "
            "save them from boundary_mask_explorer.py"
        )
    section = dict(content[projection])
    section.pop(NOTE_KEY, None)
    unknown = set(section) - set(TEXTURE_SETTING_KEYS)
    missing = set(TEXTURE_SETTING_KEYS) - set(section)
    if unknown or missing:
        raise ValueError(
            f"Texture settings for {projection!r} in {path}: "
            f"unknown keys {sorted(unknown)}, missing keys {sorted(missing)}"
        )
    return section


def save_texture_settings(path, projection, settings, note=None):
    """Write ``settings`` as the section for ``projection``, keeping the other sections."""

    if projection not in PROJECTIONS:
        raise ValueError(f"Unknown projection {projection!r}; use {list(PROJECTIONS)}")
    if set(settings) != set(TEXTURE_SETTING_KEYS):
        raise ValueError(f"Texture settings need exactly the keys {list(TEXTURE_SETTING_KEYS)}")
    path = Path(path)
    content = read_texture_settings_file(path)
    section = {key: settings[key] for key in TEXTURE_SETTING_KEYS}
    section[NOTE_KEY] = {
        **(note or {}),
        "saved": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    content[projection] = section
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(
        {name: content[name] for name in PROJECTIONS if name in content}, sort_keys=False
    ))
    return path

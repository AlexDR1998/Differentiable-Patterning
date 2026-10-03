"""Cell types from marker thresholds, and the file the rules are kept in.

A pixel is labelled from normalised (0-1) marker images: a marker is *high*
where its value is above its threshold. Each cell type has a rule giving, per
marker, ``"high"``, ``"low"`` or ``"any"`` (not checked). A pixel takes a cell
type when all of that type's rules hold; pixels matching no type are
``"Other"`` and pixels matching more than one are ``"Several"``.

The thresholds are tuned on the 48h images in
``Experiments/micropatterns/data_cleaning.py`` and exported to a cell type
file (see ``save_cell_type_rules``). They only mean something for images
normalised the same way, so the file also keeps the ``data.micropattern``
pre-processing settings they were tuned with.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

CELL_TYPES_FORMAT = "cell_type_rules_v1"
MARKER_STATES = ("high", "low", "any")
OTHER = "Other"
SEVERAL = "Several"


@dataclass(frozen=True)
class CellTypeRules:
    """Marker thresholds and cell-type rules.

    ``channels`` maps each marker to the measurement channel it was read from
    (e.g. ``"SOX17": "cell_fate_s2/SOX17"``), ``thresholds`` each marker to
    its threshold on the normalised image, and ``rules`` each cell type to
    ``{marker: "high" | "low" | "any"}``. ``hour`` is the timestep the
    thresholds were tuned on and ``preprocessing`` the ``data.micropattern``
    settings of the images; neither is used for labelling.
    """

    channels: Mapping[str, str]
    thresholds: Mapping[str, float]
    rules: Mapping[str, Mapping[str, str]]
    hour: int = 48
    preprocessing: Mapping = field(default_factory=dict)

    def __post_init__(self):
        markers = set(self.channels)
        if set(self.thresholds) != markers:
            raise ValueError("Cell type thresholds must list the same markers as channels")
        for cell_type, rule in self.rules.items():
            if cell_type in (OTHER, SEVERAL):
                raise ValueError(f"{cell_type!r} is reserved and cannot be a cell type")
            unknown = sorted(set(rule) - markers)
            if unknown:
                raise ValueError(f"Rule for {cell_type} uses unknown markers: " + ", ".join(unknown))
            bad = sorted(state for state in rule.values() if state not in MARKER_STATES)
            if bad:
                raise ValueError(
                    f"Rule for {cell_type} has states {bad}; use one of {MARKER_STATES}"
                )

    @property
    def markers(self):
        return tuple(self.channels)

    @property
    def cell_types(self):
        return tuple(self.rules)


def marker_values(images, channel_names, cell_type_rules, axis=-3):
    """Pick each marker's image out of a stacked array, as ``{marker: array}``.

    ``channel_names`` names the entries along ``axis`` (by default the
    channel axis of ``[..., channels, H, W]``). A marker is found by its
    channel in the rules (e.g. ``"cell_fate_s2/SOX17"``, for measurements) or
    else by its marker name (e.g. ``"SOX17"``, for model state channels).
    """
    images = np.asarray(images)
    channel_names = list(channel_names)
    values = {}
    for marker, channel in cell_type_rules.channels.items():
        if channel in channel_names:
            index = channel_names.index(channel)
        elif marker in channel_names:
            index = channel_names.index(marker)
        else:
            raise KeyError(f"Neither {channel!r} nor {marker!r} is among the channels {channel_names}")
        values[marker] = np.take(images, index, axis=axis)
    return values


def marker_high(values, cell_type_rules):
    """``{marker: boolean array}``, true where the marker is above its threshold."""
    return {
        marker: np.asarray(values[marker]) > cell_type_rules.thresholds[marker]
        for marker in cell_type_rules.markers
    }


def classify_cell_types(values, cell_type_rules, mask=None):
    """Label pixels from marker images ``{marker: array}``.

    Returns ``{name: boolean array}`` for every cell type, then ``"Several"``
    and ``"Other"``; each pixel inside ``mask`` (all pixels if None) is in
    exactly one of them, and pixels outside it in none.
    """
    high = marker_high(values, cell_type_rules)
    inside = np.ones_like(next(iter(high.values()))) if mask is None else np.asarray(mask, dtype=bool)
    matches = {}
    for cell_type, rule in cell_type_rules.rules.items():
        match = inside.copy()
        for marker, state in rule.items():
            if state == "high":
                match &= high[marker]
            elif state == "low":
                match &= ~high[marker]
        matches[cell_type] = match
    count = np.sum(list(matches.values()), axis=0) if matches else np.zeros(inside.shape, dtype=int)
    labels = {cell_type: match & (count == 1) for cell_type, match in matches.items()}
    labels[SEVERAL] = inside & (count > 1)
    labels[OTHER] = inside & (count == 0)
    return labels


def cell_type_fractions(labels, mask=None):
    """Fraction of the mask (or of all pixels) taken by each label."""
    first = np.asarray(next(iter(labels.values())))
    area = first.size if mask is None else int(np.sum(mask))
    return {name: float(np.sum(label)) / max(area, 1) for name, label in labels.items()}


def save_cell_type_rules(path, cell_type_rules):
    """Write a cell type file (YAML, readable and easy to compare in git)."""
    content = {
        "format": CELL_TYPES_FORMAT,
        "hour": int(cell_type_rules.hour),
        "markers": {
            marker: {
                "channel": str(cell_type_rules.channels[marker]),
                "threshold": round(float(cell_type_rules.thresholds[marker]), 4),
            }
            for marker in cell_type_rules.markers
        },
        "cell_types": {
            cell_type: {marker: str(state) for marker, state in rule.items()}
            for cell_type, rule in cell_type_rules.rules.items()
        },
        "preprocessing": dict(cell_type_rules.preprocessing),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        file.write(
            "# Cell-type rules made with Experiments/micropatterns/data_cleaning.py.\n"
            "# A marker is high where its normalised (0-1) value is above its\n"
            "# threshold; a cell type needs every listed marker state (any = not\n"
            "# checked). The thresholds hold for images processed with the\n"
            "# data.micropattern settings under preprocessing.\n"
        )
        yaml.safe_dump(content, file, sort_keys=False)


def load_cell_type_rules(path):
    """Read a cell type file written by ``save_cell_type_rules``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Cell type file {path} does not exist. Export it from "
            "Experiments/micropatterns/data_cleaning.py (section Cell types at 48h)."
        )
    with open(path) as file:
        content = yaml.safe_load(file)
    if not isinstance(content, Mapping) or content.get("format") != CELL_TYPES_FORMAT:
        raise ValueError(f"{path} is not a {CELL_TYPES_FORMAT} cell type file")
    markers = content.get("markers") or {}
    return CellTypeRules(
        channels={marker: str(entry["channel"]) for marker, entry in markers.items()},
        thresholds={marker: float(entry["threshold"]) for marker, entry in markers.items()},
        rules={
            str(cell_type): {str(marker): str(state) for marker, state in (rule or {}).items()}
            for cell_type, rule in (content.get("cell_types") or {}).items()
        },
        hour=int(content.get("hour", 48)),
        preprocessing=dict(content.get("preprocessing") or {}),
    )

"""Cell types from marker thresholds, and the file the rules are kept in.

A pixel is labelled from normalised (0-1) marker images: a marker is *high*
where its value is above its threshold. Each cell type has one or more
*clauses*, each giving, per marker, ``"high"``, ``"low"`` or ``"any"`` (not
checked). A pixel takes a cell type when every marker state of at least one
of its clauses holds (clauses are joined by "or"). Pixels matching no type
are ``"Other"`` and pixels matching more than one are ``"Several"``.

Each clause has a weight between 0 and 1 (1 by default) saying how strongly
matching it counts: a pixel matching clauses with weights w1, w2, ... belongs
to the cell type with probability 1 - (1 - w1)(1 - w2)..., a "noisy or".
With all weights 1 this is the plain "or". Weights only enter estimated
shares (``pattern_label_shares``); pixel labels use every clause with a
weight above 0.

Markers imaged in separate stains (different colonies) cannot be combined
pixel by pixel. ``stains`` records which markers were imaged together, and
``Common.dataloader.cell_type_shares`` estimates cell-type shares from them.

The thresholds are tuned on the 48h images in
``Experiments/micropatterns/data_cleaning.py`` and exported to a cell type
file (see ``save_cell_type_rules``). They only mean something for images
normalised the same way, so the file also keeps the ``data.micropattern``
pre-processing settings they were tuned with.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml

CELL_TYPES_FORMAT = "cell_type_rules_v2"
# Older files: one clause per cell type and no stains.
OLD_CELL_TYPES_FORMATS = ("cell_type_rules_v1",)
MARKER_STATES = ("high", "low", "any")
OTHER = "Other"
SEVERAL = "Several"


def _as_clauses(rule):
    """A cell type's rule as a tuple of clauses (one mapping = one clause)."""
    clauses = (rule,) if isinstance(rule, Mapping) else tuple(rule)
    return tuple({str(marker): str(state) for marker, state in clause.items()} for clause in clauses)


@dataclass(frozen=True)
class CellTypeRules:
    """Marker thresholds and cell-type rules.

    ``channels`` maps each marker to the measurement channel it is read from
    when all markers are taken from one image (e.g. ``"SOX17":
    "cell_fate_s2/SOX17"``), ``thresholds`` each marker to its threshold on
    the normalised image, and ``rules`` each cell type to a clause
    ``{marker: "high" | "low" | "any"}`` or a list of clauses joined by "or"
    (stored as a tuple of clauses). ``stains`` maps each stain (images taken
    together, e.g. ``"cell_fate_s1"``) to ``{marker: channel}``; it may be
    empty. ``clause_weights`` maps a cell type to one weight (0 to 1) per
    clause; cell types left out have weight 1 for every clause. ``hour`` is the timestep the thresholds were tuned on and
    ``preprocessing`` the ``data.micropattern`` settings of the images;
    neither is used for labelling.
    """

    channels: Mapping[str, str]
    thresholds: Mapping[str, float]
    rules: Mapping[str, Sequence[Mapping[str, str]]]
    hour: int = 48
    preprocessing: Mapping = field(default_factory=dict)
    stains: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    clause_weights: Mapping[str, Sequence[float]] = field(default_factory=dict)

    def __post_init__(self):
        markers = set(self.channels)
        if set(self.thresholds) != markers:
            raise ValueError("Cell type thresholds must list the same markers as channels")
        rules = {str(cell_type): _as_clauses(rule) for cell_type, rule in self.rules.items()}
        object.__setattr__(self, "rules", rules)
        for cell_type, clauses in rules.items():
            if cell_type in (OTHER, SEVERAL):
                raise ValueError(f"{cell_type!r} is reserved and cannot be a cell type")
            if not clauses:
                raise ValueError(f"Rule for {cell_type} has no clauses")
            for clause in clauses:
                unknown = sorted(set(clause) - markers)
                if unknown:
                    raise ValueError(f"Rule for {cell_type} uses unknown markers: " + ", ".join(unknown))
                bad = sorted(state for state in clause.values() if state not in MARKER_STATES)
                if bad:
                    raise ValueError(
                        f"Rule for {cell_type} has states {bad}; use one of {MARKER_STATES}"
                    )
        weights = {str(cell_type): tuple(float(w) for w in ws) for cell_type, ws in self.clause_weights.items()}
        object.__setattr__(self, "clause_weights", weights)
        for cell_type, ws in weights.items():
            if cell_type not in rules or len(ws) != len(rules[cell_type]):
                raise ValueError(f"Clause weights for {cell_type} must give one weight per clause")
            if any(not 0.0 <= w <= 1.0 for w in ws):
                raise ValueError(f"Clause weights for {cell_type} must be between 0 and 1")
        for stain, stain_channels in self.stains.items():
            unknown = sorted(set(stain_channels) - markers)
            if unknown:
                raise ValueError(f"Stain {stain} lists unknown markers: " + ", ".join(unknown))

    @property
    def markers(self):
        return tuple(self.channels)

    @property
    def cell_types(self):
        return tuple(self.rules)

    def weights_of(self, cell_type):
        """One weight per clause of ``cell_type`` (1 unless set)."""
        return self.clause_weights.get(cell_type, (1.0,) * len(self.rules[cell_type]))

    def active_rules(self):
        """``{cell type: clauses}`` without the clauses of weight 0."""
        return {
            cell_type: [clause for clause, w in zip(clauses, self.weights_of(cell_type)) if w > 0]
            for cell_type, clauses in self.rules.items()
        }


def marker_values(images, channel_names, cell_type_rules, axis=-3, channels=None):
    """Pick each marker's image out of a stacked array, as ``{marker: array}``.

    ``channel_names`` names the entries along ``axis`` (by default the
    channel axis of ``[..., channels, H, W]``). A marker is found by its
    channel in ``channels`` (by default the rules' ``channels``; pass a
    stain's ``{marker: channel}`` to read one stain, e.g.
    ``"cell_fate_s2/SOX17"``, for measurements) or else by its marker name
    (e.g. ``"SOX17"``, for model state channels).
    """
    images = np.asarray(images)
    channel_names = list(channel_names)
    values = {}
    for marker, channel in (cell_type_rules.channels if channels is None else channels).items():
        if channel in channel_names:
            index = channel_names.index(channel)
        elif marker in channel_names:
            index = channel_names.index(marker)
        else:
            raise KeyError(f"Neither {channel!r} nor {marker!r} is among the channels {channel_names}")
        values[marker] = np.take(images, index, axis=axis)
    return values


def marker_high(values, cell_type_rules):
    """``{marker: boolean array}``, true where the marker is above its threshold.

    Only the markers present in ``values`` are returned.
    """
    return {
        marker: np.asarray(values[marker]) > cell_type_rules.thresholds[marker]
        for marker in cell_type_rules.markers
        if marker in values
    }


def _label_pixels(clauses_by_type, high, mask):
    """Labels from ``{cell type: clauses}`` and high maps (see ``classify_cell_types``)."""
    inside = np.ones_like(next(iter(high.values()))) if mask is None else np.asarray(mask, dtype=bool)
    matches = {}
    for cell_type, clauses in clauses_by_type.items():
        match = np.zeros_like(inside)
        for clause in clauses:
            holds = inside.copy()
            for marker, state in clause.items():
                if state == "high":
                    holds &= high[marker]
                elif state == "low":
                    holds &= ~high[marker]
            match |= holds
        matches[cell_type] = match
    count = np.sum(list(matches.values()), axis=0) if matches else np.zeros(inside.shape, dtype=int)
    labels = {cell_type: match & (count == 1) for cell_type, match in matches.items()}
    labels[SEVERAL] = inside & (count > 1)
    labels[OTHER] = inside & (count == 0)
    return labels


def classify_cell_types(values, cell_type_rules, mask=None):
    """Label pixels from marker images ``{marker: array}``.

    Returns ``{name: boolean array}`` for every cell type, then ``"Several"``
    and ``"Other"``; each pixel inside ``mask`` (all pixels if None) is in
    exactly one of them, and pixels outside it in none. Every marker the
    rules use must be in ``values``; for a single stain, use
    ``classify_within_stain``.
    """
    return _label_pixels(cell_type_rules.active_rules(), marker_high(values, cell_type_rules), mask)


def classify_within_stain(values, cell_type_rules, mask=None):
    """Label pixels of one stain, using only the clauses it can check.

    ``values`` holds that stain's markers. A clause that needs a marker the
    stain lacks is left out, so a pixel can be "Other" here and still match
    a cell type through a clause checked in another stain. Returns
    ``{name: boolean array}`` as ``classify_cell_types``.
    """
    checkable = {
        cell_type: [
            clause
            for clause in clauses
            if all(marker in values for marker, state in clause.items() if state != "any")
        ]
        for cell_type, clauses in cell_type_rules.active_rules().items()
    }
    return _label_pixels(checkable, marker_high(values, cell_type_rules), mask)


def label_names(cell_type_rules):
    """All labels, in order: the cell types, then "Several" and "Other"."""
    return (*cell_type_rules.cell_types, SEVERAL, OTHER)


def pattern_label_shares(cell_type_rules, clause_weights=None, resolve_several=0.0):
    """Share of each label for every high/low pattern of the markers.

    Returns ``[2 ** markers, labels]`` (labels as in ``label_names``), where
    bit ``i`` of a pattern is ``cell_type_rules.markers[i]``. A pattern
    belongs to a cell type with probability 1 - product of (1 - weight) over
    the clauses it matches (``clause_weights`` overrides the rules' own);
    cell types are taken as independent, so a pattern is "Several" with the
    chance of two or more and "Other" with the chance of none. With weights
    of 0 and 1 every pattern has exactly one label.

    ``resolve_several`` (0 to 1) gives a pattern matching several cell
    types to the one it most likely belongs to, when that probability is
    at least ``1 - resolve_several`` above the next one. At 0 nothing
    changes; at 1 the most likely cell type always wins (the first in rule
    order on ties), so no pattern is "Several".
    """
    markers = cell_type_rules.markers
    weights = dict(cell_type_rules.clause_weights)
    weights.update(clause_weights or {})
    codes = np.arange(2 ** len(markers))
    high = {marker: (codes >> i & 1).astype(bool) for i, marker in enumerate(markers)}
    belongs = []
    for cell_type, clauses in cell_type_rules.rules.items():
        missing = np.ones(len(codes))
        for clause, w in zip(clauses, weights.get(cell_type, (1.0,) * len(clauses))):
            holds = _label_pixels({"clause": [clause]}, high, None)["clause"]
            missing *= np.where(holds, 1.0 - w, 1.0)
        belongs.append(1.0 - missing)
    belongs = np.stack(belongs, axis=1) if belongs else np.zeros((len(codes), 0))
    if resolve_several > 0 and belongs.shape[1] > 1:
        ordered = np.sort(belongs, axis=1)
        resolved = ordered[:, -1] - ordered[:, -2] >= 1.0 - resolve_several
        highest = np.arange(belongs.shape[1]) == np.argmax(belongs, axis=1)[:, None]
        belongs = np.where(highest | ~resolved[:, None], belongs, 0.0)
    none = np.prod(1.0 - belongs, axis=1)
    only = np.stack(
        [belongs[:, t] * np.prod(np.delete(1.0 - belongs, t, axis=1), axis=1) for t in range(belongs.shape[1])],
        axis=1,
    ) if belongs.shape[1] else np.zeros((len(codes), 0))
    several = 1.0 - none - only.sum(axis=1)
    return np.column_stack([only, np.clip(several, 0.0, 1.0), none])


def pixel_label_shares(values, cell_type_rules, mask=None, resolve_several=0.0):
    """Per-pixel share of each label, counted as in ``pattern_label_shares``.

    ``values`` is ``{marker: array}`` with every rule marker. Returns
    ``{name: float array}`` in ``label_names`` order; inside ``mask`` (all
    pixels if None) the shares of each pixel sum to 1, outside they are 0.
    Unlike ``classify_cell_types`` this uses the clause weights, so summing
    it over a colony gives the same shares as counting the pixels' patterns.
    ``resolve_several`` is as for ``pattern_label_shares``.
    """
    markers = cell_type_rules.markers
    high = marker_high(values, cell_type_rules)
    codes = sum(high[marker].astype(int) << i for i, marker in enumerate(markers))
    shares = pattern_label_shares(cell_type_rules, resolve_several=resolve_several)[codes]
    if mask is not None:
        shares = shares * np.asarray(mask, dtype=bool)[..., None]
    return {name: shares[..., j] for j, name in enumerate(label_names(cell_type_rules))}


def sample_pixel_labels(shares, rng):
    """Draw one label per pixel from per-pixel shares.

    ``shares`` is ``{name: float array}`` as from ``pixel_label_shares`` and
    ``rng`` a ``numpy.random.Generator``. Returns ``{name: boolean array}``;
    pixels whose shares are all 0 (outside the mask) get no label.
    """
    names = list(shares)
    stacked = np.stack([np.asarray(shares[name], dtype=float) for name in names], axis=-1)
    cumulative = np.cumsum(stacked, axis=-1)
    draw = rng.random(stacked.shape[:-1])[..., None] * cumulative[..., -1:]
    chosen = np.sum(cumulative <= draw, axis=-1)
    inside = cumulative[..., -1] > 0
    return {name: inside & (chosen == j) for j, name in enumerate(names)}


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
        "stains": {
            str(stain): {str(marker): str(channel) for marker, channel in stain_channels.items()}
            for stain, stain_channels in cell_type_rules.stains.items()
        },
        "cell_types": {
            cell_type: (
                dict(clauses[0]) if len(clauses) == 1 else [dict(clause) for clause in clauses]
            )
            for cell_type, clauses in cell_type_rules.rules.items()
        },
        "clause_weights": {
            cell_type: [round(w, 4) for w in ws]
            for cell_type, ws in cell_type_rules.clause_weights.items()
            if any(w != 1.0 for w in ws)
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
            "# checked) of at least one of its clauses (a list means \"or\"),\n"
            "# counted with the clause weights under clause_weights (1 if absent).\n"
            "# Stains list the markers imaged together in the same colony. The\n"
            "# thresholds hold for images processed with the data.micropattern\n"
            "# settings under preprocessing.\n"
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
    formats = (CELL_TYPES_FORMAT, *OLD_CELL_TYPES_FORMATS)
    if not isinstance(content, Mapping) or content.get("format") not in formats:
        raise ValueError(f"{path} is not a {CELL_TYPES_FORMAT} cell type file")
    markers = content.get("markers") or {}
    return CellTypeRules(
        channels={marker: str(entry["channel"]) for marker, entry in markers.items()},
        thresholds={marker: float(entry["threshold"]) for marker, entry in markers.items()},
        rules={
            str(cell_type): rule or {}
            for cell_type, rule in (content.get("cell_types") or {}).items()
        },
        hour=int(content.get("hour", 48)),
        preprocessing=dict(content.get("preprocessing") or {}),
        stains={
            str(stain): {str(marker): str(channel) for marker, channel in (channels or {}).items()}
            for stain, channels in (content.get("stains") or {}).items()
        },
        clause_weights={
            str(cell_type): [float(w) for w in ws]
            for cell_type, ws in (content.get("clause_weights") or {}).items()
        },
    )

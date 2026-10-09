"""Cell-type shares from markers imaged in separate stains.

Some markers are imaged in different colonies (stains): in the 260726 data
SOX2 only in stain 1 and FOXA2 only in stain 2, while TBXT and SOX17 are in
both. A rule that needs SOX2 and FOXA2 together, or an "or" of clauses
checked in different stains, cannot be checked cell by cell. Instead, the
share of every high/low pattern of all markers is estimated for each group
of pixels (e.g. one ring of one colony), and a cell type's share is the
total share of the patterns it matches.

The estimate. The markers imaged in every stain are the *shared* markers.
The share of each pattern of the shared markers is the mean over the
stains. Each stain's other markers are assumed to vary independently of the
other stains' markers once the shared markers are known: for example, among
the TBXT+/SOX17- pixels of one ring, being SOX2+ (stain 1) says nothing
about being FOXA2- (stain 2). Then

    share(all markers) = share(shared) x product over stains of
                         share(the stain's own markers | shared)

with each conditional share measured in its own stain. Groups are kept
small (rings or bands) so that the assumption only has to hold locally.
Shared patterns seen in few pixels are pulled towards the stain's overall
shares in the group by ``smoothing`` pixels' worth of weight, so that they
do not give extreme conditionals.

With a single stain holding every marker this is just counting pixels. For
model output, where all markers are in every pixel, use the same stains as
for the data so that model and data shares are estimated the same way.
"""

from typing import Mapping

import numpy as np

from Common.dataloader.cell_types import pattern_label_shares


def pattern_counts(values, thresholds, groups, n_groups):
    """Number of pixels of each high/low pattern in each group.

    ``values`` is ``[markers, pixels]``, ``thresholds`` one per marker and
    ``groups`` the group of each pixel (-1 = left out). Returns
    ``[n_groups, 2 ** markers]``, where bit ``i`` of a pattern is marker ``i``.
    """
    values = np.asarray(values)
    groups = np.asarray(groups)
    n_patterns = 2 ** len(thresholds)
    keep = groups >= 0
    bits = (1 << np.arange(len(thresholds)))[:, None]
    codes = np.sum((values[:, keep] > np.asarray(thresholds, dtype=float)[:, None]) * bits, axis=0)
    counts = np.bincount(groups[keep] * n_patterns + codes, minlength=n_groups * n_patterns)
    return counts.reshape(n_groups, n_patterns).astype(float)


def _sub_patterns(codes, markers, subset):
    """Patterns over ``subset`` read from patterns ``codes`` over ``markers``."""
    out = np.zeros_like(codes)
    for j, marker in enumerate(subset):
        out |= ((codes >> markers.index(marker)) & 1) << j
    return out


def combine_stains(stain_counts, markers, smoothing=1.0):
    """Share of every pattern of ``markers`` in each group, from stains.

    ``stain_counts`` lists ``(stain markers, counts)`` with counts from
    ``pattern_counts`` over those markers. Returns ``[groups, 2 **
    len(markers)]`` (bit ``i`` = ``markers[i]``), NaN for groups where a
    stain has no pixels. See the module docstring for the estimate.
    """
    markers = tuple(markers)
    stain_markers = [tuple(stain) for stain, _ in stain_counts]
    missing = [marker for marker in markers if not any(marker in stain for stain in stain_markers)]
    if missing:
        raise ValueError("No stain images the markers " + ", ".join(missing))
    shared = tuple(marker for marker in markers if all(marker in stain for stain in stain_markers))
    patterns = np.arange(2 ** len(markers))
    shared_of_pattern = _sub_patterns(patterns, markers, shared)
    n_groups = np.asarray(stain_counts[0][1]).shape[0]
    shared_share = np.zeros((n_groups, 2 ** len(shared)))
    shares = np.ones((n_groups, len(patterns)))
    taken = set(shared)
    for stain, counts in stain_counts:
        stain = tuple(stain)
        # A marker in several (but not all) stains is taken from the first.
        own = tuple(marker for marker in markers if marker in stain and marker not in taken)
        taken |= set(own)
        counts = np.asarray(counts, dtype=float)
        codes = np.arange(counts.shape[1])
        # counts[group, shared pattern, own pattern]
        cell = _sub_patterns(codes, stain, shared) * 2 ** len(own) + _sub_patterns(codes, stain, own)
        by_cell = counts @ np.eye(2 ** len(shared) * 2 ** len(own))[cell]
        by_cell = by_cell.reshape(n_groups, 2 ** len(shared), 2 ** len(own))
        with np.errstate(invalid="ignore", divide="ignore"):
            total = counts.sum(axis=1)[:, None]
            total = np.where(total > 0, total, np.nan)
            shared_share += by_cell.sum(axis=2) / total
            own_share = by_cell.sum(axis=1) / total
            conditional = (by_cell + smoothing * own_share[:, None, :]) / (
                by_cell.sum(axis=2, keepdims=True) + smoothing
            )
        shares *= conditional[:, shared_of_pattern, _sub_patterns(patterns, markers, own)]
    return shares * (shared_share / len(stain_counts))[:, shared_of_pattern]


def share_estimator(stain_samples, n_groups, cell_type_rules, smoothing=1.0):
    """A function from thresholds to label shares ``[n_groups, labels]``.

    ``stain_samples`` has one entry per stain, ``(markers, values,
    groups)``: the stain's markers, their values ``[markers, pixels]`` with
    the pixels of every colony of that stain pooled, and the group of each
    pixel (e.g. ``3 * colony + band``; -1 = left out). Labels are in the
    order of ``label_names``. The returned function takes thresholds as
    ``{marker: threshold}`` or one per marker of ``cell_type_rules``, and
    optionally ``clause_weights`` (``{cell type: weights}``) in place of the
    rules' own (see ``pattern_label_shares``).
    """
    markers = cell_type_rules.markers
    rule_labels = pattern_label_shares(cell_type_rules)
    samples = [(tuple(stain), np.asarray(values), np.asarray(groups)) for stain, values, groups in stain_samples]

    def shares(thresholds, clause_weights=None):
        to_labels = rule_labels if clause_weights is None else pattern_label_shares(cell_type_rules, clause_weights)
        if not isinstance(thresholds, Mapping):
            thresholds = dict(zip(markers, thresholds))
        counts = [
            (stain, pattern_counts(values, [thresholds[m] for m in stain], groups, n_groups))
            for stain, values, groups in samples
        ]
        return combine_stains(counts, markers, smoothing) @ to_labels

    return shares


def rule_stains(cell_type_rules):
    """``{stain: {marker: channel}}`` of the rules, or one stain holding
    every marker (``"all"``) for rules saved without stains."""
    return dict(cell_type_rules.stains) or {"all": dict(cell_type_rules.channels)}


def stain_counts_from_patterns(counts, markers, stains):
    """Per-stain counts (for ``combine_stains``) from counts over all ``markers``.

    For images where every pixel has every marker (model output). Each
    stain's counts are ``counts`` ``[groups, 2 ** len(markers)]`` summed over
    the markers the stain lacks, so the shares are estimated the same way as
    for the data. ``stains`` lists each stain's markers.
    """
    counts = np.asarray(counts, dtype=float)
    markers = tuple(markers)
    codes = np.arange(counts.shape[-1])
    return [
        (tuple(stain), counts @ np.eye(2 ** len(stain))[_sub_patterns(codes, markers, tuple(stain))])
        for stain in stains
    ]


def colony_label_shares(stain_counts, cell_type_rules, smoothing=1.0, resolve_several=0.0):
    """Label shares ``[labels]`` of one colony from per-ring stain counts.

    ``stain_counts`` is as for ``combine_stains``, with one group per ring
    (or band) of the colony. Each ring is estimated on its own and the rings
    are averaged, weighted by their pixel count (mean over the stains).
    Labels are in the order of ``label_names``. ``resolve_several`` is as
    for ``pattern_label_shares``.
    """
    shares = combine_stains(stain_counts, cell_type_rules.markers, smoothing) @ pattern_label_shares(
        cell_type_rules, resolve_several=resolve_several
    )
    area = np.mean([np.asarray(counts).sum(axis=1) for _, counts in stain_counts], axis=0)
    area = np.where(np.isnan(shares[:, 0]), 0.0, area)
    return np.nansum(shares * area[:, None], axis=0) / max(area.sum(), 1e-9)

"""Fit marker thresholds to target cell types in radial bands of a colony.

The colony is split into three bands by distance from its centre: a
*centre* disc, a *ring* and the *periphery*, with edges given as fractions of
the colony radius. For each band (and condition) the user names the cell type
that should be the most common there. Thresholds that make this hold exactly
for every replicate rarely exist, so the fit looks for the ones that come
closest.

Shares come from ``Common.dataloader.cell_type_shares``, which also handles
markers imaged in separate stains.

How close a band is to its target is its *margin*: the share of the target
cell type minus the share of the most common other label (cell types,
``"Several"`` and ``"Other"``). A positive margin means the target wins. The
score is the mean margin over every band with a target, each margin capped
at ``cap`` so that a band that is already won cannot make up for one that
is lost. Among thresholds with the same score, the fit prefers the larger
uncapped mean margin, i.e. the clearest wins.

Optionally, colony-wide constraints ask for a cell type's share of the
whole colony (in one condition) to be as large or as small as possible;
each adds its weight times that share to the score (see ``cell_type_score``).

Thresholds (and, optionally, clause weights) are fitted by coordinate
search on a grid: each parameter in turn takes the grid value that gives
the best score with the others held fixed, until a full pass changes
nothing. Starting from several points guards against getting stuck.
"""

import numpy as np

BANDS = ("centre", "ring", "periphery")


def relative_distance(mask):
    """Distance of each pixel from the image centre, as a fraction of the
    distance to the furthest mask pixel (the colony radius)."""
    mask = np.asarray(mask, dtype=bool)
    rows, columns = np.indices(mask.shape)
    distance = np.hypot(rows - (mask.shape[0] - 1) / 2.0, columns - (mask.shape[1] - 1) / 2.0)
    radius = distance[mask].max() if mask.any() else 1.0
    return distance / max(radius, 1e-9)


def radial_bands(mask, edges, gap=0.0):
    """Band of each pixel: 0 centre, 1 ring, 2 periphery, -1 outside.

    Distances are fractions of the colony radius (see ``relative_distance``).
    ``edges`` are the ``(centre/ring, ring/periphery)`` boundaries, and a
    strip ``gap`` wide around each boundary is left out (-1), like pixels
    outside the mask.
    """
    distance = relative_distance(mask)
    inner, outer = edges
    half = gap / 2.0
    band = np.full(distance.shape, -1, dtype=int)
    band[distance < inner - half] = 0
    band[(distance >= inner + half) & (distance < outer - half)] = 1
    band[distance >= outer + half] = 2
    band[~np.asarray(mask, dtype=bool)] = -1
    return band


def radial_rings(mask, n_rings):
    """Ring of each pixel (0 at the centre), splitting the colony radius into
    ``n_rings`` equal steps; -1 outside the mask."""
    ring = np.minimum((relative_distance(mask) * n_rings).astype(int), n_rings - 1)
    return np.where(np.asarray(mask, dtype=bool), ring, -1)


def band_margins(shares, targets):
    """Target share minus the largest other share, per group (NaN without a target).

    ``targets`` holds the target label index of each group, or -1 for none.
    """
    shares = np.asarray(shares)
    targets = np.asarray(targets)
    margins = np.full(len(shares), np.nan)
    for group, target in enumerate(targets):
        if target < 0 or np.isnan(shares[group]).all():
            continue
        others = np.delete(shares[group], target)
        margins[group] = shares[group, target] - others.max()
    return margins


def target_score(margins, cap=0.1):
    """Mean margin over groups with a target, each capped at ``cap``."""
    margins = np.asarray(margins)
    margins = margins[~np.isnan(margins)]
    return float(np.mean(np.minimum(margins, cap))) if len(margins) else float("nan")


def colony_share(shares, area, colonies, label):
    """Share of ``label`` over whole colonies, averaged over the colonies.

    ``colonies`` lists, per colony, the groups that make it up (e.g. its
    three bands); each colony's share is the area-weighted mean of its
    groups' shares (``area`` = pixels per group), ignoring empty groups.
    """
    values = []
    for groups in colonies:
        groups = np.asarray(groups)
        group_shares = shares[groups, label]
        weights = np.where(np.isnan(group_shares), 0.0, area[groups])
        if weights.sum() > 0:
            values.append(float(np.sum(np.nan_to_num(group_shares) * weights) / weights.sum()))
    return float(np.mean(values)) if values else float("nan")


def cell_type_score(shares, targets, cap=0.1, area=None, constraints=()):
    """How well label shares meet band targets and colony-wide constraints.

    The band part is ``target_score`` of the margins (0 without targets).
    Each constraint ``(colonies, label, weight)`` adds ``weight`` times the
    ``colony_share`` of ``label``: a positive weight rewards a larger share,
    a negative one a smaller share. Returns ``(score, tie-break)``, compared
    in that order, where the tie-break uses uncapped margins.
    """
    margins = band_margins(shares, targets)
    has_targets = not np.isnan(margins).all()
    capped = target_score(margins, cap) if has_targets else 0.0
    uncapped = target_score(margins, cap=np.inf) if has_targets else 0.0
    extra = 0.0
    for colonies, label, weight in constraints:
        share = colony_share(shares, area, colonies, label)
        extra += 0.0 if np.isnan(share) else weight * share
    return round(capped + extra, 12), round(uncapped + extra, 12)


def coordinate_search(objective, start, grids, max_passes=20):
    """Maximise ``objective(parameters)`` one parameter at a time.

    Each parameter in turn takes the value from its grid (``grids[i]``)
    that scores best with the others fixed, moving only for a strictly
    better score, until a full pass changes nothing. Scores may be tuples
    (compared in order). ``start`` is snapped to the grids. Returns
    ``(parameters, score)``.
    """
    grids = [np.asarray(grid, dtype=float) for grid in grids]
    parameters = np.array([grid[np.argmin(np.abs(grid - value))] for grid, value in zip(grids, start)])
    best = objective(parameters)
    for _ in range(max_passes):
        moved = False
        for index, grid in enumerate(grids):
            for candidate in grid:
                trial = parameters.copy()
                trial[index] = candidate
                trial_score = objective(trial)
                if trial_score > best:
                    best, parameters, moved = trial_score, trial, True
        if not moved:
            break
    return parameters, best


def threshold_grid(step=0.01):
    """Threshold (or weight) values from 0 to 1 in ``step``."""
    return np.round(np.arange(0.0, 1.0 + step / 2, step), 6)


def fit_thresholds(shares, targets, start, step=0.01, cap=0.1):
    """Coordinate search over thresholds alone for the best band targets.

    ``shares`` maps thresholds (one per marker) to label shares per group,
    as made by ``Common.dataloader.cell_type_shares.share_estimator``.
    Returns ``(thresholds, score)`` with the capped score.
    """
    parameters, best = coordinate_search(
        lambda thresholds: cell_type_score(shares(thresholds), targets, cap),
        start,
        [threshold_grid(step)] * len(start),
    )
    return parameters, best[0]

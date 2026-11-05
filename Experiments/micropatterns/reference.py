"""Measured 0 h colonies of a trained model, as references for new geometries.

Used by ``drawn_mask_explorer.py``, ``boundary_mask_explorer.py`` (cell density) and
``Experiments/inverse_design``.
"""

from dataclasses import replace

import numpy as np

from Experiments.micropatterns.texture_synthesis import (
    cell_coverage,
    coverage_spectrum,
    filled_reference,
)


def load_reference(data_config, data_root=None):
    """0 h measurements of a model's training replicates, scaled as in training.

    ``data_config`` is the model bundle's ``data`` section and ``data_root``
    the 260726 dataset folder (None: ``DATA_PATH_BASE/260726_nca_dataset``).
    Returns a dict with measurements ``[B, C, H, W]``, which measurements were
    taken (``available``, ``[B, C]``), colony masks (``support``, ``[B, H, W]``),
    the channel schema and the replicate labels.
    """

    from Experiments.micropatterns.config_helpers import load_data

    data_config = replace(
        data_config,
        knockout=replace(data_config.knockout, curriculum=("baseline",)),
    )
    train_replicates = data_config.micropattern.train_replicates
    if train_replicates is None:
        train_replicates = tuple(range(1, data_config.batches + 1))
    data, aux, _, boundary, _, _ = load_data(
        data_config,
        impath=None if data_root is None else str(data_root),
        replicate_indices=tuple(index - 1 for index in train_replicates),
        pool_copies_override=1,
    )
    return {
        "measurements": np.asarray(data)[:, 0],
        "available": np.asarray(aux["measurement_mask"])[:, 0],
        "support": np.asarray(boundary)[:, 0] > 0.5,
        "schema": aux["channel_schema"],
        "replicates": tuple(aux.get("batch_replicates", train_replicates)),
    }


def combine_reference(reference, mode, sample, seed):
    """One reference colony ``[C, H, W]`` and its mask ``[H, W]``.

    ``"fixed"`` uses one replicate. ``"sample_groups"`` takes each staining
    panel from a random replicate that measured it, as in
    ``boundary_mask_explorer.py``.
    """

    measurements = reference["measurements"]
    if mode == "fixed":
        index = int(sample) % measurements.shape[0]
        return measurements[index], reference["support"][index]
    rng = np.random.default_rng(int(seed))
    combined = np.zeros_like(measurements[0])
    for name, indices in zip(
        reference["schema"].group_names,
        reference["schema"].group_measurement_indices,
    ):
        indices = np.asarray(indices, dtype=int)
        eligible = np.flatnonzero(np.all(reference["available"][:, indices], axis=1))
        if not len(eligible):
            raise ValueError(f"No training replicate measures {name!r} at 0 h")
        combined[indices] = measurements[int(rng.choice(eligible)), indices]
    support = reference["support"][int(rng.choice(measurements.shape[0]))]
    return combined, support


def fill_reference_gaps(reference, markers, output_shape):
    """The reference with the gaps between cells filled, and its cell/gap statistics.

    Each staining panel of each replicate is a separate colony, so its gaps
    are found from its own generic cell markers (``markers``, e.g. LMBR and
    SOX2), or from all its channels if it has none. Returns a dict with

    - ``reference``: a copy of ``reference`` with filled measurements
    - ``gap_values``: each state channel's value in the gaps ``[C]``
    - ``measurement_gap_values``: the same for every measurement channel
    - ``amplitude``: the cell/gap amplitude spectrum on ``output_shape``,
      averaged (as power) over the panels with generic markers
    - ``density``: their mean covered fraction of the colony
    - ``coverage``: one row per panel and replicate (``replicate``, ``panel``,
      ``coverage``, ``from_markers``)
    """

    schema = reference["schema"]
    channel_markers = [channel.marker for channel in schema.measurement_channels]
    measured = np.asarray(reference["measurements"], dtype=np.float32)
    filled = measured.copy()
    replicates = reference.get("replicates", tuple(range(1, len(measured) + 1)))
    gap_sum = np.zeros(measured.shape[1], dtype=np.float64)
    gap_count = np.zeros(measured.shape[1], dtype=np.float64)
    power, fractions, rows = [], [], []
    for index, (colony, support) in enumerate(zip(measured, reference["support"])):
        for name, indices in zip(schema.group_names, schema.group_measurement_indices):
            indices = np.asarray(indices, dtype=int)
            if not np.all(reference["available"][index, indices]):
                continue
            marker_indices = [i for i in indices if channel_markers[i] in markers]
            coverage, _ = cell_coverage(colony, support, marker_indices or indices)
            filled[index, indices], gaps = filled_reference(colony[indices], support, coverage)
            gap_sum[indices] += gaps
            gap_count[indices] += 1
            fraction = float(np.mean(coverage[support]))
            rows.append({
                "replicate": replicates[index],
                "panel": name,
                "coverage": fraction,
                "from_markers": bool(marker_indices),
            })
            if marker_indices:
                fractions.append(fraction)
                power.append(coverage_spectrum(coverage, support, output_shape) ** 2)
    if not fractions:
        raise ValueError(f"No measured panel has any of the markers {list(markers)}")
    measurement_gap_values = (gap_sum / np.maximum(gap_count, 1)).astype(np.float32)
    return {
        "reference": {**reference, "measurements": filled},
        "gap_values": measurement_gap_values[np.asarray(schema.primary_measurements)],
        "measurement_gap_values": measurement_gap_values,
        "amplitude": np.sqrt(np.mean(power, axis=0)).astype(np.float32),
        "density": float(np.mean(fractions)),
        "coverage": rows,
    }

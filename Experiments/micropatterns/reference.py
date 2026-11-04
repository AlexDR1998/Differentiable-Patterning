"""Measured 0 h colonies of a trained model, as references for new geometries.

Used by ``drawn_mask_explorer.py`` and ``Experiments/inverse_design``.
"""

from dataclasses import replace

import numpy as np


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

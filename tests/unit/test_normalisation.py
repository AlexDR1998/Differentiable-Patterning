import numpy as np
import pytest

from Common.dataloader.micropattern_260726 import circular_colony_mask, source_condition
from Common.dataloader.normalisation import (
    condition_bins,
    percentile_bins,
    rescale,
    trajectory_bins,
)


SAMPLES = {
    "a": np.arange(0, 101, dtype=np.float32),
    "b": np.arange(100, 301, dtype=np.float32),
}


def test_per_replicate_bins_use_each_trajectory():
    bins = percentile_bins(SAMPLES, 0, 100, "per_replicate")
    assert bins == {"a": (0.0, 100.0), "b": (100.0, 300.0)}


def test_replicate_mean_bins_average_the_trajectories():
    bins = percentile_bins(SAMPLES, 0, 100, "replicate_mean")
    assert bins == {"a": (50.0, 200.0), "b": (50.0, 200.0)}


def test_pooled_bins_use_all_pixels():
    bins = percentile_bins(SAMPLES, 0, 100, "pooled")
    assert bins == {"a": (0.0, 300.0), "b": (0.0, 300.0)}


def test_reference_per_replicate_uses_the_reference_bounds():
    bins = percentile_bins(SAMPLES, 0, 100, "per_replicate", reference={"b": "a"})
    assert bins == {"a": (0.0, 100.0), "b": (0.0, 100.0)}


@pytest.mark.parametrize("mode", ["replicate_mean", "pooled"])
def test_reference_shared_modes_use_only_reference_trajectories(mode):
    samples = {**SAMPLES, "c": np.arange(1000, 1101, dtype=np.float32)}
    bins = percentile_bins(samples, 0, 100, mode, reference={"c": "a"})
    # Only "a" and "b" are references; "c" (e.g. a knockout) is left out.
    expected = (50.0, 200.0) if mode == "replicate_mean" else (0.0, 300.0)
    assert bins == {"a": expected, "b": expected, "c": expected}


def test_reference_without_pixels_is_an_error():
    with pytest.raises(ValueError):
        percentile_bins(SAMPLES, 0, 100, "per_replicate", reference={"b": "missing"})


def test_empty_trajectories_are_skipped():
    bins = percentile_bins({**SAMPLES, "c": np.array([])}, 0, 100, "per_replicate")
    assert set(bins) == {"a", "b"}


def _range(low, high):
    return np.arange(low, high + 1, dtype=np.float32)


# Trajectories keyed by (condition, replicate), as in the 260726 loader.
CONDITION_SAMPLES = {
    ("ctrl", 0): _range(0, 100),
    ("ctrl", 1): _range(10, 200),
    ("sl0", 0): _range(0, 20),
    ("sl0", 1): _range(10, 30),
    ("sl24", 0): _range(0, 150),
    ("sl24", 1): _range(10, 150),
}


def test_condition_bins_normalise_each_condition_on_its_own():
    bins = condition_bins(CONDITION_SAMPLES, 0, 100, "replicate_mean")
    assert bins[("ctrl", 0)] == bins[("ctrl", 1)] == (5.0, 150.0)
    assert bins[("sl0", 0)] == bins[("sl0", 1)] == (5.0, 25.0)


def test_joint_bins_keep_control_bounds_that_cover_the_knockouts():
    samples = {key: CONDITION_SAMPLES[key] for key in [("ctrl", 0), ("ctrl", 1), ("sl0", 0), ("sl0", 1)]}
    bins = trajectory_bins(samples, 0, 100, "replicate_mean", "joint")
    # The dim knockout takes the control bounds instead of its own (5, 25).
    assert set(bins.values()) == {(5.0, 150.0)}


def test_joint_bins_widen_where_a_knockout_goes_outside_control():
    bins = trajectory_bins(CONDITION_SAMPLES, 0, 100, "replicate_mean", "joint")
    # sl24 averages to (5, 150) too; a brighter sl24 widens the upper bound.
    assert set(bins.values()) == {(5.0, 150.0)}
    samples = {**CONDITION_SAMPLES, ("sl24", 1): _range(10, 350)}
    bins = trajectory_bins(samples, 0, 100, "replicate_mean", "joint")
    assert set(bins.values()) == {(5.0, 250.0)}


def test_joint_bins_per_replicate_stay_within_each_replicate():
    bins = trajectory_bins(CONDITION_SAMPLES, 0, 100, "per_replicate", "joint")
    # Replicate 0: control (0, 100) widened by sl24 (0, 150).
    assert {bins[(condition, 0)] for condition in ("ctrl", "sl0", "sl24")} == {(0.0, 150.0)}
    # Replicate 1: control (10, 200) already covers both knockouts.
    assert {bins[(condition, 1)] for condition in ("ctrl", "sl0", "sl24")} == {(10.0, 200.0)}


@pytest.mark.parametrize("mode", ["per_replicate", "replicate_mean", "pooled"])
def test_control_bins_match_a_control_reference(mode):
    reference = {key: ("ctrl", key[1]) for key in CONDITION_SAMPLES if key[0] != "ctrl"}
    expected = percentile_bins(CONDITION_SAMPLES, 1, 99, mode, reference=reference)
    assert trajectory_bins(CONDITION_SAMPLES, 1, 99, mode, "control") == expected


def test_control_bins_without_a_control_replicate_keep_their_own():
    samples = {("ctrl", 0): _range(0, 100), ("sl0", 1): _range(0, 20)}
    bins = trajectory_bins(samples, 0, 100, "per_replicate", "control")
    assert bins[("sl0", 1)] == (0.0, 20.0)


def test_selected_bins_are_percentile_bins():
    expected = percentile_bins(CONDITION_SAMPLES, 0, 100, "pooled")
    assert trajectory_bins(CONDITION_SAMPLES, 0, 100, "pooled", "selected") == expected


def test_unknown_knockout_bounds_is_an_error():
    with pytest.raises(ValueError):
        trajectory_bins(CONDITION_SAMPLES, 0, 100, "pooled", "unknown")


@pytest.mark.parametrize(
    "low, high, mode", [(50, 50, "pooled"), (-1, 50, "pooled"), (0, 100, "unknown")]
)
def test_percentile_bins_rejects_bad_arguments(low, high, mode):
    with pytest.raises(ValueError):
        percentile_bins(SAMPLES, low, high, mode)


def test_rescale_is_linear_and_clipped():
    np.testing.assert_allclose(
        rescale([0.0, 10.0, 15.0, 20.0, 30.0], 10.0, 20.0), [0.0, 0.0, 0.5, 1.0, 1.0]
    )
    # A zero-width range must not divide by zero.
    assert np.all(np.isfinite(rescale([1.0, 2.0], 1.0, 1.0)))


def test_circular_colony_mask_scales_the_fitted_radius():
    rows, columns = np.ogrid[:41, :41]
    disk = (rows - 20) ** 2 + (columns - 20) ** 2 <= 10**2
    full = circular_colony_mask(disk, "cell_fate_s1", 1.0, 1.0)
    half = circular_colony_mask(disk, "cell_fate_s1", 1.0, 0.5)
    np.testing.assert_array_equal(full, disk)
    assert half.sum() < 0.3 * disk.sum()
    assert half[20, 20]


def test_source_condition_uses_control_before_knockout():
    assert source_condition("sl24", 24) == "ctrl"
    assert source_condition("sl24", 36) == "sl24"
    assert source_condition("sl0", 0) == "ctrl"
    assert source_condition("sl0", 0, substitute_preperturbation=False) == "sl0"

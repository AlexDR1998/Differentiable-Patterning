import numpy as np
import pytest

from Common.dataloader.micropattern_260726 import circular_colony_mask, source_condition
from Common.dataloader.normalisation import percentile_bins, rescale


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

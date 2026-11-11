"""Pure parts of the cell census (counting and typing cells by nuclei)."""

import numpy as np

from Common.dataloader.cell_types import CellTypeRules, label_names
from Experiments.micropatterns.cell_census import (
    cell_mean_values,
    census_label_shares,
    panel_stain,
    point_rings,
)


def rules():
    return CellTypeRules(
        channels={"A": "s1/A", "B": "s2/B"},
        thresholds={"A": 0.5, "B": 0.5},
        rules={"TypeA": [{"A": "high", "B": "any"}], "TypeB": [{"A": "any", "B": "high"}]},
        stains={"s1": {"A": "s1/A"}, "s2": {"B": "s2/B"}},
    )


def test_panel_stain_finds_the_stain_imaged_in_a_panel():
    assert panel_stain(rules(), ["s1/A", "s1/LMBR"]) == ("s1", {"A": "s1/A"})
    assert panel_stain(rules(), ["rna/CER1"]) is None


def test_point_rings_split_the_radius_evenly():
    points = np.array([[10.0, 10.0], [10.0, 14.9], [10.0, 15.1], [10.0, 30.0]])
    np.testing.assert_array_equal(point_rings(points, (10.0, 10.0), 10.0, 2), [0, 0, 1, 1])


def test_cell_mean_values():
    cells = np.array([[1, 1, 0], [2, 0, 0]])
    channels = np.stack([np.array([[1.0, 3.0, 9.0], [5.0, 9.0, 9.0]])], axis=-1)
    np.testing.assert_allclose(cell_mean_values(cells, channels, 2), [[2.0, 5.0]])


def census_row(stain, markers, cell_counts, pixel_counts, nuclei, hour=0.0):
    return {
        "replicate": 1, "condition": "baseline", "hour": hour, "panel": stain, "stain": stain,
        "markers": markers, "nuclei": nuclei, "colony_mm2": 0.2,
        "cell_counts": np.asarray(cell_counts, dtype=float), "pixel_counts": np.asarray(pixel_counts, dtype=float),
    }


def test_shares_by_cells_and_pixels_differ_when_cell_sizes_differ():
    # One ring. Stain s1: by pixels half the colony is A-high, but A-high
    # cells are large, so only a fifth of the cells are A-high. s2: B always low.
    rows = [
        census_row("s1", ("A",), [[400, 100]], [[500, 500]], nuclei=500),
        census_row("s2", ("B",), [[500, 0]], [[1000, 0]], nuclei=500),
    ]

    shares = {row["cell_type"]: row for row in census_label_shares(rows, rules())}

    assert set(shares) == set(label_names(rules()))
    assert abs(shares["TypeA"]["share_pixels"] - 0.5) < 0.01
    assert abs(shares["TypeA"]["share_cells"] - 0.2) < 0.01
    # TypeA: half of 0.2 mm2 over 100 cells
    assert abs(shares["TypeA"]["area_per_cell_um2"] - 1000.0) < 50.0


def test_times_without_every_stain_are_left_out():
    rows = [census_row("s1", ("A",), [[1, 1]], [[1, 1]], nuclei=2)]
    assert census_label_shares(rows, rules()) == []


def test_census_file_round_trip(tmp_path):
    from Experiments.micropatterns.cell_census import load_census, save_census

    rows = [census_row("s1", ("A",), [[4, 1]], [[5, 5]], nuclei=5, hour=48.0)]
    save_census(tmp_path / "census.npz", rows, {"rings": 1})

    loaded, settings = load_census(tmp_path / "census.npz")

    assert settings == {"rings": 1}
    assert loaded[0]["markers"] == ("A",) and loaded[0]["hour"] == 48.0
    np.testing.assert_array_equal(loaded[0]["cell_counts"], [[4, 1]])

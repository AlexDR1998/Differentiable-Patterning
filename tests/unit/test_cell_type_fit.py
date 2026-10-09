import numpy as np
import pytest

from Common.dataloader.cell_type_fit import (
    band_margins,
    colony_share,
    coordinate_search,
    cell_type_score,
    fit_thresholds,
    threshold_grid,
    radial_bands,
    radial_rings,
    target_score,
)
from Common.dataloader.cell_type_shares import share_estimator
from Common.dataloader.cell_types import CellTypeRules, label_names


def _rules():
    return CellTypeRules(
        channels={"SOX17": "s2/SOX17", "TBXT": "s2/TBXT"},
        thresholds={"SOX17": 0.5, "TBXT": 0.5},
        rules={
            "Endoderm": {"SOX17": "high", "TBXT": "any"},
            "Mesoderm": {"SOX17": "low", "TBXT": "high"},
        },
    )


def test_radial_bands_split_by_fraction_of_radius():
    mask = np.hypot(*(np.indices((21, 21)) - 10.0)) <= 10
    band = radial_bands(mask, edges=(0.3, 0.7))
    assert band[10, 10] == 0
    assert band[10, 15] == 1  # distance 0.5
    assert band[10, 19] == 2  # distance 0.9
    assert band[0, 0] == -1  # outside the mask
    # A gap leaves out pixels near an edge.
    assert radial_bands(mask, edges=(0.3, 0.7), gap=0.3)[10, 13] == -1  # distance 0.3

    rings = radial_rings(mask, 4)
    assert rings[10, 10] == 0 and rings[10, 13] == 1 and rings[10, 20] == 3
    assert rings[0, 0] == -1


def test_band_shares_and_margins():
    rules = _rules()
    assert label_names(rules) == ("Endoderm", "Mesoderm", "Several", "Other")
    # Group 0: two endoderm pixels and one mesoderm; group 1: one "Other".
    values = np.array([[0.9, 0.8, 0.1, 0.1], [0.0, 0.9, 0.9, 0.1]])
    groups = np.array([0, 0, 0, 1])
    shares = share_estimator([(("SOX17", "TBXT"), values, groups)], 3, rules)([0.5, 0.5])
    np.testing.assert_allclose(shares[0], [2 / 3, 1 / 3, 0, 0])
    np.testing.assert_allclose(shares[1], [0, 0, 0, 1])
    assert np.isnan(shares[2]).all()  # empty group

    margins = band_margins(shares, targets=[0, 1, 0])
    assert margins[0] == pytest.approx(1 / 3)
    assert margins[1] == pytest.approx(-1.0)
    assert np.isnan(margins[2])
    assert target_score(margins, cap=0.1) == pytest.approx((0.1 - 1.0) / 2)


def test_fit_thresholds_finds_a_separating_threshold():
    rules = _rules()
    rng = np.random.default_rng(0)
    # Group 0 should be mesoderm (SOX17 around 0.2, TBXT high) and group 1
    # endoderm (SOX17 around 0.4). The start threshold 0.5 calls both mesoderm.
    sox17 = np.concatenate([rng.uniform(0.15, 0.25, 50), rng.uniform(0.35, 0.45, 50)])
    tbxt = np.full(100, 0.9)
    groups = np.repeat([0, 1], 50)
    shares = share_estimator([(("SOX17", "TBXT"), np.stack([sox17, tbxt]), groups)], 2, rules)
    thresholds, score = fit_thresholds(shares, targets=[1, 0], start=[0.5, 0.5])
    assert 0.25 <= thresholds[0] < 0.35
    assert score == pytest.approx(0.1)


def test_colony_constraints_and_clause_weights():
    rules = CellTypeRules(
        channels={"A": "a", "B": "b"},
        thresholds={"A": 0.5, "B": 0.5},
        rules={"X": [{"A": "high"}, {"B": "high"}]},
    )
    # One colony of two groups: group 0 (10 px) all A+, group 1 (30 px) all B+.
    values = np.concatenate([np.tile([[0.9], [0.1]], 10), np.tile([[0.1], [0.9]], 30)], axis=1)
    groups = np.repeat([0, 1], [10, 30])
    area = np.array([10.0, 30.0])
    shares = share_estimator([(("A", "B"), values, groups)], 2, rules)
    colonies = [[0, 1]]
    assert colony_share(shares([0.5, 0.5]), area, colonies, 0) == pytest.approx(1.0)
    half = shares([0.5, 0.5], {"X": (1.0, 0.5)})
    assert colony_share(half, area, colonies, 0) == pytest.approx((10 + 15) / 40)

    # Minimising X over the colony (no band targets), with only the second
    # clause's weight free, switches that clause off: X keeps group 0 only.
    def objective(parameters):
        weighted = shares([0.5, 0.5], {"X": (1.0, parameters[0])})
        return cell_type_score(weighted, targets=[-1, -1], area=area, constraints=[(colonies, 0, -1.0)])

    parameters, score = coordinate_search(objective, [1.0], [threshold_grid(0.25)])
    assert parameters[0] == 0.0
    assert score[0] == pytest.approx(-10 / 40)

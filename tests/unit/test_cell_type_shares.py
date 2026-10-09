import numpy as np
import pytest

from Common.dataloader.cell_type_shares import combine_stains, pattern_counts, share_estimator
from Common.dataloader.cell_types import CellTypeRules


def _rules():
    return CellTypeRules(
        channels={"SOX17": "s2/SOX17", "TBXT": "s2/TBXT", "SOX2": "s1/SOX2", "FOXA2": "s2/FOXA2"},
        thresholds={"SOX17": 0.5, "TBXT": 0.5, "SOX2": 0.5, "FOXA2": 0.5},
        rules={
            "Endoderm": {"SOX17": "high"},
            "Mesoderm": [
                {"TBXT": "high", "FOXA2": "low", "SOX17": "low"},
                {"TBXT": "high", "SOX2": "high"},
            ],
        },
    )


def test_pattern_counts_per_group():
    values = np.array([[0.9, 0.1, 0.9, 0.9], [0.9, 0.9, 0.1, 0.1]])
    counts = pattern_counts(values, [0.5, 0.5], groups=[0, 0, 1, -1], n_groups=2)
    # Bit 0 = first marker. Group 0: patterns 0b11 and 0b10; group 1: 0b01.
    np.testing.assert_array_equal(counts, [[0, 0, 1, 1], [0, 1, 0, 0]])


def test_one_stain_with_every_marker_is_plain_counting():
    counts = np.array([[1.0, 2.0, 3.0, 4.0]])
    shares = combine_stains([(("A", "B"), counts)], ("A", "B"))
    np.testing.assert_allclose(shares, counts / 10)


def test_two_stains_combine_through_the_shared_marker():
    # Shared marker T. Given T+, stain 1 has X+ in half the pixels and stain
    # 2 has Y+ in a quarter; given T-, neither marker is ever high.
    # Stain 1 markers (T, X): patterns bit0 = T, bit1 = X.
    stain_1 = np.array([[50.0, 25.0, 0.0, 25.0]])  # T- 50, T+X- 25, T+X+ 25
    # Stain 2 markers (T, Y): T- 50, T+Y- 37.5, T+Y+ 12.5 (of 100).
    stain_2 = np.array([[50.0, 37.5, 0.0, 12.5]])
    shares = combine_stains([(("T", "X"), stain_1), (("T", "Y"), stain_2)], ("T", "X", "Y"), smoothing=0.0)
    # Bits: T = 1, X = 2, Y = 4.
    assert shares[0].sum() == pytest.approx(1.0)
    assert shares[0, 0b000] == pytest.approx(0.5)
    assert shares[0, 0b111] == pytest.approx(0.5 * 0.5 * 0.25)
    assert shares[0, 0b011] == pytest.approx(0.5 * 0.5 * 0.75)
    assert shares[0, 0b110] == pytest.approx(0.0)


def test_estimator_combines_mesoderm_clauses_from_two_stains():
    rules = _rules()
    rng = np.random.default_rng(0)
    n = 4000
    # Every pixel TBXT+ SOX17-; SOX2+ in 50% (stain 1), FOXA2- in 40% (stain 2).
    stain_1 = np.stack([np.full(n, 0.1), np.full(n, 0.9), (rng.random(n) < 0.5) * 0.9])
    stain_2 = np.stack([np.full(n, 0.1), np.full(n, 0.9), (rng.random(n) < 0.6) * 0.9])
    shares = share_estimator(
        [
            (("SOX17", "TBXT", "SOX2"), stain_1, np.zeros(n, dtype=int)),
            (("SOX17", "TBXT", "FOXA2"), stain_2, np.zeros(n, dtype=int)),
        ],
        n_groups=1,
        cell_type_rules=rules,
    )({"SOX17": 0.5, "TBXT": 0.5, "SOX2": 0.5, "FOXA2": 0.5})
    # Labels: Endoderm, Mesoderm, Several, Other.
    # Mesoderm = 1 - P(SOX2-) P(FOXA2+) = 1 - 0.5 * 0.6.
    assert shares[0, 1] == pytest.approx(0.7, abs=0.03)
    assert shares[0, 0] == pytest.approx(0.0)
    assert shares[0].sum() == pytest.approx(1.0)


def test_empty_group_is_nan():
    counts = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    shares = combine_stains([(("T", "X"), counts), (("T", "Y"), counts)], ("T", "X", "Y"))
    assert np.isnan(shares[1]).all()
    assert not np.isnan(shares[0]).any()


def test_model_counts_split_into_stains_match_counting_each_stain():
    from Common.dataloader.cell_type_shares import stain_counts_from_patterns

    rng = np.random.default_rng(0)
    values = rng.uniform(size=(3, 50))
    groups = rng.integers(-1, 2, size=50)
    full = pattern_counts(values, [0.5] * 3, groups, 2)
    stains = [("A", "B"), ("B", "C")]
    split = stain_counts_from_patterns(full, ("A", "B", "C"), stains)
    for (stain, counts), rows in zip(split, ([0, 1], [1, 2])):
        assert stain in stains
        np.testing.assert_array_equal(counts, pattern_counts(values[rows], [0.5, 0.5], groups, 2))


def test_colony_label_shares_weights_rings_by_area():
    from Common.dataloader.cell_type_shares import colony_label_shares

    rules = CellTypeRules(
        channels={"A": "A"}, thresholds={"A": 0.5}, rules={"High": {"A": "high"}}
    )
    # Ring 0: 1 pixel, all high. Ring 1: 3 pixels, none high. Ring 2: empty.
    counts = np.array([[0.0, 1.0], [3.0, 0.0], [0.0, 0.0]])
    shares = colony_label_shares([(("A",), counts)], rules)
    # Labels: High, Several, Other.
    np.testing.assert_allclose(shares, [0.25, 0.0, 0.75])

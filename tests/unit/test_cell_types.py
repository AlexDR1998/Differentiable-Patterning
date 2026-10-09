import numpy as np
import pytest

from Common.dataloader.cell_types import (
    CellTypeRules,
    cell_type_fractions,
    classify_cell_types,
    load_cell_type_rules,
    classify_within_stain,
    label_names,
    marker_values,
    pattern_label_shares,
    pixel_label_shares,
    sample_pixel_labels,
    save_cell_type_rules,
)


def _rules(**overrides):
    settings = dict(
        channels={"SOX17": "cell_fate_s2/SOX17", "TBXT": "cell_fate_s2/TBXT"},
        thresholds={"SOX17": 0.5, "TBXT": 0.4},
        rules={
            "Endoderm": {"SOX17": "high", "TBXT": "any"},
            "Mesoderm": {"SOX17": "low", "TBXT": "high"},
            "Primitive streak": {"TBXT": "high"},
        },
        hour=48,
        preprocessing={"cleaning": {"normalisation": "replicate_mean"}},
    )
    settings.update(overrides)
    return CellTypeRules(**settings)


def test_classify_cell_types_labels_each_masked_pixel_once():
    values = {"SOX17": np.array([0.9, 0.1, 0.1, 0.9, 0.2]), "TBXT": np.array([0.1, 0.8, 0.1, 0.8, 0.9])}
    mask = np.array([True, True, True, True, False])
    labels = classify_cell_types(values, _rules(), mask)

    assert labels["Endoderm"].tolist() == [True, False, False, False, False]
    # Mesoderm and primitive streak both match pixel 1, and endoderm and
    # primitive streak pixel 3.
    assert labels["Several"].tolist() == [False, True, False, True, False]
    assert labels["Other"].tolist() == [False, False, True, False, False]
    assert not labels["Mesoderm"].any()
    stacked = np.sum(list(labels.values()), axis=0)
    assert stacked.tolist() == [1, 1, 1, 1, 0]

    fractions = cell_type_fractions(labels, mask)
    assert fractions["Several"] == pytest.approx(0.5)
    assert sum(fractions.values()) == pytest.approx(1.0)


def test_clauses_are_joined_by_or():
    rules = _rules(
        channels={"SOX17": "s2/SOX17", "TBXT": "s2/TBXT", "SOX2": "s1/SOX2"},
        thresholds={"SOX17": 0.5, "TBXT": 0.5, "SOX2": 0.5},
        rules={
            "Mesoderm": [{"TBXT": "high", "SOX17": "low"}, {"TBXT": "high", "SOX2": "high"}],
        },
    )
    assert rules.rules["Mesoderm"][1] == {"TBXT": "high", "SOX2": "high"}
    values = {
        "TBXT": np.array([0.9, 0.9, 0.9, 0.1]),
        "SOX17": np.array([0.1, 0.9, 0.9, 0.1]),
        "SOX2": np.array([0.1, 0.9, 0.1, 0.9]),
    }
    labels = classify_cell_types(values, rules)
    assert labels["Mesoderm"].tolist() == [True, True, False, False]

    # Stain 2 alone (no SOX2) can only check the first clause.
    stain_labels = classify_within_stain({"TBXT": values["TBXT"], "SOX17": values["SOX17"]}, rules)
    assert stain_labels["Mesoderm"].tolist() == [True, False, False, False]

    # Pattern bits follow rules.markers = (SOX17, TBXT, SOX2).
    names = label_names(rules)
    table = pattern_label_shares(rules)
    mesoderm, other = names.index("Mesoderm"), names.index("Other")
    assert table[0b010, mesoderm] == 1.0  # TBXT+ only
    assert table[0b111, mesoderm] == 1.0  # all high: second clause
    assert table[0b011, other] == 1.0  # TBXT+ SOX17+ SOX2-

    # A clause weight of 0.25 on the second clause: a pixel matching only
    # it is mesoderm a quarter of the time; matching both clauses, always.
    weighted = pattern_label_shares(rules, {"Mesoderm": (1.0, 0.25)})
    assert weighted[0b111, mesoderm] == pytest.approx(0.25)
    assert weighted[0b111, other] == pytest.approx(0.75)
    assert weighted[0b110, mesoderm] == pytest.approx(1.0)  # TBXT+ SOX2+ SOX17-
    np.testing.assert_allclose(weighted.sum(axis=1), 1.0)
    # Weight 0 switches a clause off for pixel labels as well.
    off = _rules(**{**rules.__dict__, "clause_weights": {"Mesoderm": (1.0, 0.0)}})
    assert classify_cell_types(values, off)["Mesoderm"].tolist() == [True, False, False, False]


def test_pixel_label_shares_use_clause_weights():
    rules = _rules(
        channels={"SOX17": "s2/SOX17", "TBXT": "s2/TBXT", "SOX2": "s1/SOX2"},
        thresholds={"SOX17": 0.5, "TBXT": 0.5, "SOX2": 0.5},
        rules={
            "Endoderm": {"SOX17": "high"},
            "Mesoderm": [{"TBXT": "high", "SOX17": "low"}, {"TBXT": "high", "SOX2": "high"}],
        },
        clause_weights={"Mesoderm": (0.4, 0.25)},
    )
    values = {
        "SOX17": np.array([0.9, 0.1, 0.1, 0.1]),
        "TBXT": np.array([0.9, 0.9, 0.1, 0.9]),
        "SOX2": np.array([0.9, 0.1, 0.1, 0.1]),
    }
    mask = np.array([True, True, True, False])
    shares = pixel_label_shares(values, rules, mask)

    assert list(shares) == list(label_names(rules))
    # Pixel 0 is endoderm and, a quarter of the time, also mesoderm.
    assert shares["Endoderm"][0] == pytest.approx(0.75)
    assert shares["Several"][0] == pytest.approx(0.25)
    assert shares["Mesoderm"][1] == pytest.approx(0.4)
    assert shares["Other"][1] == pytest.approx(0.6)
    assert shares["Other"][2] == pytest.approx(1.0)
    totals = np.sum(list(shares.values()), axis=0)
    np.testing.assert_allclose(totals, [1.0, 1.0, 1.0, 0.0])
    # Summed over the colony, the shares match counting pixel patterns.
    codes = np.array([0b111, 0b010, 0b000])
    expected = pattern_label_shares(rules)[codes].sum(axis=0)
    np.testing.assert_allclose([shares[name].sum() for name in label_names(rules)], expected)


def test_resolve_several_assigns_by_margin():
    rules = _rules(
        channels={"SOX17": "s2/SOX17", "TBXT": "s2/TBXT"},
        thresholds={"SOX17": 0.5, "TBXT": 0.5},
        rules={"Endoderm": {"SOX17": "high"}, "Mesoderm": {"TBXT": "high"}},
        clause_weights={"Mesoderm": (0.4,)},
    )
    names = label_names(rules)
    endoderm, several = names.index("Endoderm"), names.index("Several")
    # SOX17+ TBXT+ matches both: endoderm (1.0) leads mesoderm (0.4) by 0.6.
    kept = pattern_label_shares(rules)
    assert kept[0b11, several] == pytest.approx(0.4)
    np.testing.assert_allclose(pattern_label_shares(rules, resolve_several=0.3), kept)
    resolved = pattern_label_shares(rules, resolve_several=0.4)
    assert resolved[0b11, endoderm] == pytest.approx(1.0)
    assert resolved[0b11, several] == 0.0
    # Matching mesoderm alone is unchanged.
    np.testing.assert_allclose(resolved[0b10], kept[0b10])
    always = pattern_label_shares(rules, resolve_several=1.0)
    assert np.all(always[:, several] == 0.0)
    np.testing.assert_allclose(always.sum(axis=1), 1.0)

    values = {"SOX17": np.array([0.9]), "TBXT": np.array([0.9])}
    shares = pixel_label_shares(values, rules, resolve_several=1.0)
    assert shares["Endoderm"][0] == pytest.approx(1.0)


def test_sample_pixel_labels_draws_one_label_per_pixel():
    shares = {
        "A": np.array([1.0, 0.0, 0.3, 0.0]),
        "B": np.array([0.0, 1.0, 0.7, 0.0]),
    }
    labels = sample_pixel_labels(shares, np.random.default_rng(0))
    assert labels["A"][:2].tolist() == [True, False]
    assert labels["B"][:2].tolist() == [False, True]
    # Exactly one label inside, none where every share is 0.
    assert (labels["A"].astype(int) + labels["B"]).tolist() == [1, 1, 1, 0]

    many = {name: np.full(20000, share[2]) for name, share in shares.items()}
    drawn = sample_pixel_labels(many, np.random.default_rng(1))
    assert drawn["A"].mean() == pytest.approx(0.3, abs=0.02)


def test_marker_values_finds_measurement_or_marker_names():
    rules = _rules()
    images = np.arange(2 * 3 * 2 * 2).reshape(2, 3, 2, 2)
    by_measurement = marker_values(images, ["cell_fate_s2/TBXT", "x", "cell_fate_s2/SOX17"], rules)
    np.testing.assert_array_equal(by_measurement["TBXT"], images[:, 0])
    np.testing.assert_array_equal(by_measurement["SOX17"], images[:, 2])
    by_marker = marker_values(images, ["LMBR", "SOX17", "TBXT"], rules)
    np.testing.assert_array_equal(by_marker["TBXT"], images[:, 2])
    with pytest.raises(KeyError):
        marker_values(images, ["LMBR", "SOX2", "FOXA2"], rules)


def test_cell_type_file_round_trip(tmp_path):
    path = tmp_path / "cell_types" / "rules.yaml"
    rules = _rules()
    save_cell_type_rules(path, rules)
    assert load_cell_type_rules(path) == rules

    with_clauses = _rules(
        rules={"Endoderm": {"SOX17": "high"}, "Mesoderm": [{"TBXT": "high"}, {"SOX17": "low"}]},
        stains={"s2": {"SOX17": "s2/SOX17", "TBXT": "s2/TBXT"}},
        clause_weights={"Mesoderm": (1.0, 0.4)},
    )
    save_cell_type_rules(path, with_clauses)
    assert load_cell_type_rules(path) == with_clauses

    # Files from before clauses and stains still load.
    old = tmp_path / "old.yaml"
    old.write_text(
        "format: cell_type_rules_v1\nhour: 48\n"
        "markers:\n  SOX17: {channel: s2/SOX17, threshold: 0.4}\n"
        "cell_types:\n  Endoderm: {SOX17: high}\n"
    )
    assert load_cell_type_rules(old).rules == {"Endoderm": ({"SOX17": "high"},)}

    with pytest.raises(FileNotFoundError):
        load_cell_type_rules(tmp_path / "missing.yaml")
    other = tmp_path / "other.yaml"
    other.write_text("format: quality_flags_v1\n")
    with pytest.raises(ValueError):
        load_cell_type_rules(other)


@pytest.mark.parametrize(
    "overrides",
    [
        {"thresholds": {"SOX17": 0.5}},
        {"rules": {"Endoderm": {"SOX2": "high"}}},
        {"rules": {"Endoderm": {"SOX17": "positive"}}},
        {"rules": {"Other": {"SOX17": "high"}}},
        {"rules": {"Endoderm": []}},
        {"stains": {"s1": {"SOX2": "s1/SOX2"}}},
        {"clause_weights": {"Endoderm": (0.5, 0.5)}},
        {"clause_weights": {"Endoderm": (1.5,)}},
    ],
)
def test_cell_type_rules_are_checked(overrides):
    with pytest.raises(ValueError):
        _rules(**overrides)

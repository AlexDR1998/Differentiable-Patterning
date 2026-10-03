import numpy as np
import pytest

from Common.dataloader.cell_types import (
    CellTypeRules,
    cell_type_fractions,
    classify_cell_types,
    load_cell_type_rules,
    marker_values,
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
    ],
)
def test_cell_type_rules_are_checked(overrides):
    with pytest.raises(ValueError):
        _rules(**overrides)

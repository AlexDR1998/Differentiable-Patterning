"""Held-out 250K colonies: reading, normalisation, single-stain cell types, interpolation."""

from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from Common.dataloader.cell_types import load_cell_type_rules, label_names
from Experiments.micropatterns import density_test_data as dt

SUM_RULES = "Experiments/micropatterns/conf/cell_types/260726_cell_types_sum_v1.yaml"


def _write(path, values, pixels_per_um=1.72005):
    image = np.stack([np.full((6, 6), value, dtype=np.uint16) for value in values])
    tifffile.imwrite(path, image, imagej=True, resolution=(pixels_per_um, pixels_per_um))


def test_test_files_are_found_by_projection_and_hour(tmp_path):
    for name in ("MAX_KO24.tif", "SUM_KO28.tif", "SUM_KO24.tif", "SLICE_KO36.tif", "notes.tif"):
        _write(tmp_path / name, (1, 2, 3, 4))
    assert list(dt.test_image_paths(tmp_path, "sum")) == [24, 28]
    assert list(dt.test_image_paths(tmp_path, "slice")) == [36]


def test_channels_are_put_in_cell_fate_s2_order(tmp_path):
    # Files hold SOX17, TBXT, FOXA2, LMBR
    _write(tmp_path / "MAX_KO24.tif", (17, 20, 30, 40))
    image = dt.read_test_image(tmp_path / "MAX_KO24.tif")
    by_name = {name: image[0, 0, index] for name, index in dt.group_channels()}
    assert by_name == {
        "cell_fate_s2/SOX17": 17, "cell_fate_s2/TBXT": 20,
        "cell_fate_s2/FOXA2": 30, "cell_fate_s2/LMBR": 40,
    }
    assert dt.pixel_size_um(tmp_path / "MAX_KO24.tif") == pytest.approx(1 / 1.72005)


def _percentiles():
    return dt.channel_percentiles(SimpleNamespace(
        histogram_percentiles=(0.0, 100.0), cleaning=SimpleNamespace(channel_percentiles={}),
    ))


def test_joint_normalisation_keeps_differences_between_colonies():
    images = np.stack([np.full((2, 2, 4), 1.0), np.full((2, 2, 4), 3.0)])
    images[:, 0, 0] = 0.0  # a dark pixel in each
    masks = [np.ones((2, 2), dtype=bool)] * 2
    joint, _ = dt.normalise(images, masks, _percentiles(), "joint")
    per_image, _ = dt.normalise(images, masks, _percentiles(), "per_image")
    assert joint[0, 1, 1, 0] == pytest.approx(1 / 3) and joint[1, 1, 1, 0] == pytest.approx(1.0)
    assert per_image[0, 1, 1, 0] == per_image[1, 1, 1, 0] == pytest.approx(1.0)


def test_matched_normalisation_maps_onto_the_reference_range():
    images = np.stack([np.linspace(0, 10, 16).reshape(4, 4, 1).repeat(4, axis=2)])
    reference = {name: (0.2, 0.6) for name, _ in dt.group_channels()}
    matched, _ = dt.normalise(images, [np.ones((4, 4), dtype=bool)], _percentiles(), "match_150k", reference)
    assert matched.min() == pytest.approx(0.2) and matched.max() == pytest.approx(0.6)


def test_reference_reads_left_out_channels_from_another_panel():
    names = ["cell_fate_s1/SOX17", "cell_fate_s2/FOXA2", "cell_fate_s2/TBXT", "cell_fate_s2/LMBR"]
    image = np.stack([np.full((3, 3), value) for value in (0.1, 0.2, 0.3, 0.4)])
    ranges, sources = dt.reference_ranges([image], [np.ones((3, 3), dtype=bool)], names, _percentiles())
    assert sources["cell_fate_s2/SOX17"] == "cell_fate_s1/SOX17"
    assert ranges["cell_fate_s2/TBXT"] == pytest.approx((0.3, 0.3))


def test_single_stain_rules_drop_clauses_needing_other_stains():
    rules = load_cell_type_rules(SUM_RULES)
    restricted, dropped, borrowed = dt.stain_rules(rules)
    assert set(restricted.markers) == {"SOX17", "TBXT", "FOXA2"}
    assert borrowed == {}
    assert all("SOX2" not in clause for clauses in restricted.rules.values() for clause in clauses)
    assert dropped and all(clause.get("SOX2") not in (None, "any") for _, clause in dropped)
    assert restricted.cell_types == rules.cell_types


def test_model_prevalence_uses_only_the_stains_markers():
    rules = load_cell_type_rules(SUM_RULES)
    restricted, _, _ = dt.stain_rules(rules)
    markers = rules.markers
    # Every pixel TBXT high and SOX2 high, the rest low: mesoderm in the S2
    # rules (SOX17 low, TBXT high, FOXA2 low), whatever SOX2 is.
    pattern = (1 << markers.index("TBXT")) | (1 << markers.index("SOX2"))
    counts = np.zeros((2, 3, 4, 2 ** len(markers)))  # [densities, KO times, rings, patterns]
    counts[..., pattern] = 10.0
    shares = dt.model_stain_prevalence(counts, markers, restricted)
    labels = label_names(restricted)
    assert shares.shape == (2, 3, len(labels))
    weight = restricted.weights_of("Mesoderm")[0]  # the kept clause's weight
    assert np.allclose(shares[..., labels.index("Mesoderm")], weight)
    assert np.allclose(shares[..., labels.index("Other")], 1 - weight)


def test_figure_rows_hold_every_run_density_and_ko_time():
    rules = load_cell_type_rules(SUM_RULES)
    restricted, _, _ = dt.stain_rules(rules)
    markers = rules.markers
    counts = np.zeros((2, 3, 4, 2 ** len(markers)))  # [densities, KO times, rings, patterns]
    counts[..., 1 << markers.index("TBXT")] = 10.0
    result = {
        "pattern_counts": counts,
        "cell_type_markers": markers,
        "densities": [1000.0, 2000.0],
        "cells_per_mm2": [1000.0, 1800.0],  # the second density was capped
        "cells_per_colony": [200.0, 360.0],
        "covered_shares": [0.3, 0.5],
        "density_unit": "per_mm2",
        "requested_ko_hours": [0.0, 24.0, 48.0],
        "groups": {"model": "M1"},
        "model_id": "m1",
    }
    rows = dt.model_prevalence_rows([result, result], restricted, "model")
    labels = label_names(restricted)
    assert len(rows) == 2 * 2 * 3 * len(labels)
    capped = [r for r in rows if r["run"] == 1 and r["density_index"] == 1 and r["cell_type"] == "Mesoderm"]
    assert {r["density"] for r in capped} == {1800.0}
    assert {r["requested_density"] for r in capped} == {2000.0}
    assert [r["knockout_hour"] for r in capped] == [0.0, 24.0, 48.0]
    assert capped[0]["group"] == "M1"
    assert capped[0]["prevalence"] == pytest.approx(restricted.weights_of("Mesoderm")[0])


def test_prediction_is_interpolated_inside_the_sweep_only():
    grid = np.zeros((2, 2, 1))
    grid[1, :, 0] = 1.0  # density 2000: share 1
    grid[:, 1, 0] += 0.5  # KO 36 h: +0.5
    value = dt.predict_at(grid, [1000, 2000], [24, 36], 1500, 30)
    assert value[0] == pytest.approx(0.75)
    assert np.isnan(dt.predict_at(grid, [1000, 2000], [24, 36], 2500, 30)[0])


def test_test_density_scales_the_measured_density():
    results = [{"density_unit": "per_mm2", "measured_density": 3000.0}] * 2
    assert dt.test_density(results) == pytest.approx(5000.0)
    with pytest.raises(ValueError, match="cells per mm"):
        dt.test_density([{"density_unit": "share", "measured_density": 0.5}])


def test_markers_from_another_panel_are_read_from_the_test_panel():
    # The slice rules read SOX17 from cell_fate_s1; the test panel images it too.
    rules = load_cell_type_rules("Experiments/micropatterns/conf/cell_types/260726_cell_types_sliced_nca.yaml")
    restricted, _, borrowed = dt.stain_rules(rules)
    assert restricted.channels["SOX17"] == "cell_fate_s2/SOX17"
    assert borrowed == {"SOX17": "cell_fate_s1/SOX17"}
    assert {"Notochord", "Endoderm", "Mesoderm"} <= set(restricted.cell_types)


def test_panel_shares_count_each_pixel_once():
    rules = load_cell_type_rules(SUM_RULES)
    restricted, _, _ = dt.stain_rules(rules)
    mask = np.ones((8, 8), dtype=bool)
    # SOX17 high everywhere: endoderm
    values = {"SOX17": np.ones((8, 8)), "TBXT": np.zeros((8, 8)), "FOXA2": np.zeros((8, 8))}
    shares = dt.panel_label_shares(values, mask, restricted, 4)
    labels = label_names(restricted)
    assert shares[labels.index("Endoderm")] == pytest.approx(1.0)


def test_brightness_scales_one_colony_and_channel():
    image = np.ones((2, 2, 4), dtype=np.float32)
    scaled = dt.apply_brightness(image, {"cell_fate_s2/TBXT": {28: 2.0}}, 28)
    by_name = {name: scaled[0, 0, index] for name, index in dt.group_channels()}
    assert by_name["cell_fate_s2/TBXT"] == 2.0 and by_name["cell_fate_s2/SOX17"] == 1.0
    assert np.array_equal(dt.apply_brightness(image, {"cell_fate_s2/TBXT": {28: 2.0}}, 24), image)


def test_shares_follow_the_thresholds_for_data_and_model_alike():
    rules = load_cell_type_rules(SUM_RULES)
    mask = np.ones((4, 4), dtype=bool)
    # TBXT at 0.3 everywhere, nothing else: mesoderm if the TBXT threshold is below 0.3
    data = np.zeros((1, 4, 4, 4))
    data[0, [name for name, _ in dt.group_channels()].index("cell_fate_s2/TBXT")] = 0.3
    data_names = [name for name, _ in dt.group_channels()]
    model = np.zeros((1, 3, 4, 4))
    model[0, 1] = 0.3
    model_names = ["SOX17", "TBXT", "FOXA2"]
    for threshold, expected in ((0.2, True), (0.4, False)):
        data_shares, restricted = dt.shares_with(data, data_names, mask, rules, {"TBXT": threshold}, 2)
        model_shares, _ = dt.shares_with(model, model_names, mask, rules, {"TBXT": threshold}, 2)
        mesoderm = label_names(restricted).index("Mesoderm")
        assert (data_shares[0, mesoderm] > 0) == expected
        assert np.allclose(data_shares, model_shares)

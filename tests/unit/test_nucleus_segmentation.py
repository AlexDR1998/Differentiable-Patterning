"""Stored nucleus segmentations: storage, scoring against clicked nuclei, and use in counting."""

from types import SimpleNamespace

import numpy as np

from Experiments.micropatterns import nucleus_segmentation as segmentation


def discs(shape, centres, radius):
    rows, columns = np.indices(shape)
    labels = np.zeros(shape, dtype=np.int32)
    for index, (row, column) in enumerate(centres, start=1):
        labels[np.hypot(rows - row, columns - column) <= radius] = index
    return labels


def test_points_match_one_to_one_within_the_radius():
    found = [[10, 10], [10, 13], [40, 40]]
    clicked = [[10, 11], [30, 30]]

    # One click near two found points matches one of them only
    assert segmentation.match_points(found, clicked, 3.0) == (1, 2, 1)
    assert segmentation.match_points([], clicked, 3.0) == (0, 0, 2)


def test_crop_scores_count_merges_as_missed_nuclei():
    # Three nuclei clicked, but two were segmented as one
    labels = discs((60, 60), [(20, 20), (40, 40)], 4)
    labels[discs((60, 60), [(20, 28)], 4) > 0] = 1
    crop = {"top": 10, "left": 10, "size": 40}
    scores = segmentation.score_labels(labels, crop, [[20, 20], [20, 28], [40, 40]], 4.0)

    assert scores["clicked"] == 3 and scores["found"] == 2
    assert scores["missed"] == 1 and scores["extra"] == 0
    assert np.isclose(scores["count_ratio"], 2 / 3)


def test_crop_edges_neither_add_nor_lose_nuclei():
    # A clicked nucleus whose centre lands just outside the crop still matches;
    # an unclicked one centred right at the edge is not counted as extra
    labels = discs((80, 80), [(19, 40), (40, 40), (21, 60)], 4)
    crop = {"top": 20, "left": 20, "size": 40}
    scores = segmentation.score_labels(labels, crop, [[20.5, 40], [40, 40]], 4.0)

    assert (scores["matched"], scores["extra"], scores["missed"]) == (2, 0, 0)


def test_relabel_inside_keeps_nuclei_centred_in_the_colony():
    labels = discs((50, 50), [(10, 10), (25, 25), (45, 45)], 3)
    colony = discs((50, 50), [(25, 25)], 22) > 0  # (10, 10) is 21.2 px from the centre, (45, 45) 28 px

    kept = segmentation.relabel_inside(labels, colony)
    assert sorted(np.unique(kept)) == [0, 1, 2]
    assert kept[10, 10] == 1 and kept[45, 45] == 0


def test_stored_labels_are_used_for_counting(tmp_path, monkeypatch):
    from Experiments.micropatterns.cell_counts import colony_nuclei

    monkeypatch.setenv("NUCLEUS_SEGMENTATION_ROOT", str(tmp_path / "segmentation"))
    dataset = tmp_path / "260726_nca_dataset"
    image = dataset / "cell_fate_markers/ctrl_s1_slices/A1_F1_0h_slice.tif"
    image.parent.mkdir(parents=True)
    image.touch()
    relative = "cell_fate_markers/ctrl_s1_slices/A1_F1_0h_slice.tif"
    labels = discs((60, 60), [(20, 20), (30, 30), (2, 2)], 3)
    segmentation.write_labels(dataset, "test_method", relative, labels, {"method": "test"})
    alignment = SimpleNamespace(centre=lambda key: (30.0, 30.0), pattern_radius=25.0)

    assert segmentation.stored_methods(dataset) == ["test_method"]
    found, colony, _ = colony_nuclei(image, "cell_fate_s1", dataset, alignment, segmentation="test_method")
    # The nucleus at the corner lies outside the colony
    assert found.max() == 2 and colony[30, 30] and not colony[2, 2]


def test_unknown_images_fall_back_to_counting(tmp_path, monkeypatch):
    monkeypatch.setenv("NUCLEUS_SEGMENTATION_ROOT", str(tmp_path / "segmentation"))
    assert segmentation.stored_labels(tmp_path, "missing", "some/image.tif") is None


def test_settings_file_names_a_segmentation():
    from Experiments.micropatterns.texture_settings import DEFAULT_TEXTURE_SETTINGS_FILE, CellSettings, load_cell_settings

    assert load_cell_settings(DEFAULT_TEXTURE_SETTINGS_FILE, "slice").nucleus_options()["segmentation"] == "cpsam_sox2_lmbr"
    assert CellSettings(segmentation="cpsam").nucleus_options()["segmentation"] == "cpsam"


def test_sweep_refuses_a_missing_stored_segmentation(tmp_path, monkeypatch):
    import pytest

    from Experiments.micropatterns.density_ko_sweep import check_stored_segmentation

    monkeypatch.setenv("NUCLEUS_SEGMENTATION_ROOT", str(tmp_path / "segmentation"))
    dataset = tmp_path / "260726_nca_dataset"
    image = dataset / "cell_fate_markers/ctrl_s1_slices/A1_F1_0h_slice.tif"
    image.parent.mkdir(parents=True)
    image.touch()
    colonies = {
        "schema": SimpleNamespace(group_names=("cell_fate_s1", "rna_expression")),
        "source_files": np.array([[str(image), ""]]),
    }
    check_stored_segmentation(colonies, dataset, "")  # none set: nothing to check
    with pytest.raises(FileNotFoundError, match="not at"):
        check_stored_segmentation(colonies, dataset, "cpsam")
    segmentation.write_labels(dataset, "cpsam", "cell_fate_markers/ctrl_s1_slices/other.tif", np.zeros((4, 4)), {})
    with pytest.raises(FileNotFoundError, match="no labels"):
        check_stored_segmentation(colonies, dataset, "cpsam")
    segmentation.write_labels(dataset, "cpsam", "cell_fate_markers/ctrl_s1_slices/A1_F1_0h_slice.tif", np.zeros((4, 4)), {})
    check_stored_segmentation(colonies, dataset, "cpsam")

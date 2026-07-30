from pathlib import Path

import numpy as np
import skimage.io

from Common.dataloader.micropattern_260726 import (
    build_micropattern_260726_manifest,
    load_micropattern_260726,
)


GROUPS = {
    "cell_fate_s1": ("cell_fate_markers/ctrl_s1", 4),
    "cell_fate_s2": ("cell_fate_markers/ctrl_s2", 4),
    "rna_expression": ("signalling/rna_expression/ctrl", 4),
    "protein_response": ("signalling/protein_response/ctrl", 3),
}


def _write_group_image(path, channel_count, replicate):
    image = np.zeros((8, 8, channel_count), dtype=np.uint16)
    for channel in range(channel_count):
        image[2:6, 2:6, channel] = replicate * 100 + channel + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    skimage.io.imsave(path, image, check_contrast=False)


def _make_control_dataset(root, rna_replicates=3):
    root = Path(root)
    for group, (relative_directory, channel_count) in GROUPS.items():
        count = rna_replicates if group == "rna_expression" else 3
        for replicate in range(1, count + 1):
            if group == "rna_expression":
                filename = f"0h_ctrl-{replicate}.tif"
            else:
                filename = f"sample_{replicate}_0h.ome.tif"
            _write_group_image(
                root / relative_directory / filename,
                channel_count,
                replicate,
            )


def _make_empty_condition_directories(root, condition):
    directories = (
        Path("cell_fate_markers") / f"{condition}_s1",
        Path("cell_fate_markers") / f"{condition}_s2",
        Path("signalling/rna_expression") / condition,
        Path("signalling/protein_response") / condition,
    )
    for directory in directories:
        (Path(root) / directory).mkdir(parents=True, exist_ok=True)


def test_loader_places_physical_replicates_on_batch_axis(tmp_path):
    _make_control_dataset(tmp_path)
    histogram_bins = np.tile(
        np.array([[0.0, 1000.0]], dtype=np.float32), (14, 1)
    )

    data, aux, names, boundary, measurement_mask = load_micropattern_260726(
        tmp_path,
        conditions=("ctrl",),
        timesteps=(0,),
        downsample=1,
        histogram_bins=histogram_bins,
        align=False,
    )
    data = np.asarray(data)

    assert data.shape == (3, 1, 14, 8, 8)
    assert boundary.shape == (3, 1, 8, 8)
    assert measurement_mask.shape == (3, 1, 14)
    assert np.all(measurement_mask)
    assert aux["batch_replicates"] == (1, 2, 3)
    assert len(names) == 14
    assert all("DAPI" not in name for name in names)
    assert np.all(np.asarray(boundary) == np.asarray(boundary)[0:1])
    expected_group_masks = np.asarray(boundary)[:, None] & np.asarray(
        aux["group_mask"]
    )[..., None, None]
    assert np.array_equal(aux["group_boundary_masks"], expected_group_masks)

    # S1/SOX17 is source page 0; each physical image remains a distinct batch.
    assert np.allclose(data[:, 0, 0, 3, 3], [0.101, 0.201, 0.301])
    # RNA/CER1 uses source page 1, demonstrating that DAPI page 0 was skipped.
    assert np.allclose(data[:, 0, 8, 3, 3], [0.102, 0.202, 0.302])


def test_manifest_records_extra_images_without_silently_using_them(tmp_path):
    _make_control_dataset(tmp_path, rna_replicates=4)

    inventory = build_micropattern_260726_manifest(
        tmp_path,
        conditions=("ctrl",),
        timesteps=(0,),
    )

    selected_rna = inventory["selected"][("ctrl", "rna_expression", 0)]
    assert [Path(path).name for path in selected_rna] == [
        "0h_ctrl-1.tif",
        "0h_ctrl-2.tif",
        "0h_ctrl-3.tif",
    ]
    assert any(
        Path(path).name == "0h_ctrl-4.tif"
        for path in inventory["unselected_files"]
    )


def test_loader_group_selection_is_independent_of_replicate_count(tmp_path):
    _make_control_dataset(tmp_path)
    histogram_bins = np.tile(
        np.array([[0.0, 1000.0]], dtype=np.float32), (4, 1)
    )

    data, aux, names, boundary, measurement_mask = load_micropattern_260726(
        tmp_path,
        conditions=("ctrl",),
        timesteps=(0,),
        downsample=1,
        replicate_count=1,
        experiment_groups=("cell_fate_s1",),
        histogram_bins=histogram_bins,
        align=False,
    )

    assert np.asarray(data).shape == (1, 1, 4, 8, 8)
    assert np.asarray(boundary).shape == (1, 1, 8, 8)
    assert np.asarray(measurement_mask).shape == (1, 1, 4)
    assert aux["selected_experiment_groups"] == ("cell_fate_s1",)
    assert aux["batch_replicates"] == (1,)
    assert names == [
        "cell_fate_s1/SOX17",
        "cell_fate_s1/SOX2",
        "cell_fate_s1/TBXT",
        "cell_fate_s1/LMBR",
    ]
    assert set(record.group for record in aux["manifest"]) == {"cell_fate_s1"}


def test_loader_can_select_a_group_without_cell_fate_s1(tmp_path):
    _make_control_dataset(tmp_path)
    histogram_bins = np.tile(
        np.array([[0.0, 1000.0]], dtype=np.float32), (4, 1)
    )

    data, aux, names, boundary, measurement_mask = load_micropattern_260726(
        tmp_path,
        conditions=("ctrl",),
        timesteps=(0,),
        downsample=1,
        replicate_count=1,
        experiment_groups=("cell_fate_s2",),
        histogram_bins=histogram_bins,
        align=False,
    )

    assert np.asarray(data).shape == (1, 1, 4, 8, 8)
    assert np.asarray(boundary).shape == (1, 1, 8, 8)
    assert np.asarray(measurement_mask).shape == (1, 1, 4)
    assert aux["selected_experiment_groups"] == ("cell_fate_s2",)
    assert names == [
        "cell_fate_s2/SOX17",
        "cell_fate_s2/FOXA2",
        "cell_fate_s2/TBXT",
        "cell_fate_s2/LMBR",
    ]


def test_loader_batch_multiplier_repeats_selected_batches(tmp_path):
    _make_control_dataset(tmp_path)
    histogram_bins = np.tile(
        np.array([[0.0, 1000.0]], dtype=np.float32), (4, 1)
    )

    data, aux, _, boundary, measurement_mask = load_micropattern_260726(
        tmp_path,
        conditions=("ctrl",),
        timesteps=(0,),
        downsample=1,
        replicate_count=1,
        batch_multiplier=2,
        experiment_groups=("cell_fate_s1",),
        histogram_bins=histogram_bins,
        align=False,
    )

    assert np.asarray(data).shape[0] == 2
    assert np.asarray(boundary).shape[0] == 2
    assert np.asarray(measurement_mask).shape[0] == 2
    assert aux["batch_multiplier"] == 2
    assert aux["batch_replicates"] == (1, 1)


def test_sl0_initial_state_uses_corresponding_control_replicates(tmp_path):
    _make_control_dataset(tmp_path)
    _make_empty_condition_directories(tmp_path, "sl0")
    histogram_bins = np.tile(
        np.array([[0.0, 1000.0]], dtype=np.float32), (14, 1)
    )

    data, aux, _, _, measurement_mask = load_micropattern_260726(
        tmp_path,
        conditions=("sl0",),
        timesteps=(0,),
        downsample=1,
        histogram_bins=histogram_bins,
        align=False,
    )

    assert data.shape[0] == 3
    assert np.all(measurement_mask)
    assert np.all(aux["group_mask"])
    assert np.all(aux["is_substituted"])
    assert set(aux["source_conditions"].reshape(-1)) == {"ctrl"}


def test_rna_stack_is_aligned_using_discarded_dapi_channel(tmp_path):
    _make_control_dataset(tmp_path)
    rna_path = tmp_path / "signalling/rna_expression/ctrl/0h_ctrl-1.tif"
    rna = np.zeros((8, 8, 4), dtype=np.uint16)
    for channel in range(4):
        rna[1:5, 1:5, channel] = 100 + channel + 1
    skimage.io.imsave(rna_path, rna, check_contrast=False)
    histogram_bins = np.tile(
        np.array([[0.0, 1000.0]], dtype=np.float32), (14, 1)
    )

    aligned, _, names, _, _ = load_micropattern_260726(
        tmp_path,
        conditions=("ctrl",),
        timesteps=(0,),
        downsample=1,
        histogram_bins=histogram_bins,
        align=True,
    )
    aligned = np.asarray(aligned)[0, 0, names.index("rna_expression/CER1")]
    rows, columns = np.indices(aligned.shape)
    mass = aligned.sum()
    centre = (
        float((rows * aligned).sum() / mass),
        float((columns * aligned).sum() / mass),
    )

    assert np.allclose(centre, (3.5, 3.5), atol=0.25)

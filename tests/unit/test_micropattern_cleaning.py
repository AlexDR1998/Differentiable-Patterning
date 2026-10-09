from pathlib import Path

import numpy as np
import pytest
import skimage.io

from Common.dataloader.alignment import (
    ColonyAlignment,
    block_centre,
    coverage_map,
    fit_colony_centre,
    grid_position,
    load_colony_alignment,
    save_colony_alignment,
)
from Common.dataloader.micropattern_260726 import load_micropattern_260726
from Common.dataloader.micropattern_cleaning import MicropatternCleaningConfig

SIZE = 128
RADIUS = 40
# Raw colony centres (row, column), off the image centre.
CENTRES = {
    ("ctrl", 1, 0): (60.0, 70.0),
    ("ctrl", 1, 24): (68.0, 58.0),
    ("ctrl", 1, 36): (63.0, 66.0),
    ("ctrl", 2, 0): (66.0, 60.0),
    ("ctrl", 2, 24): (58.0, 64.0),
    ("ctrl", 2, 36): (70.0, 61.0),
    ("sl24", 1, 36): (61.0, 69.0),
    ("sl24", 2, 36): (67.0, 57.0),
}
# Brightness of each (condition, replicate); knockout 36h is twice as bright.
BRIGHTNESS = {("ctrl", 1): 1000.0, ("ctrl", 2): 2000.0, ("sl24", 1): 2000.0, ("sl24", 2): 4000.0}


def _colony(centre, brightness, seed):
    rows, columns = np.ogrid[:SIZE, :SIZE]
    inside = (rows - centre[0]) ** 2 + (columns - centre[1]) ** 2 <= RADIUS**2
    texture = np.random.default_rng(seed).uniform(0.5, 1.0, (SIZE, SIZE))
    channel = np.where(inside, brightness * texture, 0.0)
    # Four source channels; LMBR (page 3) is the structural stain.
    return np.stack([channel, channel, channel, np.where(inside, 3000.0, 0.0)], axis=-1).astype(
        np.uint16
    )


def _make_dataset(root):
    centres = {}
    for seed, ((condition, replicate, hour), centre) in enumerate(CENTRES.items()):
        relative = Path("cell_fate_markers") / f"{condition}_s1" / f"sample_{replicate}_{hour}h.ome.tif"
        path = Path(root) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        skimage.io.imsave(
            path, _colony(centre, BRIGHTNESS[(condition, replicate)], seed), check_contrast=False
        )
        centres[relative.as_posix()] = centre
    alignment_file = Path(root) / "alignment.yaml"
    save_colony_alignment(alignment_file, ColonyAlignment(pattern_radius=RADIUS, centres=centres))
    return alignment_file


def _cleaning(alignment_file, **changes):
    settings = dict(
        enabled=True,
        alignment_file=str(alignment_file),
        hot_pixel_thresholds={},
        background_radii={},
        normalisation="per_replicate",
        percentiles_inside_mask=True,
    )
    settings.update(changes)
    return MicropatternCleaningConfig(**settings)


def _load(root, cleaning, conditions=("ctrl",), downsample=4, **options):
    return load_micropattern_260726(
        root,
        conditions=conditions,
        timesteps=(0, 24, 36),
        downsample=downsample,
        replicate_count=2,
        experiment_groups=("cell_fate_s1",),
        hist_eqs=(0.0, 100.0),
        intensity_factors={},
        cleaning=cleaning,
        **options,
    )


def test_cleaned_images_are_centred_from_the_alignment_file(tmp_path):
    alignment_file = _make_dataset(tmp_path)
    data, _, _, boundary, _ = _load(tmp_path, _cleaning(alignment_file))
    data = np.asarray(data)
    middle = (data.shape[-1] - 1) / 2.0
    rows, columns = np.indices(data.shape[-2:])
    for batch in range(data.shape[0]):
        for time in range(data.shape[1]):
            lmbr = data[batch, time, 3]
            assert lmbr.sum() > 0
            row = (lmbr * rows).sum() / lmbr.sum()
            column = (lmbr * columns).sum() / lmbr.sum()
            assert abs(row - middle) < 0.6 and abs(column - middle) < 0.6
    # Every batch shares the centred disk of the pattern radius.
    disk = np.hypot(rows - middle, columns - middle) <= RADIUS / 4
    assert np.array_equal(np.asarray(boundary)[0, 0], disk)


def test_per_replicate_and_shared_bounds(tmp_path):
    alignment_file = _make_dataset(tmp_path)
    per_replicate, aux, names, _, _ = _load(tmp_path, _cleaning(alignment_file))
    sox17 = names.index("cell_fate_s1/SOX17")
    bounds = aux["trajectory_bounds"]
    assert aux["histogram_bins"] is None
    # Replicate 2 is twice as bright, so its own upper bound is about twice as high.
    ratio = bounds[("ctrl", 2)]["cell_fate_s1/SOX17"][1] / bounds[("ctrl", 1)]["cell_fate_s1/SOX17"][1]
    assert 1.8 < ratio < 2.2
    # Each replicate is stretched to the full range.
    for batch in range(2):
        assert np.asarray(per_replicate)[batch, :, sox17].max() == pytest.approx(1.0)

    shared, aux, _, _, _ = _load(tmp_path, _cleaning(alignment_file, normalisation="replicate_mean"))
    shared = np.asarray(shared)
    assert aux["trajectory_bounds"][("ctrl", 1)] == aux["trajectory_bounds"][("ctrl", 2)]
    # With shared bounds the brighter replicate stays brighter.
    assert shared[1, :, sox17].mean() > 1.5 * shared[0, :, sox17].mean()
    # One shared row of bounds per loaded channel (stain 1 has four).
    assert np.asarray(aux["histogram_bins"]).shape == (4, 2)


def test_knockouts_can_use_control_bounds(tmp_path):
    alignment_file = _make_dataset(tmp_path)
    own, own_aux, names, _, _ = _load(tmp_path, _cleaning(alignment_file), conditions=("sl24",))
    control, control_aux, _, _, _ = _load(
        tmp_path,
        _cleaning(alignment_file, knockouts_use_control_bounds=True),
        conditions=("sl24",),
    )
    sox17 = names.index("cell_fate_s1/SOX17")
    name = "cell_fate_s1/SOX17"
    # Control-referenced knockouts use the control replicate's bounds ...
    _, reference_aux, _, _, _ = _load(tmp_path, _cleaning(alignment_file))
    assert control_aux["trajectory_bounds"][("sl24", 1)][name] == pytest.approx(
        reference_aux["trajectory_bounds"][("ctrl", 1)][name]
    )
    # ... so the brighter knockout 36h image is clipped instead of rescaled.
    knockout_36h = np.asarray(control)[0, 2, sox17]
    assert np.mean(knockout_36h[knockout_36h > 0] >= 1.0) > 0.5
    # Its own bounds include the 36h image and rescale it.
    own_36h = np.asarray(own)[0, 2, sox17]
    assert np.mean(own_36h[own_36h > 0] >= 1.0) < 0.1
    assert own_aux["trajectory_bounds"][("sl24", 1)][name][1] > control_aux["trajectory_bounds"][("sl24", 1)][name][1]


def test_training_downsampling_beyond_four(tmp_path):
    alignment_file = _make_dataset(tmp_path)
    data, _, _, boundary, _ = _load(tmp_path, _cleaning(alignment_file), downsample=8)
    assert np.asarray(data).shape[-2:] == (16, 16)
    assert np.asarray(boundary).shape[-2:] == (16, 16)
    assert np.asarray(data).max() <= 1.0


def test_cleaning_errors(tmp_path):
    alignment_file = _make_dataset(tmp_path)
    incomplete = tmp_path / "incomplete.yaml"
    alignment = load_colony_alignment(alignment_file)
    save_colony_alignment(
        incomplete,
        ColonyAlignment(
            pattern_radius=alignment.pattern_radius,
            centres=dict(list(alignment.centres.items())[1:]),
        ),
    )
    with pytest.raises(KeyError, match="no centre"):
        _load(tmp_path, _cleaning(incomplete))
    with pytest.raises(ValueError, match="per-replicate"):
        _load(tmp_path, _cleaning(alignment_file), histogram_bins=np.ones((4, 2)) * [0, 1])
    with pytest.raises(ValueError, match="alignment_file is required"):
        MicropatternCleaningConfig(enabled=True)
    with pytest.raises(FileNotFoundError, match="Export it"):
        _load(tmp_path, _cleaning(tmp_path / "missing.yaml"))


def test_alignment_file_round_trip(tmp_path):
    alignment = ColonyAlignment(
        pattern_radius=401.234,
        centres={"a/b.tif": (540.5, 530.25)},
        details={"a/b.tif": {"automatic": (541.0, 532.0), "manual": (-0.5, -1.75), "score": 0.97}},
        settings={"edge_width": 0.08},
    )
    path = tmp_path / "nested" / "alignment.yaml"
    save_colony_alignment(path, alignment)
    loaded = load_colony_alignment(path)
    assert loaded.pattern_radius == pytest.approx(401.23)
    assert loaded.centres == {"a/b.tif": (540.5, 530.25)}
    assert loaded.details["a/b.tif"]["manual"] == [-0.5, -1.75]
    assert loaded.settings == {"edge_width": 0.08}


def test_fit_colony_centre_finds_an_off_centre_colony():
    rows, columns = np.ogrid[:1080, :1080]
    foreground = (rows - 560) ** 2 + (columns - 515) ** 2 <= 400**2
    coverage = coverage_map(foreground, downsample=4)
    centre, score = fit_colony_centre(coverage, radius=100.0)
    found = block_centre(centre, 4)
    assert np.hypot(found[0] - 560, found[1] - 515) < 4.0
    assert score > 0.9
    assert np.allclose(grid_position(block_centre((10.0, 20.0), 4), 4), (10.0, 20.0))


def test_flagged_images_are_unmeasured_and_filled_from_other_replicates(tmp_path):
    alignment_file = _make_dataset(tmp_path)
    flagged = "cell_fate_markers/ctrl_s1/sample_1_24h.ome.tif"
    clean, clean_aux, names, _, clean_mask = _load(tmp_path, _cleaning(alignment_file, normalisation="replicate_mean"))
    data, aux, _, _, mask = _load(
        tmp_path,
        _cleaning(alignment_file, normalisation="replicate_mean"),
        excluded_images=[flagged, "some/other/image.tif"],
    )
    data, mask, clean = np.asarray(data), np.asarray(mask), np.asarray(clean)
    assert aux["excluded_images"] == (flagged,)
    # Replicate 1 at 24h (time index 1) is unmeasured in every stain-1 channel ...
    assert not mask[0, 1].any() and mask[1, 1].all() and mask[0, [0, 2]].all()
    assert aux["excluded"][0, 1, 0] and aux["imputed"][0, 1, 0]
    # ... filled with replicate 2 (the only other replicate), and left out of
    # the shared bounds, so the other images are rescaled differently.
    assert np.allclose(data[0, 1], data[1, 1])
    assert clean_aux["trajectory_bounds"] != aux["trajectory_bounds"]
    # Other timesteps of the flagged replicate are unchanged in shape and kept.
    assert data[0, 0].max() > 0 and data[0, 2].max() > 0


def test_quality_flags_file_round_trip(tmp_path):
    from Common.dataloader.quality_flags import load_quality_flags, save_quality_flags

    path = tmp_path / "flags" / "quality.yaml"
    save_quality_flags(path, {"b.tif": "blurred", "a.tif": ""})
    assert load_quality_flags(path) == {"a.tif": "", "b.tif": "blurred"}
    save_quality_flags(path, {})
    assert load_quality_flags(path) == {}
    with pytest.raises(FileNotFoundError, match="Export it"):
        load_quality_flags(tmp_path / "missing.yaml")


def test_cache_gives_the_same_data_and_skips_cleaning(tmp_path, monkeypatch):
    from Common.dataloader import micropattern_260726
    from Common.dataloader.disk_cache import list_cache_folders

    dataset = tmp_path / "dataset"
    cache = tmp_path / "cache"
    alignment_file = _make_dataset(dataset)
    per_replicate = _cleaning(alignment_file, background_radii={"cell_fate_s1/SOX2": 10})
    shared = _cleaning(
        alignment_file, background_radii={"cell_fate_s1/SOX2": 10}, normalisation="replicate_mean"
    )
    expected = [_load(dataset, per_replicate), _load(dataset, shared)]
    cached = [_load(dataset, per_replicate, cache_dir=cache)]
    [entry] = list_cache_folders(cache)
    assert entry["items"] == len(CENTRES) - 2  # the knockout images are not loaded

    def no_cleaning(*args, **kwargs):
        raise AssertionError("cleaned again instead of reading the cache")

    monkeypatch.setattr(micropattern_260726, "_clean_image", no_cleaning)
    # A different normalisation still reads the same cached images.
    cached.append(_load(dataset, shared, cache_dir=cache))
    for from_cache, uncached in zip(cached, expected):
        assert np.array_equal(np.asarray(from_cache.data), np.asarray(uncached.data))
        assert np.array_equal(np.asarray(from_cache.boundary_mask), np.asarray(uncached.boundary_mask))

    # Other cleaning settings get a new folder.
    monkeypatch.undo()
    _load(dataset, _cleaning(alignment_file, background_radii={"cell_fate_s1/SOX2": 20}), cache_dir=cache)
    assert len(list_cache_folders(cache)) == 2

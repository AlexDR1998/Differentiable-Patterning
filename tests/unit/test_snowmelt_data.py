from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
import yaml

from Common.dataloader.snowmelt import (
    BANDS,
    DateSplit,
    block_mean,
    build_snowmelt_sequence,
    day_of_year,
    days_since_first,
    fill_missing,
    incidence_files,
    load_snowmelt,
    resolve_snowmelt_root,
)
from Experiments.config import config_to_dict, experiment_config_from_mapping
from Experiments.snowmelt.config import SnowmeltDataConfig
from NCA.trainer.data_augmenter.snowmelt import (
    apply_boundary_channels,
    SnowmeltAugmenter,
)


def _synthetic_raw(T=3, H=8, W=8):
    """Minimal load_snowmelt()-shaped dict with a corner outside the catchment."""
    rng = np.random.default_rng(0)
    mask = np.ones((H, W), dtype=bool)
    mask[:3, :3] = False

    def dyn(shape, low, high):
        values = rng.uniform(low, high, shape).astype(np.float32)
        values[..., ~mask] = np.nan
        return values

    raw = dyn((T, len(BANDS), H, W), 0.0, 1.2)
    raw[1, BANDS.index("B3"), 5, 5] = np.nan  # sparse no-data inside the catchment
    dem = np.where(mask, rng.uniform(2000, 3000, (H, W)), np.nan).astype(np.float32)
    return {
        "dates": ["2018-05-25", "2018-06-14", "2018-06-19"][:T],
        "bands": BANDS,
        "raw": raw,
        "ndsi": dyn((T, H, W), -1.0, 1.0),
        "ndvi": dyn((T, H, W), -1.0, 1.0),
        "sca": np.where(mask, rng.integers(0, 2, (T, H, W)), np.nan).astype(np.float32),
        "dem": dem,
        "incidence": np.where(mask, rng.uniform(0, 90, (H, W)), np.nan).astype(np.float32),
        "mask": mask,
    }


def _write_dataset(root, version, dates=("2018-05-25", "2018-06-14"), H=6, W=5):
    """Tiny on-disk dataset in the v1 or v2 folder layout; the first row is outside the catchment."""
    import tifffile

    georef = [
        (33550, "d", 3, (10.0, 10.0, 0.0)),  # ModelPixelScaleTag
        (33922, "d", 6, (0.0, 0.0, 0.0, 354530.0, 5043040.0, 0.0)),  # ModelTiepointTag
    ]
    suffix = "" if version == 1 else "_v1.0"

    def write(path, values):
        path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(path, np.asarray(values, dtype=np.float32), extratags=georef)

    inside = np.ones((H, W), dtype=bool)
    inside[0] = False
    elevation = 2500.0 + 10.0 * np.arange(H)[:, None]
    for d in dates:
        for band in BANDS:
            write(root / f"S2_rawbands{suffix}" / f"DoraNivolet_{band}_{d}.tif", np.where(inside, 0.5, -9999.0))
        for prefix in ("NDSI_Nivolet", "NDVI_Nivolet", "SCA_Nivolet_NDSIgt04"):
            write(root / f"S2_derived_indexes{suffix}" / f"{prefix}_{d}.tif", np.where(inside, 0.2, np.nan))

    topo = root / f"S2_topographic_attributes{suffix}"
    if version == 1:
        write(topo / "DEM_10mTinitaly_NivoletMask.tif", np.where(inside, elevation, 0.0))
        write(topo / "INCIDENCEANGLE_10mTinitaly_NivoletMask.tif", np.where(inside, 40.0, 0.0))
    else:
        outside = np.finfo(np.float32).min
        for name, value in (("DEM", elevation), ("SLOPE", 30.0), ("ASPECT", 180.0)):
            write(topo / f"{name}_Nivolet_10m_filled" / f"{name}_10mDTMUnicoTinitaly_NivoletMask.tif",
                  np.where(inside, value, outside if name == "DEM" else np.nan))
        for d in dates:
            for hour in (10, 11, 12):
                write(topo / "INCIDENCEANGLE_Nivolet_10m_filled" / f"INCIDENCEANGLE_DOY{day_of_year(d):03d}_H{hour:02d}.tif",
                      np.where(inside, float(hour), np.nan))
    return root


def test_load_snowmelt_reads_v1_layout(tmp_path):
    data = load_snowmelt(_write_dataset(tmp_path, version=1))

    assert data["dates"] == ["2018-05-25", "2018-06-14"]
    assert data["raw"].shape == (2, len(BANDS), 6, 5)
    assert not data["mask"][0].any() and data["mask"][1:].all()
    assert np.isnan(data["raw"][:, :, 0]).all() and not np.isnan(data["raw"][:, :, 1:]).any()
    assert data["incidence"].shape == (6, 5) and data["incidence_hour"] is None
    assert data["slope"] is None and data["aspect"] is None
    assert incidence_files(tmp_path) == {}


def test_load_snowmelt_reads_v2_layout_with_hourly_incidence(tmp_path):
    root = _write_dataset(tmp_path / "v2", version=2)
    data = load_snowmelt(root, incidence_hour=12)

    assert not data["mask"][0].any() and data["mask"][1:].all()  # float32-min outside the catchment
    assert data["slope"].shape == (6, 5) and np.isnan(data["aspect"][0]).all()
    assert data["incidence"].shape == (2, 6, 5) and data["incidence_hour"] == 12
    assert np.all(data["incidence"][:, 1:] == 12.0)
    assert set(incidence_files(root)) == {(day_of_year(d), h) for d in data["dates"] for h in (10, 11, 12)}
    with pytest.raises(FileNotFoundError):
        load_snowmelt(root, incidence_hour=3)
    # Hourly incidence is not a single static map
    with pytest.raises(ValueError):
        build_snowmelt_sequence(raw=data, static_channels=("INCIDENCE",))
    assert build_snowmelt_sequence(raw=data, static_channels=("DEM",), pad=0).data.shape == (1, 2, 1, 6, 5)
    # Slope and aspect channels, aspect as cos/sin mapped to [0, 1] (the test aspect is 180 degrees: south)
    sequence = build_snowmelt_sequence(raw=data, static_channels=("SLOPE", "NORTHNESS", "EASTNESS"), pad=0)
    assert sequence.boundary_channel_names == ("CATCHMENT", "SLOPE", "NORTHNESS", "EASTNESS")
    inside = sequence.boundary_mask[0, 0] > 0.5
    assert np.allclose(sequence.boundary_mask[0, 1][inside], 30.0 / 90.0)
    assert np.allclose(sequence.boundary_mask[0, 2][inside], 0.0, atol=1e-6)
    assert np.allclose(sequence.boundary_mask[0, 3][inside], 0.5, atol=1e-6)


def test_resolve_snowmelt_root_picks_the_version_folder(tmp_path, monkeypatch):
    _write_dataset(tmp_path / "v1", version=1)
    _write_dataset(tmp_path / "v2", version=2)
    monkeypatch.delenv("SNOWMELT_DATA_ROOT", raising=False)
    monkeypatch.delenv("DATA_PATH_BASE", raising=False)

    assert resolve_snowmelt_root(tmp_path, "v1") == tmp_path / "v1"
    assert resolve_snowmelt_root(tmp_path) == tmp_path / "v2"  # the default version
    with pytest.raises(FileNotFoundError, match="v3"):
        resolve_snowmelt_root(tmp_path, "v3")
    # v1 data has no slope/aspect, so the v2-only channels fail clearly
    with pytest.raises(ValueError, match="v2"):
        build_snowmelt_sequence(root=tmp_path, version="v1", static_channels=("SLOPE",), pad=0)
    assert build_snowmelt_sequence(root=tmp_path, version="v1", static_channels=("DEM", "INCIDENCE"), pad=0).dates == (
        "2018-05-25", "2018-06-14")


def test_day_of_year():
    assert day_of_year("2018-04-05") == 95 and day_of_year("2018-09-22") == 265


def test_days_since_first():
    assert days_since_first(["2018-05-25", "2018-06-14", "2018-06-19"]) == (0.0, 20.0, 25.0)


def test_fill_missing_uses_nearest_catchment_pixel_and_zeros_outside():
    mask = np.ones((3, 4), dtype=bool)
    mask[:, 0] = False
    values = np.array([[9.0, 1.0, 2.0, 3.0], [9.0, np.nan, 2.0, 3.0], [9.0, 1.0, 2.0, 3.0]])

    filled = fill_missing(values, mask)

    assert filled[1, 1] == 1.0
    assert np.all(filled[:, 0] == 0.0)
    assert not np.isnan(filled).any()


def test_block_mean_ignores_nans():
    values = np.array([[1.0, np.nan], [3.0, np.nan]])
    assert block_mean(values, 2)[0, 0] == pytest.approx(2.0)


def test_sequence_scaling_padding_and_boundary_layout():
    raw = _synthetic_raw()
    sequence = build_snowmelt_sequence(
        target_channels=("SCA", "NDSI", "B3"), static_channels=("DEM", "INCIDENCE"), downsample=2, pad=1, raw=raw
    )

    assert sequence.data.shape == (1, 3, 3, 6, 6)
    assert sequence.boundary_mask.shape == (1, 3, 6, 6)
    assert sequence.boundary_channel_names == ("CATCHMENT", "DEM", "INCIDENCE")
    assert sequence.observation_times == (0.0, 20.0, 25.0)
    assert not np.isnan(sequence.data).any() and not np.isnan(sequence.boundary_mask).any()
    assert sequence.data.min() >= 0.0 and sequence.data.max() <= 1.0
    assert sequence.boundary_mask.max() <= 1.0
    catchment = sequence.boundary_mask[0, 0].astype(bool)
    # Zero border and the outside-catchment corner block stay empty.
    assert not catchment[0].any() and not catchment[:, -1].any() and not catchment[1, 1]
    assert np.all(sequence.data[0][..., ~catchment] == 0.0)
    assert np.all(sequence.boundary_mask[0, 1:][:, ~catchment] == 0.0)


def test_date_split_resolves_indices_and_iso_dates():
    dates = ("2018-04-05", "2018-04-20", "2018-05-25", "2018-06-14", "2018-06-19")
    split = DateSplit.resolve(dates, exclude=(0, "2018-04-20"), hold_out=(-2,))

    assert split.used == (2, 3, 4)
    assert split.training == (2, 4)
    assert split.held_out == (3,)
    with pytest.raises(ValueError, match="both excluded and held out"):
        DateSplit.resolve(dates, exclude=(0,), hold_out=(0,))
    with pytest.raises(ValueError, match="initial condition"):
        DateSplit.resolve(dates, exclude=(0,), hold_out=(1,))
    with pytest.raises(ValueError, match="at least 2"):
        DateSplit.resolve(dates, exclude=(0, 1, 2), hold_out=(3,))
    with pytest.raises(ValueError, match="out of range"):
        DateSplit.resolve(dates, exclude=(5,))
    with pytest.raises(ValueError, match="not one of"):
        DateSplit.resolve(dates, hold_out=("2018-07-01",))


def test_sequence_excludes_and_holds_out_dates():
    raw = _synthetic_raw(T=3)
    full = build_snowmelt_sequence(target_channels=("NDSI",), static_channels=(), pad=0, raw=raw)
    excluded = build_snowmelt_sequence(
        target_channels=("NDSI",), static_channels=(), pad=0, exclude_dates=(0,), raw=raw
    )
    training = build_snowmelt_sequence(
        target_channels=("NDSI",), static_channels=(), pad=0, hold_out_dates=(1,), raw=raw
    )
    evaluation = build_snowmelt_sequence(
        target_channels=("NDSI",), static_channels=(), pad=0, hold_out_dates=(1,), include_held_out=True, raw=raw
    )

    assert excluded.dates == ("2018-06-14", "2018-06-19")
    assert excluded.observation_times == (0.0, 5.0)
    np.testing.assert_array_equal(excluded.data, full.data[:, 1:])
    assert training.dates == ("2018-05-25", "2018-06-19") and training.held_out == (False, False)
    assert training.observation_times == (0.0, 25.0)
    np.testing.assert_array_equal(training.data, full.data[:, [0, 2]])
    assert evaluation.dates == full.dates and evaluation.held_out == (False, True, False)
    np.testing.assert_array_equal(evaluation.data, full.data)


def test_sequence_rejects_unknown_channels():
    with pytest.raises(ValueError):
        build_snowmelt_sequence(target_channels=("SNOW",), raw=_synthetic_raw())


def test_snowmelt_data_config_validation():
    SnowmeltDataConfig(target_channels=("SCA", "B11"))
    SnowmeltDataConfig(version="v1", static_channels=("DEM", "INCIDENCE"))
    SnowmeltDataConfig(version="v2", static_channels=("DEM", "SLOPE", "NORTHNESS", "EASTNESS"))
    with pytest.raises(ValueError):
        SnowmeltDataConfig(target_channels=())
    with pytest.raises(ValueError):
        SnowmeltDataConfig(version="v2", static_channels=("INCIDENCE",))
    with pytest.raises(ValueError):
        SnowmeltDataConfig(version="v1", static_channels=("SLOPE",))
    with pytest.raises(ValueError):
        SnowmeltDataConfig(version="v3")
    SnowmeltDataConfig(exclude_dates=(0, 1), hold_out_dates=(-1, "2018-06-14"))
    with pytest.raises(ValueError):
        SnowmeltDataConfig(hold_out_dates=("June",))
    with pytest.raises(ValueError):
        SnowmeltDataConfig(exclude_dates=(1.5,))


def test_schema_6_snowmelt_config_upgrades_to_dataset_v1():
    value = yaml.safe_load(Path("Experiments/snowmelt/conf/base_config.yaml").read_text())
    value["schema_version"] = 6
    del value["data"]["snowmelt"]["version"]
    del value["data"]["snowmelt"]["static_channels"]
    config = experiment_config_from_mapping(value)

    assert config.data.snowmelt.version == "v1"
    assert config.data.snowmelt.static_channels == ("DEM", "INCIDENCE")


def test_snowmelt_base_config_converts_and_round_trips():
    value = yaml.safe_load(Path("Experiments/snowmelt/conf/base_config.yaml").read_text())
    config = experiment_config_from_mapping(value)

    assert config.data.dataset == "snowmelt"
    assert config.data.snowmelt.target_channels == ("SCA",)
    assert config.data.snowmelt.version == "v2"
    assert config.data.snowmelt.static_channels == ("DEM",)
    assert config.data.snowmelt.exclude_dates == (0, 1)
    assert config.run.interval_mode == "steps"
    assert experiment_config_from_mapping(config_to_dict(config)) == config
    assert config.data.micropattern is None


def test_augmenter_writes_boundary_channels_and_masks_observables():
    boundary = jnp.stack([
        jnp.array([[1.0, 0.0], [1.0, 1.0]]),
        jnp.array([[0.5, 0.0], [0.2, 0.9]]),
    ])[None]  # [B=1, m=2, H, W]
    data = jnp.ones((1, 3, 2, 2, 2))  # [B, T, C_obs=2, H, W]
    augmenter = SnowmeltAugmenter(data, 3, boundary, noise_strength=0.0)
    saved = augmenter.return_saved_data()[0]  # [T, 5, H, W]

    assert jnp.array_equal(saved[:, -2:], jnp.broadcast_to(boundary[0], (3, 2, 2, 2)))
    assert jnp.all(saved[:, :2, 0, 1] == 0.0)  # outside the catchment
    x, y = augmenter.initialize_pool(jr.PRNGKey(0))
    assert jnp.array_equal(x[0][:, -2:], jnp.broadcast_to(boundary[0], (2, 2, 2, 2)))
    assert jnp.all(x[0][:, :2, 0, 1] == 0.0)
    assert jnp.array_equal(y[0], saved[1:])


def test_apply_boundary_channels_is_idempotent():
    boundary = [jnp.ones((1, 2, 2))]
    x = [jnp.full((2, 3, 2, 2), 0.3)]
    once = apply_boundary_channels(x, boundary, 1)
    assert jnp.array_equal(once[0], apply_boundary_channels(once, boundary, 1)[0])


def test_initial_pool_starts_each_slot_from_its_own_image():
    boundary = jnp.ones((1, 1, 2, 2))
    # Distinct constant image per time point: value t at time t.
    data = jnp.broadcast_to(jnp.arange(4.0)[None, :, None, None, None], (1, 4, 1, 2, 2))
    augmenter = SnowmeltAugmenter(data, 2, boundary, noise_strength=0.0)

    x, y = augmenter.initialize_pool(jr.PRNGKey(0))

    assert jnp.array_equal(x[0][:, 0, 0, 0], jnp.array([0.0, 1.0, 2.0]))
    assert jnp.array_equal(y[0][:, 0, 0, 0], jnp.array([1.0, 2.0, 3.0]))
    # advance_pool still hands slot k's prediction to slot k+1 (reinjection off).
    no_reinject = SnowmeltAugmenter(
        data, 2, boundary, reinjection_probability=0.0, noise_strength=0.0
    )
    predictions = [x[0].at[:, 0].add(10.0)]
    advanced, _ = no_reinject.advance_pool(predictions, y, 1, jr.PRNGKey(1))
    assert jnp.array_equal(advanced[0][:, 0, 0, 0], jnp.array([0.0, 10.0, 11.0]))

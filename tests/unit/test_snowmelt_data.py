from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
import yaml

from Common.dataloader.snowmelt import (
    BANDS,
    block_mean,
    build_snowmelt_sequence,
    days_since_first,
    fill_missing,
)
from Experiments.config import config_to_dict, experiment_config_from_mapping
from Experiments.snowmelt.config import SnowmeltDataConfig
from NCA.trainer.data_augmenter.snowmelt import (
    apply_boundary_channels,
    build_snowmelt_augmenter,
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
        target_channels=("SCA", "NDSI", "B3"), downsample=2, pad=1, raw=raw
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


def test_sequence_rejects_unknown_channels():
    with pytest.raises(ValueError):
        build_snowmelt_sequence(target_channels=("SNOW",), raw=_synthetic_raw())


def test_snowmelt_data_config_validation():
    SnowmeltDataConfig(target_channels=("SCA", "B11"))
    with pytest.raises(ValueError):
        SnowmeltDataConfig(target_channels=())
    with pytest.raises(ValueError):
        SnowmeltDataConfig(static_channels=("SLOPE",))


def test_snowmelt_base_config_converts_and_round_trips():
    value = yaml.safe_load(Path("Experiments/snowmelt/conf/base_config.yaml").read_text())
    config = experiment_config_from_mapping(value)

    assert config.data.dataset == "snowmelt"
    assert config.data.snowmelt.target_channels == ("SCA",)
    assert config.data.snowmelt.static_channels == ("DEM", "INCIDENCE")
    assert config.run.interval_mode == "steps"
    assert experiment_config_from_mapping(config_to_dict(config)) == config
    assert config.data.micropattern is None


def test_augmenter_writes_boundary_channels_and_masks_observables():
    boundary = jnp.stack([
        jnp.array([[1.0, 0.0], [1.0, 1.0]]),
        jnp.array([[0.5, 0.0], [0.2, 0.9]]),
    ])[None]  # [B=1, m=2, H, W]
    data = jnp.ones((1, 3, 2, 2, 2))  # [B, T, C_obs=2, H, W]
    augmenter = build_snowmelt_augmenter(boundary, noise_strength=0.0)(
        data_true=data, hidden_channels=3
    )
    augmenter.data_init()
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
    augmenter = build_snowmelt_augmenter(boundary, noise_strength=0.0)(
        data_true=data, hidden_channels=2
    )
    augmenter.data_init()

    x, y = augmenter.initialize_pool(jr.PRNGKey(0))

    assert jnp.array_equal(x[0][:, 0, 0, 0], jnp.array([0.0, 1.0, 2.0]))
    assert jnp.array_equal(y[0][:, 0, 0, 0], jnp.array([1.0, 2.0, 3.0]))
    # advance_pool still hands slot k's prediction to slot k+1 (reinjection off).
    no_reinject = build_snowmelt_augmenter(
        boundary, reinjection_probability=0.0, noise_strength=0.0
    )(data_true=data, hidden_channels=2)
    no_reinject.data_init()
    predictions = [x[0].at[:, 0].add(10.0)]
    advanced, _ = no_reinject.advance_pool(predictions, y, 1, jr.PRNGKey(1))
    assert jnp.array_equal(advanced[0][:, 0, 0, 0], jnp.array([0.0, 10.0, 11.0]))

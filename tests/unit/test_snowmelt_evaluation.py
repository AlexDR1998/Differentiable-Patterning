from types import SimpleNamespace

import jax.random as jr
import numpy as np
import pytest

from Common.dataloader.snowmelt import build_snowmelt_sequence
from Experiments.snowmelt import evaluation
from NCA.model.config import ModelConfig
from NCA.model.factory import build_model
from tests.unit.test_snowmelt_data import _synthetic_raw


def _cfg(t=4, interval_mode="steps"):
    return SimpleNamespace(
        run=SimpleNamespace(t=t, interval_mode=interval_mode, reference_interval=None, interval_steps=None),
        trainer=SimpleNamespace(boundary_mode="soft"),
    )


def test_initial_state_places_observed_hidden_and_boundary_channels():
    observed = np.full((2, 3, 3), 0.5, dtype=np.float32)
    boundary = np.stack([np.ones((3, 3)), np.full((3, 3), 0.25)]).astype(np.float32)

    state = evaluation.initial_state(observed, boundary, n_channels=6)

    assert state.shape == (6, 3, 3)
    np.testing.assert_array_equal(state[:2], observed)
    np.testing.assert_array_equal(state[2:4], 0.0)
    np.testing.assert_array_equal(state[4:], boundary)


def test_to_physical_maps_indexes_back_to_minus_one_one():
    values = np.stack([np.full((2, 2), 0.75), np.full((2, 2), 0.75)])[None]

    physical = evaluation.to_physical(values, ("NDSI", "B3"))

    np.testing.assert_allclose(physical[0, 0], 0.5)
    np.testing.assert_allclose(physical[0, 1], 0.75)


def test_full_resolution_reference_matches_upsampled_footprint():
    raw = _synthetic_raw(H=8, W=8)
    sequence = build_snowmelt_sequence(target_channels=("SCA",), downsample=3, pad=2, raw=raw)

    upsampled = evaluation.upsample_blocks(evaluation.strip_border(sequence.data[0], 2), 3)
    observed, catchment = evaluation.full_resolution_reference(raw, ("SCA",), 3)

    assert upsampled.shape[-2:] == observed.shape[-2:] == catchment.shape == (6, 6)
    assert observed.shape[:2] == sequence.data.shape[1:3]


def test_score_is_one_for_perfect_and_zero_for_persistence():
    rng = np.random.default_rng(0)
    observed = rng.uniform(0, 1, (3, 1, 4, 4)).astype(np.float32)
    catchment = np.ones((4, 4), dtype=bool)
    dates, days = ("a", "b", "c"), (0.0, 1.0, 2.0)

    perfect = evaluation.score(observed[None], observed, catchment, ("SCA",), dates, days)
    persistence = evaluation.score(
        np.broadcast_to(observed[:1], observed.shape)[None], observed, catchment, ("SCA",), dates, days
    )

    assert [row["date"] for row in perfect] == ["b", "c"]
    assert all(row["rmse"] == 0 and row["skill"] == 1 and row["snow_csi"] == 1 for row in perfect)
    assert all(row["skill"] == pytest.approx(0.0) for row in persistence)


def test_interval_score_uses_previous_image_as_persistence():
    observed = np.arange(3, dtype=np.float32)[:, None, None, None] * np.ones((3, 1, 2, 2), np.float32)
    previous = np.concatenate([observed[:1], observed[:-1]])[None]

    rows = evaluation.score(previous, observed, np.ones((2, 2), bool), ("B3",), "abc", (0, 1, 2), mode="interval")

    assert [row["persistence_rmse"] for row in rows] == [1.0, 1.0]
    assert all(row["skill"] == pytest.approx(0.0) for row in rows)


@pytest.mark.parametrize("mode", evaluation.ROLLOUT_MODES)
def test_predict_starts_from_the_observed_image(mode):
    raw = _synthetic_raw(T=3)
    sequence = build_snowmelt_sequence(target_channels=("NDSI",), downsample=2, pad=1, raw=raw)
    model, _ = build_model(
        ModelConfig(family="NCA", channels=6, kernel_str=("ID", "LAP"), fire_rate=1.0, padding="CIRCULAR"),
        key=jr.PRNGKey(0),
    )
    cfg = _cfg()

    prediction = evaluation.predict(model, cfg, sequence, jr.PRNGKey(1), n_rollouts=2, mode=mode)

    assert evaluation.rollout_schedule(cfg, sequence).steps == (6, 2)  # 20- and 5-day intervals, median 12.5 days
    assert prediction.shape == (2, 3, 1, *sequence.data.shape[-2:])
    np.testing.assert_array_equal(prediction[:, 0], np.broadcast_to(sequence.data[0, 0], prediction[:, 0].shape))
    assert np.all(np.isfinite(prediction))


def test_trajectory_keeps_selected_channels_at_each_stride():
    raw = _synthetic_raw(T=3)
    sequence = build_snowmelt_sequence(target_channels=("NDSI",), downsample=2, pad=1, raw=raw)
    model, _ = build_model(
        ModelConfig(family="NCA", channels=6, kernel_str=("ID", "LAP"), fire_rate=1.0, padding="CIRCULAR"),
        key=jr.PRNGKey(0),
    )

    frames, steps = evaluation.trajectory(model, _cfg(), sequence, jr.PRNGKey(1), stride=3, channels=(0, 2))

    assert steps == (0, 3, 6)  # 8 NCA steps in total
    assert frames.shape == (3, 2, *sequence.data.shape[-2:])
    np.testing.assert_array_equal(frames[0, 0], sequence.data[0, 0, 0])
    np.testing.assert_array_equal(frames[0, 1], 0.0)  # hidden channels start at zero


def test_load_bundle_sequence_rejects_data_of_another_dataset_version():
    snowmelt = SimpleNamespace(
        version="v1", target_channels=("SCA",), static_channels=("DEM",), pad=0, mask_threshold=0.5,
        exclude_dates=(), hold_out_dates=(),
    )
    bundle = SimpleNamespace(id="m1", config=SimpleNamespace(data=SimpleNamespace(snowmelt=snowmelt, downsample=1)))
    raw = _synthetic_raw()

    assert evaluation.load_bundle_sequence(bundle, {**raw, "version": "v1"}, verify=False).dates == tuple(raw["dates"])
    with pytest.raises(ValueError, match="trained on snowmelt dataset v1"):
        evaluation.load_bundle_sequence(bundle, {**raw, "version": "v2"}, verify=False)


def test_load_bundle_sequence_puts_held_out_dates_back_and_score_flags_them():
    snowmelt = SimpleNamespace(
        version="v1", target_channels=("NDSI",), static_channels=("DEM",), pad=0, mask_threshold=0.5,
        exclude_dates=(), hold_out_dates=(1,),
    )
    bundle = SimpleNamespace(id="m1", config=SimpleNamespace(data=SimpleNamespace(snowmelt=snowmelt, downsample=1)))
    raw = _synthetic_raw()

    sequence = evaluation.load_bundle_sequence(bundle, raw, verify=False)
    observed = sequence.data[0]
    rows = evaluation.score(
        observed[None], observed, raw["mask"], ("NDSI",), sequence.dates, sequence.observation_times,
        held_out=sequence.held_out,
    )

    assert sequence.dates == tuple(raw["dates"]) and sequence.held_out == (False, True, False)
    assert [row["held_out"] for row in rows] == [True, False]


def test_hold_out_needs_steps_mode_and_explicit_reference_interval():
    from Experiments.snowmelt.train import load_snowmelt_training_data

    cfg = SimpleNamespace(
        data=SimpleNamespace(batches=1, snowmelt=SimpleNamespace(hold_out_dates=(3,))),
        run=SimpleNamespace(interval_mode="steps", reference_interval=None),
    )
    with pytest.raises(ValueError, match="reference_interval"):
        load_snowmelt_training_data(cfg)

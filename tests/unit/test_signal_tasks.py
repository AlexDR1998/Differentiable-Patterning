"""Signalling tasks: generators, their typed config, the procedural augmenter and the evaluation."""

from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
import yaml

from Experiments.config import config_to_dict, experiment_config_from_mapping
from Experiments.config_helpers import data_channel_count
from Experiments.signalling.config import SignalTaskDataConfig
from Experiments.signalling.tasks import (
    BLUE, RED, TASKS, TISSUE, WHITE, geodesic_distance, sample_batch,
)
from NCA.trainer.data_augmenter.procedural import ProceduralTaskAugmenter

BASE_CONFIG = Path("Experiments/signalling/conf/base_config.yaml")


def _is(image, colour):
    """[H, W] cells of an RGBA image [4, H, W] with exactly ``colour``."""
    return np.all(np.isclose(image, np.asarray(colour)[:, None, None]), axis=0)


@pytest.mark.parametrize("task", sorted(TASKS))
def test_generators_shape_range_and_determinism(task):
    data, difficulty, probe = sample_batch(task, 3, 6, size=48)
    assert data.shape == (6, 3, 4, 48, 48)
    assert data.min() >= 0 and data.max() <= 1
    assert difficulty.shape == (6,) and np.all(difficulty > 0)
    assert probe.shape == (6, 48, 48) and np.all(probe.sum(axis=(1, 2)) > 0)
    # The tissue is there from the start and doesn't change
    assert np.array_equal(data[:, 0, 3], data[:, -1, 3])
    np.testing.assert_array_equal(data, sample_batch(task, 3, 6, size=48)[0])
    assert not np.array_equal(data, sample_batch(task, 4, 6, size=48)[0])


def test_connected_routing_leaves_the_other_comb_unchanged():
    data, _, _ = sample_batch("connected_routing", 0, 8)
    for initial, final in zip(data[:, 0], data[:, -1]):
        tissue = initial[3] > 0
        cue = tissue & ~_is(initial, TISSUE)
        unchanged = tissue & _is(final, TISSUE)
        assert unchanged.any()
        assert np.isinf(geodesic_distance(tissue, cue)[unchanged]).all()


def test_french_flag_bands_scale_with_length():
    data, length, _ = sample_batch("french_flag", 0, 8)
    for final, bar_length in zip(data[:, -1], length):
        tissue = final[3] > 0
        width = tissue.sum() / bar_length
        for colour in (RED, WHITE, BLUE):
            # Each band is a third of the bar, to the nearest column
            assert abs(_is(final, colour).sum() / width - bar_length / 3) <= 1


def test_timed_front_moves_along_the_bar():
    data, _, _ = sample_batch("remote_switch", 0, 4, front_step=6, frames=3, target_repeats=1)
    assert data.shape[1] == 4
    coloured = (~_is_batch(data, TISSUE) & (data[:, :, 3] > 0)).sum(axis=(2, 3))
    assert np.all(np.diff(coloured, axis=1) >= 0)
    assert np.all(coloured[:, -1] > coloured[:, 0])


def _is_batch(data, colour):
    return np.all(np.isclose(data, np.asarray(colour)[:, None, None]), axis=2)


def _augmenter(fraction):
    sample = lambda seed, count: sample_batch("remote_switch", seed, count, size=24)[0]
    return ProceduralTaskAugmenter(sample, 4, hidden_channels=2, resample_fraction=fraction)


def test_augmenter_replaces_only_the_given_fraction():
    augmenter = _augmenter(0.5)
    x, y = augmenter.initialize_pool(jr.PRNGKey(0))
    assert x[0].shape == (2, 6, 24, 24)
    saved = list(augmenter.return_saved_data())
    predictions = [state + 0.25 for state in x]
    x_next, y_next = augmenter.advance_pool(predictions, y, 1, jr.PRNGKey(1))
    replaced = [not jnp.array_equal(a, b) for a, b in zip(augmenter.return_saved_data(), saved)]
    assert sum(replaced) == 2
    for b in range(4):
        if replaced[b]:
            new = augmenter.return_saved_data()[b]
            assert jnp.array_equal(x_next[b], new[:-1]) and jnp.array_equal(y_next[b], new[1:])
        else:
            # Moved one slot along from the true initial state
            assert jnp.array_equal(x_next[b][0], saved[b][0])
            assert jnp.array_equal(x_next[b][1], predictions[b][0])
            assert jnp.array_equal(y_next[b], y[b])


def test_augmenter_without_resampling_keeps_the_instances():
    augmenter = _augmenter(0.0)
    x, y = augmenter.initialize_pool(jr.PRNGKey(0))
    saved = list(augmenter.return_saved_data())
    augmenter.advance_pool(x, y, 1, jr.PRNGKey(1))
    assert all(jnp.array_equal(a, b) for a, b in zip(augmenter.return_saved_data(), saved))


def test_base_config_converts_and_round_trips():
    config = experiment_config_from_mapping(yaml.safe_load(BASE_CONFIG.read_text()))
    assert config.data.signal_task.task == "remote_switch"
    assert config.data.emoji is None
    assert data_channel_count(config) == 4
    assert experiment_config_from_mapping(config_to_dict(config)) == config


def test_params_are_converted_to_tuples():
    value = yaml.safe_load(BASE_CONFIG.read_text())
    value["data"]["signal_task"]["params"] = {"length": [10, 30], "front_step": 4, "frames": 2}
    config = experiment_config_from_mapping(value)
    assert config.data.signal_task.params["length"] == (10, 30)
    assert experiment_config_from_mapping(config_to_dict(config)) == config


@pytest.mark.parametrize(
    "settings",
    [
        {"task": "telepathy"},
        {"task": "majority_vote", "params": {"length": (10, 20)}},
        {"resample_fraction": 1.5},
        {"evaluation_horizons": ()},
        {"target_repeats": 0},
    ],
)
def test_config_validation(settings):
    with pytest.raises(ValueError):
        SignalTaskDataConfig(**settings)


def test_unknown_signal_task_key_is_rejected():
    value = yaml.safe_load(BASE_CONFIG.read_text())
    value["data"]["signal_task"]["speed"] = 2
    with pytest.raises(ValueError, match="Unknown configuration fields"):
        experiment_config_from_mapping(value)


def test_evaluation_runs_on_a_tiny_model():
    from Experiments.signalling.evaluation import cell_accuracy, evaluate_task
    from NCA.model.NCA_model import NCA

    image = jnp.asarray(sample_batch("remote_switch", 0, 1, size=16)[0][0, -1])
    assert float(cell_accuracy(image, image)) == 1.0
    model = NCA(N_CHANNELS=6, KERNEL_STR=["ID", "LAP"], FIRE_RATE=0.5, key=jr.PRNGKey(0))
    scalars, images = evaluate_task(
        model, jr.PRNGKey(1), task="remote_switch", size=16, params={}, target_repeats=2,
        t=2, samples=2, repeats=2, horizons=(1, 2),
    )
    assert {"accuracy_frame_1", "accuracy_x1", "accuracy_x2", "arrival_spread", "quiet_step"} <= set(scalars)
    assert all(isinstance(value, float) for value in scalars.values())
    # A fixed rate of 0.5 for 2 * 2 * 2 steps
    assert scalars["updates_per_cell"] == pytest.approx(2.0)
    assert images["difficulty"].ndim == 4  # [1, H, W, RGBA], as the logger expects

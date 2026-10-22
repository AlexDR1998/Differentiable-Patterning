from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from Experiments.config import DataConfig
from Experiments.emoji import multi_attractor as ma
from Experiments.emoji.config import EmojiDataConfig, EmojiPairConfig
from NCA.model.config import ModelConfig
from NCA.model.factory import build_model


def _cfg(pairs, pad=(1, 1, 2, 2)):
    return SimpleNamespace(
        data=DataConfig(
            dataset="emojis",
            batches=1,
            downsample=1,
            emoji=EmojiDataConfig(
                task="multi_attractor",
                pairs=tuple(EmojiPairConfig(**pair) for pair in pairs),
                target_repeats=1,
                pad=pad,
            ),
        )
    )


def _fake_images(monkeypatch, values):
    monkeypatch.setattr(
        "Experiments.emoji.config_helpers.load_emoji_sequence",
        lambda sequence, **kwargs: jnp.full((1, 1, 4, 5, 5), values[sequence[0]]),
    )


class _Identity(eqx.Module):
    N_CHANNELS: int = eqx.field(static=True, default=6)

    def __call__(self, state, boundary_callback, key):
        return boundary_callback(state)


def test_seeds_and_targets_are_padded_and_shared_targets_merged(monkeypatch):
    _fake_images(monkeypatch, {"crab.png": 0.5, "microbe.png": 1.0})
    cfg = _cfg([
        {"initial": {"image": "crab.png", "mode": "patch", "size": 1}, "target": "crab.png"},
        {"initial": {"image": "microbe.png", "mode": "pixel", "channels": [3, 5]}, "target": "microbe.png"},
        {"initial": {"image": "microbe.png", "mode": "patch", "size": 1}, "target": "crab.png"},
    ])

    seeds, targets, pair_target, data = ma.load_seeds_and_targets(cfg, model_channels=8, impath="/tmp")

    assert ma.target_names(cfg) == ["crab", "microbe"]
    assert pair_target.tolist() == [0, 1, 0]
    assert data.shape == (3, 2, 4, 5, 5)
    # pad = (top, bottom, left, right) = (1, 1, 2, 2), so the centre moves to (3, 4)
    assert seeds.shape == (3, 8, 7, 9) and targets.shape == (2, 4, 7, 9)
    assert seeds[0, :4, 3, 4].tolist() == [0.5] * 4
    assert seeds[1, :, 3, 4].tolist() == [0, 0, 0, 1, 0, 1, 0, 0]
    assert np.all(targets[1, :, 1:-1, 2:-2] == 1.0) and targets[1].sum() == 4 * 25

    footprint = ma.seed_footprint(seeds)
    assert footprint.sum() == 1 and footprint[3, 4]
    assert ma.seed_colours(seeds, footprint)[2].tolist() == [1.0] * 4


def test_coloured_seeds_keep_the_hidden_channels_of_the_base_seed():
    base = np.zeros((6, 3, 3), np.float32)
    base[5, 1, 1] = 2.0
    footprint = np.zeros((3, 3), bool)
    footprint[1, 1] = True
    rgba = ma.hsva_to_rgba([[0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.5, 0.25]])

    states = ma.coloured_seeds(base, footprint, rgba)

    assert np.allclose(states[0, :, 1, 1], [1, 0, 0, 1, 0, 2])
    assert np.allclose(states[1, :, 1, 1], [0.5, 0.5, 0.5, 0.25, 0, 2])
    assert states[:, :, ~footprint].sum() == 0
    assert np.allclose(ma.rgba_to_hsva(rgba)[0], [0.0, 1.0, 1.0, 1.0])


def test_classify_labels_targets_other_died_and_diverged():
    alive = np.full((4, 2, 2), 1.0)
    dead = np.zeros((4, 2, 2))
    distances = np.array([[0.01, 0.5], [0.5, 0.02], [0.3, 0.4], [0.01, 0.5], [np.nan, np.nan]])
    final = np.stack([alive, alive, alive, dead, alive])

    labels = ma.classify(distances, final, threshold=0.1)

    assert labels.tolist() == [0, 1, ma.OTHER, ma.DIED, ma.DIVERGED]
    assert [ma.outcome_name(label, ["crab", "microbe"]) for label in labels] == [
        "crab", "microbe", "other", "died", "diverged"
    ]


def test_rollout_frames_and_distances_agree():
    model, _ = build_model(
        ModelConfig(family="NCA", channels=6, kernel_str=("ID", "LAP"), fire_rate=0.5,
                    padding="CIRCULAR", parameter_noise_level=0.0),
        key=jax.random.PRNGKey(0),
    )
    states = np.asarray(jax.random.uniform(jax.random.PRNGKey(1), (3, 6, 8, 8)))
    targets = np.asarray(jax.random.uniform(jax.random.PRNGKey(2), (2, 4, 8, 8)))
    key = jax.random.PRNGKey(3)

    # 12 features x 8 x 8 pixels: this budget fits 2 seeds per chunk, so the last chunk is padded
    assert ma.chunk_size(model, 8, 8, memory_gb=2.5e-5) == 2
    every_step, all_frames = ma.rollout(model, states, targets, 6, key, frames=6, memory_gb=2.5e-5)
    distances, rgba = ma.rollout(model, states, targets, 6, key, frames=3)

    assert all_frames.shape == (3, 7, 4, 8, 8) and rgba.shape == (3, 4, 4, 8, 8)
    assert distances.shape == (3, 7, 2)
    assert np.allclose(rgba[:, 0], states[:, :4])
    # Same keys, so the same trajectories whatever the frame spacing and chunking
    assert np.allclose(rgba, all_frames[:, ::2], atol=1e-5)
    assert np.allclose(distances, every_step, atol=1e-5)
    assert np.allclose(distances, ma.mse(all_frames[:, :, None], targets[None, None]), atol=1e-5)
    with pytest.raises(ValueError, match="multiple of frames"):
        ma.rollout(model, states, targets, 6, key, frames=4)


def test_evaluate_seeds_scores_each_pair():
    targets = np.zeros((2, 4, 4, 4), np.float32)
    targets[0, 3] = 1.0
    targets[1] = 1.0
    # Pair 0 already is target 0, pair 1 is target 0 but should grow target 1
    seeds = np.concatenate([targets[[0, 0]], np.zeros((2, 2, 4, 4), np.float32)], axis=1)

    rows, final, labels = ma.evaluate_seeds(_Identity(), seeds, targets, np.array([0, 1]), steps=3, horizon=2,
                                            repeats=2, key=jax.random.PRNGKey(0), tolerance=0.25)

    assert [row["correct"] for row in rows] == [1.0, 0.0]
    assert rows[1]["wrong_target"] == 1.0
    assert rows[0]["mse_final"] == 0.0 and rows[1]["mse_at_t"] == ma.target_gap(targets)
    assert final.shape == (2, 2, 4, 4, 4) and np.allclose(final[1, 0], targets[0])
    assert labels.tolist() == [[0, 0], [0, 0]]


def test_to_image_draws_rgba_over_white():
    rgba = np.zeros((4, 1, 2))
    rgba[:, 0, 1] = [1.0, 0.0, 0.0, 1.0]

    image = ma.to_image(rgba)

    assert image.shape == (1, 2, 3)
    assert image[0, 0].tolist() == [1.0, 1.0, 1.0]
    assert image[0, 1].tolist() == [1.0, 0.0, 0.0]


def test_seed_type_names_the_input_space():
    assert ma.seed_type("crab.png") == "full"
    assert ma.seed_type({"image": "crab.png", "mode": "patch", "size": 1}) == "patch"
    assert ma.seed_type({"image": "crab.png", "mode": "pixel", "channels": [3, 4]}) == "pixel"

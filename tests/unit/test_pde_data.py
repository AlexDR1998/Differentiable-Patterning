"""Simulated PDE training data, its typed config and its pool augmenter."""

from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import pytest
import yaml

from Experiments.config import config_to_dict, experiment_config_from_mapping
from Experiments.config_helpers import data_channel_count
from Experiments.pde.config import PdeDataConfig
from NCA.trainer.data_augmenter.pde import PdeAugmenter
from PDE.dataset import simulate_trajectories

BASE_CONFIG = Path("Experiments/pde/conf/base_config.yaml")


def _simulate(**overrides):
    settings = dict(batches=2, size=16, dt=0.5, t_end=50.0, frames=4, initial_condition="gray_scott_shapes")
    settings.update(overrides)
    return simulate_trajectories("gray_scott", jr.PRNGKey(0), **settings)


def test_trajectory_shape_and_times():
    trajectory = _simulate()
    assert trajectory.data.shape == (2, 4, 2, 16, 16)
    assert trajectory.observation_times == pytest.approx((0.0, 50 / 3, 100 / 3, 50.0))
    assert trajectory.channel_names == ("A", "B")


def test_observed_channels_and_normalisation():
    trajectory = _simulate(observed_channels=("B",))
    assert trajectory.data.shape[2] == 1
    assert trajectory.data.min() == pytest.approx(0.0)
    assert trajectory.data.max() == pytest.approx(1.0)
    per_channel = _simulate(normalisation="channel_minmax")
    for channel in range(2):
        assert per_channel.data[:, :, channel].max() == pytest.approx(1.0)


def test_unknown_channel_is_rejected():
    with pytest.raises(ValueError, match="channels"):
        _simulate(observed_channels=("C",))


def test_diverging_simulation_is_reported():
    with pytest.raises(FloatingPointError):
        simulate_trajectories(
            "heat", jr.PRNGKey(0), parameters={"D": 10.0}, size=8, dt=1.0, t_end=20.0, frames=2,
        )


@pytest.mark.parametrize(
    "settings",
    [
        {"model": "navier_stokes"},
        {"model": "heat", "parameters": {"alpha": 0.1}},
        {"model": "heat", "observed_channels": ("A",)},
        {"initial_condition": "stripes"},
        {"solver": "rk99"},
        {"frames": 1},
        {"reinjection_probability": 1.5},
    ],
)
def test_config_validation(settings):
    with pytest.raises(ValueError):
        PdeDataConfig(**settings)


def test_base_config_converts_and_round_trips():
    config = experiment_config_from_mapping(yaml.safe_load(BASE_CONFIG.read_text()))
    assert config.data.pde.model == "gray_scott"
    assert config.data.snowmelt is None
    assert data_channel_count(config) == 1
    assert experiment_config_from_mapping(config_to_dict(config)) == config


def test_parameters_are_typed_as_floats():
    value = yaml.safe_load(BASE_CONFIG.read_text())
    value["data"]["pde"]["parameters"] = {"alpha": 0.06, "gamma": 1}
    config = experiment_config_from_mapping(value)
    assert config.data.pde.parameters == {"alpha": 0.06, "gamma": 1.0}


def test_augmenter_pool_slots_and_reinjection():
    # Distinct constant frame per time point: value t at time t.
    data = jnp.broadcast_to(jnp.arange(4.0)[None, :, None, None, None], (2, 4, 1, 2, 2))
    augmenter = PdeAugmenter(data, 3, reinjection_probability=0.0, noise_strength=0.0)

    x, y = augmenter.initialize_pool(jr.PRNGKey(0))
    assert x[0].shape == (3, 4, 2, 2)
    assert jnp.array_equal(x[0][:, 0, 0, 0], jnp.array([0.0, 1.0, 2.0]))
    assert jnp.array_equal(y[0][:, 0, 0, 0], jnp.array([1.0, 2.0, 3.0]))

    # Without reinjection, slot k's prediction becomes slot k+1's start and slot 0 is reset.
    predictions = [state.at[:, 0].add(10.0) for state in x]
    advanced, _ = augmenter.advance_pool(predictions, y, 1, jr.PRNGKey(1))
    assert jnp.array_equal(advanced[0][:, 0, 0, 0], jnp.array([0.0, 10.0, 11.0]))

    # With reinjection always on, every slot starts from its true frame again.
    always = PdeAugmenter(data, 3, reinjection_probability=1.0, noise_strength=0.0)
    advanced, _ = always.advance_pool(predictions, y, 1, jr.PRNGKey(1))
    assert jnp.array_equal(advanced[0][:, 0, 0, 0], jnp.array([0.0, 1.0, 2.0]))

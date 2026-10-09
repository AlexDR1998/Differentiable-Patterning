import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from omegaconf import OmegaConf

from Common.trainer.training_result import TrainingResult
from Experiments.config import experiment_config_from_mapping
from Experiments.config_helpers import build_model, open_registry_bundle
from Experiments.impulse.config_helpers import (
    build_impulse_optimiser,
    build_intervention,
    build_objective,
    build_pair_source,
    resolve_output_directory,
)
from Experiments.model_registry import create_model_id, publish_model_bundle
from NCA.model.NCA_model import NCA
from NCA.trainer.impulse import StableAttractorPairSource, TargetedObjective


class ConfigDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _cfg(value):
    if isinstance(value, dict):
        return ConfigDict({key: _cfg(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_cfg(item) for item in value]
    return value


def _impulse_cfg():
    return _cfg(
        {
            "data": {
                "dataset": "emojis",
                "emoji": {"data_channels": 4, "observed_channels": 4},
            },
            "impulse": {
                "pair_source": {
                    "type": "stable_attractor",
                    "source_index": 0,
                    "target_index": 1,
                    "stabilisation_steps": [2, 3],
                    "target_steps": 2,
                    "initial_time": 0,
                    "target_time": -1,
                },
                "rollout": {"scan_kind": "lax"},
                "intervention": {
                    "channels": "hidden",
                    "spatial": "global",
                    "width": 1.0,
                },
                "objective": {
                    "type": "targeted",
                    "target_weight": 1.0,
                    "tolerance": 0.01,
                    "constraint_weight": 100.0,
                    "magnitude": "l2",
                    "reward_weight": 1.0,
                },
                "optimiser": {
                    "type": "adam",
                    "learn_rate": 0.001,
                    "gradient_clip_norm": 1.0,
                },
                "output": {
                    "directory": "results",
                    "base_env": "IMPULSE_OUTPUT_PATH",
                },
            },
        }
    )


def test_registry_model_loads_by_id_from_environment_store(tmp_path):
    value = OmegaConf.to_container(
        OmegaConf.load("Experiments/emoji/conf/base_config.yaml"), resolve=True
    )
    value["model"]["channels"] = 6
    train_cfg = experiment_config_from_mapping(value)
    original, _ = build_model(train_cfg.model, key=jax.random.PRNGKey(0))
    checkpoint_path = tmp_path / "trained.eqx"
    eqx.tree_serialise_leaves(checkpoint_path, original)
    published = publish_model_bundle(
        store_root=tmp_path / "store",
        collection="tests",
        model_id=create_model_id(train_cfg),
        display_name="model",
        checkpoint_path=checkpoint_path,
        cfg=train_cfg,
        training_result=TrainingResult(checkpoint_path, 1, 0.5, True),
        repository_root=tmp_path,
    )

    bundle = open_registry_bundle(
        published.id, env={"MODEL_STORE_ROOT": str(tmp_path / "store")}
    )
    loaded = bundle.load_model(key=jax.random.PRNGKey(1))

    assert bundle.path == published.path
    assert bundle.config.data == train_cfg.data
    original_leaves = jax.tree.leaves(eqx.filter(original, eqx.is_array))
    loaded_leaves = jax.tree.leaves(eqx.filter(loaded, eqx.is_array))
    assert all(jnp.array_equal(left, right) for left, right in zip(original_leaves, loaded_leaves))


def test_impulse_builders_construct_configured_components(tmp_path):
    cfg = _impulse_cfg()
    model = NCA(6, KERNEL_STR=["ID", "LAP"], FIRE_RATE=1.0, key=jax.random.PRNGKey(2))
    trajectories = jnp.zeros((2, 3, 6, 6, 6))

    pair_source = build_pair_source(cfg.impulse, model, trajectories)
    objective = build_objective(cfg.impulse.objective)
    intervention = build_intervention(
        cfg.impulse.intervention,
        cfg.data.emoji.observed_channels,
        model,
        trajectories,
        jax.random.PRNGKey(3),
    )
    optimiser = build_impulse_optimiser(cfg.impulse.optimiser)
    output = resolve_output_directory(
        cfg.impulse.output,
        env={"IMPULSE_OUTPUT_PATH": str(tmp_path)},
    )

    assert isinstance(pair_source, StableAttractorPairSource)
    assert isinstance(objective, TargetedObjective)
    assert intervention.values.shape == (1, 2, 6, 6)
    assert optimiser is not None
    assert output == (tmp_path / "results").resolve()


def test_pair_sources_only_use_the_configured_source_and_target_patterns():
    # Three patterns; pattern k has value 10k at time 0 and 10k + 1 afterwards
    trajectories = jnp.stack(
        [jnp.stack([jnp.full((2, 3, 3), 10.0 * k), jnp.full((2, 3, 3), 10.0 * k + 1)]) for k in range(3)]
    )
    model = NCA(2, KERNEL_STR=["ID"], FIRE_RATE=1.0, key=jax.random.PRNGKey(0))
    key = jax.random.PRNGKey(1)

    def sample(pair_type):
        cfg = _impulse_cfg()
        cfg.impulse.pair_source.update(type=pair_type, source_index=2, target_index=1)
        return build_pair_source(cfg.impulse, model, trajectories).sample(4, model, key)

    external = sample("external_target")
    stored = sample("trajectory_state")
    future = sample("model_future")

    assert jnp.all(external.initial_states == 20.0)
    assert jnp.all(external.target_states == 11.0)
    assert jnp.all(stored.initial_states == 20.0)
    assert jnp.all(stored.target_states == 21.0)
    assert jnp.all(future.initial_states == 20.0)

    cfg = _impulse_cfg()
    cfg.impulse.pair_source.update(target_index=3)
    with pytest.raises(ValueError, match="outside the 3 data pairs"):
        build_pair_source(cfg.impulse, model, trajectories)

import equinox as eqx
import jax
import jax.numpy as jnp

from Experiments.config_helpers import load_model_checkpoint, resolve_checkpoint_path
from Experiments.impulse.config_helpers import (
    build_impulse_optimizer,
    build_intervention,
    build_objective,
    build_pair_source,
    resolve_output_directory,
)
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


def _impulse_cfg(checkpoint_path="model.eqx"):
    return _cfg(
        {
            "checkpoint": {
                "path": checkpoint_path,
                "base_directory": None,
                "base_env": "MODEL_SAVE_PATH",
            },
            "data": {
                "dataset": "emojis",
                "emoji": {"data_channels": 4, "observed_channels": 4},
            },
            "model": {
                "family": "NCA",
                "channels": 6,
                "kernel_str": ["ID", "LAP"],
                "activation": "relu",
                "fire_rate": 1.0,
                "padding": "CIRCULAR",
                "kernel_scale": 1,
            },
            "impulse": {
                "pair_source": {
                    "type": "stable_attractor",
                    "source_index": 0,
                    "target_index": 1,
                    "stabilisation_steps": [2, 3],
                    "target_steps": 2,
                    "initial_index": 0,
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


def test_checkpoint_helpers_resolve_environment_root_and_load_model(tmp_path):
    key = jax.random.PRNGKey(0)
    original = NCA(6, KERNEL_STR=["ID", "LAP"], FIRE_RATE=1.0, key=key)
    checkpoint_path = tmp_path / "models" / "test_model.eqx"
    checkpoint_path.parent.mkdir()
    eqx.tree_serialise_leaves(checkpoint_path, original)
    cfg = _impulse_cfg("models/test_model")

    resolved = resolve_checkpoint_path(
        cfg.checkpoint, env={"MODEL_SAVE_PATH": str(tmp_path)}
    )
    loaded, _, loaded_path = load_model_checkpoint(
        cfg.model,
        cfg.checkpoint,
        key=jax.random.PRNGKey(1),
        env={"MODEL_SAVE_PATH": str(tmp_path)},
    )

    assert resolved == checkpoint_path.resolve()
    assert loaded_path == resolved
    original_leaves = eqx.filter(original, eqx.is_array)
    loaded_leaves = eqx.filter(loaded, eqx.is_array)
    assert all(
        jnp.array_equal(left, right)
        for left, right in zip(jax.tree.leaves(original_leaves), jax.tree.leaves(loaded_leaves))
    )


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
    optimiser = build_impulse_optimizer(cfg.impulse.optimiser)
    output = resolve_output_directory(
        cfg.impulse.output,
        env={"IMPULSE_OUTPUT_PATH": str(tmp_path)},
    )

    assert isinstance(pair_source, StableAttractorPairSource)
    assert isinstance(objective, TargetedObjective)
    assert intervention.values.shape == (1, 2, 6, 6)
    assert optimiser is not None
    assert output == (tmp_path / "results").resolve()

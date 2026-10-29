import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from omegaconf import OmegaConf

from Common.trainer.training_result import TrainingResult
from Experiments.config import experiment_config_from_mapping
from Experiments.config_helpers import build_model, open_registry_bundle, resolve_store_root
from Experiments.impulse.config_helpers import (
    build_impulse_optimiser,
    build_intervention,
    build_objective,
    build_pair_source,
    build_restarts,
    intervention_from_config,
)
from Experiments.model_registry import create_model_id, publish_model_bundle
from NCA.model.NCA_model import NCA
from NCA.trainer.impulse import AttractorContrastObjective, StableAttractorPairSource, TargetedObjective
from NCA.trainer.impulse.outcomes import target_gap


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
                    "patch_size": 3,
                    "duration": 1,
                    "init_scale": 0.0,
                    "initial_location": "random",
                    "restarts": 1,
                    "scout_fraction": 0.1,
                },
                "objective": {
                    "type": "targeted",
                    "target_weight": 1.0,
                    "tolerance": 0.01,
                    "constraint_weight": 100.0,
                    "magnitude": "l2",
                    "reward_weight": 1.0,
                    "contrast_weight": 0.1,
                    "temperature": None,
                },
                "optimiser": {
                    "type": "adam",
                    "learn_rate": 0.001,
                    "gradient_clip_norm": 1.0,
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
    from_shapes = intervention_from_config(
        cfg.impulse.intervention, cfg.data.emoji.observed_channels, (6, 6, 6), jax.random.PRNGKey(3)
    )

    assert isinstance(pair_source, StableAttractorPairSource)
    assert isinstance(objective, TargetedObjective)
    assert intervention.values.shape == (1, 2, 6, 6)
    assert jnp.array_equal(from_shapes.values, intervention.values)
    assert jnp.array_equal(from_shapes.location, intervention.location)
    assert optimiser is not None


def test_store_root_comes_from_the_argument_then_the_environment_then_models():
    assert resolve_store_root("given", env={"MODEL_STORE_ROOT": "env"}) == "given"
    assert resolve_store_root(None, env={"MODEL_STORE_ROOT": "env"}) == "env"
    assert resolve_store_root(None, env={}) == "models"


def test_intervention_accepts_observed_channels_and_rejects_unknown_modes():
    model = NCA(6, KERNEL_STR=["ID", "LAP"], FIRE_RATE=1.0, key=jax.random.PRNGKey(2))
    trajectories = jnp.zeros((2, 3, 6, 6, 6))

    def build(channels, spatial="global", patch_size=3, duration=1):
        return build_intervention(
            _cfg(
                {
                    "channels": channels,
                    "spatial": spatial,
                    "width": 1.0,
                    "patch_size": patch_size,
                    "duration": duration,
                    "init_scale": 0.0,
                    "initial_location": "random",
                }
            ),
            4,
            model,
            trajectories,
            jax.random.PRNGKey(3),
        )

    # "observed" is the loss channel name for the perturbation's "obs" mode
    assert build("observed").values.shape == (1, 4, 6, 6)
    with pytest.raises(ValueError, match="intervention.channels"):
        build("visible")
    with pytest.raises(ValueError, match="intervention.spatial"):
        build("hidden", spatial="box")
    assert build("hidden", spatial="patch").values.shape == (1, 2, 3, 3)
    with pytest.raises(ValueError, match="patch_size"):
        build("hidden", spatial="patch", patch_size=7)
    assert build("hidden", spatial="patch", duration=4).values.shape == (4, 2, 3, 3)
    with pytest.raises(ValueError, match="duration"):
        build("hidden", duration=0)


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


def test_init_scale_and_initial_location_set_the_starting_intervention():
    cfg = _impulse_cfg()
    model = NCA(6, KERNEL_STR=["ID", "LAP"], FIRE_RATE=1.0, key=jax.random.PRNGKey(2))
    trajectories = jnp.zeros((2, 3, 6, 6, 6))

    def build(**options):
        cfg.impulse.intervention.update(spatial="patch", **options)
        return build_intervention(cfg.impulse.intervention, 4, model, trajectories, jax.random.PRNGKey(3))

    assert jnp.all(build().values == 0.0)
    assert float(jnp.std(build(init_scale=0.1).values)) > 0.0
    assert jnp.allclose(build(initial_location="centre").get_location(), 0.5)
    with pytest.raises(ValueError, match="initial_location"):
        build(initial_location="corner")


def test_restarts_have_their_own_centres_and_a_scouting_budget():
    cfg = _impulse_cfg()
    cfg.impulse.intervention.update(spatial="patch")
    model = NCA(6, KERNEL_STR=["ID", "LAP"], FIRE_RATE=1.0, key=jax.random.PRNGKey(2))
    trajectories = jnp.zeros((2, 3, 6, 12, 12))
    key = jax.random.PRNGKey(3)

    assert build_restarts(cfg.impulse.intervention, 100, 4, model, trajectories, key) == ((), 0)
    cfg.impulse.intervention.update(restarts=3)
    others, scout = build_restarts(cfg.impulse.intervention, 100, 4, model, trajectories, key)
    first = build_intervention(cfg.impulse.intervention, 4, model, trajectories, key)

    assert len(others) == 2 and scout == 10
    centres = [tuple(item.get_location().tolist()) for item in (first, *others)]
    assert len(set(centres)) == 3
    cfg.impulse.intervention.update(initial_location="centre")
    with pytest.raises(ValueError, match="identical"):
        build_restarts(cfg.impulse.intervention, 100, 4, model, trajectories, key)


def test_attractor_contrast_objective_needs_attractors_and_defaults_its_temperature():
    cfg = _impulse_cfg()
    cfg.impulse.objective.update(type="attractor_contrast")
    attractors = jnp.stack([jnp.zeros((4, 3, 3)), jnp.ones((4, 3, 3)), jnp.full((4, 3, 3), 0.25)])

    with pytest.raises(ValueError, match="multi_attractor"):
        build_objective(cfg.impulse.objective)
    objective = build_objective(cfg.impulse.objective, attractors, 1)

    assert isinstance(objective, AttractorContrastObjective)
    assert objective.target_index == 1
    assert objective.temperature == target_gap(attractors) == 0.0625

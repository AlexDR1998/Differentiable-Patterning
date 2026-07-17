import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from NCA.model.NCA_perturbation import perturbation
from NCA.trainer.impulse import (
    ExternalTargetPairSource,
    MaximalPreservativeObjective,
    MinimalDestructiveObjective,
    ModelFuturePairSource,
    NCAImpulseOptimiser,
    StableAttractorPairSource,
    TargetedObjective,
    TrajectoryStatePairSource,
    run_nca_batch,
)
from NCA.trainer.impulse.regularisers import intervention_metrics, weighted_regulariser


class AdditiveModel(eqx.Module):
    """Small deterministic model used by impulse optimiser tests."""

    increment: jax.Array
    N_CHANNELS: int = eqx.field(static=True)

    def __init__(self, channels=2, increment=0.0):
        self.increment = jnp.asarray(increment)
        self.N_CHANNELS = channels

    def __call__(self, state, boundary_callback=lambda x: x, key=None):
        return boundary_callback(state + self.increment)


def per_sample_l2(x, y, key=None, where=None, cache=None):
    """Return simple per-sample squared error for optimiser tests."""

    return jnp.mean((x - y) ** 2, axis=(-3, -2, -1))


def test_run_nca_batch_returns_final_state_and_batch_first_trajectory():
    model = AdditiveModel(increment=0.25)
    initial = jnp.zeros((3, 2, 4, 4))

    final, trajectory = run_nca_batch(
        model,
        initial,
        steps=4,
        key=jax.random.PRNGKey(0),
        return_trajectory=True,
    )

    assert final.shape == (3, 2, 4, 4)
    assert trajectory.shape == (3, 4, 2, 4, 4)
    assert jnp.allclose(final, 1.0)
    assert jnp.allclose(trajectory[:, 1], 0.5)


def test_pair_sources_reproduce_legacy_pair_semantics():
    model = AdditiveModel(increment=0.5)
    key = jax.random.PRNGKey(1)
    initial = jnp.zeros((1, 2, 3, 3))

    future = ModelFuturePairSource(initial, target_steps=4).sample(2, model, key)
    external = ExternalTargetPairSource(initial, jnp.ones_like(initial)).sample(2, model, key)
    trajectories = jnp.stack([jnp.zeros((2, 3, 3)), jnp.ones((2, 3, 3))])[None]
    stored = TrajectoryStatePairSource(trajectories, 0, 1).sample(2, model, key)

    assert jnp.allclose(future.initial_states, 0.0)
    assert jnp.allclose(future.target_states, 2.0)
    assert jnp.allclose(external.target_states, 1.0)
    assert jnp.allclose(stored.initial_states, 0.0)
    assert jnp.allclose(stored.target_states, 1.0)


def test_stable_attractor_source_matches_switch_pool_semantics():
    model = AdditiveModel(increment=0.5)
    conditions = jnp.stack(
        [jnp.zeros((2, 3, 3)), jnp.ones((2, 3, 3))],
    )
    source = StableAttractorPairSource(
        conditions,
        source_index=0,
        target_index=1,
        stabilisation_steps=(2, 3),
    )

    batch = source.sample(3, model, jax.random.PRNGKey(2))

    assert batch.initial_states.shape == (3, 2, 3, 3)
    assert jnp.allclose(batch.initial_states, 1.0)
    assert jnp.allclose(batch.target_states, 2.0)


def test_model_future_source_keeps_freeze_target_and_rollout_horizons_separate():
    model = AdditiveModel(increment=0.5)
    initial = jnp.zeros((1, 2, 3, 3))
    source = ModelFuturePairSource(initial, target_steps=4)
    intervention = perturbation(
        mode={"channel": "all", "spatial": "global"},
        CHANNELS=2,
        OBS_CHANNELS=2,
        x=initial,
        WIDTH=1.0,
        key=jax.random.PRNGKey(8),
    )
    optimiser = NCAImpulseOptimiser(
        model=model,
        pair_source=source,
        intervention=intervention,
        objective=TargetedObjective(),
        optimiser=optax.sgd(0.1),
        observed_channels=2,
        rollout_steps=2,
        loss_functions=[per_sample_l2],
    )
    batch = source.sample(1, model, jax.random.PRNGKey(9))

    values = optimiser.evaluate(intervention, batch, jax.random.PRNGKey(10))

    # This is the legacy freeze setup: target F^4(x), prediction F^2(x + dx).
    assert jnp.allclose(batch.target_states, 2.0)
    assert jnp.allclose(values["final_states"], 1.0)
    assert jnp.allclose(values["target_loss"], 1.0)


def test_objectives_have_explicit_targeting_destruction_and_preservation_signs():
    metrics = {"l2": jnp.asarray(2.0)}

    assert TargetedObjective()(3.0, 0.5, metrics) == 3.5
    assert MinimalDestructiveObjective()(3.0, 0.5, metrics) == -2.5
    assert MaximalPreservativeObjective(
        tolerance=1.0,
        constraint_weight=10.0,
        reward_weight=0.5,
    )(3.0, 0.5, metrics) == 39.5


def test_regularisers_measure_applied_delta():
    initial = jnp.zeros((1, 2, 4, 4))
    perturbed = initial.at[:, 0].set(2.0)

    metrics = intervention_metrics(initial, perturbed)
    total = weighted_regulariser(metrics, {"l1": 2.0, "state_range": 1.0})

    assert jnp.allclose(metrics["l1"], 1.0)
    assert jnp.allclose(metrics["linf"], 2.0)
    assert jnp.allclose(metrics["state_range"], 0.5)
    assert jnp.allclose(total, 2.5)


def test_targeted_optimiser_reduces_error_and_keeps_model_frozen():
    model = AdditiveModel(channels=2, increment=0.0)
    initial = jnp.zeros((1, 2, 4, 4))
    target = jnp.ones_like(initial)
    source = ExternalTargetPairSource(initial, target)
    intervention = perturbation(
        mode={"channel": "all", "spatial": "global"},
        CHANNELS=2,
        OBS_CHANNELS=2,
        x=initial,
        WIDTH=1.0,
        key=jax.random.PRNGKey(3),
    )
    optimiser = NCAImpulseOptimiser(
        model=model,
        pair_source=source,
        intervention=intervention,
        objective=TargetedObjective(),
        optimiser=optax.adam(0.1),
        observed_channels=2,
        rollout_steps=1,
        loss_functions=[per_sample_l2],
        regulariser_coefficients={"l2": 0.01},
    )
    batch = source.sample(2, model, jax.random.PRNGKey(4))
    initial_loss = optimiser.evaluate(intervention, batch, jax.random.PRNGKey(5))["target_loss"]
    model_before = model.increment

    result = optimiser.train(
        iterations=20,
        batch_size=2,
        key=jax.random.PRNGKey(6),
        evaluation_steps=1,
    )

    assert result.best_loss < float(initial_loss)
    assert jnp.mean((result.final_states - result.target_states) ** 2) < initial_loss
    assert jnp.array_equal(model.increment, model_before)
    assert result.baseline_trajectory.shape == (2, 1, 2, 4, 4)
    assert result.perturbed_trajectory.shape == (2, 1, 2, 4, 4)


def test_intervention_channel_mode_only_changes_permitted_channels():
    initial = jnp.zeros((1, 4, 3, 3))
    intervention = perturbation(
        mode={"channel": "hidden", "spatial": "global"},
        CHANNELS=4,
        OBS_CHANNELS=2,
        x=initial,
        WIDTH=1.0,
        key=jax.random.PRNGKey(7),
    )
    intervention = eqx.tree_at(
        lambda item: item.values,
        intervention,
        jnp.ones_like(intervention.values),
    )

    changed = intervention(initial)

    assert jnp.allclose(changed[:, :2], 0.0)
    assert jnp.allclose(changed[:, 2:], 1.0)

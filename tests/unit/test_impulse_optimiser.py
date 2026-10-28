import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from NCA.trainer.impulse.perturbation import perturbation
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
    run_with_intervention,
)
from NCA.trainer.impulse.regularisers import intervention_metrics, weighted_regulariser


class AdditiveModel(eqx.Module):
    """Small deterministic model for the optimiser tests."""

    increment: jax.Array
    N_CHANNELS: int = eqx.field(static=True)

    def __init__(self, channels=2, increment=0.0):
        self.increment = jnp.asarray(increment)
        self.N_CHANNELS = channels

    def __call__(self, state, boundary_callback=lambda x: x, key=None):
        return boundary_callback(state + self.increment)


def per_sample_l2(x, y, key=None, where=None, cache=None):
    """Per-sample squared error."""

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


def test_strided_trajectory_matches_every_nth_state_of_the_full_trajectory():
    def noisy_model(state, boundary_callback=lambda x: x, key=None):
        return boundary_callback(state + jax.random.normal(key, state.shape))

    initial = jnp.zeros((2, 3, 4, 4))
    key = jax.random.PRNGKey(0)
    final, full = run_nca_batch(noisy_model, initial, 8, key, return_trajectory=True)
    strided_final, strided = run_nca_batch(
        noisy_model,
        initial,
        8,
        key,
        return_trajectory=True,
        trajectory_every=4,
        trajectory_channels=1,
    )

    # The same random keys, so the same states: only fewer are kept
    assert jnp.allclose(strided_final, final)
    assert strided.shape == (2, 2, 1, 4, 4)
    assert jnp.allclose(strided, full[:, 3::4, :1])
    with pytest.raises(ValueError, match="multiple of trajectory_every"):
        run_nca_batch(noisy_model, initial, 6, key, return_trajectory=True, trajectory_every=4)


def test_checkpointed_rollout_with_every_state_saved_matches_lax_gradient():
    def model(state, boundary_callback=lambda x: x, key=None):
        return boundary_callback(jnp.tanh(1.5 * state) + 0.1 * jax.random.normal(key, state.shape))

    initial = jnp.linspace(-1.0, 1.0, 2 * 3 * 4 * 4).reshape((2, 3, 4, 4))
    key = jax.random.PRNGKey(0)

    def loss(x, **options):
        return jnp.sum(run_nca_batch(model, x, 6, key, **options) ** 2)

    lax_gradient = jax.grad(loss)(initial)
    checkpointed_gradient = jax.grad(lambda x: loss(x, scan_kind="checkpointed", checkpoints=6))(initial)

    assert jnp.allclose(checkpointed_gradient, lax_gradient, atol=1e-5)


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


def test_stable_attractor_source_runs_each_item_for_its_own_step_count():
    model = AdditiveModel(increment=0.5)
    conditions = jnp.stack([jnp.zeros((2, 3, 3)), jnp.ones((2, 3, 3))])
    source = StableAttractorPairSource(
        conditions,
        source_index=1,
        target_index=0,
        stabilisation_steps=(1, 6),
    )

    batch = source.sample(16, model, jax.random.PRNGKey(3))

    steps = batch.metadata["stabilisation_steps"][:, None, None, None]
    assert jnp.all((steps >= 1) & (steps < 6))
    assert len(jnp.unique(steps)) > 1
    assert jnp.allclose(batch.initial_states, 1.0 + 0.5 * steps)
    assert jnp.allclose(batch.target_states, 0.5 * steps)


def test_local_perturbation_starts_anywhere_in_the_lattice_interior():
    centres = jnp.stack(
        [
            perturbation(
                mode={"channel": "all", "spatial": "local"},
                CHANNELS=2,
                OBS_CHANNELS=2,
                x=jnp.zeros((1, 2, 4, 4)),
                WIDTH=0.1,
                key=jax.random.PRNGKey(seed),
            ).get_location()
            for seed in range(200)
        ]
    )

    assert jnp.all((centres >= 0.1) & (centres <= 0.9))
    # Uniform over [0.1, 0.9] in each coordinate, so all four quadrants are used.
    quadrants = {(bool(x > 0.5), bool(y > 0.5)) for x, y in centres}
    assert len(quadrants) == 4
    assert jnp.allclose(jnp.mean(centres, axis=0), 0.5, atol=0.05)


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
    deltas = initial.at[:, 0].set(2.0)[None]

    metrics = intervention_metrics(initial, deltas)
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
    # Without an intervention the frozen model stays at zero, one away from the target.
    assert result.baseline_evaluation_loss == 1.0
    assert result.evaluation_loss < result.baseline_evaluation_loss
    # The source is all zeros: only the intervention moves the state nearer the target.
    assert result.evaluation_switch_rate == 1.0
    assert result.baseline_switch_rate == 0.0
    # A global intervention has no centre to track
    assert result.location_history is None
    # One evaluation frame: the rollout switched at its first (and only) step
    assert result.switch_steps.tolist() == [1.0, 1.0]
    assert result.evaluation_loss_curve.shape == (1,)
    assert result.evaluation_loss_curve[0] < result.baseline_loss_curve[0] == 1.0
    norms = result.metrics["gradient_norms"]
    assert float(norms["total"]) > 0.0 and float(norms["location"]) == 0.0


def test_switch_steps_time_the_last_switch_and_skip_failed_rollouts():
    from NCA.trainer.impulse.outcomes import switch_steps

    source, target = jnp.zeros((1, 1, 2, 2)), jnp.ones((1, 1, 2, 2))
    # Nearer the target from frame 1, back at the source in frame 2, then switched for good
    frames = jnp.array([0.0, 0.9, 0.1, 0.8, 1.0])[None, :, None, None, None] * jnp.ones((1, 5, 1, 2, 2))
    failed = jnp.zeros((1, 5, 1, 2, 2))

    steps = switch_steps(
        jnp.concatenate([frames, failed]),
        jnp.concatenate([source, source]),
        jnp.concatenate([target, target]),
        frame_every=8,
    )

    # Frame 3 is the fourth saved frame: 4 * 8 steps
    assert steps[0] == 32.0 and jnp.isnan(steps[1])


def test_validation_switch_rate_is_logged_at_each_log_step():
    class RecordingLogger:
        def __init__(self):
            self.validation = []

        def log_training(self, metrics, step, log_every):
            pass

        def log_validation(self, values, step):
            self.validation.append((step, values))

    model = AdditiveModel(channels=2, increment=0.0)
    initial = jnp.zeros((1, 2, 4, 4))
    source = ExternalTargetPairSource(initial, jnp.ones_like(initial))
    intervention = perturbation(
        mode={"channel": "all", "spatial": "global"},
        CHANNELS=2,
        OBS_CHANNELS=2,
        x=initial,
        WIDTH=1.0,
        key=jax.random.PRNGKey(3),
    )
    logger = RecordingLogger()
    optimiser = NCAImpulseOptimiser(
        model=model,
        pair_source=source,
        intervention=intervention,
        objective=TargetedObjective(),
        optimiser=optax.adam(0.1),
        observed_channels=2,
        rollout_steps=1,
        loss_functions=[per_sample_l2],
        logger=logger,
    )

    optimiser.train(iterations=20, batch_size=2, key=jax.random.PRNGKey(6), evaluation_steps=2, log_every=5)

    assert [step for step, _ in logger.validation] == [0, 5, 10, 15]
    # Before any update the intervention is zero: nothing switches
    first, last = logger.validation[0][1], logger.validation[-1][1]
    assert first["switch_rate"] == 0.0 and first["evaluation_loss"] == 1.0
    assert last["switch_rate"] == 1.0 and last["evaluation_loss"] < first["evaluation_loss"]
    assert "outcomes" not in first


def test_outcome_fractions_label_each_final_state():
    from NCA.trainer.impulse.outcomes import outcome_fractions

    attractors = jnp.stack([jnp.zeros((4, 3, 3)), jnp.ones((4, 3, 3))])
    final = jnp.stack([
        attractors[1],
        attractors[1] + 0.01,
        jnp.full((4, 3, 3), 0.5),  # halfway: other
        jnp.zeros((4, 3, 3)).at[3].set(0.5),  # alive, near attractor 0
    ])

    fractions = outcome_fractions(final, attractors, tolerance=0.25)

    # attractor 0, attractor 1, other, died, diverged
    assert fractions.tolist() == [0.25, 0.5, 0.25, 0.0, 0.0]


def test_validation_reports_outcomes_of_a_multi_attractor_model():
    class RecordingLogger:
        def __init__(self):
            self.validation = []

        def log_training(self, metrics, step, log_every):
            pass

        def log_validation(self, values, step):
            self.validation.append(values)

    model = AdditiveModel(channels=4, increment=0.0)
    initial = jnp.zeros((1, 4, 4, 4)).at[:, 3].set(1.0)
    target = jnp.ones_like(initial)
    attractors = jnp.stack([initial[0], target[0], jnp.full((4, 4, 4), 0.5).at[3].set(1.0)])
    logger = RecordingLogger()
    optimiser = NCAImpulseOptimiser(
        model=model,
        pair_source=ExternalTargetPairSource(initial, target),
        intervention=perturbation(
            mode={"channel": "all", "spatial": "global"},
            CHANNELS=4,
            OBS_CHANNELS=4,
            x=initial,
            WIDTH=1.0,
            key=jax.random.PRNGKey(3),
        ),
        objective=TargetedObjective(),
        optimiser=optax.adam(0.1),
        observed_channels=4,
        rollout_steps=1,
        loss_functions=[per_sample_l2],
        attractors=attractors,
        logger=logger,
    )

    optimiser.train(iterations=40, batch_size=2, key=jax.random.PRNGKey(6), evaluation_steps=1, log_every=39)

    # Attractors 0 (source), 1 (target), 2, then other, died, diverged. Before
    # any update every rollout stays at the source; later it reaches the target.
    assert logger.validation[0]["outcomes"].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert logger.validation[-1]["outcomes"].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_patch_optimiser_records_the_centre_at_every_step():
    model = AdditiveModel(channels=2, increment=0.0)
    initial = jnp.zeros((1, 2, 8, 8))
    source = ExternalTargetPairSource(initial, jnp.ones_like(initial))
    intervention = perturbation(
        mode={"channel": "all", "spatial": "patch"},
        CHANNELS=2,
        OBS_CHANNELS=2,
        x=initial,
        WIDTH=1.0,
        key=jax.random.PRNGKey(3),
        PATCH_SIZE=3,
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
    )

    result = optimiser.train(iterations=5, batch_size=1, key=jax.random.PRNGKey(6), evaluation_steps=1)

    assert result.location_history.shape == (6, 2)
    assert jnp.allclose(result.location_history[0], intervention.get_location())
    assert jnp.allclose(result.location_history[result.best_step], result.best_intervention.get_location())


def test_loss_window_averages_target_error_over_final_rollout_states():
    # States after each step are 0.25, 0.5, 0.75, 1.0; the target is 1.0.
    model = AdditiveModel(channels=2, increment=0.25)
    initial = jnp.zeros((1, 2, 4, 4))
    batch = ExternalTargetPairSource(initial, jnp.ones_like(initial)).sample(
        3, model, jax.random.PRNGKey(0)
    )
    intervention = perturbation(
        mode={"channel": "all", "spatial": "global"},
        CHANNELS=2,
        OBS_CHANNELS=2,
        x=initial,
        WIDTH=1.0,
        key=jax.random.PRNGKey(1),
    )

    def target_loss(loss_window):
        optimiser = NCAImpulseOptimiser(
            model=model,
            pair_source=None,
            intervention=intervention,
            objective=TargetedObjective(),
            optimiser=optax.adam(0.1),
            observed_channels=2,
            rollout_steps=4,
            loss_window=loss_window,
            loss_functions=[per_sample_l2],
        )
        values = optimiser.evaluate(intervention, batch, jax.random.PRNGKey(2))
        assert values["loss_per_sample"].shape == (3,)
        return float(values["target_loss"])

    assert target_loss(1) == 0.0
    assert jnp.isclose(target_loss(2), (0.25**2 + 0.0) / 2)
    assert jnp.isclose(target_loss(4), (0.75**2 + 0.5**2 + 0.25**2 + 0.0) / 4)


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


def _patch_impulse(size, centre, lattice=9, channel="hidden", duration=1):
    """A patch perturbation of ones on hidden channels 2 and 3, at ``centre``."""

    intervention = perturbation(
        mode={"channel": channel, "spatial": "patch"},
        CHANNELS=4,
        OBS_CHANNELS=2,
        x=jnp.zeros((1, 4, lattice, lattice)),
        WIDTH=1.0,
        key=jax.random.PRNGKey(0),
        PATCH_SIZE=size,
        duration=duration,
    )
    location = jax.scipy.special.logit(jnp.asarray(centre, dtype=float))
    intervention = eqx.tree_at(lambda item: item.location, intervention, location)
    return eqx.tree_at(lambda item: item.values, intervention, jnp.ones_like(intervention.values))


@pytest.mark.parametrize("size", [1, 3, 5])
def test_patch_covers_exactly_k_by_k_pixels_at_a_whole_pixel_centre(size):
    # Pixel i is at i / 8 on a 9 x 9 lattice, so (0.5, 0.5) is pixel (4, 4).
    intervention = _patch_impulse(size, (0.5, 0.5))

    changed = intervention(jnp.zeros((2, 4, 9, 9)))

    assert intervention.values.shape == (1, 2, size, size)
    assert jnp.allclose(changed[:, :2], 0.0)
    expected = jnp.zeros((9, 9)).at[4 - size // 2 : 5 + size // 2, 4 - size // 2 : 5 + size // 2].set(1.0)
    assert jnp.allclose(changed[:, 2:], expected, atol=1e-5)


def test_patch_between_pixels_covers_at_most_k_plus_one_squared_pixels():
    intervention = _patch_impulse(3, (0.53, 0.41))

    changed = intervention(jnp.zeros((1, 4, 9, 9)))

    footprint = jnp.abs(changed[0, 2]) > 1e-6
    assert int(jnp.sum(footprint)) <= 16
    # Bilinear placement keeps the total of the patch
    assert jnp.allclose(jnp.sum(changed[0, 2]), 9.0, atol=1e-4)


def test_patch_centre_has_a_gradient_and_can_be_moved():
    from Experiments.impulse.analysis import move_intervention

    intervention = _patch_impulse(3, (0.53, 0.41))
    weights = jnp.arange(81.0).reshape((9, 9))

    @eqx.filter_grad
    def gradient(candidate):
        return jnp.sum(candidate(jnp.zeros((1, 4, 9, 9)))[0, 2] * weights)

    assert jnp.all(jnp.abs(gradient(intervention).location) > 0)
    moved = move_intervention(intervention, (0.25, 0.75))(jnp.zeros((1, 4, 9, 9)))
    assert jnp.allclose(moved[0, 2, 1:4, 5:8], 1.0, atol=1e-5)
    assert jnp.allclose(jnp.sum(moved[0, 2]), 9.0, atol=1e-4)


def _step_values(intervention):
    """Patch values 1, 2, 3, ... for steps 0, 1, 2, ..."""

    steps = jnp.arange(1.0, intervention.duration + 1)[:, None, None, None]
    return eqx.tree_at(lambda item: item.values, intervention, steps * jnp.ones_like(intervention.values))


def test_multistep_patch_has_one_delta_per_step_at_the_same_place():
    intervention = _step_values(_patch_impulse(3, (0.5, 0.5), duration=3))

    deltas = intervention.deltas(jnp.zeros((2, 4, 9, 9)))

    assert intervention.values.shape == (3, 2, 3, 3)
    assert deltas.shape == (3, 1, 4, 9, 9)
    assert jnp.allclose(deltas[:, :, :2], 0.0)
    footprint = jnp.zeros((9, 9)).at[3:6, 3:6].set(1.0)
    for step in range(3):
        assert jnp.allclose(deltas[step, 0, 2:], (step + 1) * footprint, atol=1e-5)


def test_multistep_intervention_adds_each_delta_once_in_order():
    # The model doubles the state, so a delta added before update s is doubled
    # by every later update: the final state shows when each delta was added.
    class DoublingModel(eqx.Module):
        N_CHANNELS: int = eqx.field(static=True, default=4)

        def __call__(self, state, boundary_callback=lambda x: x, key=None):
            return boundary_callback(2.0 * state)

    intervention = _step_values(_patch_impulse(1, (0.5, 0.5), duration=3))
    states = jnp.zeros((2, 4, 9, 9))

    final = run_with_intervention(DoublingModel(), intervention, states, 4, jax.random.PRNGKey(0))

    # x0' = d0, then d1 after update 0 and d2 after update 1: 1*16 + 2*8 + 3*4
    expected = jnp.zeros((9, 9)).at[4, 4].set(16.0 + 16.0 + 12.0)
    assert jnp.allclose(final[:, 2:], expected, atol=1e-4)
    assert jnp.allclose(final[:, :2], 0.0)


def test_one_step_intervention_matches_the_single_shot_rollout():
    model = AdditiveModel(channels=4, increment=0.1)
    intervention = _step_values(_patch_impulse(3, (0.4, 0.6)))
    states = jnp.zeros((2, 4, 9, 9))
    key = jax.random.PRNGKey(1)

    final = run_with_intervention(model, intervention, states, 5, key)

    assert jnp.allclose(final, run_nca_batch(model, intervention(states), 5, key))
    single_shot = intervention_metrics(states, (intervention(states) - states)[:1][None])
    multistep = intervention_metrics(states, intervention.deltas(states))
    for name in single_shot:
        assert jnp.allclose(single_shot[name], multistep[name])


def test_l2_sums_impulse_energy_over_steps():
    states = jnp.zeros((1, 2, 4, 4))
    one_step = jnp.ones((1, 1, 2, 4, 4))

    single = intervention_metrics(states, one_step)["l2"]
    double = intervention_metrics(states, jnp.concatenate([one_step, one_step]))["l2"]

    assert jnp.allclose(double, jnp.sqrt(2.0) * single)


def test_intervention_must_end_before_the_loss_window():
    model = AdditiveModel(channels=4)
    initial = jnp.zeros((1, 4, 9, 9))
    arguments = dict(
        model=model,
        pair_source=ExternalTargetPairSource(initial, initial),
        objective=TargetedObjective(),
        optimiser=optax.sgd(0.1),
        observed_channels=2,
        rollout_steps=8,
        loss_window=4,
        loss_functions=[per_sample_l2],
    )

    NCAImpulseOptimiser(intervention=_patch_impulse(1, (0.5, 0.5), duration=5), **arguments)
    with pytest.raises(ValueError, match="loss window"):
        NCAImpulseOptimiser(intervention=_patch_impulse(1, (0.5, 0.5), duration=6), **arguments)

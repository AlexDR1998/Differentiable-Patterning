from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest

from Common.model.boundary import no_boundary
from Experiments.config import RunConfig
from NCA.model.NCA_model import NCA
from NCA.trainer.interval_schedule import (
    IntervalSchedule,
    build_interval_schedule,
    uniform_schedule,
)
from NCA.trainer.trainer import NcaTrainer


SNOWMELT_DAYS = (0.0, 20.0, 25.0, 45.0, 65.0)
MICROPATTERN_HOURS = (0.0, 12.0, 24.0, 36.0, 48.0)


# --- schedule resolution -----------------------------------------------------


def test_uniform_mode_ignores_times_and_matches_scalar_t():
    schedule = build_interval_schedule(32, 4, mode="uniform", times=SNOWMELT_DAYS)

    assert schedule.steps == (32, 32, 32, 32)
    assert schedule.is_uniform
    assert schedule.scan_length == 32
    assert schedule.observation_steps == (0, 32, 64, 96, 128)


def test_uniform_12h_grid_reproduces_hour_arithmetic():
    t = 64
    schedule = build_interval_schedule(t, 4, times=MICROPATTERN_HOURS)

    for hours in (0, 6, 12, 23, 24, 36, 47, 48, 60):
        slot = min(hours // 12, 4)
        assert schedule.slot_for_time(hours) == slot
        assert schedule.step_for_time(hours) == slot * t


def test_steps_mode_is_proportional_to_median_interval():
    schedule = build_interval_schedule(80, 4, mode="steps", times=SNOWMELT_DAYS)

    assert schedule.steps == (80, 20, 80, 80)
    assert schedule.scale == (1.0,) * 4
    assert not schedule.is_uniform
    assert schedule.scan_length == 80
    assert schedule.observation_steps == (0, 80, 100, 180, 260)
    assert schedule.slot_for_time(22.0) == 1
    assert schedule.step_for_time(30.0) == 100


def test_steps_mode_reference_interval_and_minimum_one_step():
    schedule = build_interval_schedule(
        4, 3, mode="steps", times=(0.0, 1.0, 11.0, 51.0), reference_interval=20.0
    )

    assert schedule.steps == (1, 2, 8)


def test_explicit_steps_override():
    schedule = build_interval_schedule(
        32, 4, mode="steps", times=SNOWMELT_DAYS, explicit_steps=[7, 3, 7, 9]
    )

    assert schedule.steps == (7, 3, 7, 9)


def test_duplicated_final_slot_gets_reference_steps():
    schedule = build_interval_schedule(40, 5, mode="steps", times=SNOWMELT_DAYS)

    assert schedule.steps == (40, 10, 40, 40, 40)


def test_fire_rate_and_dt_modes_resolve_scales():
    fire_rate = build_interval_schedule(16, 4, mode="fire_rate", times=SNOWMELT_DAYS)
    dt = build_interval_schedule(16, 4, mode="dt", times=SNOWMELT_DAYS)

    assert fire_rate.steps == dt.steps == (16,) * 4
    assert fire_rate.scale == (1.0, 0.25, 1.0, 1.0)
    assert dt.scale == (1.0, 0.25, 1.0, 1.0)
    assert not fire_rate.is_uniform


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "steps"},  # missing times
        {"mode": "steps", "times": (0.0, 5.0, 5.0)},  # not increasing
        {"mode": "steps", "times": (0.0, 1.0, 2.0, 3.0)},  # 3 intervals for 2 slots
        {"mode": "uniform", "explicit_steps": (1, 2)},
        {"mode": "fire_rate", "times": (0.0, 1.0, 2.0), "explicit_steps": (1, 2)},
        {"mode": "bogus"},
    ],
)
def test_invalid_schedules_raise(kwargs):
    with pytest.raises(ValueError):
        build_interval_schedule(8, 2, **kwargs)


def test_for_slot_is_single_uniform_rollout():
    schedule = build_interval_schedule(80, 4, mode="steps", times=SNOWMELT_DAYS)

    assert schedule.for_slot(1) == IntervalSchedule("steps", (20,), (1.0,))
    assert schedule.for_slot(1).is_uniform


def test_loop_config_validates_interval_fields():
    assert RunConfig().interval_mode == "uniform"
    RunConfig(interval_mode="steps", interval_steps=(3, 1))
    with pytest.raises(ValueError):
        RunConfig(interval_mode="sometimes")
    with pytest.raises(ValueError):
        RunConfig(interval_steps=(3, 1))
    with pytest.raises(ValueError):
        RunConfig(interval_mode="steps", reference_interval=0.0)


# --- masked rollout ------------------------------------------------------------


def _randomised_nca(key, channels=3):
    model = NCA(channels, KERNEL_STR=["ID", "LAP"], PADDING="CIRCULAR", FIRE_RATE=0.5, key=key)
    params, static = eqx.partition(model, eqx.is_inexact_array)
    leaves, treedef = jtu.tree_flatten(params)
    keys = jr.split(jr.fold_in(key, 1), len(leaves))
    leaves = [0.3 * jr.normal(k, leaf.shape, leaf.dtype) for k, leaf in zip(keys, leaves)]
    return eqx.combine(jtu.tree_unflatten(treedef, leaves), static)


def _rollout(model, states, schedule, key):
    trainer = SimpleNamespace(batch_count=len(states))
    vmapped = jax.vmap(model, in_axes=(0, None, 0))

    def vv_nca(x, callbacks, key_array):
        return jtu.tree_map(vmapped, x, callbacks, key_array)

    execution = SimpleNamespace(boundary_callbacks=lambda: [no_boundary()] * len(states))
    contexts = []

    def record_regularisers(totals, before, after, context, reg_key):
        contexts.append(context)
        return totals

    _, final, _ = NcaTrainer._run_nca_steps(
        trainer, model, vv_nca, states, {}, schedule, key, "lax", record_regularisers, execution
    )
    return final, contexts


@pytest.fixture
def rollout_inputs():
    key = jr.PRNGKey(3)
    model = _randomised_nca(key)
    states = [jr.uniform(jr.fold_in(key, b), (3, 3, 6, 6)) for b in range(2)]  # B=2, slots=3
    return model, states, jr.PRNGKey(11)


def test_masked_slots_match_unmasked_rollouts_of_their_own_length(rollout_inputs):
    model, states, key = rollout_inputs
    schedule = IntervalSchedule("steps", (3, 1, 2), (1.0,) * 3)

    masked, _ = _rollout(model, states, schedule, key)

    for slot, steps in enumerate(schedule.steps):
        reference, _ = _rollout(model, states, uniform_schedule(steps, 3), key)
        for batch in range(len(states)):
            assert jnp.array_equal(masked[batch][slot], reference[batch][slot])
    # Longer rollouts genuinely differ, so the equality above is not vacuous.
    longer, _ = _rollout(model, states, uniform_schedule(3, 3), key)
    assert not jnp.array_equal(masked[0][1], longer[0][1])


def test_int_and_uniform_schedule_give_identical_rollouts(rollout_inputs):
    model, states, key = rollout_inputs

    from_int, int_contexts = _rollout(model, states, 4, key)
    from_schedule, schedule_contexts = _rollout(model, states, uniform_schedule(4, 3), key)

    for a, b in zip(from_int, from_schedule):
        assert jnp.array_equal(a, b)
    assert "active_slots" not in int_contexts[0]
    assert "active_slots" not in schedule_contexts[0]


def test_non_uniform_rollout_exposes_active_slots(rollout_inputs):
    model, states, key = rollout_inputs
    schedule = IntervalSchedule("steps", (3, 1, 2), (1.0,) * 3)

    _, contexts = _rollout(model, states, schedule, key)

    assert "active_slots" in contexts[0]
    assert contexts[0]["active_slots"].shape == (3,)


def test_schedule_slot_count_must_match_rollout(rollout_inputs):
    model, states, key = rollout_inputs
    with pytest.raises(ValueError):
        _rollout(model, states, IntervalSchedule("steps", (3, 1), (1.0, 1.0)), key)


def test_masked_rollout_is_differentiable(rollout_inputs):
    model, states, key = rollout_inputs
    schedule = IntervalSchedule("steps", (3, 1, 2), (1.0,) * 3)
    params, static = eqx.partition(model, eqx.is_inexact_array)

    def loss(params):
        final, _ = _rollout(eqx.combine(params, static), states, schedule, key)
        return sum(jnp.mean(x ** 2) for x in final)

    grads = jax.grad(loss)(params)
    grad_leaves = jtu.tree_leaves(grads)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in grad_leaves)
    assert any(bool(jnp.any(g != 0)) for g in grad_leaves)


# --- stage 2: time arithmetic for interventions and logging --------------------

HOURS_6H = (0.0, 6.0, 12.0, 24.0, 36.0, 48.0)


def test_continuous_time_step_conversion_matches_uniform_hour_rule():
    t = 64
    schedule = build_interval_schedule(t, 4, times=MICROPATTERN_HOURS)

    for hours in (0.0, 5.0, 12.0, 30.0, 48.0):
        assert schedule.step_at_time(hours) == pytest.approx(hours * t / 12.0)
        assert schedule.time_at_step(hours * t / 12.0) == pytest.approx(hours)


def test_continuous_time_step_conversion_is_piecewise_linear():
    schedule = build_interval_schedule(80, 4, mode="steps", times=SNOWMELT_DAYS)

    assert schedule.step_at_time(22.5) == pytest.approx(90.0)  # halfway through the 5-day slot
    assert schedule.step_at_time(55.0) == pytest.approx(220.0)
    assert schedule.time_at_step(90.0) == pytest.approx(22.5)


def test_schedule_from_old_config_is_uniform():
    from NCA.trainer.interval_schedule import interval_schedule_from_config

    old_loop = SimpleNamespace(t=32)  # saved before interval fields existed
    config = SimpleNamespace(run=old_loop)

    assert interval_schedule_from_config(config, 4) == uniform_schedule(32, 4)


def test_intervention_slot_matches_12h_rule_and_follows_observation_times():
    from NCA.trainer.intervention import intervention_slot, nodal_read_block_mask

    for hours in (0, 11, 12, 24, 36, 47):
        assert int(intervention_slot(hours, MICROPATTERN_HOURS)) == hours // 12
    # 6h first frame: 12h is the start of slot 2, not 1.
    assert int(intervention_slot(12, HOURS_6H)) == 2
    assert int(intervention_slot(24, HOURS_6H)) == 3
    assert jnp.array_equal(
        nodal_read_block_mask(24, 5, observation_times=HOURS_6H),
        jnp.array([False, False, False, True, True]),
    )
    assert jnp.array_equal(
        nodal_read_block_mask(24, 4, observation_times=MICROPATTERN_HOURS),
        nodal_read_block_mask(24, 4),
    )


def test_legacy_nodal_zeroing_follows_observation_times():
    import Experiments.micropatterns.config_helpers as micropattern_helpers

    x = [jnp.ones((5, 9, 1, 1), dtype=jnp.float32)]
    x_true = [100.0 * jnp.ones((5, 9, 1, 1), dtype=jnp.float32)]
    mask = jnp.zeros((1, 4, 9), dtype=jnp.float32)
    args = (x, x_true, 9, jr.PRNGKey(0), mask, jnp.array([24], dtype=jnp.int32), 1.0)

    default = micropattern_helpers.masked_reinject_callback_bit(*args)[0]
    on_12h = micropattern_helpers.masked_reinject_callback_bit(
        *args, observation_times=MICROPATTERN_HOURS + (60.0,)
    )[0]
    on_6h = micropattern_helpers.masked_reinject_callback_bit(*args, observation_times=HOURS_6H)[0]

    assert jnp.array_equal(default, on_12h)
    assert jnp.all(on_6h[:3, 7] != 0.0) and jnp.all(on_6h[3:, 7] == 0.0)
    assert jnp.all(default[:2, 7] != 0.0) and jnp.all(default[2:, 7] == 0.0)


def test_logging_rollout_helpers_reproduce_uniform_indexing():
    from NCA.trainer.logging.tensorboard import (
        _frame_indices,
        _knockout_step,
        _rollout_schedule,
        _steps_at_image_index,
        _trajectory_snapshot_channels,
    )

    t, images = 16, 4
    schedule = _rollout_schedule(t, images)
    dense = jnp.arange((t * images + 1) * 2, dtype=jnp.float32).reshape(-1, 2, 1, 1)
    augmenter = type("Augmenter", (), {"OBS_CHANNELS": 2})()

    assert jnp.array_equal(
        _trajectory_snapshot_channels(dense, augmenter, _frame_indices(schedule, dense.shape[0])),
        _trajectory_snapshot_channels(dense, augmenter, t),
    )
    assert _frame_indices(schedule, t * images) == list(range(0, t * images, t))
    for hours in (-1, None, 0, 12, 24, 36, 60):
        expected = (
            t * images + 1 if hours is None or hours < 0 else min((hours // 12) * t, t * images)
        )
        assert _knockout_step(schedule, hours) == expected
    assert _steps_at_image_index(schedule, 2) == 2 * t
    assert _steps_at_image_index(schedule, 1.5) == pytest.approx(1.5 * t)


def test_logging_rollout_helpers_follow_non_uniform_schedule():
    from NCA.trainer.logging.tensorboard import (
        _frame_indices,
        _knockout_step,
        _rollout_schedule,
    )

    schedule = _rollout_schedule(
        build_interval_schedule(80, 4, mode="steps", times=SNOWMELT_DAYS), 4
    )

    assert schedule.total_steps == 260
    assert _frame_indices(schedule, 261) == [0, 80, 100, 180, 260]
    assert _knockout_step(schedule, 25) == 100  # knockout at the third image
    with pytest.raises(ValueError):
        _rollout_schedule(build_interval_schedule(80, 4, mode="steps", times=SNOWMELT_DAYS), 3)

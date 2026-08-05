from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from NCA.trainer.optimizer import build_learning_rate_schedule, build_optimizer


def _cfg(schedule_type="exponential", **schedule_overrides):
    schedule = {
        "type": schedule_type,
        "warmup_init_lr": 1e-6,
        "final_factor": 0.1,
        "transition_fraction": 0.75,
        **schedule_overrides,
    }
    return SimpleNamespace(
        run=SimpleNamespace(iterations=100),
        optimiser=SimpleNamespace(
            type="nadam",
            learn_rate=1e-3,
            warmup_steps=20,
            decay_rate=0.5,
            schedule=SimpleNamespace(**schedule),
            gradient_clip_norm=None,
            blocknorm=False,
            sam=False,
            apply_if_finite=False,
        ),
    )


def test_constant_schedule_preserves_peak_after_warmup():
    schedule, name = build_learning_rate_schedule(_cfg("constant").optimiser, 100)

    assert name == "const"
    assert jnp.isclose(schedule(0), 1e-6)
    assert jnp.isclose(schedule(20), 1e-3)
    assert jnp.isclose(schedule(100), 1e-3)


def test_cosine_schedule_reaches_configured_final_factor():
    schedule, name = build_learning_rate_schedule(
        _cfg("cosine", final_factor=0.1).optimiser, 100
    )

    assert name == "cos0.1"
    assert jnp.isclose(schedule(20), 1e-3)
    assert jnp.isclose(schedule(60), 5.5e-4)
    assert jnp.isclose(schedule(100), 1e-4)


def test_late_step_schedule_switches_at_post_warmup_fraction():
    schedule, name = build_learning_rate_schedule(
        _cfg("late_step", transition_fraction=0.75, final_factor=0.2).optimiser,
        100,
    )

    assert name == "step0.75x0.2"
    assert jnp.isclose(schedule(79), 1e-3)
    assert jnp.isclose(schedule(80), 2e-4)
    assert jnp.isclose(schedule(100), 2e-4)


def test_exponential_schedule_preserves_legacy_transition_length():
    schedule, name = build_learning_rate_schedule(_cfg("exponential").optimiser, 100)

    assert name == "exp0.5"
    assert jnp.isclose(schedule(20), 1e-3)
    assert jnp.isclose(schedule(120), 5e-4)


def test_optimizer_name_identifies_schedule():
    _, name = build_optimizer(_cfg("cosine", final_factor=0.1).optimiser, 100)

    assert name.startswith("nadam_schedcos0.1")


def test_optimizer_can_return_the_exact_schedule_used_for_updates():
    _, name, schedule = build_optimizer(
        _cfg("cosine", final_factor=0.1).optimiser,
        100,
        return_schedule=True,
    )

    assert name.startswith("nadam_schedcos0.1")
    assert jnp.isclose(schedule(60), 5.5e-4)


@pytest.mark.parametrize(
    ("schedule_type", "overrides", "message"),
    [
        ("cosine", {"final_factor": 1.5}, "final_factor"),
        ("late_step", {"transition_fraction": 1.0}, "transition_fraction"),
        ("unknown", {}, "Unsupported"),
    ],
)
def test_invalid_schedule_configuration_fails_clearly(
    schedule_type, overrides, message
):
    with pytest.raises(ValueError, match=message):
        build_learning_rate_schedule(_cfg(schedule_type, **overrides).optimiser, 100)

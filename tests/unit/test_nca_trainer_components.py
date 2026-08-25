from types import SimpleNamespace

import jax.numpy as jnp

from Common.trainer.config import LossConfig, PointwiseLossConfig
from NCA.trainer.objective import resolve_objective
from NCA.trainer.pool import PoolAdmissionController, TimePoolAdmissionController
from NCA.trainer.runner import _merge_advanced_states


def test_objective_is_resolved_from_typed_loss_config():
    config = LossConfig(
        terms=(
            PointwiseLossConfig(type="l2", weight=2.0),
            PointwiseLossConfig(type="spectral", weight=1.0),
        ),
        regularisers={"boundary": 0.25},
    )

    objective = resolve_objective(config)

    assert objective.names == ("l2", "spectral")
    assert objective.arguments["component_weights"] == [2.0, 1.0]
    assert "layers" not in objective.arguments
    assert objective.regulariser_coefficients == {"boundary": 0.25}


def test_pool_admission_state_is_explicit():
    config = SimpleNamespace(
        enabled=True,
        relative_threshold=1.25,
        previous_relative_threshold=1.1,
        absolute_threshold=None,
        ema_decay=0.5,
        warmup=0,
    )
    controller = PoolAdmissionController(config, default_warmup=0)

    first = controller.decide(1.0, iteration=0)
    controller.update(first, 1.0)
    second = controller.decide(2.0, iteration=1)

    assert first.admit
    assert not second.admit
    assert second.reject_relative
    assert second.reject_previous_relative


def test_pool_admission_reference_reset_preserves_counts():
    config = SimpleNamespace(
        enabled=True,
        relative_threshold=1.25,
        previous_relative_threshold=1.1,
        absolute_threshold=None,
        ema_decay=0.5,
        warmup=0,
    )
    controller = PoolAdmissionController(config, default_warmup=0)
    first = controller.decide(1.0, iteration=0)
    controller.update(first, 1.0)

    controller.reset_references()
    after_reset = controller.decide(100.0, iteration=1)

    assert after_reset.admit
    assert controller.state.admitted == 1
    assert controller.state.rejected == 0


def test_time_pool_admission_tracks_each_transition_independently():
    config = SimpleNamespace(
        enabled=True,
        relative_threshold=1.25,
        previous_relative_threshold=1.1,
        absolute_threshold=None,
        ema_decay=0.5,
        warmup=0,
    )
    controller = TimePoolAdmissionController(config, default_warmup=0)
    initial = controller.decide((1.0, 10.0), iteration=0)
    controller.update(initial, (1.0, 10.0))

    decisions = controller.decide((2.0, 10.5), iteration=1)

    assert not decisions[0].admit
    assert decisions[1].admit


def test_rejected_transition_restores_destination_from_pre_rollout_pool():
    previous = [jnp.arange(4, dtype=jnp.float32).reshape(4, 1, 1, 1)]
    advanced = [jnp.array([10.0, 20.0, 30.0, 40.0]).reshape(4, 1, 1, 1)]

    merged = _merge_advanced_states(
        advanced,
        previous,
        source_admitted=(True, False, True),
    )

    assert jnp.array_equal(
        merged[0][:, 0, 0, 0],
        jnp.array([10.0, 20.0, 2.0, 40.0]),
    )


def test_rejected_transition_merge_handles_sycl_tile_axis():
    previous = [jnp.zeros((2, 4, 1, 1, 1))]
    advanced = [jnp.ones((2, 4, 1, 1, 1))]

    merged = _merge_advanced_states(
        advanced,
        previous,
        source_admitted=(False, True, False),
    )

    assert jnp.array_equal(
        merged[0][:, :, 0, 0, 0],
        jnp.array([[1.0, 0.0, 1.0, 0.0]] * 2),
    )

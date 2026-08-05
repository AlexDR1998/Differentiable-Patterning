from types import SimpleNamespace

from Common.trainer.config import LossConfig, PointwiseLossConfig
from NCA.trainer.objective import resolve_objective
from NCA.trainer.pool import PoolAdmissionController


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

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("wandb")

import Common.trainer.wandb_logger as wandb_logger
from NCA.trainer.impulse.types import ImpulseResult
from NCA.trainer.logging.impulse_wandb_log import ImpulseLogger


@pytest.fixture
def logged(monkeypatch):
    """Replace the W&B module with one that records what is logged."""

    records = {}
    fake = SimpleNamespace(
        login=lambda **kwargs: None,
        init=lambda **kwargs: SimpleNamespace(config=kwargs),
        log=lambda values, step=None: records.update(values),
        Image=lambda image, file_type=None: np.asarray(image),
        Video=lambda video, fps=None, format=None: video,
        Histogram=lambda values: values,
        finish=lambda: None,
    )
    monkeypatch.setattr(wandb_logger, "wandb", fake)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    return records


def test_impulse_logger_logs_training_scalars_and_result(logged):
    batch, steps, channels, size = 2, 5, 6, 8
    logger = ImpulseLogger(
        np.zeros((batch, 3, channels, size, size)),
        observed_channels=4,
        wandb_config={"project": "test", "group": "test"},
    )
    logger.log_training(
        {
            "total_loss": 1.0,
            "target_loss": 0.5,
            "regulariser": 0.25,
            "intervention_metrics": {"l2": 0.1, "linf": 0.2},
        },
        step=0,
        log_every=1,
    )
    states = np.random.default_rng(0).uniform(size=(batch, channels, size, size))
    logger.log_result(
        ImpulseResult(
            best_intervention=None,
            best_step=3,
            best_loss=0.1,
            metrics={},
            initial_states=states,
            target_states=states,
            perturbed_initial_states=states + 0.1,
            final_states=states,
            baseline_trajectory=np.zeros((batch, steps, channels, size, size)),
            perturbed_trajectory=np.zeros((batch, steps, channels, size, size)),
            evaluation_loss=0.2,
            baseline_evaluation_loss=0.9,
        )
    )

    assert logged["Loss/target"] == 0.5
    assert logged["Intervention/linf"] == 0.2
    assert logged["Result/baseline_evaluation_loss"] == 0.9
    assert logged["Result/source | perturbed | final | target"].shape == (
        batch * size,
        4 * size,
        3,
    )
    assert logged["Result/perturbed trajectory"].shape == (steps, 3, size, size)

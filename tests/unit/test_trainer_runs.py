"""The full training loop runs on a tiny synthetic problem (CPU, a few seconds)."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
import yaml

from Experiments.config import experiment_config_from_mapping
from Experiments.emoji.config_helpers import build_data_augmenter
from NCA.model.factory import build_model
from NCA.trainer.context import TrainerContext
from NCA.trainer.trainer import build_trainer


def _tiny_emoji_config():
    value = yaml.safe_load(Path("Experiments/emoji/conf/base_config.yaml").read_text())
    value["data"]["batches"] = 1
    value["data"]["emoji"]["pad"] = [0, 0, 0, 0]
    value["data"]["emoji"]["shift_amount"] = 0
    value["model"]["channels"] = 8
    value["run"].update({"t": 2, "iterations": 4, "checkpoint_warmup": 0})
    value["optimiser"]["warmup_steps"] = 1
    value["logging"]["backend"] = "none"
    value["model_store"]["enabled"] = False
    return experiment_config_from_mapping(value)


@pytest.mark.parametrize("loop_autodiff", ["checkpointed", "lax"])
def test_training_loop_runs_and_saves_a_checkpoint(tmp_path, loop_autodiff):
    config = _tiny_emoji_config()
    config = replace(config, trainer=replace(config.trainer, loop_autodiff=loop_autodiff))
    data = jax.random.uniform(jax.random.PRNGKey(0), (1, 3, 4, 8, 8))
    augmenter, _ = build_data_augmenter(config.data)
    model, _ = build_model(config.model, key=jax.random.PRNGKey(1))
    context = TrainerContext(
        run_name="tiny",
        model_directory=str(tmp_path),
        data_augmenter=augmenter,
        storage_id="tiny-run",
        observed_channels=4,
        data_channels=4,
    )

    result = build_trainer(config, model, data, context).train(key=jax.random.PRNGKey(2))

    assert result.error_code == 0
    assert result.completed
    assert result.checkpoint_path is not None and result.checkpoint_path.is_file()
    assert jnp.isfinite(result.best_loss)

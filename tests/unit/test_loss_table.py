from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import pytest
import yaml

from Common.trainer.config import LOSS_TERM_CONFIGS
from Common.trainer.loss_table import LOSSES, build_loss_functions, build_loss_initialiser
from Experiments.config import experiment_config_from_mapping
from NCA.trainer.objective import resolve_objective


def _emoji_config(terms, grad_loss=False):
    value = yaml.safe_load(Path("Experiments/emoji/conf/base_config.yaml").read_text())
    value["loss"]["terms"] = terms
    value["trainer"]["grad_loss"] = grad_loss
    return value


def test_every_config_loss_has_a_function():
    assert set(LOSSES) == set(LOSS_TERM_CONFIGS) - {"multi_target"}


@pytest.mark.parametrize("name", ["l2", "l1", "spectral", "radial_profile", "channel_correlation"])
def test_plain_losses_return_one_value_per_sample(name):
    x = jr.uniform(jr.PRNGKey(0), (3, 4, 8, 8))
    loss = build_loss_functions([name], {})[0]

    assert loss(x, x + 0.1, jr.PRNGKey(1), None, None).shape == (3,)


def test_only_vgg_losses_precompute_target_features():
    assert build_loss_initialiser(["l2", "sliced_wasserstein_channel"], {}) is None
    assert build_loss_initialiser(["l2", "vgg"], {}) is not None


def test_unknown_loss_is_rejected():
    with pytest.raises(ValueError, match="Unknown losses"):
        build_loss_functions(["emd_loss"], {})
    with pytest.raises(ValueError, match="Unsupported loss type"):
        experiment_config_from_mapping(_emoji_config([{"type": "emd_loss"}]))


@pytest.mark.parametrize("loss_type", ["vgg", "vgg_grouped", "multi_target"])
def test_vgg_losses_cannot_use_grad_loss(loss_type):
    with pytest.raises(ValueError, match="grad_loss"):
        experiment_config_from_mapping(_emoji_config([{"type": loss_type}], grad_loss=True))


def test_grad_loss_is_allowed_with_other_losses():
    config = experiment_config_from_mapping(_emoji_config([{"type": "l2"}], grad_loss=True))

    assert config.trainer.grad_loss


@pytest.mark.parametrize(
    ("loss_type", "option"),
    [("vgg", "metric"), ("multi_target", "metric"), ("ott", "internal_loss_func")],
)
def test_metric_option_reaches_the_loss_that_uses_it(loss_type, option):
    config = experiment_config_from_mapping(
        _emoji_config([{"type": loss_type, "metric": "otch" if option == "metric" else "l1"}])
    )

    arguments = resolve_objective(config.loss).arguments

    assert arguments[option] == ("otch" if option == "metric" else "l1")

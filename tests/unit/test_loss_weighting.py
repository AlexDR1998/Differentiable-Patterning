import jax.numpy as jnp
import pytest
from types import SimpleNamespace

from Common.trainer.loss import build_loss_functions, l2_colony_grouped
from Common.trainer.loss_vgg import grouped_vgg_triplet_weights
from Experiments.config_helpers import build_loss_args, build_loss_filename
from Experiments.config_workflow import generate_manifest
from NCA.trainer.objective import (
    combine_loss_components,
    resolve_loss_component_weights,
)


BASE_GROUPED_L2_WEIGHTS = jnp.array(
    [0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]
)
BASE_VGG_WEIGHTS = jnp.array([0.5, 1.0, 0.5, 1.0, 1.0, 1.0])


def test_grouped_l2_uniform_importance_preserves_legacy_value():
    x = jnp.zeros((2, 9, 2, 2))
    y = jnp.arange(2 * 12 * 2 * 2, dtype=jnp.float32).reshape(2, 12, 2, 2)
    where = jnp.ones((2, 12, 1, 1), dtype=bool)

    legacy = jnp.mean((y**2) * BASE_GROUPED_L2_WEIGHTS[None, :, None, None], axis=(1, 2, 3))
    weighted = l2_colony_grouped(
        x,
        y,
        None,
        where,
        aux={"channel_importance": [1.0] * 12},
    )

    assert jnp.allclose(weighted, legacy)


def test_grouped_l2_sox2_importance_increases_sox2_only_error():
    x = jnp.zeros((1, 9, 1, 1))
    y = jnp.zeros((1, 12, 1, 1)).at[:, 3].set(1.0)
    where = jnp.ones((1, 12, 1, 1), dtype=bool)
    uniform = l2_colony_grouped(
        x, y, None, where, aux={"channel_importance": [1.0] * 12}
    )
    sox2_importance = [1.0] * 12
    sox2_importance[3] = 4.0
    emphasized = l2_colony_grouped(
        x, y, None, where, aux={"channel_importance": sox2_importance}
    )

    assert emphasized[0] > uniform[0]
    assert l2_colony_grouped(
        x, jnp.zeros_like(y), None, where, aux={"channel_importance": sox2_importance}
    )[0] == 0.0


def test_sox2_importance_maps_to_isolated_second_vgg_triplet():
    where = jnp.ones((1, 12, 1, 1), dtype=bool)
    uniform = grouped_vgg_triplet_weights([1.0] * 12, where=where)
    sox2_importance = [1.0] * 12
    sox2_importance[3] = 4.0
    emphasized = grouped_vgg_triplet_weights(sox2_importance, where=where)

    assert jnp.allclose(uniform[:, 0], BASE_VGG_WEIGHTS)
    assert emphasized[1, 0] > uniform[1, 0]
    assert jnp.all(emphasized[jnp.array([0, 2, 3, 4, 5]), 0] < uniform[jnp.array([0, 2, 3, 4, 5]), 0])
    assert jnp.isclose(jnp.sum(emphasized[:, 0]), jnp.sum(BASE_VGG_WEIGHTS))


def test_component_weights_form_normalized_weighted_mean():
    losses = [jnp.array([2.0, 4.0]), jnp.array([8.0, 10.0])]
    weights = resolve_loss_component_weights([1.0, 2.0], 2)

    combined = combine_loss_components(losses, weights)

    assert jnp.allclose(combined, jnp.array([6.0, 8.0]))
    assert jnp.allclose(
        combine_loss_components(losses, resolve_loss_component_weights(None, 2)),
        jnp.array([5.0, 7.0]),
    )


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ([1.0], "one value per configured loss"),
        ([1.0, -1.0], "negative"),
        ([0.0, 0.0], "at least one positive"),
    ],
)
def test_invalid_component_weights_fail_clearly(weights, message):
    with pytest.raises(ValueError, match=message):
        resolve_loss_component_weights(weights, 2)


def test_channel_importance_validation_rejects_bad_grouped_configuration():
    args = {
        "channel_importance": [1.0] * 11,
        "random_crop": False,
        "random_channel_shuffle": False,
    }
    with pytest.raises(ValueError, match="12 target-channel weights"):
        build_loss_functions("l2_grouped", args)

    args["channel_importance"] = [1.0] * 12
    with pytest.raises(ValueError, match="only supported for grouped"):
        build_loss_functions("l2", args)


def test_loss_config_plumbing_and_filename_include_non_default_weights():
    cfg = SimpleNamespace(
        loss=SimpleNamespace(
            terms=[
                SimpleNamespace(
                    type="vgg_grouped", weight=1.0,
                    metric="l2", random_crop=False,
                    random_channel_shuffle=False,
                    channel_importance=[1.0, 1.0, 1.0, 4.0] + [1.0] * 8,
                ),
                SimpleNamespace(type="l2_grouped", weight=2.0),
            ],
            regularisers={},
        )
    )

    args = build_loss_args(cfg.loss)
    filename = build_loss_filename(cfg.loss)

    assert args["channel_importance"][3] == 4.0
    assert args["component_weights"] == [1.0, 2.0]
    assert "ci4x4.0" in filename
    assert "cw1.0-2.0" in filename


def test_manifest_treats_weight_vectors_as_individual_sweep_values(tmp_path):
    uniform = [1.0] * 12
    sox2 = [1.0, 1.0, 1.0, 4.0] + [1.0] * 8
    manifest = generate_manifest(
        {"loss": {"terms": [{"type": "l2_grouped", "channel_importance": None}]}},
        {
            "experiment_name": "weight-test",
            "grid": {"loss.terms.0.channel_importance": [uniform, sox2]},
        },
        tmp_path,
    )

    assert manifest["count"] == 2
    assert manifest["configs"][0]["config"]["experiment"]["name"] == "weight-test"
    assert manifest["configs"][0]["config"]["logging"]["wandb"]["group"] == "weight-test"
    assert manifest["configs"][0]["config"]["loss"]["terms"][0]["channel_importance"] == uniform
    assert manifest["configs"][1]["config"]["loss"]["terms"][0]["channel_importance"] == sox2


def test_manifest_preserves_explicit_wandb_group(tmp_path):
    manifest = generate_manifest(
        {},
        {
            "experiment_name": "job-name",
            "grid": {"logging.wandb.group": ["shared-analysis-group"]},
        },
        tmp_path,
    )

    config = manifest["configs"][0]["config"]
    assert config["experiment"]["name"] == "job-name"
    assert config["logging"]["wandb"]["group"] == "shared-analysis-group"


def test_manifest_preserves_baseline_values_not_set_by_selected_branch(tmp_path):
    manifest = generate_manifest(
        {"model": {"channels": 8, "fire_rate": 0.5}},
        {
            "grid": {"model.family": ["NCA"]},
            "branches": [
                {
                    "when": {"model.family": "NCA"},
                    "grid": {"model.channels": [12]},
                },
            ],
        },
        tmp_path,
    )

    model = manifest["configs"][0]["config"]["model"]
    assert model == {"channels": 12, "fire_rate": 0.5, "family": "NCA"}

from pathlib import Path

import jax.numpy as jnp
import pytest
import yaml

from Common.trainer.config import (
    LossConfig,
    LossWeightScheduleConfig,
    MultiTargetLossConfig,
    PointwiseLossConfig,
)
from Experiments.config import config_to_dict, experiment_config_from_mapping
from Experiments.config_helpers import build_loss_filename
from NCA.trainer.loss_schedule import (
    build_loss_weight_schedule,
    final_transition_iteration,
    schedule_factor,
)


def test_linear_loss_schedule_holds_interpolates_and_holds():
    schedule = LossWeightScheduleConfig(
        type="linear",
        initial_factor=0.05,
        final_factor=1.0,
        start_fraction=0.3,
        end_fraction=0.7,
    )

    assert jnp.isclose(schedule_factor(schedule, 0, 101), 0.05)
    assert jnp.isclose(schedule_factor(schedule, 30, 101), 0.05)
    assert jnp.isclose(schedule_factor(schedule, 50, 101), 0.525)
    assert jnp.isclose(schedule_factor(schedule, 70, 101), 1.0)
    assert jnp.isclose(schedule_factor(schedule, 100, 101), 1.0)


def test_cosine_loss_schedule_has_smooth_midpoint():
    schedule = LossWeightScheduleConfig(
        type="cosine",
        initial_factor=1.0,
        final_factor=0.25,
        start_fraction=0.3,
        end_fraction=0.7,
    )

    assert jnp.isclose(schedule_factor(schedule, 50, 101), 0.625)


def test_staged_cosine_schedule_has_eight_constant_levels():
    schedule = LossWeightScheduleConfig(
        type="cosine",
        initial_factor=0.0,
        final_factor=1.0,
        start_fraction=0.0,
        end_fraction=1.0,
        stages=8,
    )

    values = [
        float(schedule_factor(schedule, iteration, 801))
        for iteration in range(801)
    ]
    levels = sorted(set(values))

    assert len(levels) == 8
    assert jnp.isclose(levels[0], 0.0)
    assert jnp.isclose(levels[-1], 1.0)
    assert values[99] == values[0]
    assert values[100] != values[99]


def test_builder_schedules_terms_and_multi_target_components():
    config = LossConfig(
        terms=(
            MultiTargetLossConfig(
                type="multi_target",
                weight=2.0,
                schedule=LossWeightScheduleConfig(
                    type="linear", initial_factor=0.5, final_factor=1.0
                ),
                multi_target_weights={"texture": 2.0, "radial": 1.0},
                multi_target_schedules={
                    "texture": LossWeightScheduleConfig(
                        type="linear", initial_factor=0.0, final_factor=1.0
                    )
                },
            ),
        )
    )
    weights = build_loss_weight_schedule(config, 101)(50)

    assert jnp.allclose(weights.terms, jnp.array([1.5]))
    assert jnp.isclose(weights.multi_target["texture"], 1.0)
    assert jnp.isclose(weights.multi_target["radial"], 1.0)
    assert tuple(weights.multi_target) == (
        "l2",
        "texture",
        "channel_mean",
        "radial",
        "correlation",
    )


def test_multi_target_normalization_preserves_configured_weight_sum():
    config = LossConfig(
        terms=(
            MultiTargetLossConfig(
                type="multi_target",
                multi_target_weights={
                    "l2": 1.0,
                    "texture": 1.0,
                    "channel_mean": 0.0,
                    "radial": 0.0,
                    "correlation": 0.0,
                },
                multi_target_schedules={
                    "l2": LossWeightScheduleConfig(
                        type="cosine", initial_factor=1.0, final_factor=0.25
                    ),
                    "texture": LossWeightScheduleConfig(
                        type="cosine", initial_factor=0.05, final_factor=1.0
                    ),
                },
                normalize_weights=True,
            ),
        )
    )
    schedule = build_loss_weight_schedule(config, 101)

    for iteration in (0, 50, 100):
        weights = schedule(iteration).multi_target
        assert jnp.isclose(sum(weights.values()), 2.0)
    assert jnp.isclose(
        schedule(0).multi_target["l2"]
        / schedule(0).multi_target["texture"],
        20.0,
    )
    assert jnp.isclose(
        schedule(100).multi_target["l2"]
        / schedule(100).multi_target["texture"],
        0.25,
    )


def test_unknown_multi_target_schedule_component_fails_before_training():
    term = MultiTargetLossConfig(
        type="multi_target",
        multi_target_schedules={"unknown": LossWeightScheduleConfig()},
    )
    with pytest.raises(ValueError, match="Unknown multi-target"):
        build_loss_weight_schedule(LossConfig(terms=(term,)), 100)


@pytest.mark.parametrize(
    "schedule",
    [
        {"type": "invalid"},
        {"initial_factor": -1.0},
        {"type": "linear", "start_fraction": 0.8, "end_fraction": 0.2},
    ],
)
def test_invalid_loss_schedule_configuration_fails(schedule):
    with pytest.raises(ValueError):
        LossWeightScheduleConfig(**schedule)


def test_typed_micropattern_config_parses_nested_loss_schedules():
    value = yaml.safe_load(
        Path("Experiments/micropatterns/conf/base_config.yaml").read_text()
    )
    value["loss"]["terms"] = [
        {
            "type": "multi_target",
            "schedule": {
                "type": "linear",
                "initial_factor": 0.5,
                "final_factor": 1.0,
                "start_fraction": 0.0,
                "end_fraction": 0.5,
            },
            "multi_target_schedules": {
                "texture": {
                    "type": "cosine",
                    "initial_factor": 0.05,
                    "final_factor": 1.0,
                    "start_fraction": 0.3,
                    "end_fraction": 0.75,
                }
            },
        }
    ]
    value["loss"]["schedule_label"] = "cos_macro"

    config = experiment_config_from_mapping(value)
    term = config.training.loss.terms[0]

    assert term.schedule.type == "linear"
    assert term.multi_target_schedules["texture"].type == "cosine"
    assert config.training.loss.schedule_label == "cos_macro"
    assert final_transition_iteration(
        config.training.loss, config.training.loop.iterations
    ) == round(0.75 * (config.training.loop.iterations - 1))
    assert experiment_config_from_mapping(config_to_dict(config)) == config


def test_static_loss_configuration_keeps_existing_weights():
    config = LossConfig(
        terms=(
            PointwiseLossConfig(type="l2", weight=1.0),
            PointwiseLossConfig(type="spectral", weight=2.0),
        )
    )

    schedule = build_loss_weight_schedule(config, 100)

    assert jnp.allclose(schedule(0).terms, jnp.array([1.0, 2.0]))
    assert jnp.allclose(schedule(99).terms, jnp.array([1.0, 2.0]))


def test_schedule_label_is_parsed_and_added_to_loss_filename():
    config = LossConfig(
        terms=(PointwiseLossConfig(type="l2"),),
        schedule_label="cos_macro",
    )

    assert build_loss_filename(config) == "l2_lscos_macro"


@pytest.mark.parametrize("label", ["", "has space", "path/name"])
def test_invalid_schedule_label_fails_early(label):
    with pytest.raises(ValueError, match="schedule_label"):
        LossConfig(
            terms=(PointwiseLossConfig(type="l2"),),
            schedule_label=label,
        )

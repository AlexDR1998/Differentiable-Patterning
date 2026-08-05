from pathlib import Path

import pytest
import yaml

from Experiments.config import (
    ExperimentConfig,
    ImpulseExperimentConfig,
    config_to_dict,
    experiment_config_from_mapping,
    impulse_experiment_config_from_mapping,
)


@pytest.mark.parametrize(
    "path",
    [
        Path("Experiments/emoji/conf/base_config.yaml"),
        Path("Experiments/micropatterns/conf/base_config.yaml"),
    ],
)
def test_active_base_configs_convert_and_round_trip(path):
    config = experiment_config_from_mapping(yaml.safe_load(path.read_text()))

    assert isinstance(config, ExperimentConfig)
    assert experiment_config_from_mapping(config_to_dict(config)) == config


def test_unknown_fields_are_rejected_with_their_section():
    value = yaml.safe_load(
        Path("Experiments/emoji/conf/base_config.yaml").read_text()
    )
    value["model"]["chanels"] = value["model"]["channels"]

    with pytest.raises(ValueError, match="model.*chanels"):
        experiment_config_from_mapping(value)


def test_normalized_nca_configuration_is_rejected():
    value = yaml.safe_load(
        Path("Experiments/emoji/conf/base_config.yaml").read_text()
    )
    value["model"]["family"] = "NormalizedNCA"

    with pytest.raises(ValueError, match="Unsupported model family"):
        experiment_config_from_mapping(value)


def test_non_kan_model_does_not_expose_kan_configuration():
    value = yaml.safe_load(
        Path("Experiments/emoji/conf/base_config.yaml").read_text()
    )

    config = experiment_config_from_mapping(value)

    assert config.model.family == "NCA"
    assert not hasattr(config.model, "kan")


def test_kan_configuration_is_owned_by_kan_model():
    value = yaml.safe_load(
        Path("Experiments/emoji/conf/base_config.yaml").read_text()
    )
    value["model"]["family"] = "FastKaNCA"
    value["model"]["kan"] = {"basis": "linear_spline", "num_basis": 12}

    config = experiment_config_from_mapping(value)

    assert config.model.kan.basis == "linear_spline"
    assert config.model.kan.num_basis == 12


def test_non_kan_model_rejects_kan_configuration():
    value = yaml.safe_load(
        Path("Experiments/emoji/conf/base_config.yaml").read_text()
    )
    value["model"]["kan"] = {"num_basis": 12}

    with pytest.raises(ValueError, match="model.*kan"):
        experiment_config_from_mapping(value)


def test_non_upsampling_model_does_not_expose_upsampler_configuration():
    value = yaml.safe_load(
        Path("Experiments/emoji/conf/base_config.yaml").read_text()
    )

    config = experiment_config_from_mapping(value)

    assert config.model.family == "NCA"
    assert not hasattr(config.model, "upsampler")


@pytest.mark.parametrize("family", ["uNCA", "isouNCA"])
def test_upsampling_model_families_are_rejected(family):
    value = yaml.safe_load(
        Path("Experiments/emoji/conf/base_config.yaml").read_text()
    )
    value["model"]["family"] = family

    with pytest.raises(ValueError, match="Unsupported model family"):
        experiment_config_from_mapping(value)


def test_non_upsampling_model_rejects_upsampler_configuration():
    value = yaml.safe_load(
        Path("Experiments/emoji/conf/base_config.yaml").read_text()
    )
    value["model"]["upsampler"] = {"depth": 2}

    with pytest.raises(ValueError, match="model.*upsampler"):
        experiment_config_from_mapping(value)


def test_augmentation_schedule_is_owned_by_data_config():
    value = yaml.safe_load(
        Path("Experiments/micropatterns/conf/base_config.yaml").read_text()
    )
    value["data"]["micropattern"]["intermediate_reinjection_probability"] = 0.2
    value["data"]["micropattern"]["intermediate_reinjection_probability_end"] = 0.8

    config = experiment_config_from_mapping(value)

    assert config.data.micropattern.intermediate_reinjection_probability == 0.2
    assert config.data.micropattern.intermediate_reinjection_probability_end == 0.8


def test_trainer_backend_defaults_to_unconstrained_jax():
    value = yaml.safe_load(
        Path("Experiments/emoji/conf/base_config.yaml").read_text()
    )

    config = experiment_config_from_mapping(value)

    assert config.training.trainer.backend.type == "none"


def test_sycl_trainer_settings_are_scoped_to_sycl_backend():
    value = yaml.safe_load(
        Path("Experiments/micropatterns/conf/base_config.yaml").read_text()
    )
    value["trainer"]["backend"] = {"type": "sycl", "fused_steps": 4}
    value["model"]["family"] = "NCA_sycl"

    config = experiment_config_from_mapping(value)

    assert config.training.trainer.backend.type == "sycl"
    assert config.training.trainer.backend.fused_steps == 4


def test_default_backend_rejects_sycl_only_settings():
    value = yaml.safe_load(
        Path("Experiments/micropatterns/conf/base_config.yaml").read_text()
    )
    value["trainer"]["backend"] = {"type": "none", "fused_steps": 4}

    with pytest.raises(ValueError, match="training.trainer.backend.*fused_steps"):
        experiment_config_from_mapping(value)


def test_nvidia_backend_uses_the_standard_jax_trainer_configuration():
    value = yaml.safe_load(
        Path("Experiments/emoji/conf/base_config.yaml").read_text()
    )
    value["trainer"]["backend"] = {"type": "nvidia"}

    config = experiment_config_from_mapping(value)

    assert config.training.trainer.backend.type == "nvidia"


def test_impulse_workflow_has_a_separate_typed_root():
    value = yaml.safe_load(
        Path("Experiments/impulse/conf/base_config.yaml").read_text()
    )

    config = impulse_experiment_config_from_mapping(value)

    assert isinstance(config, ImpulseExperimentConfig)
    assert config.impulse.rollout.steps == 64

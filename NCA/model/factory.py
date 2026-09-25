"""Build NCA models from their typed ``ModelConfig``.

``build_model`` is the one place that turns ``model.family`` into a model
class. To add a new family, add its class and any extra constructor arguments
to ``MODEL_FAMILIES``, and its short name to ``build_model_config_string``.
"""

from dataclasses import asdict

import jax

from NCA.model.NCA_fast_KAN_model import FastKaNCA
from NCA.model.NCA_gated_model import gNCA
from NCA.model.NCA_gated_noise_model import gnNCA
from NCA.model.NCA_hierarchical import HNCA
from NCA.model.NCA_model import NCA
from NCA.model.NCA_noise_model import nNCA


# Retired model families, and the family that now builds them. Their saved
# parameters have the same layout, so old bundles still load.
PORTABLE_MODEL_FAMILIES = {
    "NCA_sycl": "NCA",
    "NCA_fast": "NCA",
    "gNCA_sycl": "gNCA",
}

ACTIVATIONS = {
    "relu": jax.nn.relu,
    "tanh": jax.nn.tanh,
    "swish": jax.nn.swish,
    "gelu": jax.nn.gelu,
    "linear": lambda x: x,
}


def portable_family(family):
    """Return the family that builds ``family`` today."""
    return PORTABLE_MODEL_FAMILIES.get(family, family)


def _no_extra_arguments(model_config):
    return {}


def _noise_arguments(model_config):
    return {"PARAMETER_NOISE_LEVEL": model_config.parameter_noise_level}


def _kan_arguments(model_config):
    kan_aux = asdict(model_config.kan)
    if kan_aux["hidden_features"] is None:
        del kan_aux["hidden_features"]
    return {"KAN_AUX": kan_aux}


def _hnca_arguments(model_config):
    return {
        "SCALE": model_config.scale,
        "OBS_CHANNELS": model_config.obs_channels,
        "PARENT_LEARNABLE_KERNELS": model_config.parent_learnable_kernels,
        "CHILD_GATED": model_config.child_gated,
        "PARENT_GATED": model_config.parent_gated,
        "ACTUATOR_GATED": model_config.actuator_gated,
    }


# family -> (model class, function giving its extra constructor arguments)
MODEL_FAMILIES = {
    "NCA": (NCA, _no_extra_arguments),
    "gNCA": (gNCA, _no_extra_arguments),
    "nNCA": (nNCA, _noise_arguments),
    "gnNCA": (gnNCA, _noise_arguments),
    "FastKaNCA": (FastKaNCA, _kan_arguments),
    "HNCA": (HNCA, _hnca_arguments),
}


def build_model(model_config, key=None):
    """Construct a model and its short run-name string from a ModelConfig."""
    family = portable_family(model_config.family)
    if family not in MODEL_FAMILIES:
        raise ValueError(f"Unknown model family {family}")
    activation_name = model_config.activation or "relu"
    if activation_name not in ACTIVATIONS:
        raise ValueError(f"Unsupported activation {activation_name}")

    model_class, extra_arguments = MODEL_FAMILIES[family]
    model = model_class(
        N_CHANNELS=model_config.channels,
        KERNEL_STR=model_config.kernel_str,
        ACTIVATION=ACTIVATIONS[activation_name],
        FIRE_RATE=model_config.fire_rate,
        PADDING=model_config.padding,
        KERNEL_SCALE=model_config.kernel_scale,
        key=key,
        **extra_arguments(model_config),
    )
    return model, build_model_config_string(model_config, family=family)


def build_model_config_string(model_config, family=None):
    """Short model description used in run names, e.g. ``gNCA_c48``."""
    family = model_config.family if family is None else family
    cfg_str = f"{family}_c{model_config.channels}"
    if model_config.activation not in {None, "relu"}:
        cfg_str += f"_act{model_config.activation}"
    if model_config.kernel_scale != 1:
        cfg_str += f"_ks{model_config.kernel_scale}"
    if family in {"nNCA", "gnNCA"}:
        cfg_str += f"_pn{model_config.parameter_noise_level}"
    if family == "FastKaNCA":
        kan = model_config.kan
        cfg_str += f"_kb{kan.num_basis}"
        if kan.basis == "linear_spline":
            cfg_str += "_klin"
        elif kan.basis != "rbf":
            cfg_str += f"_k{kan.basis}"
        if kan.extrapolation != "constant":
            cfg_str += f"_kex{kan.extrapolation}"
        if kan.hidden_features is not None:
            cfg_str += f"_kh{kan.hidden_features}"
        if kan.base_activation != "identity":
            cfg_str += f"_kbase{kan.base_activation}"
        if not kan.use_layernorm:
            cfg_str += "_noln"
        if not kan.final_zero_init:
            cfg_str += "_nozero"
    if family == "HNCA":
        cfg_str += f"_s{model_config.scale}"
        if model_config.parent_learnable_kernels:
            cfg_str += "_plk"
        if model_config.child_gated:
            cfg_str += "_cg"
        if model_config.parent_gated:
            cfg_str += "_pg"
        if model_config.actuator_gated:
            cfg_str += "_ag"
    return cfg_str

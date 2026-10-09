

import hashlib
import os
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path

# build_model is re-exported here because saved model bundles record
# "Experiments.config_helpers:build_model" as their model factory.
from NCA.model.factory import build_model, build_model_config_string  # noqa: F401


MAX_WANDB_TAG_LENGTH = 64
EXCLUDED_WANDB_TAG_KEYS = {
    "logging.wandb.project",
    "logging.wandb.group",
    "logging.wandb.tags",
    "logging.wandb.tag_keys",
    "model_store.root",
    "model_store.collection",
    "model_store.model_factory",
}
WANDB_TAG_KEY_ALIASES = {
    "loss.terms.0.multi_target_schedules": "loss_schedule",
    "loss.terms.0.multi_target_weights": "loss_weight",
}


def data_channel_count(cfg):
    """Return the measurement channel count for the configured data domain."""

    if cfg.data.dataset == "emojis":
        return cfg.data.emoji.data_channels
    if cfg.data.dataset == "snowmelt":
        return len(cfg.data.snowmelt.target_channels)
    if cfg.data.dataset == "pde":
        return len(cfg.data.pde.channel_names)
    return cfg.data.micropattern.data_channels


def _compact_value(value):
    if isinstance(value, Enum):
        value = value.value
    if value is None:
        return "none"
    if isinstance(value, (list, tuple)):
        return "-".join(_compact_value(item) for item in value)
    return str(value)


def _sequence_alias(sequence):
    aliases = []
    for filename in _as_list(sequence):
        basename = str(filename).rsplit("/", 1)[-1].split(".", 1)[0]
        alias = basename[:2].lower()
        if aliases and aliases[-1] == alias:
            continue
        aliases.append(alias)
    return "_".join(aliases)


def _safe_wandb_tag(tag, max_length=MAX_WANDB_TAG_LENGTH):
    if max_length is None:
        return tag
    if len(tag) <= max_length:
        return tag
    digest = hashlib.sha1(tag.encode("utf-8")).hexdigest()[:8]
    return f"{tag[: max_length - 9]}~{digest}"


def _wandb_tag_key(key):
    """Shorten known deep configuration paths before length protection."""

    for prefix, alias in WANDB_TAG_KEY_ALIASES.items():
        if key == prefix:
            return alias
        if key.startswith(f"{prefix}."):
            return f"{alias}{key[len(prefix):]}"
    return key


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def compact_nonzero_config_string(values, aliases=None):
    aliases = aliases or {}
    parts = []
    for key, value in values.items():
        if value is None or value == 0:
            continue
        parts.append(f"{aliases.get(key, key)}{_compact_value(value)}")
    return "_".join(parts)


def loss_terms(loss_config):
    return list(loss_config.terms)


def loss_names(loss_config):
    return [term.type for term in loss_config.terms]


def loss_weights(loss_config):
    return [float(term.weight) for term in loss_config.terms]


def build_loss_filename(loss_config, include_loss_args=False):
    terms = loss_terms(loss_config)
    names = loss_names(loss_config)
    loss_str = "_".join(names).lower()
    for term in terms:
        # Loss term classes have different fields, so optional ones use getattr.
        name = term.type
        if "vgg" in name:
            loss_str += f"_vgg{str(term.metric).lower()}"
            if term.random_crop: loss_str += "_rc"
            if term.random_channel_shuffle: loss_str += "_chshuffle"
        channel_importance = getattr(term, "channel_importance", None)
        if channel_importance is not None:
            non_default = [f"{i + 1}x{_compact_value(w)}" for i, w in enumerate(channel_importance) if float(w) != 1.0]
            if non_default: loss_str += "_ci" + "-".join(non_default)
        multi_target_weights = term.multi_target_weights if name == "multi_target" else None
        if multi_target_weights is not None:
            loss_str += (
                f"_mtw_tex{multi_target_weights.get('texture', 1.0):g}"
                f"_cm{multi_target_weights.get('channel_mean', 0.0):g}"
                f"_corr{multi_target_weights.get('correlation', 0.0):g}"
                f"_rad{multi_target_weights.get('radial', 0.0):g}"
                f"_rchsh{int(bool(term.random_channel_shuffle))}"
                f"_rcr{int(bool(term.random_crop))}"
            )
            l2_weight = multi_target_weights.get("l2", 0.0)
            if float(l2_weight) != 0.0:
                loss_str += f"_l2{l2_weight:g}"
        if include_loss_args:
            keys = ("S", "K", "D", "epsilon", "sharpen", "samples")
            arg_str = compact_nonzero_config_string({key: getattr(term, key, None) for key in keys})
            if arg_str: loss_str += f"_{arg_str}"

    weights = loss_weights(loss_config)
    if any(weight != 1.0 for weight in weights):
        loss_str += "_cw" + "-".join(_compact_value(weight) for weight in weights)

    if loss_config.schedule_label:
        loss_str += f"_ls{loss_config.schedule_label}"

    reg_str = compact_nonzero_config_string(
        loss_config.regularisers,
        aliases={
            "boundary": "bd",
            "contiguous_growth": "cg",
            "intermediate_state": "is",
            "hidden_state_size": "hs",
            "localised_hidden": "lh",
            "perturbation_conservation": "pc",
            "update_sensitivity": "us",
        },
    )
    if reg_str:
        loss_str += f"_{reg_str}"
    return loss_str


def open_registry_bundle(model_id, store_root=None, env=None):
    """Open a model bundle by ID or alias.

    ``store_root`` defaults to ``MODEL_STORE_ROOT`` and then ``./models``,
    matching the registry CLI and explorer defaults.
    """

    from Experiments.model_registry import ModelRegistry

    environment = os.environ if env is None else env
    resolved_store_root = store_root or environment.get("MODEL_STORE_ROOT") or "models"
    return ModelRegistry(resolved_store_root).get(model_id)


def load_model_registry_list(
    export_path,
    store_root=None,
    key=None,
    implementation="portable",
    env=None,
):
    """Load the NCA models named by a model-registry explorer YAML export.

    Models are returned in export order. Each registry bundle verifies its
    checkpoint checksum before reconstructing the architecture and loading its
    Equinox leaves. Portable loading is the default and replaces an archived
    SYCL implementation with its equivalent standard JAX model.

    ``store_root`` defaults to ``MODEL_STORE_ROOT`` and then ``./models``,
    matching the registry CLI and explorer defaults.
    """

    from omegaconf import OmegaConf

    path = Path(export_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model registry export not found: {path}")

    document = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    if not isinstance(document, Mapping):
        raise TypeError("Model registry export must contain a YAML mapping")
    if document.get("schema_version") != 1:
        raise ValueError(
            "Unsupported model registry export schema version "
            f"{document.get('schema_version')!r}; expected 1"
        )

    entries = document.get("models")
    if not isinstance(entries, list):
        raise TypeError("Model registry export 'models' must be a list")

    model_ids = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise TypeError(f"Model registry export models[{index}] must be a mapping")
        model_id = entry.get("model_id")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError(
                f"Model registry export models[{index}] has no valid model_id"
            )
        model_ids.append(model_id)

    declared_count = document.get("model_count")
    if declared_count is not None and declared_count != len(model_ids):
        raise ValueError(
            f"Model registry export declares {declared_count} models but contains "
            f"{len(model_ids)}"
        )

    return [
        open_registry_bundle(model_id, store_root, env).load_model(
            key=key,
            implementation=implementation,
        )
        for model_id in model_ids
    ]


def _selected(tag_key, keys):
    """Whether ``tag_key`` is one of ``keys`` or lies below one (None selects all)."""
    return keys is None or any(tag_key == key or tag_key.startswith(f"{key}.") for key in keys)


def build_tags(cfg, prefix="", max_length=MAX_WANDB_TAG_LENGTH, keys=None):
    """One ``key:value`` tag per setting; with ``keys``, only for those settings."""
    tags = []
    if isinstance(cfg, Mapping) or hasattr(cfg, "items"):
        items = cfg.items()
    elif is_dataclass(cfg):
        items = ((item.name, getattr(cfg, item.name)) for item in fields(cfg))
    else:
        raise TypeError(
            f"W&B tag configuration nodes must be mappings or dataclasses, got {type(cfg).__name__}"
        )

    for key, value in items:
        if key == "seed":
            continue
        tag_key = f"{prefix}{key}"
        if tag_key in EXCLUDED_WANDB_TAG_KEYS:
            continue
        if value is None:
            continue
        if (
            isinstance(value, Mapping)
            or hasattr(value, "items")
            or is_dataclass(value)
        ):
            tags.extend(
                build_tags(value, prefix=f"{tag_key}.", max_length=max_length, keys=keys)
            )
        elif isinstance(value, (list, tuple)) and any(
            isinstance(item, Mapping)
            or hasattr(item, "items")
            or is_dataclass(item)
            for item in value
        ):
            for index, item in enumerate(value):
                if (
                    isinstance(item, Mapping)
                    or hasattr(item, "items")
                    or is_dataclass(item)
                ):
                    tags.extend(
                        build_tags(
                            item,
                            prefix=f"{tag_key}.{index}.",
                            max_length=max_length,
                            keys=keys,
                        )
                    )
                elif _selected(f"{tag_key}.{index}", keys):
                    tags.append(
                        _safe_wandb_tag(
                            f"{_wandb_tag_key(f'{tag_key}.{index}')}:{_compact_value(item)}",
                            max_length=max_length,
                        )
                    )
        elif _selected(tag_key, keys):
            if tag_key == "data.emoji.sequence":
                value = _sequence_alias(value)
            else:
                value = _compact_value(value)
            tags.append(
                _safe_wandb_tag(
                    f"{_wandb_tag_key(tag_key)}:{value}", max_length=max_length
                )
            )
    return tags


def build_wandb_tags(cfg):
    """Automatic tags for the settings in ``logging.wandb.tag_keys``, plus hand-written tags.

    Sweep manifests set ``tag_keys`` to the settings that differ between
    their runs, so the W&B tags show only what distinguishes a run. Without
    ``tag_keys`` (e.g. a single config run by hand) every setting is tagged.
    """

    wandb = getattr(getattr(cfg, "logging", None), "wandb", None)
    keys = getattr(wandb, "tag_keys", None)
    automatic_tags = build_tags(cfg, keys=None if keys is None else tuple(keys))
    return list(dict.fromkeys((*automatic_tags, *_explicit_tags(cfg))))


def build_registry_tags(cfg):
    """Build readable, untruncated tags for the local model registry."""

    automatic_tags = build_tags(cfg, max_length=None)
    return list(dict.fromkeys((*automatic_tags, *_explicit_tags(cfg))))


def _explicit_tags(cfg):
    """Tags written by hand in logging.wandb.tags."""
    wandb = getattr(getattr(cfg, "logging", None), "wandb", None)
    return [str(tag) for tag in (getattr(wandb, "tags", None) or ())]

import os

import jax
import jax.numpy as jnp
import numpy as np

from Common.dataloader.emoji import load_emoji_sequence
from Experiments.config_helpers import (
    _as_list,
    _cfg_get,
    _sequence_alias,
    build_loss_filename as _shared_build_loss_filename,
)
from NCA.trainer.data_augmenter_nca_terminal import TerminalCarryDataAugmenter


def _pad_tuple(value):
    if value is None:
        return None
    if isinstance(value, int):
        return [value, value, value, value]
    return list(value)


def build_loss_filename(cfg, include_layers=True):
    include_layers = include_layers and cfg.model.family in {"uNCA", "isouNCA"}
    include_loss_args = cfg.model.family in {"uNCA", "isouNCA"}
    return _shared_build_loss_filename(
        cfg,
        include_layers=include_layers,
        include_loss_args=include_loss_args,
    )


def build_data_config_string(cfg):
    terminal_cfg = _cfg_get(cfg.data, "terminal_carry", None)
    regeneration_cfg = _cfg_get(cfg.data, "regeneration", None)
    terminal_str = ""
    if _cfg_get(terminal_cfg, "enabled", False):
        terminal_str = (
            f"_tc{_cfg_get(terminal_cfg, 'initial_probability', 0.0)}"
            f"-{_cfg_get(terminal_cfg, 'final_probability', 0.0)}"
        )
    regeneration_str = ""
    regeneration_enabled = _cfg_get(
        regeneration_cfg, "enabled", _cfg_get(cfg.data, "regenerate", False)
    )
    regeneration_initial = _cfg_get(regeneration_cfg, "initial_probability", 1.0)
    regeneration_final = _cfg_get(regeneration_cfg, "final_probability", regeneration_initial)
    legacy_regeneration = (
        _cfg_get(cfg.data, "regenerate", False)
        and regeneration_initial == 1.0
        and regeneration_final == 1.0
        and _cfg_get(regeneration_cfg, "start_iteration", 0) == 0
        and _cfg_get(regeneration_cfg, "schedule_iterations", 0) == 0
    )
    if regeneration_enabled and not legacy_regeneration:
        regeneration_str = f"_rg{regeneration_initial}-{regeneration_final}"
    task = _cfg_get(cfg.data, "task", "sequence")
    if task == "multi_attractor":
        pairs = _as_list(_cfg_get(cfg.data, "pairs", None))
        aliases = []
        for pair in pairs:
            initial = _cfg_get(pair, "initial", None)
            initial_image = (
                initial if isinstance(initial, str) else _cfg_get(initial, "image", None)
            )
            target = _cfg_get(pair, "target", None)
            aliases.append(
                f"{_sequence_alias([initial_image])}2{_sequence_alias([target])}"
            )
        pair_alias = "-".join(aliases)
        return (
            f"data_multi_{pair_alias}"
            f"_b{cfg.data.batches}"
            f"_ds{cfg.data.downsample}"
            f"_regen{cfg.data.regenerate}{terminal_str}{regeneration_str}"
        )
    if task != "sequence":
        raise ValueError(f"Unknown emoji data.task {task!r}")
    return (
        f"data_{_sequence_alias(cfg.data.sequence)}"
        f"_b{cfg.data.batches}"
        f"_ds{cfg.data.downsample}"
        f"_regen{cfg.data.regenerate}{terminal_str}{regeneration_str}"
    )


def _load_single_emoji(filename, cfg, impath):
    if not filename:
        raise ValueError("Every multi-attractor initial condition and target needs an image filename")
    return load_emoji_sequence(
        [filename],
        impath_emojis=impath,
        downsample=cfg.data.downsample,
        crop_square=_cfg_get(cfg.data, "crop_square", False),
    )[0, 0]


def _build_initial_condition(initial_cfg, cfg, impath):
    if isinstance(initial_cfg, str):
        initial_cfg = {"image": initial_cfg, "mode": "full"}
    if initial_cfg is None or not hasattr(initial_cfg, "get"):
        raise ValueError("multi-attractor pair.initial must be a filename or a mapping")

    image = _load_single_emoji(_cfg_get(initial_cfg, "image", None), cfg, impath)
    mode = _cfg_get(initial_cfg, "mode", "full")
    if mode == "full":
        return image
    if mode == "patch":
        patch_size = int(_cfg_get(initial_cfg, "size", 12))
        height, width = image.shape[-2:]
        if patch_size <= 0 or patch_size > min(height, width):
            raise ValueError(
                f"initial patch size must be in [1, {min(height, width)}], got {patch_size}"
            )
        top = (height - patch_size) // 2
        left = (width - patch_size) // 2
        initial = np.zeros_like(image)
        initial[:, top : top + patch_size, left : left + patch_size] = image[
            :, top : top + patch_size, left : left + patch_size
        ]
        return initial
    if mode == "pixel":
        channel = int(_cfg_get(initial_cfg, "channel", 0))
        value = float(_cfg_get(initial_cfg, "value", 1.0))
        if not 0 <= channel < image.shape[0]:
            raise ValueError(
                f"initial pixel channel must be in [0, {image.shape[0] - 1}], got {channel}"
            )
        initial = np.zeros_like(image)
        initial[channel, image.shape[-2] // 2, image.shape[-1] // 2] = value
        return initial
    raise ValueError(f"Unknown multi-attractor initial mode {mode!r}")


def _load_multi_attractor_data(cfg, impath):
    pairs = _as_list(_cfg_get(cfg.data, "pairs", None))
    if not pairs:
        raise ValueError("data.pairs must contain at least one pair for data.task=multi_attractor")
    target_repeats = int(_cfg_get(cfg.data, "target_repeats", 2))
    if target_repeats < 1:
        raise ValueError("data.target_repeats must be at least 1")

    trajectories = []
    expected_shape = None
    for index, pair in enumerate(pairs):
        if not hasattr(pair, "get"):
            raise ValueError(f"data.pairs[{index}] must be a mapping")
        initial = _build_initial_condition(_cfg_get(pair, "initial", None), cfg, impath)
        target = _load_single_emoji(_cfg_get(pair, "target", None), cfg, impath)
        if initial.shape != target.shape:
            raise ValueError(
                f"data.pairs[{index}] initial and target shapes differ: "
                f"{initial.shape} != {target.shape}"
            )
        if expected_shape is not None and target.shape != expected_shape:
            raise ValueError(
                "All multi-attractor pairs must have the same channel and spatial shape; "
                f"pair {index} has {target.shape}, expected {expected_shape}"
            )
        expected_shape = target.shape
        trajectories.append(np.stack([initial] + [target] * target_repeats))
    return np.stack(trajectories)


def load_data(cfg, impath=None):
    custom_impath = impath is not None
    if impath is None:
        data_path_base = os.getenv("DATA_PATH_BASE")
        if data_path_base is None:
            raise ValueError("DATA_PATH_BASE must be set when load_data is called without impath.")
        impath = os.path.join(data_path_base, "Emojis", "")
    task = _cfg_get(cfg.data, "task", "sequence")
    if task == "sequence":
        data = load_emoji_sequence(
            _as_list(cfg.data.sequence),
            impath_emojis=impath,
            downsample=cfg.data.downsample,
            crop_square=_cfg_get(cfg.data, "crop_square", False),
        )
    elif task == "multi_attractor":
        data = _load_multi_attractor_data(cfg, impath)
    else:
        raise ValueError(f"Unknown emoji data.task {task!r}")
    cfg_str = build_data_config_string(cfg)
    if custom_impath:
        cfg_str += "_custompath"
    return data, cfg_str


def build_data_augmenter(cfg):
    pad = _pad_tuple(cfg.data.pad)
    batches = cfg.data.batches
    shift_amount = cfg.data.shift_amount
    noise_strength = cfg.data.noise_strength
    regenerate = cfg.data.regenerate
    noise_mode = _cfg_get(cfg.data, "noise_mode", "full")
    terminal_cfg = _cfg_get(cfg.data, "terminal_carry", None)
    regeneration_cfg = _cfg_get(cfg.data, "regeneration", None)

    terminal_enabled = _cfg_get(terminal_cfg, "enabled", False)
    terminal_start = _cfg_get(terminal_cfg, "start_iteration", 0)
    terminal_schedule = _cfg_get(terminal_cfg, "schedule_iterations", 0)
    terminal_initial = _cfg_get(terminal_cfg, "initial_probability", 0.0)
    terminal_final = _cfg_get(terminal_cfg, "final_probability", terminal_initial)

    regeneration_enabled = _cfg_get(regeneration_cfg, "enabled", regenerate)
    regeneration_start = _cfg_get(regeneration_cfg, "start_iteration", 0)
    regeneration_schedule = _cfg_get(regeneration_cfg, "schedule_iterations", 0)
    regeneration_initial = _cfg_get(regeneration_cfg, "initial_probability", 1.0)
    regeneration_final = _cfg_get(regeneration_cfg, "final_probability", regeneration_initial)

    class EmojiDataAugmenter(TerminalCarryDataAugmenter):
        TERMINAL_CARRY_ENABLED = terminal_enabled
        TERMINAL_CARRY_START = terminal_start
        TERMINAL_CARRY_SCHEDULE = terminal_schedule
        TERMINAL_CARRY_INITIAL = terminal_initial
        TERMINAL_CARRY_FINAL = terminal_final

        def data_init(self, SHARDING=None):
            data = self.return_saved_data()
            data = self.duplicate_batches(data, batches)
            if pad is not None:
                data = self.pad(data, pad)
            self.save_data(data)
            return None

        def data_callback(self, x, y, i, key):
            if shift_amount and hasattr(self, "PREVIOUS_KEY"):
                x = self.unshift(x, shift_amount, self.PREVIOUS_KEY)
                y = self.unshift(y, shift_amount, self.PREVIOUS_KEY)

            x_true, _ = self.split_x_y(1)
            x = self.propagate_with_terminal_carry(x, x_true, i, key)

            if shift_amount:
                x = self.shift(x, shift_amount, key=key)
                y = self.shift(y, shift_amount, key=key)
            if regeneration_enabled:
                probability = self.scheduled_probability(
                    i,
                    regeneration_start,
                    regeneration_schedule,
                    regeneration_initial,
                    regeneration_final,
                )
                damaged = self.zero_random_circle(x, key=key)
                damage_mask = jax.random.bernoulli(
                    jax.random.fold_in(key, 2), probability, (len(x),)
                )
                if hasattr(x, "ndim"):
                    x = jnp.where(
                        damage_mask[:, None, None, None, None], damaged, x
                    )
                else:
                    for batch_index in range(len(x)):
                        x[batch_index] = jnp.where(
                            damage_mask[batch_index], damaged[batch_index], x[batch_index]
                        )
            if noise_strength:
                x = self.noise(x, noise_strength, mode=noise_mode, key=key)

            self.PREVIOUS_KEY = key
            return x, y

    cfg_str = "da"
    return EmojiDataAugmenter, cfg_str


def resolve_run_t(cfg):
    if not _cfg_get(cfg.run, "derive_t_from_fire_rate", False):
        return cfg.run.t
    numerator = _cfg_get(cfg.run, "fire_rate_step_numerator", None)
    if numerator is None:
        numerator = 32 if cfg.model.channels == 32 else 64
    return int(numerator / cfg.model.fire_rate)


def build_filename(cfg, model_cfg_str, data_cfg_str, data_augmenter_cfg_str):
    filename_mode = _cfg_get(cfg.run, "filename_mode", "hydra")
    if filename_mode == "legacy_train":
        return build_legacy_training_filename(cfg)
    if filename_mode != "hydra":
        raise ValueError(f"Unknown filename_mode {filename_mode}")

    loss_str = build_loss_filename(cfg)
    train_str = (
        f"_t{resolve_run_t(cfg)}"
        # f"_iters{cfg.run.iterations}"
        f"_lr{cfg.optimiser.learn_rate}"
        f"_dr{cfg.optimiser.decay_rate}"
        f"_batch{_cfg_get(cfg.trainer, 'batch_mode', 'tree')}"
    )
    return "_".join([model_cfg_str, data_cfg_str, loss_str, train_str])


def build_legacy_training_filename(cfg):
    sequence_alias = _sequence_alias(cfg.data.sequence)
    regen_str = "regenerate_" if cfg.data.regenerate else ""
    loss_mode = "_".join(_as_list(cfg.loss.primary)).lower()
    return (
        f"emoji_{sequence_alias}_{loss_mode}_{cfg.model.family}_{regen_str}"
        f"ch{cfg.model.channels}_ds{cfg.data.downsample}"
        f"_steps{resolve_run_t(cfg)}"
        f"_iters{cfg.run.iterations}"
        f"_igc{cfg.loss.regulariser_coeffs.intermediate_state}"
        f"_brc{cfg.loss.regulariser_coeffs.boundary}"
        f"_cgc{cfg.loss.regulariser_coeffs.contiguous_growth}"
        f"_pcc{cfg.loss.regulariser_coeffs.perturbation_conservation}"
        f"_usc{cfg.loss.regulariser_coeffs.update_sensitivity}"
    )

import os

import jax
import jax.numpy as jnp
import numpy as np

from Common.dataloader.emoji import load_emoji_sequence
from Experiments.config_helpers import (
    _sequence_alias,
    build_loss_filename as _shared_build_loss_filename,
)
from NCA.trainer.data_augmenter.nca_terminal import TerminalCarryDataAugmenter


def _pad_tuple(value):
    if value is None:
        return None
    if isinstance(value, int):
        return [value, value, value, value]
    return list(value)


def build_loss_filename(cfg):
    return _shared_build_loss_filename(
        cfg.loss,
        include_loss_args=False,
    )


def build_data_config_string(data_config):
    emoji = data_config.emoji
    terminal = emoji.terminal_carry
    regeneration = emoji.regeneration
    terminal_str = ""
    if terminal.enabled:
        terminal_str = f"_tc{terminal.initial_probability}-{terminal.final_probability}"
    # Only a non-default regeneration schedule is spelled out in the name.
    regeneration_str = ""
    default_regeneration = (
        regeneration.initial_probability == 1.0
        and regeneration.final_probability == 1.0
        and regeneration.start_iteration == 0
        and regeneration.schedule_iterations == 0
    )
    if regeneration.enabled and not default_regeneration:
        regeneration_str = (
            f"_rg{regeneration.initial_probability}-{regeneration.final_probability}"
        )
    suffix = (
        f"_b{data_config.batches}"
        f"_ds{data_config.downsample}"
        f"_regen{regeneration.enabled}{terminal_str}{regeneration_str}"
    )
    if emoji.task == "multi_attractor":
        aliases = []
        for pair in emoji.pairs:
            initial = pair.initial
            initial_image = initial if isinstance(initial, str) else initial.get("image")
            aliases.append(
                f"{_sequence_alias([initial_image])}2{_sequence_alias([pair.target])}"
            )
        return f"data_multi_{'-'.join(aliases)}{suffix}"
    if emoji.task != "sequence":
        raise ValueError(f"Unknown data.emoji.task {emoji.task!r}")
    return f"data_{_sequence_alias(emoji.sequence)}{suffix}"


def _load_single_emoji(filename, data_config, impath):
    if not filename:
        raise ValueError("Every multi-attractor initial condition and target needs an image filename")
    loaded = load_emoji_sequence(
        [filename],
        impath_emojis=impath,
        downsample=data_config.downsample,
        crop_square=data_config.emoji.crop_square,
    )
    data = loaded.data if hasattr(loaded, "data") else loaded
    return data[0, 0]


def _build_initial_condition(initial_cfg, data_config, impath):
    if isinstance(initial_cfg, str):
        initial_cfg = {"image": initial_cfg, "mode": "full"}
    if initial_cfg is None or not hasattr(initial_cfg, "get"):
        raise ValueError("multi-attractor pair.initial must be a filename or a mapping")

    image = _load_single_emoji(initial_cfg.get("image"), data_config, impath)
    mode = initial_cfg.get("mode", "full")
    if mode == "full":
        return image
    if mode == "patch":
        patch_size = int(initial_cfg.get("size", 12))
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
        channel = int(initial_cfg.get("channel", 0))
        value = float(initial_cfg.get("value", 1.0))
        if not 0 <= channel < image.shape[0]:
            raise ValueError(
                f"initial pixel channel must be in [0, {image.shape[0] - 1}], got {channel}"
            )
        initial = np.zeros_like(image)
        initial[channel, image.shape[-2] // 2, image.shape[-1] // 2] = value
        return initial
    raise ValueError(f"Unknown multi-attractor initial mode {mode!r}")


def _load_multi_attractor_data(data_config, impath):
    pairs = data_config.emoji.pairs
    if not pairs:
        raise ValueError("data.emoji.pairs must contain at least one pair for data.emoji.task=multi_attractor")
    target_repeats = data_config.emoji.target_repeats
    if target_repeats < 1:
        raise ValueError("data.emoji.target_repeats must be at least 1")

    trajectories = []
    expected_shape = None
    for index, pair in enumerate(pairs):
        initial = _build_initial_condition(pair.initial, data_config, impath)
        target = _load_single_emoji(pair.target, data_config, impath)
        if initial.shape != target.shape:
            raise ValueError(
                f"data.emoji.pairs[{index}] initial and target shapes differ: "
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


def load_data(data_config, impath=None):
    custom_impath = impath is not None
    if impath is None:
        data_path_base = os.getenv("DATA_PATH_BASE")
        if data_path_base is None:
            raise ValueError("DATA_PATH_BASE must be set when load_data is called without impath.")
        impath = os.path.join(data_path_base, "Emojis", "")
    task = data_config.emoji.task
    if task == "sequence":
        dataset = load_emoji_sequence(
            list(data_config.emoji.sequence),
            impath_emojis=impath,
            downsample=data_config.downsample,
            crop_square=data_config.emoji.crop_square,
        )
        data = dataset.data
    elif task == "multi_attractor":
        data = _load_multi_attractor_data(data_config, impath)
    else:
        raise ValueError(f"Unknown data.emoji.task {task!r}")
    cfg_str = build_data_config_string(data_config)
    if custom_impath:
        cfg_str += "_custompath"
    return data, cfg_str


def build_data_augmenter(data_config):
    emoji = data_config.emoji
    pad = _pad_tuple(emoji.pad)
    batches = data_config.batches
    shift_amount = emoji.shift_amount
    noise_strength = emoji.noise_strength
    noise_mode = emoji.noise_mode
    terminal = emoji.terminal_carry
    regeneration = emoji.regeneration

    class EmojiDataAugmenter(TerminalCarryDataAugmenter):
        TERMINAL_CARRY_ENABLED = terminal.enabled
        TERMINAL_CARRY_START = terminal.start_iteration
        TERMINAL_CARRY_SCHEDULE = terminal.schedule_iterations
        TERMINAL_CARRY_INITIAL = terminal.initial_probability
        TERMINAL_CARRY_FINAL = terminal.final_probability

        def data_init(self, SHARDING=None):
            data = self.return_saved_data()
            data = self.duplicate_batches(data, batches)
            if pad is not None:
                data = self.pad(data, pad)
            self.save_data(data)
            return None

        def advance_pool(self, x, y, i, key):
            if shift_amount and hasattr(self, "PREVIOUS_KEY"):
                x = self.unshift(x, shift_amount, self.PREVIOUS_KEY)
                y = self.unshift(y, shift_amount, self.PREVIOUS_KEY)

            x_true, _ = self.split_x_y(1)
            x = self.propagate_with_terminal_carry(x, x_true, i, key)

            if shift_amount:
                x = self.shift(x, shift_amount, key=key)
                y = self.shift(y, shift_amount, key=key)
            if regeneration.enabled:
                probability = self.scheduled_probability(
                    i,
                    regeneration.start_iteration,
                    regeneration.schedule_iterations,
                    regeneration.initial_probability,
                    regeneration.final_probability,
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
    if not cfg.run.derive_t_from_fire_rate:
        return cfg.run.t
    numerator = cfg.run.fire_rate_step_numerator
    if numerator is None:
        numerator = 32 if cfg.model.channels == 32 else 64
    return int(numerator / cfg.model.fire_rate)


def build_filename(cfg, model_cfg_str, data_cfg_str, data_augmenter_cfg_str):
    loss_str = build_loss_filename(cfg)
    train_str = (
        f"_t{resolve_run_t(cfg)}"
        f"_lr{cfg.optimiser.learn_rate}"
        f"_dr{cfg.optimiser.decay_rate}"
    )
    return "_".join([model_cfg_str, data_cfg_str, loss_str, train_str])

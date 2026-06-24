import os

from Common.dataloader.emoji import load_emoji_sequence
from Experiments.config_helpers import (
    _as_list,
    _compact_value,
    _cfg_get,
    build_loss_args,
    build_loss_filename as _shared_build_loss_filename,
)
from NCA.trainer.data_augmenter_nca import DataAugmenter
from NCA.trainer.data_augmenter_nca_basic import jittable_callback_bit


def _pad_tuple(value):
    if value is None:
        return None
    if isinstance(value, int):
        return [value, value, value, value]
    return list(value)


def build_loss_filename(cfg, include_layers=True):
    return _shared_build_loss_filename(
        cfg,
        include_layers=include_layers,
        include_loss_args=True,
    )


def build_data_config_string(cfg):
    return (
        f"data_{_compact_value(_as_list(cfg.data.sequence))}"
        f"_b{cfg.data.batches}"
        f"_ds{cfg.data.downsample}"
        f"_pad{_compact_value(_pad_tuple(cfg.data.pad))}"
        f"_regen{cfg.data.regenerate}"
        f"_shift{cfg.data.shift_amount}"
        f"_noise{cfg.data.noise_strength}"
    )


def load_data(cfg, impath=None):
    custom_impath = impath is not None
    if impath is None:
        data_path_base = os.getenv("DATA_PATH_BASE")
        if data_path_base is None:
            raise ValueError("DATA_PATH_BASE must be set when load_data is called without impath.")
        impath = os.path.join(data_path_base, "Emojis", "")
    data = load_emoji_sequence(
        _as_list(cfg.data.sequence),
        impath_emojis=impath,
        downsample=cfg.data.downsample,
        crop_square=_cfg_get(cfg.data, "crop_square", False),
    )
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

    class EmojiDataAugmenter(DataAugmenter):
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
            x = jittable_callback_bit(x, x_true, self.OBS_CHANNELS)

            if shift_amount:
                x = self.shift(x, shift_amount, key=key)
                y = self.shift(y, shift_amount, key=key)
            if regenerate:
                x = self.zero_random_circle(x, key=key)
            if noise_strength:
                x = self.noise(x, noise_strength, mode=noise_mode, key=key)

            self.PREVIOUS_KEY = key
            return x, y

    cfg_str = (
        f"da_pad{_compact_value(pad)}"
        f"_regen{regenerate}"
        f"_shift{shift_amount}"
        f"_noise{noise_strength}"
    )
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
        f"train_t{resolve_run_t(cfg)}"
        f"_iters{cfg.run.iterations}"
        f"_lr{cfg.optimiser.learn_rate}"
        f"_dr{cfg.optimiser.decay_rate}"
    )
    return "_".join([model_cfg_str, data_cfg_str, data_augmenter_cfg_str, loss_str, train_str])


def build_legacy_training_filename(cfg):
    sequence_alias = _cfg_get(cfg.data, "sequence_alias", "al_mi_ro")
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


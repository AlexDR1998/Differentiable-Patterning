import math
import os
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from einops import repeat

from Common.dataloader.disk_cache import cache_dir_from_environment
from Common.dataloader.micropattern import (
    load_micropattern_260726,
    load_micropattern_circle_4ch_individual,
    load_micropattern_circle_nodal_knockout_9ch_explicit_colony,
)
from Common.dataloader.quality_flags import load_quality_flags
from Common.dataloader.results import MicropatternDataset
from Experiments.config_helpers import (
    _compact_value,
    build_loss_filename,
)
from Common.dataloader.micropattern_schemas import (
    MICROPATTERN_4CH_SCHEMA,
    MICROPATTERN_GROUPED_12CH_SCHEMA,
)
from NCA.trainer.data_augmenter.micropattern import MicropatternAugmenter


# Relative alignment files in data.micropattern.cleaning are relative to here.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _repository_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _cleaning_for_loader(cleaning):
    """The cleaning config with its alignment file resolved, or None when off."""
    if not cleaning.enabled:
        return None
    return replace(cleaning, alignment_file=str(_repository_path(cleaning.alignment_file)))


def _excluded_images(micropattern):
    """Image paths flagged as low quality, from ``quality_flags_file`` (none if unset)."""
    if micropattern.quality_flags_file is None:
        return ()
    return tuple(load_quality_flags(_repository_path(micropattern.quality_flags_file)))


def _reuses_control_bounds(data_config):
    """Whether knockout loads take their clipping bounds from the control data.

    Without cleaning this has always been so. With cleaning, only when
    ``knockouts_use_control_bounds`` is set and the bounds are shared; with
    per-replicate bounds the loader pairs each knockout replicate with its
    control replicate itself.
    """
    cleaning = data_config.micropattern.cleaning
    if not cleaning.enabled:
        return True
    return cleaning.knockouts_use_control_bounds and cleaning.shares_bounds


CURRICULUM_CONDITIONS = {
    "baseline": ("ctrl", -1),
    "ko_0h": ("sl0", 0),
    "ko_24h": ("sl24", 24),
}


def resolve_knockout_curriculum(intervention):
    curriculum = intervention.curriculum
    if curriculum is None:
        return ("baseline",)
    curriculum = tuple(curriculum)
    if not curriculum:
        raise ValueError("knockout.curriculum cannot be empty")
    unknown = sorted(set(curriculum) - CURRICULUM_CONDITIONS.keys())
    if unknown:
        raise ValueError(f"Unknown knockout curriculum entries: {unknown}")
    return curriculum


def _coerce_dataset_result(result):
    """Accept old tuple-returning test doubles while loaders migrate."""

    if isinstance(result, MicropatternDataset):
        return result
    data, aux, names, boundary, measurement_mask = result
    return MicropatternDataset(
        data=data,
        aux=aux,
        channel_names=tuple(names),
        boundary_mask=boundary,
        measurement_mask=measurement_mask,
    )


def build_knockout_times(mode, knockout_time, batches):
    if mode is None:
        return [None] * batches
    if mode == "only_one_ko":
        pattern = [knockout_time]
    elif mode == "one_ko_and_baseline":
        pattern = [knockout_time, None]
    elif mode == "both_ko_and_baseline":
        pattern = [0, 24, None]
    elif mode == "only_both_ko":
        pattern = [0, 24]
    else:
        raise ValueError(f"Unknown knockout mode {mode}")
    return [pattern[i % len(pattern)] for i in range(batches)]


def expand_channel_timestep_mask_for_loss(
    data_config, channel_timestep_mask, channel_schema=None
):
    mask = jnp.asarray(channel_timestep_mask)
    if (
        data_config.dataset != "micropatterns_260726"
        and data_config.micropattern.data_channels == 12
        and mask.shape[-1] == 9
    ):
        mask = jnp.concatenate(
            [
                mask[..., 0:4],
                mask[..., 0:3],
                mask[..., 4:8],
                mask[..., 8:9],
            ],
            axis=-1,
        )
    return mask


def build_data_augmenter(
    data_config,
    total_iterations,
    data,
    model_channels,
    channel_timestep_mask=None,
    channel_schema=None,
    intervention_times=None,
    observation_times=None,
):
    """Build the micropattern augmenter for ``data`` [batch, time, measurement, H, W].

    ``observation_times`` (hours) map knockout times to transition slots for
    non-uniform interval schedules; ``None`` keeps the 12-hour convention.
    Returns the augmenter and its short run-name string.
    """
    micropattern = data_config.micropattern
    reinjection_options = dict(
        noise_strength=micropattern.noise_strength,
        reinjection_probability=micropattern.intermediate_reinjection_probability,
        reinjection_probability_end=micropattern.intermediate_reinjection_probability_end,
        reinjection_decay_start_fraction=micropattern.intermediate_reinjection_decay_start_fraction,
        total_iterations=total_iterations,
    )
    if data_config.dataset == "micropatterns_260726":
        if (
            intervention_times is not None
            and "NODAL" not in channel_schema.state_channels
        ):
            raise ValueError(
                "Knockout curricula require NODAL in the selected state schema"
            )
        augmenter = MicropatternAugmenter(
            data,
            channel_schema,
            model_channels,
            reinjection="group",
            measurement_mask=channel_timestep_mask,
            intervention_times=intervention_times,
            **reinjection_options,
        )
        return augmenter, (
            f"da_snapshot_noise{micropattern.noise_strength}"
            f"_irp{micropattern.intermediate_reinjection_probability}"
        )

    # Older colony datasets: reinjection masked by measured channels, with
    # the NODAL knockouts set by data.knockout.mode / data.knockout.time.
    data_channels = micropattern.data_channels
    if data_channels == 4 and data_config.knockout.mode is not None:
        raise ValueError("data.micropattern.data_channels=4 is only supported for no-knockout group-A data.")
    if data_channels == 4:
        schema = MICROPATTERN_4CH_SCHEMA
    elif data_channels == 12:
        schema = MICROPATTERN_GROUPED_12CH_SCHEMA
    else:
        raise ValueError(f"Unsupported data.micropattern.data_channels={data_channels}. Expected 4 or 12.")
    batch_count = len(data)
    augmenter = MicropatternAugmenter(
        data,
        schema,
        model_channels,
        reinjection="masked",
        measurement_mask=channel_timestep_mask,
        knockout_times=build_knockout_times(
            data_config.knockout.mode, data_config.knockout.time, batch_count
        ),
        observation_times=observation_times,
        **reinjection_options,
    )
    cfg_str = (
        f"da_ko{_compact_value(data_config.knockout.mode)}"
        f"_kot{_compact_value(data_config.knockout.time)}"
        f"_noise{micropattern.noise_strength}"
        f"_irp{micropattern.intermediate_reinjection_probability}"
        f"-{micropattern.intermediate_reinjection_probability_end}"
    )
    return augmenter, cfg_str


def load_train_validation_data(data_config, impath=None):
    """Load disjoint physical-replicate splits with train-fitted scaling."""

    configured_train = data_config.micropattern.train_replicates
    configured_validation = data_config.micropattern.validation_replicates
    train_replicates = (
        tuple(range(1, data_config.batches + 1))
        if configured_train is None
        else tuple(configured_train)
    )
    validation_replicates = (
        () if configured_validation is None else tuple(configured_validation)
    )
    if not train_replicates or min(train_replicates) < 1:
        raise ValueError("train_replicates must contain positive, one-based IDs")
    if len(set(train_replicates)) != len(train_replicates):
        raise ValueError("train_replicates cannot contain duplicates")
    if validation_replicates and min(validation_replicates) < 1:
        raise ValueError(
            "validation_replicates must contain positive, one-based IDs"
        )
    if len(set(validation_replicates)) != len(validation_replicates):
        raise ValueError("validation_replicates cannot contain duplicates")
    overlap = set(train_replicates) & set(validation_replicates)
    if overlap:
        raise ValueError(
            f"Training and validation replicates overlap: {sorted(overlap)}"
        )
    if data_config.dataset != "micropatterns_260726" and validation_replicates:
        raise ValueError(
            "Replicate-held-out validation is currently supported only for "
            "micropatterns_260726"
        )

    histogram_bins = None
    curriculum = resolve_knockout_curriculum(data_config.knockout)
    if (
        data_config.dataset == "micropatterns_260726"
        and curriculum != ("baseline",)
        and _reuses_control_bounds(data_config)
    ):
        baseline_config = replace(
            data_config,
            knockout=replace(data_config.knockout, curriculum=("baseline",)),
        )
        baseline = load_data(
            baseline_config,
            impath,
            replicate_indices=tuple(value - 1 for value in train_replicates),
            pool_copies_override=1,
        )
        histogram_bins = baseline[1]["histogram_bins"]

    train = load_data(
        data_config,
        impath,
        replicate_indices=tuple(value - 1 for value in train_replicates),
        histogram_bins=histogram_bins,
    )
    if not validation_replicates:
        return train, None
    validation = load_data(
        data_config,
        impath,
        replicate_indices=tuple(value - 1 for value in validation_replicates),
        histogram_bins=train[1]["histogram_bins"],
        pool_copies_override=1,
    )
    return train, validation


def load_data(
    data_config,
    impath=None,
    *,
    replicate_indices=None,
    histogram_bins=None,
    pool_copies_override=None,
):
    custom_impath = impath is not None
    data_channels = data_config.micropattern.data_channels
    pool_copies = (
        data_config.micropattern.pool_copies
        if pool_copies_override is None
        else pool_copies_override
    )
    if pool_copies <= 0 or int(pool_copies) != pool_copies:
        raise ValueError("data.micropattern.pool_copies must be a positive integer")
    pool_copies = int(pool_copies)
    if data_config.dataset == "micropatterns_260726":
        curriculum = resolve_knockout_curriculum(data_config.knockout)
        if curriculum != ("baseline",) and data_channels != 14:
            raise ValueError("Knockout curricula require the full 14-channel schema")
        if impath is None:
            data_path_base = os.getenv("DATA_PATH_BASE")
            if data_path_base is None:
                raise ValueError("DATA_PATH_BASE must be set when load_data is called without impath.")
            impath = os.path.join(data_path_base, "260726_nca_dataset")
        conditions = tuple(dict.fromkeys(
            CURRICULUM_CONDITIONS[item][0] for item in curriculum
        ))
        dataset = _coerce_dataset_result(load_micropattern_260726(
            impath,
            conditions=conditions,
            timesteps=tuple(data_config.micropattern.timesteps),
            downsample=data_config.downsample,
            replicate_count=data_config.batches,
            replicate_indices=replicate_indices,
            histogram_bins=histogram_bins,
            pool_copies=1,
            experiment_groups=data_config.micropattern.experiment_groups,
            hist_eqs=data_config.micropattern.histogram_percentiles,
            intensity_factors=data_config.micropattern.intensity_factors,
            cleaning=_cleaning_for_loader(data_config.micropattern.cleaning),
            excluded_images=_excluded_images(data_config.micropattern),
            # Local notebooks set DATA_CACHE_DIR to reuse cleaned images.
            cache_dir=cache_dir_from_environment(),
        ))
        data = dataset.data
        aux = dataset.aux
        names = dataset.channel_names
        boundary = dataset.boundary_mask
        mask = dataset.measurement_mask
        replicates_per_condition = data.shape[0] // len(conditions)
        condition_slices = {
            condition: slice(
                index * replicates_per_condition,
                (index + 1) * replicates_per_condition,
            )
            for index, condition in enumerate(conditions)
        }
        selections = [
            condition_slices[CURRICULUM_CONDITIONS[item][0]]
            for item in curriculum
        ]
        selection_indices = tuple(
            batch_index
            for selection in selections
            for batch_index in range(selection.start, selection.stop)
        )
        data = jnp.concatenate(
            [data[selection] for selection in selections], axis=0
        )
        boundary = jnp.concatenate(
            [boundary[selection] for selection in selections], axis=0
        )
        mask = jnp.concatenate(
            [mask[selection] for selection in selections], axis=0
        )
        intervention_times = tuple(
            time
            for item in curriculum
            for time in [CURRICULUM_CONDITIONS[item][1]] * replicates_per_condition
        )
        if pool_copies > 1:
            data = jnp.concatenate([data] * pool_copies, axis=0)
            boundary = jnp.concatenate([boundary] * pool_copies, axis=0)
            mask = jnp.concatenate([mask] * pool_copies, axis=0)
            intervention_times *= pool_copies
        final_indices = selection_indices * pool_copies
        batch_aux_keys = (
            "group_boundary_masks",
            "group_mask",
            "source_conditions",
            "is_substituted",
            "source_files",
            "batch_conditions",
            "batch_replicates",
        )
        for key in batch_aux_keys:
            value = aux.get(key)
            if value is None:
                continue
            if isinstance(value, tuple):
                aux[key] = tuple(value[index] for index in final_indices)
            else:
                aux[key] = value[np.asarray(final_indices)]
        aux["pool_copies"] = pool_copies
        aux["measurement_mask"] = mask
        aux["loss_measurement_mask"] = mask[:, 1:]
        aux["curriculum"] = curriculum
        aux["intervention_times"] = (
            None if curriculum == ("baseline",) else intervention_times
        )
        selected_schema = dataset.schema
        selected_channel_count = getattr(
            selected_schema, "n_measurement_channels", data.shape[2]
        )
        if data_channels is not None and data_channels != selected_channel_count:
            raise ValueError(
                "data.micropattern.data_channels does not match the selected 260726 "
                f"experiment groups ({data_channels} != "
                f"{selected_channel_count})"
            )
        group_str = _compact_value(
            list(getattr(selected_schema, "group_names", ()))
        )
        cfg_str = (
            f"data_b{data_config.batches}"
            f"_c{selected_channel_count}"
            f"_pc{pool_copies}"
            f"_g{group_str}"
            f"_ds{data_config.downsample}"
            f"_ts{_compact_value(list(data_config.micropattern.timesteps))}"
            f"_cur{_compact_value(curriculum)}"
        )
        if custom_impath:
            cfg_str += "_custompath"
        return data, aux, names, boundary, mask[:, 1:], cfg_str
    if data_channels not in {4, 12}:
        raise ValueError(f"Unsupported data.micropattern.data_channels={data_channels}. Expected 4 or 12.")
    if data_channels == 4 and data_config.knockout.mode is not None:
        raise ValueError("data.micropattern.data_channels=4 is only supported for no-knockout group-A data.")

    if impath is None:
        data_path_base = os.getenv("DATA_PATH_BASE")
        if data_path_base is None:
            raise ValueError("DATA_PATH_BASE must be set when load_data is called without impath.")
        impath = data_path_base + "Timecourse_seperate_colonies/"

    if data_config.knockout.mode is None and data_channels == 4:
        dataset = _coerce_dataset_result(load_micropattern_circle_4ch_individual(
            impath=os.path.join(impath, "A/*"),
            BATCHES=data_config.batches,
            DOWNSAMPLE=data_config.downsample,
            TIMESTEPS=list(data_config.micropattern.timesteps),
            PROCESSING_MODES=(
                "map_to_0_1",
                "downsample"
            )
        ))
        data = dataset.data
        aux = dataset.aux
        CHANNEL_NAMES = list(dataset.channel_names)
        boundary_mask = dataset.boundary_mask
        CHANNEL_TIMESTEP_MASK = dataset.measurement_mask
        CHANNEL_NAMES = [
            channel_name if channel_name.startswith("A-") else f"A-{channel_name}"
            for channel_name in CHANNEL_NAMES
        ]
        if len(CHANNEL_TIMESTEP_MASK.shape) == 2:
            CHANNEL_TIMESTEP_MASK = repeat(
                CHANNEL_TIMESTEP_MASK,
                "t c -> b t c",
                b=data_config.batches,
            )
    elif data_config.knockout.mode is None:
        dataset = _coerce_dataset_result(load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=impath,
            FILTER_KN_TIME=data_config.knockout.time,
            BATCHES=data_config.batches,
            DOWNSAMPLE=data_config.downsample,
            TIMESTEPS=list(data_config.micropattern.timesteps),
            PROCESSING_MODES=(
                "map_to_0_1",
                "downsample"
            )
        ))
        data = dataset.data
        aux = dataset.aux
        CHANNEL_NAMES = list(dataset.channel_names)
        boundary_mask = dataset.boundary_mask
        CHANNEL_TIMESTEP_MASK = dataset.measurement_mask
    elif data_config.knockout.mode=="only_one_ko":
        dataset = _coerce_dataset_result(load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=impath,
            FILTER_KN_TIME=data_config.knockout.time,
            BATCHES=data_config.batches,
            DOWNSAMPLE=data_config.downsample,
            TIMESTEPS=list(data_config.micropattern.timesteps),
            PROCESSING_MODES=(
                "map_to_0_1",
                "downsample"
            )
        ))
        data = dataset.data
        aux = dataset.aux
        CHANNEL_NAMES = list(dataset.channel_names)
        boundary_mask = dataset.boundary_mask
        CHANNEL_TIMESTEP_MASK = dataset.measurement_mask
    
    elif data_config.knockout.mode=="one_ko_and_baseline":
        
        dataset_ko = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=impath,
            FILTER_KN_TIME=data_config.knockout.time,
            BATCHES=1,
            DOWNSAMPLE=data_config.downsample,
            TIMESTEPS=list(data_config.micropattern.timesteps),
            PROCESSING_MODES=(
                "map_to_0_1",
                "downsample"
            )
        )
        dataset_base = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=impath,
            FILTER_KN_TIME=None, # type: ignore
            BATCHES=1,
            DOWNSAMPLE=data_config.downsample,
            TIMESTEPS=list(data_config.micropattern.timesteps),
            PROCESSING_MODES=(
                "map_to_0_1",
                "downsample"
            )
        )
        dataset_ko = _coerce_dataset_result(dataset_ko)
        dataset_base = _coerce_dataset_result(dataset_base)
        data_ko = dataset_ko.data
        boundary_mask_ko = dataset_ko.boundary_mask
        CHANNEL_TIMESTEP_MASK_KO = dataset_ko.measurement_mask
        data_base = dataset_base.data
        aux = dataset_base.aux
        CHANNEL_NAMES = list(dataset_base.channel_names)
        boundary_mask_base = dataset_base.boundary_mask
        CHANNEL_TIMESTEP_MASK_BASE = dataset_base.measurement_mask
        data = jnp.concatenate([data_ko,data_base],axis=0)
        boundary_mask = jnp.concatenate([boundary_mask_ko,boundary_mask_base],axis=0)
        CHANNEL_TIMESTEP_MASK = jnp.concatenate([CHANNEL_TIMESTEP_MASK_KO,CHANNEL_TIMESTEP_MASK_BASE],axis=0)
        if data_config.batches>2:
            data = repeat(data,"b ... -> (nb b) ...",nb=math.ceil(data_config.batches/2))[:data_config.batches]
            boundary_mask = repeat(boundary_mask,"b ... -> (nb b) ...",nb=math.ceil(data_config.batches/2))[:data_config.batches]
            CHANNEL_TIMESTEP_MASK = repeat(CHANNEL_TIMESTEP_MASK,"b ... -> (nb b) ...",nb=math.ceil(data_config.batches/2))[:data_config.batches]

    elif data_config.knockout.mode=="both_ko_and_baseline":
        dataset_ko_0 = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=impath,
            FILTER_KN_TIME=0,
            BATCHES=1,
            DOWNSAMPLE=data_config.downsample,
            TIMESTEPS=list(data_config.micropattern.timesteps),
            PROCESSING_MODES=(
                "map_to_0_1",
                "downsample"
            )
        )

        dataset_ko_24 = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=impath,
            FILTER_KN_TIME=24,
            BATCHES=1,
            DOWNSAMPLE=data_config.downsample,
            TIMESTEPS=list(data_config.micropattern.timesteps),
            PROCESSING_MODES=(
                "map_to_0_1",
                "downsample"
            )
        )
        dataset_base = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=impath,
            FILTER_KN_TIME=None, # pyright: ignore[reportArgumentType]
            BATCHES=1,
            DOWNSAMPLE=data_config.downsample,
            TIMESTEPS=list(data_config.micropattern.timesteps),
            PROCESSING_MODES=(
                "map_to_0_1",
                "downsample"
            )
        )

        dataset_ko_0 = _coerce_dataset_result(dataset_ko_0)
        dataset_ko_24 = _coerce_dataset_result(dataset_ko_24)
        dataset_base = _coerce_dataset_result(dataset_base)
        data_ko_0 = dataset_ko_0.data
        boundary_mask_ko_0 = dataset_ko_0.boundary_mask
        CHANNEL_TIMESTEP_MASK_KO_0 = dataset_ko_0.measurement_mask
        data_ko_24 = dataset_ko_24.data
        boundary_mask_ko_24 = dataset_ko_24.boundary_mask
        CHANNEL_TIMESTEP_MASK_KO_24 = dataset_ko_24.measurement_mask
        data_base = dataset_base.data
        aux = dataset_base.aux
        CHANNEL_NAMES = list(dataset_base.channel_names)
        boundary_mask_base = dataset_base.boundary_mask
        CHANNEL_TIMESTEP_MASK_BASE = dataset_base.measurement_mask


        data = jnp.concatenate([data_ko_0,data_ko_24,data_base],axis=0)
        boundary_mask = jnp.concatenate([boundary_mask_ko_0,boundary_mask_ko_24,boundary_mask_base],axis=0)
        CHANNEL_TIMESTEP_MASK = jnp.concatenate([CHANNEL_TIMESTEP_MASK_KO_0,CHANNEL_TIMESTEP_MASK_KO_24,CHANNEL_TIMESTEP_MASK_BASE],axis=0)
        if data_config.batches>3:
            data = repeat(data,"b ... -> (nb b) ...",nb=math.ceil(data_config.batches/3))[:data_config.batches]
            boundary_mask = repeat(boundary_mask,"b ... -> (nb b) ...",nb=math.ceil(data_config.batches/3))[:data_config.batches]
            CHANNEL_TIMESTEP_MASK = repeat(CHANNEL_TIMESTEP_MASK,"b ... -> (nb b) ...",nb=math.ceil(data_config.batches/3))[:data_config.batches]

    elif data_config.knockout.mode=="only_both_ko":
        dataset_ko_0 = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=impath,
            FILTER_KN_TIME=0,
            BATCHES=1,
            DOWNSAMPLE=data_config.downsample,
            TIMESTEPS=list(data_config.micropattern.timesteps),
            PROCESSING_MODES=(
                "map_to_0_1",
                "downsample"
            )
        )

        dataset_ko_24 = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=impath,
            FILTER_KN_TIME=24,
            BATCHES=1,
            DOWNSAMPLE=data_config.downsample,
            TIMESTEPS=list(data_config.micropattern.timesteps),
            PROCESSING_MODES=(
                "map_to_0_1",
                "downsample"
            )
        )

        dataset_ko_0 = _coerce_dataset_result(dataset_ko_0)
        dataset_ko_24 = _coerce_dataset_result(dataset_ko_24)
        data_ko_0 = dataset_ko_0.data
        boundary_mask_ko_0 = dataset_ko_0.boundary_mask
        CHANNEL_TIMESTEP_MASK_KO_0 = dataset_ko_0.measurement_mask
        data_ko_24 = dataset_ko_24.data
        aux = dataset_ko_24.aux
        CHANNEL_NAMES = list(dataset_ko_24.channel_names)
        boundary_mask_ko_24 = dataset_ko_24.boundary_mask
        CHANNEL_TIMESTEP_MASK_KO_24 = dataset_ko_24.measurement_mask

        data = jnp.concatenate([data_ko_0,data_ko_24],axis=0)
        boundary_mask = jnp.concatenate([boundary_mask_ko_0,boundary_mask_ko_24],axis=0)
        CHANNEL_TIMESTEP_MASK = jnp.concatenate([CHANNEL_TIMESTEP_MASK_KO_0,CHANNEL_TIMESTEP_MASK_KO_24],axis=0)
        if data_config.batches>2:
            data = repeat(data,"b ... -> (nb b) ...",nb=math.ceil(data_config.batches/2))[:data_config.batches]
            boundary_mask = repeat(boundary_mask,"b ... -> (nb b) ...",nb=math.ceil(data_config.batches/2))[:data_config.batches]
            CHANNEL_TIMESTEP_MASK = repeat(CHANNEL_TIMESTEP_MASK,"b ... -> (nb b) ...",nb=math.ceil(data_config.batches/2))[:data_config.batches]
    else:
        raise ValueError(f"Unknown knockout mode {data_config.knockout.mode}")
    # if H["knockout"] is not None and H["knockout_mode"]=="both":
    
        # NCA_hyperparameters["FIRE_RATE"]=1.0 # For fine tuning on both WT and KO data, we want to use all the data and not drop any updates randomly, as the dataset is already small.
    
    if data_config.micropattern.duplicate_final_timestep:
        data = jnp.concatenate([data, data[:, -1:]], axis=1)
        if len(CHANNEL_TIMESTEP_MASK.shape) == 2:
            CHANNEL_TIMESTEP_MASK = jnp.concatenate(
                [CHANNEL_TIMESTEP_MASK, CHANNEL_TIMESTEP_MASK[-1:]],
                axis=0,
            )
        elif len(CHANNEL_TIMESTEP_MASK.shape) == 3:
            CHANNEL_TIMESTEP_MASK = jnp.concatenate(
                [CHANNEL_TIMESTEP_MASK, CHANNEL_TIMESTEP_MASK[:, -1:]],
                axis=1,
            )
        else:
            raise ValueError(
                "CHANNEL_TIMESTEP_MASK must have shape [T, C] or [B, T, C] "
                f"when duplicate_final_timestep is enabled. Got {CHANNEL_TIMESTEP_MASK.shape}."
            )

    # Pool cardinality is a data-assembly concern: duplicate every aligned
    # batch component together before handing the result to the augmenter.
    if pool_copies > 1:
        data = jnp.concatenate([data] * pool_copies, axis=0)
        boundary_mask = jnp.concatenate([boundary_mask] * pool_copies, axis=0)
        if CHANNEL_TIMESTEP_MASK.ndim == 3:
            CHANNEL_TIMESTEP_MASK = jnp.concatenate(
                [CHANNEL_TIMESTEP_MASK] * pool_copies, axis=0
            )

    # Data and boundary_mask are [B,T,C,H,W] and [B,1,H,W]. Keep the
    # historical six-pixel border unless a benchmark/experiment explicitly
    # requests aligned spatial dimensions.
    pad_multiple = data_config.micropattern.pad_multiple
    if pad_multiple is None:
        height_padding = (6, 6)
        width_padding = (6, 6)
    else:
        pad_multiple = int(pad_multiple)
        if pad_multiple <= 0:
            raise ValueError("data.micropattern.pad_multiple must be a positive integer or null")

        def aligned_padding(size):
            extra = (-size) % pad_multiple
            return extra // 2, extra - extra // 2

        height_padding = aligned_padding(data.shape[-2])
        width_padding = aligned_padding(data.shape[-1])

    data = jnp.pad(
        data,
        ((0, 0), (0, 0), (0, 0), height_padding, width_padding),
    )
    boundary_mask = jnp.pad(
        boundary_mask,
        ((0, 0), (0, 0), height_padding, width_padding),
    )
    

    cfg_str = (
        f"data_b{data_config.batches}"
        f"_c{data_channels}"
        f"_pc{pool_copies}"
        f"_ds{data_config.downsample}"
        f"_ts{_compact_value(list(data_config.micropattern.timesteps))}"
        f"_ko{_compact_value(data_config.knockout.mode)}"
        f"_kot{_compact_value(data_config.knockout.time)}"
    )
    if custom_impath:
        cfg_str += "_custompath"

    return data,aux,CHANNEL_NAMES,boundary_mask,CHANNEL_TIMESTEP_MASK,cfg_str

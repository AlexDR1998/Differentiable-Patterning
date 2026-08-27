import math
import os
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import repeat

from Common.dataloader.micropattern import (
    load_micropattern_260726,
    load_micropattern_circle_4ch_individual,
    load_micropattern_circle_nodal_knockout_9ch_explicit_colony,
)
from Common.dataloader.results import MicropatternDataset
from Experiments.config_helpers import (
    _compact_value,
    build_loss_filename,
    build_model,
)
from NCA.trainer.data_augmenter.colony_4ch import DataAugmenter as DataAugmenter4Ch
from NCA.trainer.data_augmenter.colony_9ch import DataAugmenter as DataAugmenterGrouped
from NCA.trainer.data_augmenter.micropattern import DataAugmenter as DataAugmenter260726


NODAL_CHANNEL = 7
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


def clamp_nodal(x, intervention_times, nodal_channel, global_batch_indices=None):
    times = jnp.asarray(intervention_times, dtype=jnp.int32)
    if global_batch_indices is not None:
        times = times[jnp.asarray(global_batch_indices)]
    values = list(x)
    for batch, knockout_time in enumerate(times):
        knockout_index = knockout_time // 12
        zero = (knockout_time >= 0) & (
            jnp.arange(values[batch].shape[0]) >= knockout_index
        )
        nodal = jnp.where(
            zero[:, None, None], 0.0, values[batch][:, nodal_channel]
        )
        values[batch] = values[batch].at[:, nodal_channel].set(nodal)
    return values


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


def _as_tree(data):
    if isinstance(data, list):
        return data
    try:
        return [data[i] for i in range(data.shape[0])]
    except AttributeError:
        return list(data)


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


@eqx.filter_jit
def masked_reinject_callback_bit(
    x,
    x_true,
    obs_channels,
    key,
    channel_timestep_mask,
    knockout_times,
    probability,
    global_batch_indices=None,
    global_batch_count=None,
):
    if hasattr(x, "ndim"):
        B, T = x.shape[:2]
        x = x.at[:, 1:].set(x[:, :-1])
        x = x.at[:, 0].set(x_true[:, 0])
        global_B = B if global_batch_count is None else global_batch_count
        indices = jnp.arange(B) if global_batch_indices is None else global_batch_indices
        inject = jax.random.bernoulli(
            key, probability, shape=(global_B, T - 1)
        )[indices, :, None]
        if T > 1:
            measured = channel_timestep_mask[:, : T - 1]
            if measured.shape[2] < obs_channels:
                measured = jnp.pad(
                    measured,
                    ((0, 0), (0, 0), (0, obs_channels - measured.shape[2])),
                )
            mask = (inject & measured[:, :, :obs_channels].astype(bool))[..., None, None]
            observed = jnp.where(
                mask,
                x_true[:, 1:, :obs_channels],
                x[:, 1:, :obs_channels],
            )
            x = x.at[:, 1:, :obs_channels].set(observed)

        knockout_index = knockout_times // 12
        zero_mask = (
            (knockout_times[:, None] >= 0)
            & (jnp.arange(T)[None] >= knockout_index[:, None])
        )
        nodal = jnp.where(zero_mask[..., None, None], 0.0, x[:, :, NODAL_CHANNEL])
        return x.at[:, :, NODAL_CHANNEL].set(nodal)

    propagate_xn = lambda xi: xi.at[1:].set(xi[:-1])
    reset_x0 = lambda xi, xi_true: xi.at[0].set(xi_true[0])

    x = jax.tree_util.tree_map(propagate_xn, x)
    x = jax.tree_util.tree_map(reset_x0, x, x_true)

    B = len(x)
    T = x[0].shape[0]
    global_B = B if global_batch_count is None else global_batch_count
    indices = jnp.arange(B) if global_batch_indices is None else global_batch_indices
    inject_mask = jax.random.bernoulli(
        key, probability, shape=(global_B, T - 1)
    )[indices]

    if T > 1:
        for b in range(B):
            measured = channel_timestep_mask[b, : T - 1]
            if measured.shape[1] < obs_channels:
                measured = jnp.pad(
                    measured,
                    ((0, 0), (0, obs_channels - measured.shape[1])),
                    constant_values=0,
                )
            measured = measured[:, :obs_channels]
            mask = inject_mask[b, :, None] & measured.astype(bool)
            mask = mask[:, :, None, None]
            x_obs = jnp.where(
                mask,
                x_true[b][1:, :obs_channels],
                x[b][1:, :obs_channels],
            )
            x[b] = x[b].at[1:, :obs_channels].set(x_obs)

    for b in range(B):
        knockout_time = knockout_times[b]
        knockout_index = knockout_time // 12
        zero_mask = (knockout_time >= 0) & (jnp.arange(T) >= knockout_index)
        nodal = jnp.where(zero_mask[:, None, None], 0.0, x[b][:, NODAL_CHANNEL])
        x[b] = x[b].at[:, NODAL_CHANNEL].set(nodal)
    return x


def build_data_augmenter(
    data_config,
    total_iterations,
    channel_timestep_mask=None,
    channel_schema=None,
    intervention_times=None,
):
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        data=data_config,
        knockout=data_config.intervention,
        run=SimpleNamespace(iterations=total_iterations),
    )
    data_channels = cfg.data.micropattern.data_channels
    if cfg.data.get("dataset", "micropatterns") == "micropatterns_260726":
        if (
            intervention_times is not None
            and "NODAL" not in channel_schema.state_channels
        ):
            raise ValueError(
                "Knockout curricula require NODAL in the selected state schema"
            )

        class DA_subclass(DataAugmenter260726):
            noise_strength = cfg.data.micropattern.noise_strength

            def __init__(self, *args, **kwargs):
                kwargs["schema"] = channel_schema
                kwargs["intermediate_reinjection_probability"] = cfg.data.micropattern.intermediate_reinjection_probability
                kwargs["intermediate_reinjection_probability_end"] = cfg.data.micropattern.get(
                    "intermediate_reinjection_probability_end",
                    kwargs["intermediate_reinjection_probability"],
                )
                kwargs["intermediate_reinjection_decay_start_fraction"] = cfg.data.micropattern.get(
                    "intermediate_reinjection_decay_start_fraction", 0.25
                )
                kwargs["intermediate_reinjection_total_iterations"] = cfg.run.iterations
                super().__init__(*args, **kwargs)

            def initialize_pool(self, key):
                x, y = super().initialize_pool(key)
                if intervention_times is not None:
                    x = clamp_nodal(
                        x,
                        intervention_times,
                        self.schema.state_channels.index("NODAL"),
                    )
                return x, y

            def advance_pool(self, x, y, i, key):
                x, y = super().advance_pool(x, y, i, key)
                if intervention_times is not None:
                    x = clamp_nodal(
                        x,
                        intervention_times,
                        self.schema.state_channels.index("NODAL"),
                        getattr(self, "_global_batch_indices", None),
                    )
                return x, y

        return DA_subclass, (
            f"da_snapshot_noise{cfg.data.micropattern.noise_strength}"
            f"_irp{cfg.data.micropattern.get('intermediate_reinjection_probability', 0.5)}"
        )
    if data_channels == 4 and cfg.knockout.mode is not None:
        raise ValueError("data.micropattern.data_channels=4 is only supported for no-knockout group-A data.")
    if data_channels == 4:
        data_augmenter_base = DataAugmenter4Ch
    elif data_channels == 12:
        data_augmenter_base = DataAugmenterGrouped
    else:
        raise ValueError(f"Unsupported data.micropattern.data_channels={data_channels}. Expected 4 or 12.")

    if cfg.knockout.mode is not None:
        build_knockout_times(cfg.knockout.mode, cfg.knockout.time, cfg.data.batches)
    
    class DA_subclass(data_augmenter_base):
        supports_global_reinjection_mask = True

        def __init__(self, *args, **kwargs):
            kwargs["intermediate_reinjection_probability"] = cfg.data.micropattern.intermediate_reinjection_probability
            kwargs["intermediate_reinjection_probability_end"] = cfg.data.micropattern.get(
                "intermediate_reinjection_probability_end",
                kwargs["intermediate_reinjection_probability"],
            )
            kwargs["intermediate_reinjection_decay_start_fraction"] = cfg.data.micropattern.get(
                "intermediate_reinjection_decay_start_fraction", 0.25
            )
            kwargs["intermediate_reinjection_total_iterations"] = cfg.run.iterations
            super().__init__(*args, **kwargs)
            if channel_timestep_mask is None:
                mask = jnp.ones(
                    (
                        len(self.data_true),
                        self.data_true[0].shape[0] - 1,
                        self.OBS_CHANNELS,
                    ),
                    dtype=jnp.float32,
                )
            else:
                mask = jnp.asarray(channel_timestep_mask, dtype=jnp.float32)
            self.channel_timestep_mask = mask
            knockout_times = build_knockout_times(
                cfg.knockout.mode,
                cfg.knockout.time,
                len(self.data_true),
            )
            self.knockout_times = jnp.array(
                [-1 if knockout_time is None else knockout_time for knockout_time in knockout_times],
                dtype=jnp.int32,
            )

        def advance_pool(self,x,y,i,key):
            x_true,_ =self.split_x_y(1)	
            reinjection_key = getattr(self, "_sharded_global_key", key)
            x = masked_reinject_callback_bit(
                x,
                x_true,
                self.OBS_CHANNELS,
                jax.random.fold_in(reinjection_key, 0),
                self.channel_timestep_mask,
                self.knockout_times,
                self.reinjection_probability(i),
                getattr(self, "_global_batch_indices", None),
                getattr(self, "_global_batch_count", None),
            )
            x = self.noise(x,cfg.data.micropattern.noise_strength,key=key)
            self.PREVIOUS_KEY = key
            return x,y
    cfg_str = (
        f"da_ko{_compact_value(cfg.knockout.mode)}"
        f"_kot{_compact_value(cfg.knockout.time)}"
        f"_noise{cfg.data.micropattern.noise_strength}"
        f"_irp{cfg.data.micropattern.get('intermediate_reinjection_probability', 0.5)}"
        f"-{cfg.data.micropattern.get('intermediate_reinjection_probability_end', cfg.data.micropattern.get('intermediate_reinjection_probability', 0.5))}"
    )
    return DA_subclass, cfg_str


def load_train_validation_data(data_config, impath=None):
    """Load disjoint physical-replicate splits with train-fitted scaling."""

    configured_train = data_config.micropattern.get("train_replicates", None)
    configured_validation = data_config.micropattern.get(
        "validation_replicates", None
    )
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
    curriculum = resolve_knockout_curriculum(data_config.intervention)
    if data_config.dataset == "micropatterns_260726" and curriculum != ("baseline",):
        baseline_config = replace(
            data_config,
            intervention=replace(data_config.intervention, curriculum=("baseline",)),
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
    from types import SimpleNamespace

    cfg = SimpleNamespace(data=data_config, knockout=data_config.intervention)
    custom_impath = impath is not None
    data_channels = cfg.data.micropattern.data_channels
    pool_copies = (
        cfg.data.micropattern.get("pool_copies", 1)
        if pool_copies_override is None
        else pool_copies_override
    )
    if pool_copies <= 0 or int(pool_copies) != pool_copies:
        raise ValueError("data.micropattern.pool_copies must be a positive integer")
    pool_copies = int(pool_copies)
    if cfg.data.get("dataset", "micropatterns") == "micropatterns_260726":
        curriculum = resolve_knockout_curriculum(cfg.knockout)
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
            timesteps=tuple(cfg.data.micropattern.timesteps),
            downsample=cfg.data.downsample,
            replicate_count=cfg.data.batches,
            replicate_indices=replicate_indices,
            histogram_bins=histogram_bins,
            pool_copies=1,
            experiment_groups=cfg.data.micropattern.get("experiment_groups", None),
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
            f"data_b{cfg.data.batches}"
            f"_c{selected_channel_count}"
            f"_pc{pool_copies}"
            f"_g{group_str}"
            f"_ds{cfg.data.downsample}"
            f"_ts{_compact_value(list(cfg.data.micropattern.timesteps))}"
            f"_cur{_compact_value(curriculum)}"
        )
        if custom_impath:
            cfg_str += "_custompath"
        return data, aux, names, boundary, mask[:, 1:], cfg_str
    if data_channels not in {4, 12}:
        raise ValueError(f"Unsupported data.micropattern.data_channels={data_channels}. Expected 4 or 12.")
    if data_channels == 4 and cfg.knockout.mode is not None:
        raise ValueError("data.micropattern.data_channels=4 is only supported for no-knockout group-A data.")

    if impath is None:
        data_path_base = os.getenv("DATA_PATH_BASE")
        if data_path_base is None:
            raise ValueError("DATA_PATH_BASE must be set when load_data is called without impath.")
        impath = data_path_base + "Timecourse_seperate_colonies/"

    if cfg.knockout.mode is None and data_channels == 4:
        dataset = _coerce_dataset_result(load_micropattern_circle_4ch_individual(
            impath=os.path.join(impath, "A/*"),
            BATCHES=cfg.data.batches,
            DOWNSAMPLE=cfg.data.downsample,
            TIMESTEPS=list(cfg.data.micropattern.timesteps),
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
                b=cfg.data.batches,
            )
    elif cfg.knockout.mode is None:
        dataset = _coerce_dataset_result(load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=impath,
            FILTER_KN_TIME=cfg.knockout.time,
            BATCHES=cfg.data.batches,
            DOWNSAMPLE=cfg.data.downsample,
            TIMESTEPS=list(cfg.data.micropattern.timesteps),
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
    elif cfg.knockout.mode=="only_one_ko":
        dataset = _coerce_dataset_result(load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=impath,
            FILTER_KN_TIME=cfg.knockout.time,
            BATCHES=cfg.data.batches,
            DOWNSAMPLE=cfg.data.downsample,
            TIMESTEPS=list(cfg.data.micropattern.timesteps),
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
    
    elif cfg.knockout.mode=="one_ko_and_baseline":
        
        dataset_ko = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=impath,
            FILTER_KN_TIME=cfg.knockout.time,
            BATCHES=1,
            DOWNSAMPLE=cfg.data.downsample,
            TIMESTEPS=list(cfg.data.micropattern.timesteps),
            PROCESSING_MODES=(
                "map_to_0_1",
                "downsample"
            )
        )
        dataset_base = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=impath,
            FILTER_KN_TIME=None, # type: ignore
            BATCHES=1,
            DOWNSAMPLE=cfg.data.downsample,
            TIMESTEPS=list(cfg.data.micropattern.timesteps),
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
        if cfg.data.batches>2:
            data = repeat(data,"b ... -> (nb b) ...",nb=math.ceil(cfg.data.batches/2))[:cfg.data.batches]
            boundary_mask = repeat(boundary_mask,"b ... -> (nb b) ...",nb=math.ceil(cfg.data.batches/2))[:cfg.data.batches]
            CHANNEL_TIMESTEP_MASK = repeat(CHANNEL_TIMESTEP_MASK,"b ... -> (nb b) ...",nb=math.ceil(cfg.data.batches/2))[:cfg.data.batches]

    elif cfg.knockout.mode=="both_ko_and_baseline":
        dataset_ko_0 = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=impath,
            FILTER_KN_TIME=0,
            BATCHES=1,
            DOWNSAMPLE=cfg.data.downsample,
            TIMESTEPS=list(cfg.data.micropattern.timesteps),
            PROCESSING_MODES=(
                "map_to_0_1",
                "downsample"
            )
        )

        dataset_ko_24 = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=impath,
            FILTER_KN_TIME=24,
            BATCHES=1,
            DOWNSAMPLE=cfg.data.downsample,
            TIMESTEPS=list(cfg.data.micropattern.timesteps),
            PROCESSING_MODES=(
                "map_to_0_1",
                "downsample"
            )
        )
        dataset_base = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=impath,
            FILTER_KN_TIME=None, # pyright: ignore[reportArgumentType]
            BATCHES=1,
            DOWNSAMPLE=cfg.data.downsample,
            TIMESTEPS=list(cfg.data.micropattern.timesteps),
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
        if cfg.data.batches>3:
            data = repeat(data,"b ... -> (nb b) ...",nb=math.ceil(cfg.data.batches/3))[:cfg.data.batches]
            boundary_mask = repeat(boundary_mask,"b ... -> (nb b) ...",nb=math.ceil(cfg.data.batches/3))[:cfg.data.batches]
            CHANNEL_TIMESTEP_MASK = repeat(CHANNEL_TIMESTEP_MASK,"b ... -> (nb b) ...",nb=math.ceil(cfg.data.batches/3))[:cfg.data.batches]

    elif cfg.knockout.mode=="only_both_ko":
        dataset_ko_0 = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=impath,
            FILTER_KN_TIME=0,
            BATCHES=1,
            DOWNSAMPLE=cfg.data.downsample,
            TIMESTEPS=list(cfg.data.micropattern.timesteps),
            PROCESSING_MODES=(
                "map_to_0_1",
                "downsample"
            )
        )

        dataset_ko_24 = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=impath,
            FILTER_KN_TIME=24,
            BATCHES=1,
            DOWNSAMPLE=cfg.data.downsample,
            TIMESTEPS=list(cfg.data.micropattern.timesteps),
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
        if cfg.data.batches>2:
            data = repeat(data,"b ... -> (nb b) ...",nb=math.ceil(cfg.data.batches/2))[:cfg.data.batches]
            boundary_mask = repeat(boundary_mask,"b ... -> (nb b) ...",nb=math.ceil(cfg.data.batches/2))[:cfg.data.batches]
            CHANNEL_TIMESTEP_MASK = repeat(CHANNEL_TIMESTEP_MASK,"b ... -> (nb b) ...",nb=math.ceil(cfg.data.batches/2))[:cfg.data.batches]
    else:
        raise ValueError(f"Unknown knockout mode {cfg.knockout.mode}")
    # if H["knockout"] is not None and H["knockout_mode"]=="both":
    
        # NCA_hyperparameters["FIRE_RATE"]=1.0 # For fine tuning on both WT and KO data, we want to use all the data and not drop any updates randomly, as the dataset is already small.
    
    if cfg.data.micropattern.get("duplicate_final_timestep", False):
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
    pad_multiple = cfg.data.micropattern.get("pad_multiple", None)
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
        f"data_b{cfg.data.batches}"
        f"_c{data_channels}"
        f"_pc{pool_copies}"
        f"_ds{cfg.data.downsample}"
        f"_ts{_compact_value(list(cfg.data.micropattern.timesteps))}"
        f"_ko{_compact_value(cfg.knockout.mode)}"
        f"_kot{_compact_value(cfg.knockout.time)}"
    )
    if custom_impath:
        cfg_str += "_custompath"

    return data,aux,CHANNEL_NAMES,boundary_mask,CHANNEL_TIMESTEP_MASK,cfg_str

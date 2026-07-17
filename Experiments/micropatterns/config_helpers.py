import math
import os

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import repeat

from Common.dataloader.micropattern import (
    load_micropattern_circle_4ch_individual,
    load_micropattern_circle_nodal_knockout_9ch_explicit_colony,
)
from Experiments.config_helpers import (
    _compact_value,
    build_loss_filename,
    build_model,
)
from NCA.trainer.data_augmenter_4ch_colony import DataAugmenter as DataAugmenter4Ch
from NCA.trainer.data_augmenter_9ch_colony import DataAugmenter as DataAugmenterGrouped


NODAL_CHANNEL = 7


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


def expand_channel_timestep_mask_for_loss(cfg, channel_timestep_mask):
    mask = jnp.asarray(channel_timestep_mask)
    if cfg.data.data_channels == 12 and mask.shape[-1] == 9:
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
    n_inject=None,
):
    propagate_xn = lambda xi: xi.at[1:].set(xi[:-1])
    reset_x0 = lambda xi, xi_true: xi.at[0].set(xi_true[0])

    x = jax.tree_util.tree_map(propagate_xn, x)
    x = jax.tree_util.tree_map(reset_x0, x, x_true)

    B = len(x)
    T = x[0].shape[0]
    N_ELIGIBLE = B * (T - 1)
    N_INJECT = N_ELIGIBLE // 2 if n_inject is None else n_inject

    if N_INJECT > 0:
        scores = jax.random.uniform(key, shape=(N_ELIGIBLE,))
        inject_inds = jnp.argsort(scores)[:N_INJECT]
        inject_mask = jnp.zeros((N_ELIGIBLE,), dtype=bool).at[inject_inds].set(True)
        inject_mask = inject_mask.reshape((B, T - 1))

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


def build_data_augmenter(cfg, channel_timestep_mask=None):
    data_channels = cfg.data.data_channels
    if data_channels == 4 and cfg.knockout.mode is not None:
        raise ValueError("data.data_channels=4 is only supported for no-knockout group-A data.")
    if data_channels == 4:
        data_augmenter_base = DataAugmenter4Ch
    elif data_channels == 12:
        data_augmenter_base = DataAugmenterGrouped
    else:
        raise ValueError(f"Unsupported data.data_channels={data_channels}. Expected 4 or 12.")

    if cfg.knockout.mode is not None:
        build_knockout_times(cfg.knockout.mode, cfg.knockout.time, cfg.data.batches)
    
    class DA_subclass(data_augmenter_base):
        supports_sharded_inject_count = True

        def __init__(self, *args, **kwargs):
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

        def data_callback(self,x,y,i,key):
            x_true,_ =self.split_x_y(1)	
            x = masked_reinject_callback_bit(
                _as_tree(x),
                _as_tree(x_true),
                self.OBS_CHANNELS,
                jax.random.fold_in(key, 0),
                self.channel_timestep_mask,
                self.knockout_times,
                getattr(self, "_sharded_n_inject", None),
            )
            x = self.noise(x,cfg.data.noise_strength,key=key)
            self.PREVIOUS_KEY = key
            return x,y
    cfg_str = (
        f"da_ko{_compact_value(cfg.knockout.mode)}"
        f"_kot{_compact_value(cfg.knockout.time)}"
        f"_noise{cfg.data.noise_strength}"
    )
    return DA_subclass, cfg_str


def load_data(cfg, impath=None):
    custom_impath = impath is not None
    data_channels = cfg.data.data_channels
    if data_channels not in {4, 12}:
        raise ValueError(f"Unsupported data.data_channels={data_channels}. Expected 4 or 12.")
    if data_channels == 4 and cfg.knockout.mode is not None:
        raise ValueError("data.data_channels=4 is only supported for no-knockout group-A data.")

    if impath is None:
        data_path_base = os.getenv("DATA_PATH_BASE")
        if data_path_base is None:
            raise ValueError("DATA_PATH_BASE must be set when load_data is called without impath.")
        impath = data_path_base + "Timecourse_seperate_colonies/"

    if cfg.knockout.mode is None and data_channels == 4:
        data,aux,CHANNEL_NAMES,boundary_mask,CHANNEL_TIMESTEP_MASK = load_micropattern_circle_4ch_individual(
            impath=os.path.join(impath, "A/*"),
            BATCHES=cfg.data.batches,
            DOWNSAMPLE=cfg.data.downsample,
            TIMESTEPS=list(cfg.data.timesteps),
            PROCESSING_MODES={
                "map_to_0_1",
                "downsample"
            }
        )
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
        data,aux,CHANNEL_NAMES,boundary_mask,CHANNEL_TIMESTEP_MASK = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=impath,
            FILTER_KN_TIME=cfg.knockout.time,
            BATCHES=cfg.data.batches,
            DOWNSAMPLE=cfg.data.downsample,
            TIMESTEPS=list(cfg.data.timesteps),
            PROCESSING_MODES={
                "map_to_0_1",
                "downsample"
            }
        )
    elif cfg.knockout.mode=="only_one_ko":
        data,aux,CHANNEL_NAMES,boundary_mask,CHANNEL_TIMESTEP_MASK = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=impath,
            FILTER_KN_TIME=cfg.knockout.time,
            BATCHES=cfg.data.batches,
            DOWNSAMPLE=cfg.data.downsample,
            TIMESTEPS=list(cfg.data.timesteps),
            PROCESSING_MODES={
                "map_to_0_1",
                "downsample"
            }
        )
    
    elif cfg.knockout.mode=="one_ko_and_baseline":
        
        data_ko,aux,CHANNEL_NAMES,boundary_mask_ko,CHANNEL_TIMESTEP_MASK_KO = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=impath,
            FILTER_KN_TIME=cfg.knockout.time,
            BATCHES=1,
            DOWNSAMPLE=cfg.data.downsample,
            TIMESTEPS=list(cfg.data.timesteps),
            PROCESSING_MODES={
                "map_to_0_1",
                "downsample"
            }
        )
        data_base,aux,CHANNEL_NAMES,boundary_mask_base,CHANNEL_TIMESTEP_MASK_BASE = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=impath,
            FILTER_KN_TIME=None, # type: ignore
            BATCHES=1,
            DOWNSAMPLE=cfg.data.downsample,
            TIMESTEPS=list(cfg.data.timesteps),
            PROCESSING_MODES={
                "map_to_0_1",
                "downsample"
            }
        )
        data = jnp.concatenate([data_ko,data_base],axis=0)
        boundary_mask = jnp.concatenate([boundary_mask_ko,boundary_mask_base],axis=0)
        CHANNEL_TIMESTEP_MASK = jnp.concatenate([CHANNEL_TIMESTEP_MASK_KO,CHANNEL_TIMESTEP_MASK_BASE],axis=0)
        if cfg.data.batches>2:
            data = repeat(data,"b ... -> (nb b) ...",nb=math.ceil(cfg.data.batches/2))[:cfg.data.batches]
            boundary_mask = repeat(boundary_mask,"b ... -> (nb b) ...",nb=math.ceil(cfg.data.batches/2))[:cfg.data.batches]
            CHANNEL_TIMESTEP_MASK = repeat(CHANNEL_TIMESTEP_MASK,"b ... -> (nb b) ...",nb=math.ceil(cfg.data.batches/2))[:cfg.data.batches]

    elif cfg.knockout.mode=="both_ko_and_baseline":
        data_ko_0,aux,CHANNEL_NAMES,boundary_mask_ko_0,CHANNEL_TIMESTEP_MASK_KO_0 = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=impath,
            FILTER_KN_TIME=0,
            BATCHES=1,
            DOWNSAMPLE=cfg.data.downsample,
            TIMESTEPS=list(cfg.data.timesteps),
            PROCESSING_MODES={
                "map_to_0_1",
                "downsample"
            }
        )

        data_ko_24,aux,CHANNEL_NAMES,boundary_mask_ko_24,CHANNEL_TIMESTEP_MASK_KO_24 = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=impath,
            FILTER_KN_TIME=24,
            BATCHES=1,
            DOWNSAMPLE=cfg.data.downsample,
            TIMESTEPS=list(cfg.data.timesteps),
            PROCESSING_MODES={
                "map_to_0_1",
                "downsample"
            }
        )
        data_base,aux,CHANNEL_NAMES,boundary_mask_base,CHANNEL_TIMESTEP_MASK_BASE = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=impath,
            FILTER_KN_TIME=None, # pyright: ignore[reportArgumentType]
            BATCHES=1,
            DOWNSAMPLE=cfg.data.downsample,
            TIMESTEPS=list(cfg.data.timesteps),
            PROCESSING_MODES={
                "map_to_0_1",
                "downsample"
            }
        )


        data = jnp.concatenate([data_ko_0,data_ko_24,data_base],axis=0)
        boundary_mask = jnp.concatenate([boundary_mask_ko_0,boundary_mask_ko_24,boundary_mask_base],axis=0)
        CHANNEL_TIMESTEP_MASK = jnp.concatenate([CHANNEL_TIMESTEP_MASK_KO_0,CHANNEL_TIMESTEP_MASK_KO_24,CHANNEL_TIMESTEP_MASK_BASE],axis=0)
        if cfg.data.batches>3:
            data = repeat(data,"b ... -> (nb b) ...",nb=math.ceil(cfg.data.batches/3))[:cfg.data.batches]
            boundary_mask = repeat(boundary_mask,"b ... -> (nb b) ...",nb=math.ceil(cfg.data.batches/3))[:cfg.data.batches]
            CHANNEL_TIMESTEP_MASK = repeat(CHANNEL_TIMESTEP_MASK,"b ... -> (nb b) ...",nb=math.ceil(cfg.data.batches/3))[:cfg.data.batches]

    elif cfg.knockout.mode=="only_both_ko":
        data_ko_0,aux,CHANNEL_NAMES,boundary_mask_ko_0,CHANNEL_TIMESTEP_MASK_KO_0 = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=impath,
            FILTER_KN_TIME=0,
            BATCHES=1,
            DOWNSAMPLE=cfg.data.downsample,
            TIMESTEPS=list(cfg.data.timesteps),
            PROCESSING_MODES={
                "map_to_0_1",
                "downsample"
            }
        )

        data_ko_24,aux,CHANNEL_NAMES,boundary_mask_ko_24,CHANNEL_TIMESTEP_MASK_KO_24 = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=impath,
            FILTER_KN_TIME=24,
            BATCHES=1,
            DOWNSAMPLE=cfg.data.downsample,
            TIMESTEPS=list(cfg.data.timesteps),
            PROCESSING_MODES={
                "map_to_0_1",
                "downsample"
            }
        )

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
    
    if cfg.data.get("duplicate_final_timestep", False):
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

    #Data and boundary_mask is of size [B,T,C,W,H].
    # W and H are 500, we want to pad them to 512.
    data = jnp.pad(data,((0,0),(0,0),(0,0),(6,6),(6,6)))
    boundary_mask = jnp.pad(boundary_mask,((0,0),(0,0),(6,6),(6,6)))
    

    cfg_str = (
        f"data_b{cfg.data.batches}"
        f"_c{data_channels}"
        f"_ds{cfg.data.downsample}"
        f"_ts{_compact_value(list(cfg.data.timesteps))}"
        f"_ko{_compact_value(cfg.knockout.mode)}"
        f"_kot{_compact_value(cfg.knockout.time)}"
    )
    if custom_impath:
        cfg_str += "_custompath"

    return data,aux,CHANNEL_NAMES,boundary_mask,CHANNEL_TIMESTEP_MASK,cfg_str

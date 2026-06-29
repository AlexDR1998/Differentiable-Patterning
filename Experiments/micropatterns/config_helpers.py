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
from NCA.trainer.data_augmenter_nca_basic import (
    jittable_callback_bit as baseline_jittable_callback_bit,
)
from NCA.trainer.data_augmenter_4ch_colony import DataAugmenter as DataAugmenter4Ch
from NCA.trainer.data_augmenter_9ch_colony import DataAugmenter as DataAugmenterGrouped


def build_data_augmenter(cfg):
    data_channels = cfg.data.data_channels
    if data_channels == 4 and cfg.knockout.mode is not None:
        raise ValueError("data.data_channels=4 is only supported for no-knockout group-A data.")
    if data_channels == 4:
        data_augmenter_base = DataAugmenter4Ch
    elif data_channels == 12:
        data_augmenter_base = DataAugmenterGrouped
    else:
        raise ValueError(f"Unsupported data.data_channels={data_channels}. Expected 4 or 12.")

    if cfg.knockout.mode is None:
        jittable_callback_bit = baseline_jittable_callback_bit
        
        
    else:
        # _KNOCKOUT = H["knockout"]//12
        _KNOCKOUT = cfg.knockout.time//12 # Convert knockout time in hours to index (assuming 12h between each timepoint)
        if cfg.knockout.mode=="only_one_ko":
            @eqx.filter_jit
            def jittable_callback_bit(x,x_true,OBS_CHANNELS): # pyright: ignore[reportRedeclaration]
                propagate_xn = lambda x:x.at[1:].set(x[:-1])
                reset_x0 = lambda x,x_true:x.at[0].set(x_true[0])
                knockout_nodal = lambda x:x.at[_KNOCKOUT:,7].set(0.0) # Set nodal channel to 0 at and after knockout time
                x = jax.tree_util.tree_map(propagate_xn,x) # Set initial condition at each X[n] at next iteration to be final state from X[n-1] of this iteration
                x = jax.tree_util.tree_map(reset_x0,x,x_true) # Keep first initial x correct
                x = jax.tree_util.tree_map(knockout_nodal,x)
                
                return x
        elif cfg.knockout.mode=="one_ko_and_baseline":
            @eqx.filter_jit
            def jittable_callback_bit(x,x_true,OBS_CHANNELS): # pyright: ignore[reportRedeclaration]
                propagate_xn = lambda x:x.at[1:].set(x[:-1])
                reset_x0 = lambda x,x_true:x.at[0].set(x_true[0])
                # knockout_nodal = lambda x:x.at[_KNOCKOUT:,7].set(0.0) # Set nodal channel to 0 at and after knockout time
                x = jax.tree_util.tree_map(propagate_xn,x) 
                x = jax.tree_util.tree_map(reset_x0,x,x_true)
                
                for b in range(len(x)//3):
                    x[b*3] = x[b*3].at[_KNOCKOUT:,7].set(0.0) # Set nodal channel to 0 at and after knockout time for every even batch
                    x[b*3+1] = x[b*3+1].at[:,:OBS_CHANNELS].set(x_true[b*3+1][:,:OBS_CHANNELS]) # Set every other batch of intermediate initial conditions to correct initial conditions
                
                return x
        elif cfg.knockout.mode=="both_ko_and_baseline":
            @eqx.filter_jit
            def jittable_callback_bit(x,x_true,OBS_CHANNELS): # pyright: ignore[reportRedeclaration]
                # Here we only want 9 channels - no duplicates - as this is what the NCA sees.
                propagate_xn = lambda x:x.at[1:].set(x[:-1])
                reset_x0 = lambda x,x_true:x.at[0].set(x_true[0])
                x = jax.tree_util.tree_map(propagate_xn,x) # Set initial condition at each X[n] at next iteration to be final state from X[n-1] of this iteration
                x = jax.tree_util.tree_map(reset_x0,x,x_true) # Keep first initial x correct
                # x[0] = x[0].at[0:,7].set(0.0) # 0h nodal knockout batch
                # x[1] = x[1].at[2:,7].set(0.0) # 24h nodal knockout batch
                for b in range(len(x)//4):
                    x[b*4] = x[b*4].at[0:,7].set(0.0) # Set nodal channel to 0 at and after knockout time for every even batch
                    x[b*4+1] = x[b*4+1].at[2:,7].set(0.0) # Set nodal channel to 0 at and after knockout time for every odd batch
                    x[b*4+2] = x[b*4+2].at[:,:OBS_CHANNELS].set(x_true[b*4+2][:,:OBS_CHANNELS]) # Set every other batch of intermediate initial conditions to correct initial conditions
                return x
        elif cfg.knockout.mode=="only_both_ko":
            @eqx.filter_jit
            def jittable_callback_bit(x,x_true,OBS_CHANNELS): # pyright: ignore[reportRedeclaration]
                propagate_xn = lambda x:x.at[1:].set(x[:-1])
                reset_x0 = lambda x,x_true:x.at[0].set(x_true[0])
                x = jax.tree_util.tree_map(propagate_xn,x) # Set initial condition at each X[n] at next iteration to be final state from X[n-1] of this iteration
                x = jax.tree_util.tree_map(reset_x0,x,x_true) # Keep first initial x correct
                for b in range(len(x)//2):
                    x[b*2] = x[b*2].at[0:,7].set(0.0) # Set nodal channel to 0 at and after knockout time for every even batch
                    x[b*2+1] = x[b*2+1].at[2:,7].set(0.0) # Set nodal channel to 0 at and after knockout time for every odd batch
                return x
        else:
            raise ValueError(f"Unknown knockout mode {cfg.knockout.mode}")
    
    class DA_subclass(data_augmenter_base):
        def data_callback(self,x,y,i,key):
            x_true,_ =self.split_x_y(1)	
            if cfg.knockout.mode is None:
                x = jittable_callback_bit(
                    x,
                    x_true,
                    self.OBS_CHANNELS,
                    jax.random.fold_in(key, 0),
                )
            else:
                x = jittable_callback_bit(x,x_true,self.OBS_CHANNELS)
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

import sys
import os
import jax
jax.config.update("jax_default_matmul_precision", "tensorfloat32" ) # Decent speedup on H100
import jax.numpy as jnp
import equinox as eqx
import optax
import math
from dotenv import load_dotenv
load_dotenv()
CODE_PATH = os.getenv("PVC_PATH")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH")
DATA_PATH = os.getenv("DATA_PATH_BASE") + "Timecourse_seperate_colonies/" # type: ignore
sys.path.append(CODE_PATH)  # type: ignore
os.chdir(CODE_PATH) # type: ignore

from einops import rearrange,repeat,reduce
from NCA.model.NCA_model import NCA
from NCA.model.NCA_upsample_isotropic_model import uNCA as isouNCA
from NCA.model.NCA_upsample_model import uNCA
from NCA.trainer.NCA_trainer import NCA_Trainer
from NCA.trainer.optimizer import build_optimizer
from NCA.trainer.data_augmenter_9ch_colony import DataAugmenter as DataAugmenterGrouped
from Experiments.config_helpers import build_tags

from Common.dataloader.micropattern import load_micropattern_circle_nodal_knockout_9ch_explicit_colony

def build_model(cfg, key=None):
    if cfg.model.family == "NCA":
        model = NCA(
            N_CHANNELS=cfg.model.channels,
            KERNEL_STR=cfg.model.kernel_str,
            FIRE_RATE=cfg.model.fire_rate,
            PADDING=cfg.model.padding,
            key=key,
        )
    elif cfg.model.family == "uNCA":
        model = uNCA(
            N_CHANNELS=cfg.model.channels,
            O_CHANNELS=cfg.data.data_channels,
            KERNEL_STR=cfg.model.kernel_str,
            FIRE_RATE=cfg.model.fire_rate,
            PADDING=cfg.model.padding,
            # SPATIAL_UPSAMPLE = cfg.model.upscale_factor,
            UPSAMPLER_AUX = {
                "depth": cfg.model.upsampler.depth,
                "width_factor": cfg.model.upsampler.width_factor,
                "fourier_modes" : cfg.model.upsampler.fourier_modes,
                "upsample_factor": cfg.model.upscale_factor
            },
            key=key,
            
        )
    elif cfg.model.family == "isouNCA":
        model = isouNCA(
            N_CHANNELS=cfg.model.channels,
            O_CHANNELS=cfg.data.data_channels,
            KERNEL_STR=cfg.model.kernel_str,
            FIRE_RATE=cfg.model.fire_rate,
            PADDING=cfg.model.padding,
            # SPATIAL_UPSAMPLE = cfg.model.upscale_factor,
            # RADIUS=cfg.model.upsampler.radius
            UPSAMPLER_AUX = {
                "depth": cfg.model.upsampler.depth,
                "width_factor": cfg.model.upsampler.width_factor,
                "radius" : cfg.model.upsampler.radius,
                "upsample_factor": cfg.model.upscale_factor
            },
            key=key,
        )
    else:
        raise ValueError(f"Unknown model family {cfg.model.family}")
    
    if cfg.knockout.mode is not None:
        model_path = build_filename(cfg)
        model = eqx.tree_deserialise_leaves(model_path, model)
    
    return model

def build_filename(cfg):
    kernel_str = "_".join(cfg.model.kernel_str).lower()
    loss_str = "_".join(cfg.loss.primary).lower()
    loss_str += "_".join(cfg.loss.layers).lower()
    loss_str += f"_{cfg.loss.vgg_internal.lower()}"
    loss_str += f"_rc_{cfg.loss.random_crop}"
    loss_str += f"_lcm{cfg.loss.regulariser_coeffs.latent_channel_match}_is{cfg.loss.regulariser_coeffs.intermediate_state}_ls{cfg.loss.regulariser_coeffs.latent_size}"
    if cfg.model.family == "NCA":
        filename = f"{cfg.model.family}{kernel_str}_c{cfg.model.channels}_{loss_str}_{cfg.run.scaling}_t{cfg.run.t}"
    else:
        filename = f"{cfg.model.family}{kernel_str}_c{cfg.model.channels}_{loss_str}_{cfg.run.scaling}_t{cfg.run.t}_up{cfg.model.upscale_factor}_ud{cfg.model.upsampler.depth}_uw{cfg.model.upsampler.width_factor}"
        if cfg.model.family == "isouNCA":
            filename += f"_rad{cfg.model.upsampler.radius}"
        elif cfg.model.family == "uNCA":
            filename += f"_fm{cfg.model.upsampler.fourier_modes}"
    filename+=f"_lr{cfg.optimiser.learn_rate}_dr{cfg.optimiser.decay_rate}"
    return filename


def build_data_augmenter(cfg):
    if cfg.knockout.mode is None:
        @eqx.filter_jit
        def jittable_callback_bit(x,x_true,OBS_CHANNELS): # pyright: ignore[reportRedeclaration]
            # Here we only want 9 channels - no duplicates - as this is what the NCA sees.
            propagate_xn = lambda x:x.at[1:].set(x[:-1])
            reset_x0 = lambda x,x_true:x.at[0].set(x_true[0])
            x = jax.tree_util.tree_map(propagate_xn,x) # Set initial condition at each X[n] at next iteration to be final state from X[n-1] of this iteration
            x = jax.tree_util.tree_map(reset_x0,x,x_true) # Keep first initial x correct
            for b in range(len(x)//2):
                x[b*2] = x[b*2].at[:,:OBS_CHANNELS].set(x_true[b*2][:,:OBS_CHANNELS]) # Set every other batch of intermediate initial conditions to correct initial conditions
            return x
        
        
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
        else:
            raise ValueError(f"Unknown knockout mode {cfg.knockout.mode}")
    
    class DA_subclass(DataAugmenterGrouped):
        def data_callback(self,x,y,i,key):
            x_true,_ =self.split_x_y(1)	
            x = jittable_callback_bit(x,x_true,self.OBS_CHANNELS)
            x = self.noise(x,cfg.data.noise_strength,key=key)
            self.PREVIOUS_KEY = key
            return x,y
    return DA_subclass



def load_data(cfg):
    if cfg.knockout.mode is None:
        data,aux,CHANNEL_NAMES,boundary_mask,CHANNEL_TIMESTEP_MASK = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=DATA_PATH,
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
            impath=DATA_PATH,
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
            impath=DATA_PATH,
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
            impath=DATA_PATH,
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
            impath=DATA_PATH,
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
            impath=DATA_PATH,
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
            impath=DATA_PATH,
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
            impath=DATA_PATH,
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
            impath=DATA_PATH,
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
    

    return data,aux,CHANNEL_NAMES,boundary_mask,CHANNEL_TIMESTEP_MASK
    

def run(cfg):
    key = jax.random.PRNGKey(cfg.seed)
    model_key, train_key = jax.random.split(key)
    model = build_model(cfg, key=model_key)
    run_name = build_filename(cfg)
    optimiser,opt_name = build_optimizer(cfg)
    run_name += f"_{opt_name}"
    data,_,CHANNEL_NAMES,boundary_mask,CHANNEL_TIMESTEP_MASK = load_data(cfg)
    trainer = NCA_Trainer(
        NCA_model=model,
        data=data,
        model_filename=run_name,
        BOUNDARY_MASK=boundary_mask,
        LOSS_TIME_CHANNEL_MASK=None,
        DATA_AUGMENTER=build_data_augmenter(cfg),  # pyright: ignore[reportArgumentType]
        GRAD_LOSS=cfg.trainer.grad_loss,
        MODEL_DIRECTORY=MODEL_SAVE_PATH + cfg.logging.wandb.group + "/", # type: ignore
    )
    trainer.train(
        t=cfg.run.t,
        iters=cfg.run.iterations,
        REGULARISER_COEFFS={**cfg.loss.regulariser_coeffs},
        WARMUP=cfg.run.warmup,
        optimiser=optimiser,
        WRITE_IMAGES=cfg.run.write_images,
        LOSS_FUNC_STR=cfg.loss.primary,
        KNOCKOUT_ARGS={
            "channel": cfg.knockout.channel,
            "time": cfg.knockout.time
        },
        wandb_args={
            "project":cfg.logging.wandb.project,
            "group":cfg.logging.wandb.group,
            # "group":"baseline-9ch-train-1",
            "tags":build_tags(cfg),
            "name":run_name
        },
        LOSS_ARGS={
            "metric":cfg.loss.vgg_internal,
            "channels":None,
            "experiment_groups":None,
            "S":1024,
            "K":5,
            "D":3,
            "sharpen":True,
            "epsilon":0.1,
            "internal_loss_func":"l2",
            "samples":128,
            "random_crop":cfg.loss.random_crop,
            "layers":cfg.loss.layers
        },
        
        LOG_EVERY=cfg.trainer.log_every,
        CLEAR_CACHE_EVERY=cfg.trainer.clear_cache_every,
        LOOP_AUTODIFF=cfg.trainer.loop_autodiff,
        POOL_ADMISSION_CONFIG={
            "enabled": cfg.trainer.get("pool_admission_enabled", True),
            "relative_threshold": cfg.trainer.get("pool_admission_relative_threshold", 1.25),
            "previous_relative_threshold": cfg.trainer.get("pool_admission_previous_relative_threshold", 1.10),
            "absolute_threshold": cfg.trainer.get("pool_admission_absolute_threshold", None),
            "ema_decay": cfg.trainer.get("pool_admission_ema_decay", 0.95),
            "warmup": cfg.trainer.get("pool_admission_warmup", None),
        },
        key=train_key,
    )

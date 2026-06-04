import sys
import os
import jax
import jax.numpy as jnp
import optax
from dotenv import load_dotenv
load_dotenv()
CODE_PATH = os.getenv("PVC_PATH")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH")
DATA_PATH = os.getenv("DATA_PATH_BASE") + "Emojis/" # type: ignore
sys.path.append(CODE_PATH)  # type: ignore
os.chdir(CODE_PATH) # type: ignore

from Common.dataloader.emoji import load_emoji_sequence
from einops import rearrange,repeat,reduce
from NCA.trainer.data_augmenter_nca import DataAugmenter
# from NCA.model.NCA_gated_model import gNCA
from NCA.model.NCA_model import NCA
from NCA.model.NCA_upsample_isotropic_model import uNCA as isouNCA
from NCA.model.NCA_upsample_model import uNCA
from NCA.trainer.NCA_trainer import NCA_Trainer
from NCA.trainer.optimizer import build_optimizer
from Experiments.config_helpers import build_tags



def build_model(cfg):
    if cfg.model.family == "NCA":
        model = NCA(
            N_CHANNELS=cfg.model.channels,
            KERNEL_STR=cfg.model.kernel_str,
            FIRE_RATE=cfg.model.fire_rate,
            PADDING=cfg.model.padding,
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
            }
            
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
            }
        )
    else:
        raise ValueError(f"Unknown model family {cfg.model.family}")
    return model





def build_filename(cfg):
    kernel_str = "_".join(cfg.model.kernel_str).lower()
    loss_str = "_".join(cfg.loss.primary).lower()
    if cfg.model.family == "NCA":
        filename = f"{cfg.model.family}{kernel_str}_c{cfg.model.channels}_{loss_str}_t{cfg.run.t}"
    else:
        filename = f"{cfg.model.family}{kernel_str}_c{cfg.model.channels}_{loss_str}_t{cfg.run.t}_up{cfg.model.upscale_factor}_ud{cfg.model.upsampler.depth}_uw{cfg.model.upsampler.width_factor}_lcm{cfg.loss.regulariser_coeffs.latent_channel_match}"
        if cfg.model.family == "isouNCA":
            filename += f"_rad{cfg.model.upsampler.radius}"
        elif cfg.model.family == "uNCA":
            filename += f"_fm{cfg.model.upsampler.fourier_modes}"
    return filename
def run(cfg):
    class data_augmenter_subclass(DataAugmenter):
        #Redefine how data is pre-processed before training
        def data_init(self,SHARDING=None):
            data = self.return_saved_data()
            data = self.duplicate_batches(data, cfg.data.batches)
            data = self.pad(data, [20,20,20,20]) 		
            self.save_data(data)
            return None


    data = load_emoji_sequence(
        ["alien_monster.png","microbe.png","rooster.png","rooster.png"],
        impath_emojis=DATA_PATH,
        downsample=cfg.data.downsample,
    )
    model = build_model(cfg)
    run_name = build_filename(cfg)
    optimiser,opt_name = build_optimizer(cfg)
    trainer = NCA_Trainer(
        NCA_model=model,
        data=data,
        model_filename=run_name,
        DATA_AUGMENTER=data_augmenter_subclass,
        GRAD_LOSS=cfg.trainer.grad_loss,
        MODEL_DIRECTORY=MODEL_SAVE_PATH + cfg.logging.wandb.group + "/", # type: ignore
    )
    try:
        trainer.train(
            t=cfg.run.t,
            iters=cfg.run.iterations,
            REGULARISER_COEFFS={**cfg.loss.regulariser_coeffs},
            WARMUP=cfg.run.warmup,
            optimiser=optimiser,
            WRITE_IMAGES=cfg.run.write_images,
            LOSS_FUNC_STR=cfg.loss.primary,
            wandb_args={
                "project":cfg.logging.wandb.project,
                "group":cfg.logging.wandb.group,
                # "group":"baseline-9ch-train-1",
                "tags":build_tags(cfg),
                "name":run_name
            },
            # KNOCKOUT_ARGS=KNOCKOUT_ARGS,
            LOG_EVERY=100,
            CLEAR_CACHE_EVERY=500,
            LOOP_AUTODIFF="lax"
        )
    except Exception as e:
        print(f"Error during training: {e}")
        print("Retrying with checkpointed autodiff in case of OOM...")
        trainer.train(
            t=cfg.run.t,
            iters=cfg.run.iterations,
            REGULARISER_COEFFS={**cfg.loss.regulariser_coeffs},
            WARMUP=cfg.run.warmup,
            optimiser=optimiser,
            WRITE_IMAGES=cfg.run.write_images,
            LOSS_FUNC_STR=cfg.loss.primary,
            wandb_args={
                "project":cfg.logging.wandb.project,
                "group":cfg.logging.wandb.group,
                # "group":"baseline-9ch-train-1",
                "tags":build_tags(cfg),
                "name":run_name
            },
            # KNOCKOUT_ARGS=KNOCKOUT_ARGS,
            LOG_EVERY=100,
            CLEAR_CACHE_EVERY=500,
            LOOP_AUTODIFF="checkpointed"
        )
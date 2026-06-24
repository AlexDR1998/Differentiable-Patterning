import marimo

__generated_with = "0.18.4"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo
    import sys
    import os
    sys.path.append('/home/alex/PhD/Differentiable-Patterning/')
    # print(sys.path)
    import jax 

    # jax.config.update('jax_platform_name', 'cpu')
    from Common.utils import key_pytree_gen
    import jax.numpy as np
    import numpy as onp
    import jax.random as jr
    import equinox as eqx
    import optax
    from tqdm.notebook import tqdm
    import time
    from einops import rearrange,repeat,reduce
    from NCA.trainer.data_augmenter_nca import DataAugmenter
    # from NCA.model.NCA_gated_model import gNCA
    from NCA.model.NCA_model import NCA
    from NCA.model.NCA_upsample_isotropic_model import uNCA
    from NCA.trainer.NCA_trainer import NCA_Trainer
    # from NCA.model.NCA_multi_scale import mNCA
    # from NCA.model.NCA_noise_model import nNCA
    # from NCA.model.NCA_gated_noise_model import gnNCA
    # from Common.dataloader.micropattern import load_micropattern_circle_8ch_individual,load_micropattern_circle_8ch_individual_explicit_colony
    from Common.model.boundary import model_boundary
    from Common.save_to_video import save_to_video_rgb
    from Common.dataloader.emoji import load_emoji_sequence
    # from Experiments.emoji.time_gate_stability_comparison import H_to_filename as H_to_filename_gate
    # from Experiments.emoji.parameter_noise_sweep import H_to_filename as H_to_filename_noise
    # from Experiments.emoji.fire_rate_sweep import H_to_filename as H_to_filename_fr
    # from Experiments.emoji.local_perturbation import run as run_local_perturbations
    from marimo_utils import plot_matrix,generate_hyperparameter_combinations,generate_hyperparameter_combinations_indexed
    import matplotlib.pyplot as plt
    from pprint import pprint
    from skimage import measure
    import matplotlib.style
    matplotlib.style.use(
        "default"
    )
    import wandb


@app.cell
def _():
    from dotenv import load_dotenv
    load_dotenv()
    CODE_PATH = os.getenv("PVC_PATH")
    DATA_PATH = os.getenv("DATA_PATH_BASE") + "Emojis/"
    return CODE_PATH, DATA_PATH


@app.cell
def _():
    CHANNELS = 32
    BATCHES = 1
    # TRAIN_ITERS = 2000
    H = {
        "steps_between_images":64,
        "iters":100,
        "intermediate_growth_coeff":1.0,
        "boundary_reg_coeff":0.0,
        "contiguous_growth_coeff":0.0,
        "perturbation_conservation_coeff":0.0,
        "update_sensitivity_coeff":0.0,
        "latent_channel_match":1.0
    }
    loss_str = ["l2"]
    SPATIAL_UPSAMPLE = 8
    DATA_DOWNSAMPLE = 1
    FILENAME = "emoji_uNCA_test_14"
    # DATA_PATH = "../Data/Emojis/"
    return (
        BATCHES,
        CHANNELS,
        DATA_DOWNSAMPLE,
        FILENAME,
        H,
        SPATIAL_UPSAMPLE,
        loss_str,
    )


@app.cell
def _(CHANNELS, SPATIAL_UPSAMPLE):
    model = uNCA(
        N_CHANNELS=CHANNELS,
        O_CHANNELS = 4,
        KERNEL_STR=["ID","GRAD","LAP"],
        ACTIVATION=jax.nn.relu,
        PADDING="CIRCULAR",
        UPSAMPLER_AUX={
            "depth":3,
            "width_factor":1,
            "radius":4,
            "upsample_factor":SPATIAL_UPSAMPLE
        },
        # SPATIAL_UPSAMPLE=SPATIAL_UPSAMPLE,
        FIRE_RATE=0.5,
        key=jr.PRNGKey(7)
    )
    base_model = NCA(
        N_CHANNELS=CHANNELS,
        # O_CHANNELS = 3,
        KERNEL_STR=["ID","GRAD","LAP"],
        ACTIVATION=jax.nn.relu,
        PADDING="CIRCULAR",
        # SPATIAL_UPSAMPLE=SPATIAL_UPSAMPLE,
        FIRE_RATE=0.5,
        key=jr.PRNGKey(6)
    )
    return (model,)


@app.cell
def _(model):
    # print(model.upsample.layers[0].weight)
    print(len(model.get_weights()))
    print(model.get_weights()[0].shape)
    model_diff,model_static = model.partition()
    print(model_static)
    # print(model_diff)
    return


@app.cell(column=1)
def _(DATA_DOWNSAMPLE, DATA_PATH, H):

    data = load_emoji_sequence(
        ["alien_monster.png","microbe.png","rooster.png","rooster.png"],
        impath_emojis=DATA_PATH,
        downsample=DATA_DOWNSAMPLE,
    )

    schedule = optax.exponential_decay(1e-3, transition_steps=H["iters"], decay_rate=0.99)
    optimiser = optax.chain(optax.scale_by_param_block_norm(),
                            optax.nadam(schedule))
    # # plt.imshow(data[0,0,0])
    # data = np.concatenate([data,onp.zeros((
    #     data.shape[0],
    #     data.shape[1],
    #     CHANNELS-4,
    #     data.shape[3],
    #     data.shape[4]))],axis=2)
    # data = np.pad(data,((0,0),(0,0),(0,CHANNELS-4),(4,4),(4,4)))
    # print(data.shape)
    # plt.imshow(rearrange(data[0,0,:4],"c h w -> h w c"))
    # plt.imshow(rearrange(data[0,1,:4],"c h w -> h w c"))
    return data, optimiser


@app.cell
def _(BATCHES):
    class data_augmenter_subclass(DataAugmenter):
        #Redefine how data is pre-processed before training
        def data_init(self,SHARDING=None):
            data = self.return_saved_data()
            data = self.duplicate_batches(data, BATCHES)
            data = self.pad(data, [20,20,20,20]) 		
            self.save_data(data)
            return None
    return (data_augmenter_subclass,)


@app.cell
def _():
    wandb.finish()
    return


@app.cell
def _(CODE_PATH, FILENAME, data, data_augmenter_subclass, model):
    opt = NCA_Trainer(
        model,
        data,
        model_filename=FILENAME,
        MODEL_DIRECTORY=CODE_PATH+"models/",
        DATA_AUGMENTER=data_augmenter_subclass,
        GRAD_LOSS=True
    )
    return (opt,)


@app.cell
def _(FILENAME, H, loss_str, opt, optimiser):
    opt.train(
        t=H["steps_between_images"],
        iters=H["iters"],
        REGULARISER_COEFFS={
            "intermediate_state":H["intermediate_growth_coeff"],
            "boundary":H["boundary_reg_coeff"],
            "contiguous_growth":H["contiguous_growth_coeff"],
            "perturbation_conservation":H["perturbation_conservation_coeff"],
            "update_sensitivity":H["update_sensitivity_coeff"],
            "latent_channel_match":H["latent_channel_match"]
        },
        WARMUP=10,
        optimiser=optimiser,
        WRITE_IMAGES=True,
        LOSS_FUNC_STR=loss_str,
        # LOSS_ARGS={
        #     "channels":None,
        #     "experiment_groups":None,
        #     "S":H["ott_S"],
        #     "K":H["ott_K"],
        #     "D":H["ott_D"],
        #     "sharpen":H["ott_sharpen"],
        #     "epsilon":H["ott_epsilon"],
        #     "internal_loss_func":H["ott_internal_loss_func"],
        # },
        wandb_args={
            "project":"nca-upsample",
            "group":"upsample-testing",
            # "group":"baseline-9ch-train-1",
            "tags":[f"{k}:{v}" for k,v in H.items()],
            "name":FILENAME
        },
        # KNOCKOUT_ARGS=KNOCKOUT_ARGS,
        LOG_EVERY=10,
        CLEAR_CACHE_EVERY=500,
    )
    return


@app.cell
def _():
    return


@app.cell
def _():

    # print(x["latent"].shape)
    # print(x["output"].shape)
    # plt.imshow(x["output"][1,:4].T)
    return


if __name__ == "__main__":
    app.run()

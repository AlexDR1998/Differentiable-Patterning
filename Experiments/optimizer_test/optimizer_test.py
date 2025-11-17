PVC_PATH = "/mnt/ceph/ar-dp/"
import sys
import os
sys.path.append(PVC_PATH)
os.chdir(PVC_PATH)
import jax
import jax.numpy as np
import jax.random as jr
import optax
import equinox as eqx
import sys
from einops import repeat,rearrange
import glob
from NCA.trainer.data_augmenter_nca_basic import DataAugmenter as MicropatternDataAugmenter
from NCA.trainer.data_augmenter_nca import DataAugmenter as EmojiDataAugmenter
from NCA.trainer.optimizer import sam_optimizer,muon_optimizer
# from NCA.model.NCA_gated_model import gNCA
from NCA.model.NCA_model import NCA
from NCA.trainer.NCA_trainer import NCA_Trainer
from Common.dataloader.micropattern import load_micropattern_circle_8ch_individual
from Common.dataloader.emoji import load_emoji_sequence
from Common.utils import index_to_param_list
import time
from pprint import pprint
import argparse


"""
    This script is for properly testing different optimizers and their configurations. We consider 4 optimizers:
    - Optimistic Adam
    - Nadam
    - Muon
    - Sharpness-Aware Minimization (SAM) wrapped around Nadam
    Each optimizer can be run with or without parameter-wise block normalization, and with different numbers of
    gradient accumulation steps. The optimizers are tested on two tasks: micropattern training and emoji training.

    We use an exponential learning rate decay schedule with a linear warmup for all optimizers.
"""

def build_opt(optimiser_hparams,schedule):
    opt_dict = {
        "optimistic_adam": optax.optimistic_adam_v2(schedule),
        "nadam": optax.nadam(schedule),
        "muon": muon_optimizer(schedule),
        "sam": sam_optimizer(base_optimizer=optax.nadam(schedule),rho=1e-4,sync_period=2),
    }
    if optimiser_hparams["optimizer"] not in opt_dict.keys():
        raise ValueError(f"Optimizer {optimiser_hparams['optimizer']} not recognized. Available optimizers: {list(opt_dict.keys())}")
    optimiser = opt_dict[optimiser_hparams["optimizer"]]
    modestr = optimiser_hparams["optimizer"]
    if optimiser_hparams["multistep"]>1:
        optimiser = optax.MultiSteps(optimiser, optimiser_hparams["multistep"])
        modestr += f"_multistep{optimiser_hparams['multistep']}"
    if optimiser_hparams["block_norm"]:
        optimiser = optax.chain(optax.scale_by_param_block_norm(), optimiser)
        modestr += "_blocknorm"
    optimiser = optax.apply_if_finite(optimiser, max_consecutive_errors=5) # Always wrap in this, just to handle any occasional NaNs
    return optimiser,modestr

def emoji_task(CHANNELS,BATCHES,TRAINING_ITERATIONS,key,optimiser_hparams,warmup_steps,schedule):
    class data_augmenter_subclass(EmojiDataAugmenter):
        def data_init(self, SHARDING=None):
            data = self.return_saved_data()
            data = self.pad(data, [10, 10, 10, 10])
            self.save_data(data)
            return None
    STEPS_BETWEEN_IMAGES = 128
    NCA_hyperparameters = {
        "N_CHANNELS":CHANNELS,
        "KERNEL_STR":["ID","LAP","GRAD"],
        "FIRE_RATE":0.5,
        "PADDING":"circular",
        "key":key
    }

    
    data = load_emoji_sequence(
        [
            "crab.png",
            "microbe.png",
            # "avocado.png",
            # "alien_monster.png",
            # "butterfly.png",
            # "lizard.png",
            # "mushroom.png",
        ],
        downsample=1,
        impath_emojis=PVC_PATH+"Data/Emojis/",
    )
    data_filename = "cr_mi_av_al_bt_li_mu"
    # data_filename = "cr_mi"

    data = rearrange(data, "B T C W H -> T B C W H")

    data = repeat(data, "B T C W H -> (B b) T C W H", b=BATCHES)

    initial_condition = np.array(data)

    W = initial_condition.shape[-2]
    H = initial_condition.shape[-1]

    initial_condition = initial_condition.at[:, :, :, : W // 2 - 6].set(0)
    initial_condition = initial_condition.at[:, :, :, W // 2 + 5 :].set(0)
    initial_condition = initial_condition.at[:, :, :, :, : H // 2 - 6].set(0)
    initial_condition = initial_condition.at[:, :, :, :, H // 2 + 5 :].set(0)
    data = np.concatenate(
        [initial_condition, data, data], axis=1
    )  # Join initial condition and data along the time axis
    print("(Batch, Time, Channels, Width, Height): " + str(data.shape))
    nca = NCA(**NCA_hyperparameters)
    opt,optmode = build_opt(optimiser_hparams,schedule)
    name = "optimizer_test_"+optmode+"_emoji_grad"
    print("-----------------------------------------------------------------------------------------------------")
    print(f"Training {optmode} on emoji task")
    trainer = NCA_Trainer(
        nca,
        data,
        DATA_AUGMENTER=data_augmenter_subclass,
        model_filename=name,
        MODEL_DIRECTORY= "models/",
        LOG_DIRECTORY= "logs/",
    )
    
    trainer.train(
        t=STEPS_BETWEEN_IMAGES, 
        iters=TRAINING_ITERATIONS,
        REGULARISER_COEFFS={
            "intermediate_state":0.1,
            "boundary":0.0,
            "contiguous_growth":1.0,
            "update_sensitivity":0.0,
            "perturbation_conservation":0.0,
        }, 
        LOOP_AUTODIFF="checkpointed", 
        optimiser=opt,
        WARMUP=warmup_steps,
        LOG_EVERY=200,
        CLEAR_CACHE_EVERY=500,
        wandb_args={
            "project":"nca_optimizer_test",
            "name":name,
            "tags":["multi_species","NCA","emoji","long","optimizer_test","grad"],
            "group":"nca_optimizer_test_run_3"}
    )



def micropattern_task(CHANNELS,BATCHES,TRAINING_ITERATIONS,key,optimiser_hparams,warmup_steps,schedule):
    NCA_hyperparameters = {
        "N_CHANNELS":CHANNELS,
        "KERNEL_STR":["ID","LAP","DIFF"],
        "FIRE_RATE":0.5,
        "PADDING":"circular",
        "key":key
    }

    DOWNSAMPLE = 4
    STEPS_BETWEEN_IMAGES = int(256 / np.sqrt(DOWNSAMPLE))
    # FILENAME = f"micropattern_circle_8ch_individual_gNCA_t{STEPS_BETWEEN_IMAGES}_ch{CHANNELS}_ds{DOWNSAMPLE}_v4_long"
    DATA_PATH = PVC_PATH+"Data/Timecourse_individual_images/*"
    data, aux, CHANNEL_NAMES, boundary_mask = load_micropattern_circle_8ch_individual(
        impath=DATA_PATH, 
        BATCHES=BATCHES, 
        DOWNSAMPLE=DOWNSAMPLE,
        TIMESTEPS=[0,12,24,36,48],
        PROCESSING_MODES=["map_to_0_1","downsample"],
    )
    print("Data shape = " + str(data.shape))
    print("Boundary mask shape = " + str(boundary_mask.shape))


    MASK = np.array([
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1]])
        # [1,1,1,1,1,0,0,0]])

    print("-----------------------------------------------------------------------------------------------------")
    # print(f"Training gNCA on with STEPS_BETWEEN_IMAGES: {STEPS_BETWEEN_IMAGES} CHANNELS: {CHANNELS}")
    # nca = gNCA(**NCA_hyperparameters)
    nca = NCA(**NCA_hyperparameters)
    opt,optmode = build_opt(optimiser_hparams,schedule)
    name = f"optimizer_test_{optmode}_micropattern_t{STEPS_BETWEEN_IMAGES}_ch{CHANNELS}_ds{DOWNSAMPLE}_48h_experiment_groupby"
    print("-----------------------------------------------------------------------------------------------------")
    print(f"Training {optmode} on micropattern task")
    trainer = NCA_Trainer(
        nca,
        data,
        model_filename=name,
        DATA_AUGMENTER=MicropatternDataAugmenter,
        MODEL_DIRECTORY="models/",
        LOG_DIRECTORY="logs/",
        BOUNDARY_MASK=boundary_mask,
        BOUNDARY_MODE="soft",
        LOSS_TIME_CHANNEL_MASK=MASK
    )
    trainer.train(
        t=STEPS_BETWEEN_IMAGES,
        iters=TRAINING_ITERATIONS,
        REGULARISER_COEFFS={
            "intermediate_state":0.1,
            "boundary": 1.0,
            "contiguous_growth":0.0,
            "update_sensitivity":0.0,
            "perturbation_conservation":0.0,
        },
        WARMUP=warmup_steps,
        optimiser=opt,
        WRITE_IMAGES=True,
        LOSS_FUNC_STR="vgg",
        LOSS_ARGS = {
            "channels":None,
            "experiment_groups":None
        },
        wandb_args={
            "project":"nca_optimizer_test",
            "name":name,
            "tags":["NCA","micropattern","optimizer_test","long"],
            "group":"nca_optimizer_test_run_4",
            
        },
        LOG_EVERY=200,
        CLEAR_CACHE_EVERY=500
    )




def main():
    key = jr.PRNGKey(int(time.time()))
    index = int(sys.argv[1])
    key = jr.fold_in(key,index)
    FULL_HYPERPARAMETERS = {
        # "optimizer":["optimistic_adam","nadam","muon","sam"],
        # "optimizer":["nadam","muon","sam"],
        "optimizer":["optimistic_adam","nadam","muon"],
        "block_norm":[True,False],
        "multistep":[1,2,4],
        "task":["micropattern"]
        
    }
    HPARAMS = index_to_param_list(index,4,FULL_HYPERPARAMETERS)



    for H in HPARAMS:
        print("-----------------------------------------------------------------------------------------------------")
        print("Running experiment with hyperparameters:")
        pprint(H)

        # print(glob.glob(DATA_PATH+"*.tif"))
        BATCHES = 2
        CHANNELS = 32
        TRAINING_ITERATIONS = 10000


        warmup_steps = 100  # number of steps for warmup
        init_lr = 1e-6      # starting learning rate
        target_lr = 1e-3    # learning rate after warmup

        warmup_fn = optax.linear_schedule(
            init_value=init_lr,
            end_value=target_lr,
            transition_steps=warmup_steps,
        )

        decay_fn = optax.exponential_decay(
            init_value=target_lr,
            transition_steps=TRAINING_ITERATIONS,
            decay_rate=0.98,
        )

        schedule = optax.join_schedules(
            schedules=[warmup_fn, decay_fn],
            boundaries=[warmup_steps],
        )

        # try:
        if H["task"]=="micropattern":
            micropattern_task(
                CHANNELS = CHANNELS,
                BATCHES = BATCHES,
                TRAINING_ITERATIONS = TRAINING_ITERATIONS,
                key = key,
                optimiser_hparams = H,
                warmup_steps = warmup_steps,
                schedule = schedule
            )
        elif H["task"]=="emoji":
            emoji_task(
                CHANNELS = CHANNELS,
                BATCHES = BATCHES,
                TRAINING_ITERATIONS = TRAINING_ITERATIONS,
                key = key,
                optimiser_hparams = H,
                warmup_steps = warmup_steps,
                schedule = schedule
            )
        # except Exception as e:
        #     print(f"Experiment with hyperparameters {H} failed with exception: {e}")
        #     continue
        #     # raise e

if __name__ == "__main__":
    main()


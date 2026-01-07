"""

    Takes a pre-trained NCA model on full 9 channels (TBXT, LMBR ...) without any knockout, and fine tunes
    it on the partial nodal knockout data.
"""

import os
from dotenv import load_dotenv
load_dotenv()
PVC_PATH = os.getenv("PVC_PATH")
DATA_PATH_BASE = os.getenv("DATA_PATH_BASE")
import jax
import jax.numpy as np
import numpy as onp
import jax.random as jr
import optax
import equinox as eqx
import sys
from einops import repeat,rearrange
import glob
from pprint import pprint
sys.path.append(PVC_PATH)
os.chdir(PVC_PATH)
from NCA.trainer.data_augmenter_9ch_colony import DataAugmenter as DataAugmenterGrouped
from NCA.model.NCA_gated_model import gNCA
from NCA.model.NCA_model import NCA
from NCA.model.NCA_multi_scale import mNCA
from NCA.trainer.NCA_trainer import NCA_Trainer
from Common.dataloader.micropattern import load_micropattern_circle_nodal_knockout_9ch_explicit_colony
from Common.utils import index_to_param_list
from Experiments.optimizer_test.optimizer_test import build_opt 
import time
import argparse
from pathlib import Path
import os
key = jax.random.PRNGKey(int(time.time()))
index = int(sys.argv[1])
TOTAL_JOBS = int(sys.argv[2])
DATA_PATH_INDIVIDUAL = DATA_PATH_BASE + "Timecourse Individual Images/*"
DATA_PATH_GROUPED= DATA_PATH_BASE + "Timecourse Seperate Colonies/*"
BATCHES = 2
# DOWNSAMPLE = 2
TRAINING_ITERATIONS = 5000

FULL_HYPERPARAMETERS = {
    # "loss_mode":["ott_grouped_and_l2","vgg_grouped_and_l2"],
    "loss_mode":["vgg_grouped_and_l2"],
    "model":["NCA"],
    "optimizer":["nadam"],
    "block_norm":[True],
    "noise_strength":[0.005],
    "multistep":[1],
    "channels":[16,24,32,48,64],
    "ott_S":[2048],
    "ott_K":[5],
    "ott_D":[4],
    "learn_rate":[1e-3],
    "downsample":[4],
    "ott_sharpen":[True],
    "ott_epsilon":[0.01],
    "ott_internal_loss_func":["l1"],
    "intermediate_growth":[1.0],
    "boundary_reg":[1.0],
    "contiguous_growth":[1.0],
    "knockout":[0,24] # or "knockout"
}

HPARAMS = index_to_param_list(index,TOTAL_JOBS,FULL_HYPERPARAMETERS)

@eqx.filter_jit
def jittable_callback_bit(x,x_true,OBS_CHANNELS):
    propagate_xn = lambda x:x.at[1:].set(x[:-1])
    reset_x0 = lambda x,x_true:x.at[0].set(x_true[0])
    x = jax.tree_util.tree_map(propagate_xn,x) # Set initial condition at each X[n] at next iteration to be final state from X[n-1] of this iteration
    x = jax.tree_util.tree_map(reset_x0,x,x_true) # Keep first initial x correct
    for b in range(len(x)//2):
        x[b*2] = x[b*2].at[:,:OBS_CHANNELS].set(x_true[b*2][:,:OBS_CHANNELS]) # Set every other batch of intermediate initial conditions to correct initial conditions
    return x


def train(H,key):
    
    assert H["knockout"] in [None,0,24], "Invalid knockout parameter"
    MODEL = H["model"]
    if MODEL == "NCA":
         model = NCA
    elif MODEL == "gNCA":
         model = gNCA
    elif MODEL == "mNCA":
         model = mNCA
    else:
        raise ValueError("Invalid MODEL")
    CHANNELS = H["channels"]
    LOSS_MODE = H["loss_mode"]
    # GRAD_LOSS = H["grad_loss"]
    DOWNSAMPLE = H["downsample"]
    NOISE_STRENGTH = H["noise_strength"]
    INIT_LR = H["learn_rate"]
    BOUNDARY_REG_COEFF = H["boundary_reg"]
    INTERMEDIATE_GROWTH_COEFF = H["intermediate_growth"]
    CONTIGUOUS_GROWTH_COEFF = H["contiguous_growth"]
    STEPS_BETWEEN_IMAGES = int(256 / np.sqrt(DOWNSAMPLE))
    NCA_hyperparameters = {
        "N_CHANNELS":CHANNELS,
        "KERNEL_STR":["ID","LAP","DIFF"],
        "FIRE_RATE":0.5,
        "PADDING":"circular",
        "key":key
    }
    data,aux,CHANNEL_NAMES,boundary_mask,CHANNEL_TIMESTEP_MASK = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
        impath=DATA_PATH_GROUPED,
        FILTER_KN_TIME=H["knockout"],
        BATCHES=BATCHES,
        DOWNSAMPLE=DOWNSAMPLE,
        TIMESTEPS=[0,12,24,36,48],
        PROCESSING_MODES={
            "map_to_0_1",
            "downsample"
        }
    )
    print(f"Channel timestep mask shape: {CHANNEL_TIMESTEP_MASK.shape}")
    print(f"Boundary mask shape: {boundary_mask.shape}")
    DATA_CHANNELS = 12
    OBS_CHANNELS = 9
    data =np.concatenate([data,data[:,-1:]],axis=1) # Duplicate last time step to enforce stability at the end of run
    CHANNEL_TIMESTEP_MASK = np.concatenate([CHANNEL_TIMESTEP_MASK,CHANNEL_TIMESTEP_MASK[-1:]],axis=0)
    class DA_subclass(DataAugmenterGrouped):
        def data_callback(self,x,y,i,key):
            x_true,_ =self.split_x_y(1)	
            x = jittable_callback_bit(x,x_true,self.OBS_CHANNELS)
            x = self.noise(x,NOISE_STRENGTH,key=key)
            self.PREVIOUS_KEY = key
            return x,y
    
    
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

    optimiser,opt_str = build_opt(H,schedule)
    nca = model(**NCA_hyperparameters)
    loss_str = {
        "l2":["l2"],
        "vgg":["vgg"],
        "vgg_grouped":["vgg_grouped"],
        "vgg_and_l2":["vgg","l2"],
        "vgg_grouped_and_l2":["vgg_grouped_and_l2"],
        "l2_grad":["l2"],
        "vgg_grad":["vgg"],
        "vgg_and_l2_grad":["vgg","l2"],
        "ott":["ott"],
        "ott_grad":["ott"],
        "ott_chstack":["ott_chstack"],
        "ott_chstack_grad":["ott_chstack"],
        "ott_and_l2":["ott","l2"],
        "ott_and_l2_grad":["ott","l2"],
        "ott_chstack_and_l2":["ott_chstack","l2"],
        "ott_chstack_and_l2_grad":["ott_chstack","l2"],
        "ott_grouped":["ott_grouped"],
        "ott_grouped_and_l2":["ott_grouped_and_l2"],
        # "ott_grouped_grad":["ott_grouped"],
    }[LOSS_MODE]
    GRAD_LOSS = "_grad" in LOSS_MODE
    if "ott" in LOSS_MODE:
        loss_name = f"{LOSS_MODE}_S{H['ott_S']}K{H['ott_K']}D{H['ott_D']}shp{H['ott_sharpen']}ep{H['ott_epsilon']}{H['ott_internal_loss_func']}"
    else:
        loss_name = LOSS_MODE
    FILENAME = f"baseline_9ch_{MODEL}_{loss_name}_steps{STEPS_BETWEEN_IMAGES}_ds{DOWNSAMPLE}_ch{CHANNELS}_opt{opt_str}_ns{NOISE_STRENGTH}_ig{INTERMEDIATE_GROWTH_COEFF}_br{BOUNDARY_REG_COEFF}_cg{CONTIGUOUS_GROWTH_COEFF}"
    if H["knockout"] in [0,24]:
        nca = nca.load(PVC_PATH+f"models/{FILENAME}.eqx")
        FILENAME += f"_knockout{H['knockout']}"
    
    

    
    opt = NCA_Trainer(
        nca,
        data,
        model_filename=FILENAME,
        DATA_AUGMENTER=DA_subclass,
        MODEL_DIRECTORY=PVC_PATH+"models/",
        LOG_DIRECTORY=PVC_PATH+"logs/",
        OBS_CHANNELS=OBS_CHANNELS,
        DATA_CHANNELS=DATA_CHANNELS,
        BOUNDARY_MASK=boundary_mask,
        BOUNDARY_MODE="soft",
        GRAD_LOSS=GRAD_LOSS,
        LOSS_TIME_CHANNEL_MASK=CHANNEL_TIMESTEP_MASK,
    )

    # try:
    opt.train(
        t=STEPS_BETWEEN_IMAGES,
        iters=TRAINING_ITERATIONS,
        REGULARISER_COEFFS={
            "intermediate_state":INTERMEDIATE_GROWTH_COEFF,
            "boundary":BOUNDARY_REG_COEFF,
            "contiguous_growth":CONTIGUOUS_GROWTH_COEFF,
        },
        WARMUP=warmup_steps,
        optimiser=optimiser,
        WRITE_IMAGES=True,
        LOSS_FUNC_STR=loss_str,
        LOSS_ARGS={
            "channels":None,
            "experiment_groups":None,
            "S":H["ott_S"],
            "K":H["ott_K"],
            "D":H["ott_D"],
            "sharpen":H["ott_sharpen"],
            "epsilon":H["ott_epsilon"],
            "internal_loss_func":H["ott_internal_loss_func"],
        },
        wandb_args={
            "project":"nca-micropatterns-nodal-knockout",
            "group":"baseline-9ch-train-test",
            "tags":[f"{k}:{v}" for k,v in H.items()],
            "name":FILENAME
        },
        LOG_EVERY=100,
        CLEAR_CACHE_EVERY=500,
    )
    # except Exception as e:
    #     print(f"Training failed with hyperparameters {H} exception: {e}")
    return key

    
    

    

print(f"JOB {index}/{TOTAL_JOBS} RUNNING WITH {len(HPARAMS)} SETS OF HPARAMS EACH")
# pprint(HPARAMS)
# print("---------------------------------------------------")
# data, aux,CHANNEL_NAMES,boundary_mask,augmenter,DATA_CHANNELS = load_data(HPARAMS[0]["downsample"],HPARAMS[0]["loss_mode"] in ["vgg_grouped","vgg_grouped_and_l2"])
# print("Data loaded to test for errors, shape = " + str(data.shape))
# print("Channel names: " + str(CHANNEL_NAMES))
for H in HPARAMS:
    print("---------------------------------------------------")
    print(f"RUNNING WITH HYPERPARAMS:")
    pprint(H)
    key = train(H,key)
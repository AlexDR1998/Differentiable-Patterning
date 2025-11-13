PVC_PATH = "/mnt/ceph/ar-dp/"
import jax
import jax.numpy as np
import numpy as onp
import jax.random as jr
import optax
import equinox as eqx
import sys
import os
from einops import repeat,rearrange
import glob
sys.path.append(PVC_PATH)
os.chdir(PVC_PATH)
print(sys.path)
# from NCA.trainer.data_augmenter_nca_basic import DataAugmenter
from NCA.trainer.data_augmenter_micropattern_vgg_colony import DataAugmenter as DataAugmenterGrouped
from NCA.trainer.data_augmenter_nca_basic import DataAugmenter as DataAugmenterBasic
from NCA.model.NCA_gated_model import gNCA
from NCA.model.NCA_model import NCA
from NCA.model.NCA_multi_scale import mNCA
from NCA.trainer.NCA_trainer import NCA_Trainer
from Common.dataloader.micropattern import load_micropattern_circle_8ch_individual,load_micropattern_circle_8ch_individual_explicit_colony
from Common.utils import index_to_param_list
from Experiments.optimizer_test.optimizer_test import build_opt 
import time
import argparse
from pathlib import Path
import os
key = jax.random.PRNGKey(int(time.time()))



index = int(sys.argv[1])
# DATA_PATH = "/projects/u5be/alex_data/Micropatterns/Timecourse_individual_images/*"
DATA_PATH_INDIVIDUAL = "Data/Timecourse_individual_images/*"
DATA_PATH_GROUPED= "Data/Timecourse Seperate Colonies/*"
BATCHES = 2
# DOWNSAMPLE = 2
TRAINING_ITERATIONS = 10000
# CHANNELS = 32

FULL_HYPERPARAMETERS = {
    # "model":["mNCA","gNCA","NCA"],
    "loss_mode":["vgg","vgg_grouped","vgg_and_l2","vgg_grouped_and_l2"],
    "model":["NCA","gNCA"],
    "optimizer":["nadam"],
    "block_norm":[True],
    # "noise_strength":[0.001,0.01,0.1],
    "noise_strength":[0.1],
    "multistep":[1],
    "channels":[32],
    "intermediate_growth":[0.0,0.2,1.0,2.0],
    "contiguous_growth":[0.0],
    "downsample":[8],
    "grad_loss": [False]
}

HPARAMS = index_to_param_list(index,4,FULL_HYPERPARAMETERS)


def load_data(DOWNSAMPLE,GROUPED):
    if GROUPED:        
        data, aux, CHANNEL_NAMES, boundary_mask = load_micropattern_circle_8ch_individual_explicit_colony(
            impath=PVC_PATH+DATA_PATH_GROUPED, 
            BATCHES=BATCHES, 
            DOWNSAMPLE=DOWNSAMPLE,
            TIMESTEPS=[0,12,24,36,48],
            PROCESSING_MODES=["map_to_0_1","downsample"],
        )
        augmenter = "grouped_colony"
        DATA_CHANNELS = 11
    else:
        data, aux, CHANNEL_NAMES, boundary_mask = load_micropattern_circle_8ch_individual(
            impath=PVC_PATH+DATA_PATH_INDIVIDUAL, 
            BATCHES=BATCHES, 
            DOWNSAMPLE=DOWNSAMPLE,
            TIMESTEPS=[0,12,24,36,48],
            PROCESSING_MODES=["map_to_0_1","downsample"],
        )
        augmenter = "individual"
        DATA_CHANNELS = 8
    return data, aux,CHANNEL_NAMES,boundary_mask,augmenter,DATA_CHANNELS


def run_training(H,key):
    key = jr.fold_in(key,index) 
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
    GRAD_LOSS = H["grad_loss"]
    DOWNSAMPLE = H["downsample"]
    NOISE_STRENGTH = H["noise_strength"]
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
    if MODEL == "mNCA":
        NCA_hyperparameters["SCALES"] = [1,2,4,8]
    

    OBS_CHANNELS = 8

    data, aux,CHANNEL_NAMES,boundary_mask,augmenter,DATA_CHANNELS = load_data(DOWNSAMPLE,LOSS_MODE in ["vgg_grouped","vgg_grouped_and_l2"])
    
    data =np.concatenate([data,data[:,-1:]],axis=1) # Duplicate last time step to enforce stability at the end of run
    
    DA = {
        "individual": DataAugmenterBasic,
        "grouped_colony": DataAugmenterGrouped,
    }[augmenter]

    @eqx.filter_jit
    def jittable_callback_bit(x,x_true,OBS_CHANNELS):
        propagate_xn = lambda x:x.at[1:].set(x[:-1])
        reset_x0 = lambda x,x_true:x.at[0].set(x_true[0])
        x = jax.tree_util.tree_map(propagate_xn,x) # Set initial condition at each X[n] at next iteration to be final state from X[n-1] of this iteration
        x = jax.tree_util.tree_map(reset_x0,x,x_true) # Keep first initial x correct
        for b in range(len(x)//2):
            x[b*2] = x[b*2].at[:,:OBS_CHANNELS].set(x_true[b*2][:,:OBS_CHANNELS]) # Set every other batch of intermediate initial conditions to correct initial conditions
        return x
    
    class DA_subclass(DA):
        def data_callback(self,x,y,i,key):
            x_true,_ =self.split_x_y(1)	
            x = jittable_callback_bit(x,x_true,self.OBS_CHANNELS)
            x = self.noise(x,NOISE_STRENGTH,key=key)
            #y = self.noise(y,0.01,key=jax.random.fold_in(key,2*i))
            self.PREVIOUS_KEY = key
            return x,y
        
    print("Data shape = " + str(data.shape))
    print("Boundary mask shape = " + str(boundary_mask.shape))
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

    MASK = np.array([
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1],])
        # [1,1,1,1,1,0,0,0]])


    print("-----------------------------------------------------------------------------------------------------")
    nca = model(**NCA_hyperparameters)
    print(f"Training {nca.get_config()['MODEL']} on with STEPS_BETWEEN_IMAGES: {STEPS_BETWEEN_IMAGES} CHANNELS: {CHANNELS}")
    # if INTERMEDIATE_GROWTH_COEFF ==:
        # reg_str = "_int"
    # else:
    #     reg_str = ""
    # if CONTIGUOUS_GROWTH_COEFF ==1.0:
    #     reg_str += "_contig"

    FILENAME = f"micropattern_circle_8ch_3colony_individual_{LOSS_MODE}_{opt_str}_int{INTERMEDIATE_GROWTH_COEFF}_contig_{CONTIGUOUS_GROWTH_COEFF}_noise{NOISE_STRENGTH}_{nca.get_config()['MODEL']}_t{STEPS_BETWEEN_IMAGES}_ch{CHANNELS}_ds{DOWNSAMPLE}_48h_stable"
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
        LOSS_TIME_CHANNEL_MASK=MASK,
    )

    if LOSS_MODE == "l2":
        loss_str = ["l2"]
    elif LOSS_MODE == "vgg":
        loss_str = ["vgg"]
    elif LOSS_MODE == "vgg_grouped":
        loss_str = ["vgg_grouped"]
    elif LOSS_MODE == "vgg_and_l2":
        loss_str = ["vgg","l2"]
    elif LOSS_MODE == "vgg_grouped_and_l2":
        loss_str = ["vgg_grouped_and_l2"]
    # elif LOSS_MODE == "both_average":
        # loss_str = ["vgg","l2"]
    else:
        raise ValueError("Invalid LOSS_MODE")
    # try:
    opt.train(
        t=STEPS_BETWEEN_IMAGES,
        iters=TRAINING_ITERATIONS,
        REGULARISER_COEFFS={
            "intermediate_state":INTERMEDIATE_GROWTH_COEFF,
            "boundary": 1.0,
            "contiguous_growth":CONTIGUOUS_GROWTH_COEFF,
        },
        WARMUP=warmup_steps,
        optimiser=optimiser,
        WRITE_IMAGES=True,
        LOSS_FUNC_STR=loss_str,
        wandb_args={
            "project":"nca-micropatterns",
            "group":"ind_8ch_3colony_loss_channel_opt_48h_stable_intreg",
            # "tags":["training",nca.get_config()['MODEL'],str(CHANNELS)+"ch",str(DOWNSAMPLE)+"x_downsample",LOSS_MODE],
            "tags":[f"{k}:{v}" for k,v in H.items()],
            "name":FILENAME
        },
        LOG_EVERY=100,
        CLEAR_CACHE_EVERY=500,
    )
    # except Exception as e:
        # print(f"Training failed with hyperparameters {H} exception: {e}")
    return key


for H in HPARAMS:
    key = run_training(H,key)
"""

    Takes a pre-trained NCA model on full 9 channels (TBXT, LMBR ...) without any knockout, and fine tunes
    it on the partial nodal knockout data.
"""

import os
from dotenv import load_dotenv
load_dotenv()
PVC_PATH = os.getenv("PVC_PATH")
DATA_PATH_BASE = os.getenv("DATA_PATH_BASE")
DATA_PATH_INDIVIDUAL = DATA_PATH_BASE + "Timecourse Individual Images/*"
DATA_PATH_GROUPED= DATA_PATH_BASE + "Timecourse Seperate Colonies/*"
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
# import os

BATCHES = 2
# DOWNSAMPLE = 2


def H_to_filename(H):
    if "ott" in H["loss_mode"]:
        loss_name = f"{H['loss_mode']}_S{H['ott_S']}K{H['ott_K']}D{int(4-onp.log2(H['downsample']))}shp{H['ott_sharpen']}ep{H['ott_epsilon']}{H['ott_internal_loss_func']}"
    elif "vgg" in H["loss_mode"]:
        loss_name = f"{H['loss_mode']}_{H['metric']}"
        if H["metric"] in ["emdsp","emdfull"]:
            loss_name += f"_ep{H['ott_epsilon']}{H['ott_internal_loss_func']}{H['loss_normalize']}"
    elif "clip" in H["loss_mode"]:
        loss_name = f"{H['loss_mode']}_{H['metric']}_{H['loss_normalize']}"
    else:
        loss_name = H["loss_mode"]

    opt_str = H["optimizer"]
    opt_str_base = H["optimizer"]
    if H["multistep"]>1:
        opt_str += f"_multistep{H['multistep']}"
    if H["block_norm"]:
        opt_str += "_blocknorm"
        opt_str_base += "_blocknorm"
    # _STEPS_AT_DS8 = 64

    if H["stepsize_scaling"]=="convective":
        STEPS_BETWEEN_IMAGES = int(H["steps_at_ds8"]*(8/H["downsample"])) # Scale steps between images with downsample factor, linearly like for sliving hyperbolic PDEs
    if H["stepsize_scaling"]=="diffusive":
        STEPS_BETWEEN_IMAGES = int(H["steps_at_ds8"]*((8/H["downsample"])**2)) # Scale steps between images with downsample factor squared, like for diffusive PDEs
    # STEPS_BETWEEN_IMAGES = int(512 / H["downsample"])
    # FILENAME = f"baseline_9ch_{MODEL}_{loss_name}_steps{STEPS_BETWEEN_IMAGES}_ds{DOWNSAMPLE}_ch{CHANNELS}_opt{opt_str}_ns{NOISE_STRENGTH}_ig{INTERMEDIATE_GROWTH_COEFF}_br{BOUNDARY_REG_COEFF}_cg{CONTIGUOUS_GROWTH_COEFF}"
    FILENAME_BASE = f"baseline_9ch_{H['model']}_{loss_name}_ds{H['downsample']}_t{STEPS_BETWEEN_IMAGES}_ch{H['channels']}_opt{opt_str_base}_good"
    if H["knockout"] in [0,24]:
    
        FILENAME_KO = f"ftko_{H['knockout_mode']}_{H['knockout']}_9ch_{H['model']}_{loss_name}_ds{H['downsample']}_t{STEPS_BETWEEN_IMAGES}_ch{H['channels']}_opt{opt_str}_{H['TRAINING_ITERATIONS']}iters_lr{H['finetune_lr']}"
    else:
        FILENAME_KO = None
    return {"base":FILENAME_BASE,"ko":FILENAME_KO,"timesteps":STEPS_BETWEEN_IMAGES}
    
    



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
    FINE_LR = H["finetune_lr"]
    BOUNDARY_REG_COEFF = H["boundary_reg"]
    INTERMEDIATE_GROWTH_COEFF = H["intermediate_growth"]
    CONTIGUOUS_GROWTH_COEFF = H["contiguous_growth"]
    # STEPS_BETWEEN_IMAGES = int(512 / DOWNSAMPLE)
    NCA_hyperparameters = {
        "N_CHANNELS":CHANNELS,
        "KERNEL_STR":["ID","LAP","DIFF"],
        "FIRE_RATE":0.5,
        "PADDING":"circular",
        "key":key
    }
    if H["knockout"] is not None and H["knockout_mode"]=="both":
        # NCA_hyperparameters["FIRE_RATE"]=1.0 # For fine tuning on both WT and KO data, we want to use all the data and not drop any updates randomly, as the dataset is already small.
        data_ko,aux,CHANNEL_NAMES,boundary_mask_ko,CHANNEL_TIMESTEP_MASK_KO = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=DATA_PATH_GROUPED,
            FILTER_KN_TIME=H["knockout"],
            BATCHES=1,
            DOWNSAMPLE=DOWNSAMPLE,
            TIMESTEPS=[0,12,24,36,48],
            PROCESSING_MODES={
                "map_to_0_1",
                "downsample"
            }
        )
        data_base,aux,CHANNEL_NAMES,boundary_mask_base,CHANNEL_TIMESTEP_MASK_BASE = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=DATA_PATH_GROUPED,
            FILTER_KN_TIME=None,
            BATCHES=1,
            DOWNSAMPLE=DOWNSAMPLE,
            TIMESTEPS=[0,12,24,36,48],
            PROCESSING_MODES={
                "map_to_0_1",
                "downsample"
            }
        )
        data = np.concatenate([data_ko,data_base],axis=0)
        boundary_mask = np.concatenate([boundary_mask_ko,boundary_mask_base],axis=0)
        CHANNEL_TIMESTEP_MASK = np.concatenate([CHANNEL_TIMESTEP_MASK_KO,CHANNEL_TIMESTEP_MASK_BASE],axis=0)

    elif H["knockout"] is not None and H["knockout_mode"]=="all":
        data_ko_0,aux,CHANNEL_NAMES,boundary_mask_ko_0,CHANNEL_TIMESTEP_MASK_KO_0 = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=DATA_PATH_GROUPED,
            FILTER_KN_TIME=0,
            BATCHES=1,
            DOWNSAMPLE=DOWNSAMPLE,
            TIMESTEPS=[0,12,24,36,48],
            PROCESSING_MODES={
                "map_to_0_1",
                "downsample"
            }
        )

        data_ko_24,aux,CHANNEL_NAMES,boundary_mask_ko_24,CHANNEL_TIMESTEP_MASK_KO_24 = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=DATA_PATH_GROUPED,
            FILTER_KN_TIME=24,
            BATCHES=1,
            DOWNSAMPLE=DOWNSAMPLE,
            TIMESTEPS=[0,12,24,36,48],
            PROCESSING_MODES={
                "map_to_0_1",
                "downsample"
            }
        )
        data_base,aux,CHANNEL_NAMES,boundary_mask_base,CHANNEL_TIMESTEP_MASK_BASE = load_micropattern_circle_nodal_knockout_9ch_explicit_colony(
            impath=DATA_PATH_GROUPED,
            FILTER_KN_TIME=None,
            BATCHES=1,
            DOWNSAMPLE=DOWNSAMPLE,
            TIMESTEPS=[0,12,24,36,48],
            PROCESSING_MODES={
                "map_to_0_1",
                "downsample"
            }
        )


        data = np.concatenate([data_ko_0,data_ko_24,data_base],axis=0)
        boundary_mask = np.concatenate([boundary_mask_ko_0,boundary_mask_ko_24,boundary_mask_base],axis=0)
        CHANNEL_TIMESTEP_MASK = np.concatenate([CHANNEL_TIMESTEP_MASK_KO_0,CHANNEL_TIMESTEP_MASK_KO_24,CHANNEL_TIMESTEP_MASK_BASE],axis=0)
    else:
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
    CHANNEL_TIMESTEP_MASK = np.concatenate([CHANNEL_TIMESTEP_MASK,CHANNEL_TIMESTEP_MASK[:,-1:]],axis=1)
    
    
    
    
    if H["knockout"] is None:
        @eqx.filter_jit
        def jittable_callback_bit(x,x_true,OBS_CHANNELS):
            # Here we only want 9 channels - no duplicates - as this is what the NCA sees.
            propagate_xn = lambda x:x.at[1:].set(x[:-1])
            reset_x0 = lambda x,x_true:x.at[0].set(x_true[0])
            # knockout_nodal = lambda x:x.at[_KNOCKOUT:,7].set(0.0) # Set nodal channel to 0 at and after knockout time
            x = jax.tree_util.tree_map(propagate_xn,x) # Set initial condition at each X[n] at next iteration to be final state from X[n-1] of this iteration
            x = jax.tree_util.tree_map(reset_x0,x,x_true) # Keep first initial x correct
            # x = jax.tree_util.tree_map(knockout_nodal,x)
            # Set nodal to 0 at knockout time
            # x = x.at[:]

            for b in range(len(x)//2):
                x[b*2] = x[b*2].at[:,:OBS_CHANNELS].set(x_true[b*2][:,:OBS_CHANNELS]) # Set every other batch of intermediate initial conditions to correct initial conditions
            return x
        KNOCKOUT_ARGS={
            "time":None,
            "channel":None
        }
        target_lr = INIT_LR    # learning rate after warmup
        warmup_steps = 100  # number of steps for warmup
    else:
        _KNOCKOUT = H["knockout"]//12
        if H["knockout_mode"]=="pure":
            @eqx.filter_jit
            def jittable_callback_bit(x,x_true,OBS_CHANNELS):
                # Here we only want 9 channels - no duplicates - as this is what the NCA sees.
                propagate_xn = lambda x:x.at[1:].set(x[:-1])
                reset_x0 = lambda x,x_true:x.at[0].set(x_true[0])
                knockout_nodal = lambda x:x.at[_KNOCKOUT:,7].set(0.0) # Set nodal channel to 0 at and after knockout time
                x = jax.tree_util.tree_map(propagate_xn,x) # Set initial condition at each X[n] at next iteration to be final state from X[n-1] of this iteration
                x = jax.tree_util.tree_map(reset_x0,x,x_true) # Keep first initial x correct
                x = jax.tree_util.tree_map(knockout_nodal,x)
                # Set nodal to 0 at knockout time
                # x = x.at[:]

                # for b in range(len(x)//2):
                    # x[b*2] = x[b*2].at[:,:OBS_CHANNELS].set(x_true[b*2][:,:OBS_CHANNELS]) # Set every other batch of intermediate initial conditions to correct initial conditions
                return x
        elif H["knockout_mode"]=="both":
            @eqx.filter_jit
            def jittable_callback_bit(x,x_true,OBS_CHANNELS):
                # Here we only want 9 channels - no duplicates - as this is what the NCA sees.
                propagate_xn = lambda x:x.at[1:].set(x[:-1])
                reset_x0 = lambda x,x_true:x.at[0].set(x_true[0])
                # knockout_nodal = lambda x:x.at[_KNOCKOUT:,7].set(0.0) # Set nodal channel to 0 at and after knockout time
                x = jax.tree_util.tree_map(propagate_xn,x) # Set initial condition at each X[n] at next iteration to be final state from X[n-1] of this iteration
                x = jax.tree_util.tree_map(reset_x0,x,x_true) # Keep first initial x correct
                # x = jax.tree_util.tree_map(knockout_nodal,x)
                # Apply knockour_nodal to every even batch
                for b in range(len(x)//2):
                    x[b*2] = x[b*2].at[_KNOCKOUT:,7].set(0.0) # Set nodal channel to 0 at and after knockout time for every even batch
    
                
                return x
        elif H["knockout_mode"]=="all":
            @eqx.filter_jit
            def jittable_callback_bit(x,x_true,OBS_CHANNELS):
                # Here we only want 9 channels - no duplicates - as this is what the NCA sees.
                propagate_xn = lambda x:x.at[1:].set(x[:-1])
                reset_x0 = lambda x,x_true:x.at[0].set(x_true[0])
                x = jax.tree_util.tree_map(propagate_xn,x) # Set initial condition at each X[n] at next iteration to be final state from X[n-1] of this iteration
                x = jax.tree_util.tree_map(reset_x0,x,x_true) # Keep first initial x correct
                x[0] = x[0].at[0:,7].set(0.0) # 0h nodal knockout batch
                x[1] = x[1].at[2:,7].set(0.0) # 24h nodal knockout batch
                # x[2] = x[2].at[:,:OBS_CHANNELS].set(x_true[2][:,:OBS_CHANNELS]) # WT batch - set all channels to correct initial conditions
                return x
        KNOCKOUT_ARGS={
            "time":H["knockout"]//12,
            "channel":7
        }
        warmup_steps=1
        target_lr = FINE_LR    # learning rate after warmup
    
    
    
    class DA_subclass(DataAugmenterGrouped):
        def advance_pool(self,x,y,i,key):
            x_true,_ =self.split_x_y(1)	
            x = jittable_callback_bit(x,x_true,self.OBS_CHANNELS)
            x = self.noise(x,NOISE_STRENGTH,key=key)
            self.PREVIOUS_KEY = key
            return x,y
    
    
    
    init_lr = 1e-6      # starting learning rate
    
    warmup_fn = optax.linear_schedule(
        init_value=init_lr,
        end_value=target_lr,
        transition_steps=warmup_steps,
    )

    decay_fn = optax.exponential_decay(
        init_value=target_lr,
        transition_steps=H['TRAINING_ITERATIONS'],
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
        "l2_grouped":["l2_grouped"],
        "vgg":["vgg"],
        "vgg_grouped":["vgg_grouped"],
        "vgg_and_l2":["vgg","l2"],
        "vgg_grouped_and_l2":["vgg_grouped_and_l2"],
        "l2_grad":["l2"],
        "vgg_grad":["vgg"],
        "clip":["clip"],
        "clip_grouped":["clip_grouped"],
        "clip_grouped_and_l2":["clip_grouped_and_l2"],
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

    filename_dict = H_to_filename(H)
    FILENAME_BASE = filename_dict["base"]
    FILENAME_KO = filename_dict["ko"]
    STEPS_BETWEEN_IMAGES = filename_dict["timesteps"]
    
    if H["knockout"] in [0,24]:
        nca = nca.load(PVC_PATH+f"models/{FILENAME_BASE}.eqx")
        FILENAME = FILENAME_KO
    else:
        FILENAME = FILENAME_BASE
    
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
        iters=H['TRAINING_ITERATIONS'],
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
            "D":int(4-onp.log2(H["downsample"])),
            "sharpen":H["ott_sharpen"],
            "epsilon":H["ott_epsilon"],
            "internal_loss_func":H["ott_internal_loss_func"],
            "metric":H["metric"],
            "tau":1.0,
            "normalize":H["loss_normalize"],
            "samples":H["samples"],

        },
        wandb_args={
            "project":"nca-micropatterns-nodal-knockout",
            # "group":"baseline-9ch-texture-train",
            # "group":"baseline-9ch-clip-test-1",
            # "group":"baseline-9ch-train-final-diffusive-scaling",
            # "group":"baseline-9ch-train-final-ott-downsample-fix-2",
            "group":"ko-all-l2-vgg-9ch-final-diffusive-scaling",
            "tags":[f"{k}:{v}" for k,v in H.items()],
            "name":FILENAME
        },
        KNOCKOUT_ARGS=KNOCKOUT_ARGS,
        LOG_EVERY=100,
        CLEAR_CACHE_EVERY=500,
    )
    # except Exception as e:
    #     print(f"Training failed with hyperparameters {H} exception: {e}")
    return key

    
    

def main():

    key = jax.random.PRNGKey(int(time.time()))
    index = int(sys.argv[1])
    TOTAL_JOBS = int(sys.argv[2])

    FULL_HYPERPARAMETERS = {
        # "loss_mode":["ott_grouped_and_l2","vgg_grouped_and_l2"],
        
        "model":["NCA"],
        "optimizer":["nadam"],
        "block_norm":[True],
        "noise_strength":[0.005],
        "multistep":[1,2,4,8],
        "channels":[64],
        "ott_S":[512],
        # "ott_D":[3],
        "learn_rate":[1e-3],
        "downsample":[4,8],
        "ott_sharpen":[True],
        "ott_epsilon":[0.01],
        "samples":[64],
        
        "loss_mode":["vgg_grouped_and_l2"],
        "metric":["l2"],
        "ott_internal_loss_func":["l1"],
        "ott_K":[5],
        "loss_normalize":[False],
        "stepsize_scaling":["diffusive"],
        "steps_at_ds8":[32],
        
        # "loss_mode":["ott_grouped_and_l2","ott_grouped"],
        # "ott_internal_loss_func":["l1","l2"],
        # "metric":["l2"],
        # "ott_K":[7,5,3,2],
        # "loss_normalize":[False],
        # # "stepsize_scaling":["convective","diffusive"],
        # "stepsize_scaling":["convective"],
        # "steps_at_ds8":[64],
        
        # "loss_mode":["clip_grouped_and_l2","clip_grouped"],
        # "ott_internal_loss_func":["l2"],
        # "metric":["l2","l1"],
        # "ott_K":[1],
        # "loss_normalize":[False,True],
        # "stepsize_scaling":["convective","diffusive"],
        # "steps_at_ds8":[64],

        # "loss_mode":["l2_grouped"],
        # "metric":["l2"],
        # "ott_internal_loss_func":["l2"],
        # "ott_K":[5],
        # "loss_normalize":[False],
        # "stepsize_scaling":["convective","diffusive"],
        # "steps_at_ds8":[64],

        "intermediate_growth":[1.0],
        "boundary_reg":[5.0],
        "contiguous_growth":[0.0],
        # "TRAINING_ITERATIONS": [8000],
        # "knockout":[None], # or None for baseline model training
        # "finetune_lr":[1e-4],
        
        "TRAINING_ITERATIONS": [10,100,1000,5000],
        "knockout":[0], # or None for baseline model training
        "knockout_mode":["all"],# "both" or "pure" - whether to fine tune on KO data alone or to also include baseline
        "finetune_lr":[1e-4,1e-5,1e-6],
    }

    HPARAMS = index_to_param_list(index,TOTAL_JOBS,FULL_HYPERPARAMETERS)

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

if __name__ == "__main__":
    main()
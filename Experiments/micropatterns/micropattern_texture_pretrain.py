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
from NCA.trainer.data_augmenter_nca_basic import DataAugmenter
from NCA.model.NCA_gated_model import gNCA
from NCA.model.NCA_model import NCA
from NCA.model.NCA_multi_scale import mNCA
from NCA.trainer.NCA_trainer import NCA_Trainer
from Common.dataloader.micropattern import load_micropattern_circle_8ch_individual
from Common.utils import index_to_param_list
import time
import argparse
from pathlib import Path
import os
key = jax.random.PRNGKey(int(time.time()))



class data_augmenter_subclass(DataAugmenter):
    def data_callback(self,x,y,i,key):
        """
        Called after every training iteration to perform data augmentation and processing		


        Parameters
        ----------
        x : PyTree [BATCHES] f32[N-N_steps,CHANNELS,WIDTH,HEIGHT]
            Initial conditions
        y : PyTree [BATCHES] f32[N-N_steps,CHANNELS,WIDTH,HEIGHT]
            Final states
        i : int
            Current training iteration - useful for scheduling mid-training data augmentation

        Returns
        -------
        x : PyTree [BATCHES] f32[N-N_steps,CHANNELS,WIDTH,HEIGHT]
            Initial conditions
        y : PyTree [BATCHES] f32[N-N_steps,CHANNELS,WIDTH,HEIGHT]
            Final states

        """

        
        x_true,_ =self.split_x_y(1)
        x = jittable_callback_bit(x,x_true,self.OBS_CHANNELS)
        x = self.noise(x,0.01,key=key)
        self.PREVIOUS_KEY = key
        return x,y
		

@eqx.filter_jit
def jittable_callback_bit(x,x_true,OBS_CHANNELS):
	propagate_xn = lambda x:x.at[1:].set(x[:-1])
	reset_x0 = lambda x,x_true:x.at[0].set(x_true[0])
	
	x = jax.tree_util.tree_map(propagate_xn,x) # Set initial condition at each X[n] at next iteration to be final state from X[n-1] of this iteration
	x = jax.tree_util.tree_map(reset_x0,x,x_true) # Keep first initial x correct
			
	for b in range(len(x)//2):
		x[b*2] = x[b*2].at[:,:OBS_CHANNELS].set(x_true[b*2][:,:OBS_CHANNELS]) # Set every other batch of intermediate initial conditions to correct initial conditions
	return x


# argparser = argparse.ArgumentParser()
# argparser.add_argument('--downsample', type=int, help='Resolution downsampling factor', default=1)
# argparser.add_argument('--channels', type=int, help='Number of channels in NCA', default=16)
# args = argparser.parse_args()







index = int(sys.argv[1])
# DATA_PATH = "/projects/u5be/alex_data/Micropatterns/Timecourse_individual_images/*"
DATA_PATH = "Data/Timecourse_individual_images/*"
BATCHES = 2
DOWNSAMPLE = 2
TRAINING_ITERATIONS = 20000
STEPS_BETWEEN_IMAGES = 256#int(256 / np.sqrt(DOWNSAMPLE))
CHANNELS = 32

FULL_HYPERPARAMETERS = {
    # "model":["mNCA","gNCA","NCA"],
    "model":["NCA"],
    "channels":[32],
    "loss_mode":["l2","vgg","both_average"],
    "grad_loss": [True, False]
}

HPARAMS = index_to_param_list(index,4,FULL_HYPERPARAMETERS)


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
    NCA_hyperparameters = {
        "N_CHANNELS":CHANNELS,
        "KERNEL_STR":["ID","LAP","DIFF"],
        "FIRE_RATE":0.5,
        "PADDING":"circular",
        "key":key
    }
    if MODEL == "mNCA":
        NCA_hyperparameters["SCALES"] = [1,2,4,8]
    
    data, aux, CHANNEL_NAMES, boundary_mask = load_micropattern_circle_8ch_individual(
        impath=PVC_PATH+DATA_PATH, 
        BATCHES=BATCHES, 
        DOWNSAMPLE=DOWNSAMPLE,
        TIMESTEPS=[0,12,24,36,48],
        PROCESSING_MODES=["map_to_0_1","downsample"],
    )
    OBS_CHANNELS = 8
    _p = 3 # Takes 250 -> 256, which is nicely divisible by 8
    data = np.pad(data,((0,0),(0,0),(0,0),(_p,_p),(_p,_p)))
    boundary_mask = np.pad(boundary_mask,((0,0),(0,0),(_p,_p),(_p,_p)))
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

    optimiser = optax.chain(optax.scale_by_param_block_norm(), optax.nadam(schedule))
    optimiser = optax.apply_if_finite(optimiser,max_consecutive_errors=5)

    MASK = np.array([
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1]])
        # [1,1,1,1,1,0,0,0]])


    print("-----------------------------------------------------------------------------------------------------")
    print(f"Training gNCA on with STEPS_BETWEEN_IMAGES: {STEPS_BETWEEN_IMAGES} CHANNELS: {CHANNELS}")
    nca = model(**NCA_hyperparameters)
    FILENAME = f"micropattern_circle_8ch_individual_loss_comparison_{nca.get_config()['MODEL']}_t{STEPS_BETWEEN_IMAGES}_ch{CHANNELS}_ds{DOWNSAMPLE}_loss_{LOSS_MODE}_grad_{GRAD_LOSS}_48h"
    opt = NCA_Trainer(
        nca,
        data,
        model_filename=FILENAME,
        DATA_AUGMENTER=data_augmenter_subclass,
        MODEL_DIRECTORY=PVC_PATH+"models/",
        LOG_DIRECTORY=PVC_PATH+"logs/",
        BOUNDARY_MASK=boundary_mask,
        BOUNDARY_MODE="soft",
        GRAD_LOSS=GRAD_LOSS,
        LOSS_TIME_CHANNEL_MASK=MASK,
    )

    if LOSS_MODE == "l2":
        loss_str = ["l2"]
        # loss_channel_func = None
    elif LOSS_MODE == "vgg":
        loss_str = ["vgg"]
        # loss_channel_func = None
    elif LOSS_MODE == "both_average":
        loss_str = ["vgg","l2"]
        # loss_channel_func = None
    # elif LOSS_MODE == "both_split":
    #     loss_str = ["vgg","l2"]
    #     loss_channel_func = onp.ones((OBS_CHANNELS),dtype=np.int32)
    #     loss_channel_func[:OBS_CHANNELS//2]= 0 # Apply vgg to first 4 channels, l2 to other channels
    else:
        raise ValueError("Invalid LOSS_MODE")
    try:
        opt.train(
            t=STEPS_BETWEEN_IMAGES,
            iters=TRAINING_ITERATIONS,
            REGULARISER_COEFFS={
                "intermediate_state":0.1,
                "boundary": 1.0,
                "contiguous_growth":0.0,
            },
            WARMUP=warmup_steps,
            optimiser=optimiser,
            WRITE_IMAGES=True,
            LOSS_FUNC_STR=loss_str,
            wandb_args={
                "project":"nca-micropatterns",
                "group":"individual_8ch_loss_model_comparison_48h",
                "tags":["training",nca.get_config()['MODEL'],str(CHANNELS)+"ch",str(DOWNSAMPLE)+"x_downsample"],
                "name":FILENAME
            },
            LOG_EVERY=100,
            CLEAR_CACHE_EVERY=500,
        )
    except Exception as e:
        print(f"Training failed with hyperparameters {H} exception: {e}")
    return key


for H in HPARAMS:
    key = run_training(H,key)
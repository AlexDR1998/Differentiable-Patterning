import jax
import jax.random as jr
import jax.numpy as np
import equinox as eqx
import os
import sys
import time
import optax
from pprint import pprint
from dotenv import load_dotenv
load_dotenv()
CODE_PATH = os.getenv("PVC_PATH")
DATA_PATH = os.getenv("DATA_PATH_BASE") + "Emojis/"
sys.path.append(CODE_PATH)
os.chdir(CODE_PATH)

from NCA.model.NCA_model import NCA
from NCA.model.NCA_gated_model import gNCA
from NCA.model.NCA_KAN_model import kaNCA
from NCA.trainer.NCA_trainer import NCA_Trainer
from Common.dataloader.emoji import load_emoji_sequence
from Common.utils import index_to_param_list
from NCA.trainer.data_augmenter_nca import DataAugmenter

index = int(sys.argv[1])
TOTAL_JOBS = int(sys.argv[2])
BATCHES = 4
# CHANNELS=32
# DOWNSAMPLE = 1
# STEPS_BETWEEN_IMAGES=64
# iters=8000


def H_to_filename(H):
    if H["regenerate"]:
        regen_str = "regenerate_"
    else:
        regen_str = ""
    FILENAME = f"emoji_al_mi_ro_{H['loss_mode']}_{H['model']}_{regen_str}ch{H['channels']}_ds{H['downsample']}_steps{H['steps_between_images']}_iters{H['iters']}_igc{H['intermediate_growth_coeff']}_brc{H['boundary_reg_coeff']}_cgc{H['contiguous_growth_coeff']}_pcc{H['perturbation_conservation_coeff']}_usc{H['update_sensitivity_coeff']}"
    return FILENAME

key = jr.PRNGKey(int(time.time()))
key = jr.fold_in(key,index)

class data_augmenter_subclass(DataAugmenter):
    #Redefine how data is pre-processed before training
    def data_init(self,SHARDING=None):
        data = self.return_saved_data()
        data = self.duplicate_batches(data, BATCHES)
        data = self.pad(data, [10,10,10,10]) 		
        self.save_data(data)
        return None


def run(H,key):
    
    if H["regenerate"]:
        class data_augmenter_subclass(DataAugmenter):
            #Redefine how data is pre-processed before training
            def data_init(self,SHARDING=None):
                data = self.return_saved_data()
                data = self.duplicate_batches(data, BATCHES)
                data = self.pad(data, [10,10,10,10]) 		
                self.save_data(data)
                return None
    else: # Redifine data_callback to not have regeneration
        class data_augmenter_subclass(DataAugmenter):
            #Redefine how data is pre-processed before training
            def data_init(self,SHARDING=None):
                data = self.return_saved_data()
                data = self.duplicate_batches(data, BATCHES)
                data = self.pad(data, [10,10,10,10]) 		
                self.save_data(data)
                return None
            def data_callback(self, x, y, i, key):
                am=10
                if hasattr(self,"PREVIOUS_KEY"):
                    x = self.unshift(x, am, self.PREVIOUS_KEY)
                    y = self.unshift(y, am, self.PREVIOUS_KEY)
                x_true,_ =self.split_x_y(1)
                x = jittable_callback_bit(x,x_true,self.OBS_CHANNELS)
                x = self.shift(x,am,key=key)
                y = self.shift(y,am,key=key)
                # x = self.zero_random_circle(x,key=key)
                x = self.noise(x,0.005,key=key)

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
	

    loss_str = {
        "l2":["l2"],
        "l1":["l1"],
        "euclidean":["euclidean"],
        "spectral":["spectral"],
        "spectral_no_phase":["spectral_no_phase"],
        "spectral_phase":["spectral_phase"],
        "spectral_euclidean":["spectral","euclidean"],
        "sliced_wasserstein_spatial":["sliced_wasserstein_spatial"],
        "sliced_wasserstein_channel":["sliced_wasserstein_channel"],
        "bhattacharyya":["bhattacharyya"],
        "kl_divergence":["kl_divergence"],
        "hellinger":["hellinger"],
        "bhattacharyya_modified":["bhattacharyya","average_amplitude"],
        "hellinger_modified":["hellinger","average_amplitude"],
        "kl_divergence_modified":["kl_divergence","average_amplitude"],
        "cosine":["cosine"],
        "vgg_3ch":["vgg_3ch"],
    }[H["loss_mode"]]
    
    
    data = load_emoji_sequence(
        ["alien_monster.png","microbe.png","rooster.png","rooster.png"],
        impath_emojis=DATA_PATH,
        downsample=H["downsample"]
    )

    schedule = optax.exponential_decay(1e-3, transition_steps=H["iters"], decay_rate=0.99)
    optimiser = optax.chain(optax.scale_by_param_block_norm(),
                            optax.nadam(schedule))


    print("Training anisotropic nca")
    if H["model"] == "NCA":
         model = NCA
    elif H["model"] == "gNCA":
         model = gNCA
    nca = model(
        H["channels"],
        KERNEL_STR=["ID","LAP","GRAD"],
        KERNEL_SCALE=1,
        FIRE_RATE=0.5,
        PADDING="REPLICATE",
        key=key
    )
    FILENAME = H_to_filename(H)
    opt = NCA_Trainer(nca,
                        data,
                        model_filename=FILENAME,
                        MODEL_DIRECTORY=CODE_PATH+"models/",
                        DATA_AUGMENTER=data_augmenter_subclass,
                        GRAD_LOSS=True)

    opt.train(
        t=H["steps_between_images"],
        iters=H["iters"],
        REGULARISER_COEFFS={
            "intermediate_state":H["intermediate_growth_coeff"],
            "boundary":H["boundary_reg_coeff"],
            "contiguous_growth":H["contiguous_growth_coeff"],
            "perturbation_conservation":H["perturbation_conservation_coeff"],
            "update_sensitivity":H["update_sensitivity_coeff"],
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
            "project":"nca-emojis-thesis-ch1",
            "group":"time-contiguous-comparisons-1",
            # "group":"baseline-9ch-train-1",
            "tags":[f"{k}:{v}" for k,v in H.items()],
            "name":FILENAME
        },
        # KNOCKOUT_ARGS=KNOCKOUT_ARGS,
        LOG_EVERY=100,
        CLEAR_CACHE_EVERY=500,
    )


def main():
    
    HYPERPARAMETERS = {
        "loss_mode":["l2"],
        "model":["NCA"],
        "channels":[32],
        "downsample":[1],
        "steps_between_images":[16,32,64,96,128,192,256],
        "iters":[8000],
        "intermediate_growth_coeff":[0.0],
        "boundary_reg_coeff":[0.0], # emoji data doesn't have a boundary mask
        "contiguous_growth_coeff":[0.0,0.001,0.01,0.1,0.5,1.0,2.0,10.0],
        "perturbation_conservation_coeff":[0.0],
        "update_sensitivity_coeff":[0.0],
        "regenerate":[False,True],
    }

    HPARAMS = index_to_param_list(index,TOTAL_JOBS,HYPERPARAMETERS)

    for H in HPARAMS:
        print("---------------------------------------------------")
        print(f"RUNNING WITH HYPERPARAMS:")
        pprint(H)
        key = jr.fold_in(key,index)
        run(H,key)
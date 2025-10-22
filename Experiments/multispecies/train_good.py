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
from NCA.trainer.data_augmenter_nca import DataAugmenter as EmojiDataAugmenter
from NCA.model.NCA_gated_model import gNCA
from NCA.model.NCA_model import NCA
from NCA.trainer.NCA_trainer import NCA_Trainer
from Common.dataloader.emoji import load_emoji_sequence
from Common.utils import index_to_param_list
import time
from pprint import pprint
class data_augmenter_subclass_regrowth(EmojiDataAugmenter):
    def data_init(self, SHARDING=None):
        data = self.return_saved_data()
        data = self.pad(data, [10, 10, 10, 10])
        self.save_data(data)
        return None
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
        am=10
        if hasattr(self,"PREVIOUS_KEY"):
            x = self.unshift(x, am, self.PREVIOUS_KEY)
            y = self.unshift(y, am, self.PREVIOUS_KEY)
        x_true,_ =self.split_x_y(1)
        x = jittable_callback_bit(x,x_true,self.OBS_CHANNELS)
        x = self.shift(x,am,key=key)
        y = self.shift(y,am,key=key)
        if i > 5000:
            x = self.zero_random_circle(x,key=key)
        x = self.noise(x,0.005,key=key)
        self.PREVIOUS_KEY = key
        return x,y


class data_augmenter_subclass(EmojiDataAugmenter):
    def data_init(self, SHARDING=None):
        data = self.return_saved_data()
        data = self.pad(data, [10, 10, 10, 10])
        self.save_data(data)
        return None
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
        am=10
        if hasattr(self,"PREVIOUS_KEY"):
            x = self.unshift(x, am, self.PREVIOUS_KEY)
            y = self.unshift(y, am, self.PREVIOUS_KEY)
        x_true,_ =self.split_x_y(1)
        x = jittable_callback_bit(x,x_true,self.OBS_CHANNELS)
        x = self.shift(x,am,key=key)
        y = self.shift(y,am,key=key)
        # if i > 10000:
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

    return x


def prepare_data(BATCHES,mode):
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
    # data_filename = "cr_mi_av_al_bt_li_mu"
    data_filename = f"cr_mi_{mode}"
    data = rearrange(data, "T B C W H -> B T C W H") # We are swapping time and batch here. Rather than learning image-image sequence we are learning each image in parallel
    if mode=="patch":
        ic = np.array(data)
        W = ic.shape[-2]
        H = ic.shape[-1]
        ic = ic.at[:, :, :, : W // 2 - 6].set(0)
        ic = ic.at[:, :, :, W // 2 + 5 :].set(0)
        ic = ic.at[:, :, :, :, : H // 2 - 6].set(0)
        ic = ic.at[:, :, :, :, H // 2 + 5 :].set(0)
    elif mode=="pixel":
        ic = np.zeros_like(data)
        center_w = data.shape[-2] // 2
        center_h = data.shape[-1] // 2
        # ic = rearrange(ic, "(B b) T C W H -> B b T C W H", b=BATCHES) # Need to un-rearrange to set pixel correctly
        # for B in range(ic.shape[0]):
            # ic = ic.at[B,:,B+4,center_w,center_h].set(1.0) # For each different species, set different channel to 1 for IC
        # ic = ic.at[:,:,:3,center_w,center_h].set(1.0) # Set first 3 channels to 1 at center pixel
        ic = ic.at[0,:,0,center_w,center_h].set(1.0) # Species 1
        ic = ic.at[1,:,1,center_w,center_h].set(1.0) # Species 2

        ic = ic.at[:,:,4:,center_w,center_h].set(1.0) 
        # ic = rearrange(ic, "B b T C W H -> (B b) T C W H") # Rearrange back
    else:
        raise ValueError("Invalid mode")
    data = np.concatenate(
        [ic, data, data], axis=1
    )  # Join initial condition and data along the time axis
    # Repeat for batches
    data = repeat(data, "B T C W H -> (B b) T C W H", b=BATCHES)
    print("(Batch, Time, Channels, Width, Height): " + str(data.shape))
    return data, data_filename

def build_opt(TRAINING_ITERATIONS):
    init_lr = 1e-6      # starting learning rate
    target_lr = 1e-3    # learning rate after warmup
    warmup_steps = 100
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
    optimiser = optax.chain(optax.scale_by_param_block_norm(),optax.nadam(schedule))
    optimiser = optax.apply_if_finite(optimiser, max_consecutive_errors=5)
    return optimiser

def emoji_task(key,hparams):
    CHANNELS = 32
    BATCHES = 4
    REG_MODE = hparams["regulariser_coeffs"]
    NCA_MODEL = hparams["nca_model"]
    DATA_MODE = hparams["data_mode"]
    DATA_AUGMENTER = hparams["data_augmenter"]
    TRAINING_ITERATIONS = 20000
    warmup_steps = 100
    STEPS_BETWEEN_IMAGES = 128
    NCA_hyperparameters = {
        "N_CHANNELS":CHANNELS,
        "KERNEL_STR":["ID","LAP","GRAD"],
        "FIRE_RATE":0.5,
        "PADDING":"circular",
        "key":key
    }

    if REG_MODE == 0:
        REGULARISER_COEFFS={
            "intermediate_state":0.0,
            "boundary":0.0,
            "contiguous_growth":0.0,
            "update_sensitivity":0.0,
            "perturbation_conservation":0.0,
        } 
        REG_STR = ""
    elif REG_MODE == 1:
        REGULARISER_COEFFS={
            "intermediate_state":0.1,
            "boundary":0.0,
            "contiguous_growth":0.0,
            "update_sensitivity":0.0,
            "perturbation_conservation":0.0,
        }
        REG_STR = "_intermediate_reg"
    elif REG_MODE == 2:
        REGULARISER_COEFFS={
            "intermediate_state":0.1,
            "boundary":0.0,
            "contiguous_growth":1.0,
            "update_sensitivity":0.0,
            "perturbation_conservation":0.0,
        }
        REG_STR = "_intermediate_contiguous_reg"
    else:
        raise ValueError("Invalid REG_MODE")


    if DATA_AUGMENTER == "regrowth":
        DA = data_augmenter_subclass_regrowth
    elif DATA_AUGMENTER == "standard":
        DA = data_augmenter_subclass
    else:
        raise ValueError("Invalid DATA_AUGMENTER")
    data, data_filename = prepare_data(BATCHES,DATA_MODE)
    nca = NCA_MODEL(**NCA_hyperparameters)
    name = f"good_emoji_multi_species_{data_filename}_{nca.get_config()['MODEL']}{REG_STR}_32ch_t128_{DATA_AUGMENTER}"
    
    trainer = NCA_Trainer(
        nca,
        data,
        DATA_AUGMENTER=DA,
        model_filename=name,
        MODEL_DIRECTORY= "models/",
        LOG_DIRECTORY= "logs/",
    )
    
    trainer.train(
        t=STEPS_BETWEEN_IMAGES, 
        iters=TRAINING_ITERATIONS,
        REGULARISER_COEFFS=REGULARISER_COEFFS,
        LOOP_AUTODIFF="checkpointed", 
        optimiser=build_opt(TRAINING_ITERATIONS),
        LOSS_FUNC_STR=["l2"],
        WARMUP=warmup_steps,
        LOG_EVERY=200,
        CLEAR_CACHE_EVERY=500,
        wandb_args={
            "project":"nca_multi_species_signalling_stability",
            "name":name,
            "tags":["multi_species",nca.get_config()['MODEL'],"emoji","long","grad"],
            "group":"nca_dual_species_baseline_models_v2",}
    )

def main():
    key = jr.PRNGKey(int(time.time()))
    index = int(sys.argv[1])
    FULL_HYPERPARAMETERS = {
        # "regulariser_coeffs":[0,1,2],
        "regulariser_coeffs":[2],
        "nca_model":[NCA,gNCA],
        # "data_mode":["pixel","patch"],
        "data_mode":["pixel"],
        "data_augmenter":["standard","regrowth"],
    }
    hparam_list = index_to_param_list(index,4,FULL_HYPERPARAMETERS)
    for hparams in hparam_list:
        key = jr.fold_in(key, index)
        pprint(hparams)
        try:
            emoji_task(key,hparams)
        except Exception as e:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"Error occurred while processing {hparams}: {e}")

main()
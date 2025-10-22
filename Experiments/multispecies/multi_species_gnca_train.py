PVC_PATH = "/mnt/ceph/ar-dp/"
import sys
import os
sys.path.append(PVC_PATH)
os.chdir(PVC_PATH)
import argparse
from NCA.trainer.NCA_trainer import NCA_Trainer
from Common.dataloader.emoji import load_emoji_sequence
from Common.eddie_indexer import index_to_data_nca_type_multi_species
from NCA.trainer.data_augmenter_nca import DataAugmenter
from NCA.model.NCA_model import NCA
from NCA.model.NCA_gated_model import gNCA
from einops import rearrange, repeat
import time
import jax
import jax.numpy as np
import optax
from optax.contrib import muon
import matplotlib.pyplot as plt
# import sys

class data_augmenter_subclass(DataAugmenter):
    def data_init(self, SHARDING=None):
        data = self.return_saved_data()
        data = self.pad(data, [10,10])
        self.save_data(data)
        return None
# PVC_PATH = "/mnt/ceph/ar-dp/"

TRAINING_STEPS = 10000  # How many steps to train for
DOWNSAMPLE = 1  # How much to downsample the image by
NCA_STEPS = 128  # How many NCA steps between each image in the data sequence
BATCH = 2
# CONTIGUOUS_REGULARISER = 0.1  # Weighting for the contiguous regulariser term in the loss function

parser = argparse.ArgumentParser(description="RECORD model training script")
parser.add_argument("--contiguous_regulariser", type=float, default=0.1, help="Weighting for the contiguous regulariser term in the loss function")
args = parser.parse_args()

REGULARISER = args.contiguous_regulariser
key = jax.random.PRNGKey(int(time.time()))
# key = jax.random.fold_in(key, index)

# CHANNELS = [32,48,64,32,48,64][index]  # How many channels to use in the model
# MODE = ["gNCA", "gNCA", "gNCA", "NCA", "NCA", "NCA"][index]  # What mode to use for the model
CHANNELS = 64
MODE = "gNCA"

nca_filename = f"{MODE}_grad_{CHANNELS}ch_contiguous_{REGULARISER}_wide_perturbation_0.1_muon"


    # Redefine how data is pre-processed before training

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
    downsample=DOWNSAMPLE,
    impath_emojis="/projects/u5be/alex_data/Emojis/",
)
# data_filename = "cr_mi_av_al_bt_li_mu"
data_filename = "cr_mi"

data = rearrange(data, "B T C W H -> T B C W H")

data = repeat(data, "B T C W H -> (B b) T C W H", b=BATCH)

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



if MODE == "gNCA":
    print("Training gated NCA model on " + data_filename)

    model = gNCA(
        N_CHANNELS=CHANNELS,
        KERNEL_STR=["ID", "GRAD", "LAP"],
        ACTIVATION=jax.nn.relu,
        PADDING="CIRCULAR",
        FIRE_RATE=0.5,
        key=key,
    )
else:
    print("Training NCA model on " + data_filename)
    model = NCA(
        N_CHANNELS=CHANNELS,
        KERNEL_STR=["ID", "GRAD", "LAP"],
        ACTIVATION=jax.nn.relu,
        PADDING="CIRCULAR",
        FIRE_RATE=0.5,
        key=key,
    )
trainer = NCA_Trainer(
    model,
    data,
    DATA_AUGMENTER=data_augmenter_subclass,
    model_filename="multi_species_stable_" + nca_filename + "_" + data_filename+ "_ds_" + str(DOWNSAMPLE)+"_long",
    MODEL_DIRECTORY= "models/",
    LOG_DIRECTORY= "logs/",
)

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
    transition_steps=TRAINING_STEPS,
    decay_rate=0.98,
)

schedule = optax.join_schedules(
    schedules=[warmup_fn, decay_fn],
    boundaries=[warmup_steps],
)

optimiser = optax.chain(optax.scale_by_param_block_norm(), optax.nadam(schedule))



trainer.train(
    t=NCA_STEPS, 
    iters=TRAINING_STEPS+warmup_steps,
    REGULARISER_COEFFS={
        "intermediate_state":0.1,
        "boundary":0.0,
        "contiguous_growth":1.0,
        "update_sensitivity":0.0,
        "perturbation_conservation":1.0,
    }, 
    LOOP_AUTODIFF="checkpointed", 
    optimiser=optimiser,
    CONTIGUOUS_REGULARISER=0.0,
    WARMUP=warmup_steps,
    LOG_EVERY=200,
    CLEAR_CACHE_EVERY=200,
    wandb_args={
        "project":"multi_species_patterning",
        "name":f"{MODE}_grad_{CHANNELS}ch_multi_species_stable_contiguous_{REGULARISER}_perturbation_0.1_muon",
        "tags":["multi_species","gNCA","grad","emoji","long","hyperparam_sweep"],
        "group":"multi_species_gNCA_stable_contiguous_sweep"}
)

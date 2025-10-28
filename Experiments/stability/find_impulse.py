# PVC_PATH = "/home/alex/PhD/Differentiable-Patterning/"
PVC_PATH = "/mnt/ceph/ar-dp/"
import sys
import os
sys.path.append(PVC_PATH)
os.chdir(PVC_PATH)

from NCA.trainer.NCA_impulse_optimiser import NCA_impulse_optimiser
from Common.dataloader.emoji import load_emoji_sequence
from einops import rearrange
from Common.trainer.abstract_data_augmenter_tree import DataAugmenterAbstract
from Experiments.multispecies.train_good import prepare_data
import jax
import jax.numpy as np
import jax.random as jr
import optax
from NCA.model.NCA_gated_model import gNCA
from einops import repeat
#-------------------- TESTING CODE --------------------



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
data_filename = "cr_mi"

data = rearrange(data, "B T C W H -> T B C W H")

data = repeat(data, "B T C W H -> (B b) T C W H", b=1)

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


# index = int(sys.argv[1])

data,_ = prepare_data(BATCHES=1,mode="patch")
print("Prepared data shape: "+str(data.shape))

CHANNELS = 32
OBS_CHANNELS = 4
key = jr.PRNGKey(0)
nca = gNCA(
    N_CHANNELS=CHANNELS,
    KERNEL_STR = ["ID","LAP","GRAD"],
    ACTIVATION=jax.nn.relu,
    PADDING="CIRCULAR",
    FIRE_RATE=0.5,
    key=key,)

# nca = nca.load("models/multi_species_stable_gNCA_grad_64ch_contiguous_1.0_wide_perturbation_0.1_cr_mi_ds_1_long.eqx")
nca = nca.load("models/good_emoji_multi_species_cr_mi_patch_gNCA_intermediate_contiguous_reg_32ch_t128_regrowth.eqx")
ITERATIONS = 10000
schedule = optax.exponential_decay(
    init_value=1e-3,
    transition_steps=ITERATIONS,
    decay_rate=0.99,
)
opt = NCA_impulse_optimiser(
    NCA_model = nca,
    data = data,
    DATA_AUGMENTER=DataAugmenterAbstract,
    STEPS_TO_STABLE=[128,256],
    FILENAME="test_impulse_hidden",
    MODEL_DIRECTORY= "models/",
    LOG_DIRECTORY= "logs/",
    BOUNDARY_MASK = None,
    BOUNDARY_MODE = "soft",
    OBS_CHANNELS = 4, 
    wandb_args = {
        "name":"test_impulse_hidden",
        "project":"NCA_impulse_optimiser",
        "group":"test_runs"}
)
opt.train(
    iters=ITERATIONS, 
    optimiser=optax.adam(schedule), 
    log_interval=100,
    perturbation_mode = {"channel":"hidden","spatial":"full"},
    )
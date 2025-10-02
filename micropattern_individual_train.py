import jax
import jax.numpy as np
import optax
import equinox as eqx
import sys
from einops import repeat,rearrange
import glob
from NCA.trainer.data_augmenter_nca_basic import DataAugmenter
from NCA.model.NCA_gated_model import gNCA
from NCA.trainer.NCA_trainer import NCA_Trainer
from Common.dataloader.micropattern import load_micropattern_circle_8ch_individual
import time
import argparse
key = jax.random.PRNGKey(int(time.time()))
print(sys.path)


argparser = argparse.ArgumentParser()
argparser.add_argument('--downsample', type=int, help='Resolution downsampling factor', default=1)
argparser.add_argument('--channels', type=int, help='Number of channels in NCA', default=16)
args = argparser.parse_args()


DATA_PATH = "/projects/u5be/alex_data/Micropatterns/Timecourse_individual_images/*"
# print(glob.glob(DATA_PATH+"*.tif"))
BATCHES = 2
DOWNSAMPLE = args.downsample
TRAINING_ITERATIONS = 5000
STEPS_BETWEEN_IMAGES = 256 // DOWNSAMPLE
CHANNELS = args.channels

NCA_hyperparameters = {
    "N_CHANNELS":CHANNELS,
    "KERNEL_STR":["ID","LAP","DIFF"],
    "FIRE_RATE":0.5,
    "PADDING":"circular",
    "key":key
}
FILENAME = f"micropattern_circle_8ch_individual_gNCA_t{STEPS_BETWEEN_IMAGES}_ch{CHANNELS}_ds{DOWNSAMPLE}_v1"


data, aux, CHANNEL_NAMES, boundary_mask = load_micropattern_circle_8ch_individual(
    impath=DATA_PATH, 
    BATCHES=BATCHES, 
    DOWNSAMPLE=DOWNSAMPLE,
    TIMESTEPS=[0,12,24,36,48,60],
    PROCESSING_MODES=["map_to_0_1","downsample"],
)
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

MASK = np.array([
    [1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1],
    [1,1,1,1,1,0,0,0]])


# print("Testing data augmenter...")
# data_augmenter = DataAugmenter(data,hidden_channels=CHANNELS-8)
# x,y = data_augmenter.data_load(key=key)
# print("x shape = ",len(x),x[0].shape)
# print("y shape = ",len(y),y[0].shape)

print("-----------------------------------------------------------------------------------------------------")
print(f"Training gNCA on with STEPS_BETWEEN_IMAGES: {STEPS_BETWEEN_IMAGES} CHANNELS: {CHANNELS}")
nca = gNCA(**NCA_hyperparameters)
opt = NCA_Trainer(
    nca,
    data,
    model_filename=FILENAME,
    DATA_AUGMENTER=DataAugmenter,
    MODEL_DIRECTORY="models/",
    LOG_DIRECTORY="logs/",
    BOUNDARY_MASK=boundary_mask,
    BOUNDARY_MODE="soft",
    LOSS_TIME_CHANNEL_MASK=MASK
)
opt.train(
    t=STEPS_BETWEEN_IMAGES,
    iters=TRAINING_ITERATIONS,
    REGULARISER_COEFFS={
        "intermediate_state":1.0,
        "boundary": 1.0,
        # "contiguous_growth":1.0,
    },
    WARMUP=warmup_steps,
    optimiser=optimiser,
    WRITE_IMAGES=True,
    LOSS_FUNC_STR="euclidean",
    wandb_args={
        "project":"nca-micropatterns",
        "group":"individual_8ch_sampling_channel_sweep",
        "tags":["training","gNCA",str(CHANNELS)+"ch",str(DOWNSAMPLE)+"x_downsample"],
        "name":FILENAME
    },
    LOG_EVERY=100
)
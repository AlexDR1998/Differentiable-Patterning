import jax
import jax.numpy as np
import optax
import equinox as eqx
import sys
from einops import repeat,rearrange
import glob
from NCA.trainer.data_augmenter_nca_basic import DataAugmenter as MicropatternDataAugmenter
from NCA.trainer.data_augmenter_nca import DataAugmenter as EmojiDataAugmenter
from NCA.trainer.optimizer import sam_optimizer,muon_optimizer
from NCA.model.NCA_gated_model import gNCA
from NCA.trainer.NCA_trainer import NCA_Trainer
from Common.dataloader.micropattern import load_micropattern_circle_8ch_individual
from Common.dataloader.emoji import load_emoji_sequence
import time
from pprint import pprint
import argparse
key = jax.random.PRNGKey(int(time.time()))
print(sys.path)


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




def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values are
    'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if 'val' is
    anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError(f"invalid truth value {val}")
argparser = argparse.ArgumentParser()

argparser.add_argument('--optimizer', type=str, help='Optimizer to use (optimistic_adam, nadam, muon, sam)', default="nadam")
argparser.add_argument('--block_norm', type=strtobool, help='Extra transforms to use (True/False)', default=True)
argparser.add_argument('--multistep', type=int, help='Number of steps to accumulate gradients over', default=1)
argparser.add_argument('--task', type=str, help='Task to run (micropattern or emoji)', default="micropattern")
args = argparser.parse_args()

print("Parsed arguments:")
pprint(args)





# print(glob.glob(DATA_PATH+"*.tif"))
BATCHES = 2
CHANNELS = 32
TRAINING_ITERATIONS = 10000
NCA_hyperparameters = {
    "N_CHANNELS":CHANNELS,
    "KERNEL_STR":["ID","LAP","DIFF"],
    "FIRE_RATE":0.5,
    "PADDING":"circular",
    "key":key
}

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
def build_opt(args):
    opt_dict = {
        "optimistic_adam": optax.optimistic_adam_v2(schedule),
        "nadam": optax.nadam(schedule),
        "muon": muon_optimizer(schedule),
        "sam": sam_optimizer(base_optimizer=optax.nadam(schedule),rho=1e-4,sync_period=2),
    }
    if args.optimizer not in opt_dict.keys():
        raise ValueError(f"Optimizer {args.optimizer} not recognized. Available optimizers: {list(opt_dict.keys())}")
    optimiser = opt_dict[args.optimizer]
    modestr = args.optimizer
    if args.multistep>1:
        optimiser = optax.MultiSteps(optimiser, args.multistep)
        modestr += f"_multistep{args.multistep}"
    if args.block_norm:
        optimiser = optax.chain(optax.scale_by_param_block_norm(), optimiser)
        modestr += "_blocknorm"
    optimiser = optax.apply_if_finite(optimiser, max_consecutive_errors=5) # Always wrap in this, just to handle any occasional NaNs
    return optimiser,modestr

def emoji_task():
    class data_augmenter_subclass(EmojiDataAugmenter):
        def data_init(self, SHARDING=None):
            data = self.return_saved_data()
            data = self.pad(data, 10)
            self.save_data(data)
            return None
    STEPS_BETWEEN_IMAGES = 128
    data = load_emoji_sequence(
        [
            "crab.png",
            "microbe.png",
            "avocado.png",
            "alien_monster.png",
            "butterfly.png",
            "lizard.png",
            "mushroom.png",
        ],
        downsample=1,
        impath_emojis="/projects/u5be/alex_data/Emojis/",
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
    nca = gNCA(**NCA_hyperparameters)
    opt,optmode = build_opt(args)
    name = "optimizer_test_"+optmode+"_emoji"
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
            "tags":["multi_species","gNCA","emoji","long","optimizer_test"],
            "group":"multispecies_emoji_morph"}
    )



def micropattern_task():
    DOWNSAMPLE = 4
    STEPS_BETWEEN_IMAGES = int(256 / np.sqrt(DOWNSAMPLE))
    # FILENAME = f"micropattern_circle_8ch_individual_gNCA_t{STEPS_BETWEEN_IMAGES}_ch{CHANNELS}_ds{DOWNSAMPLE}_v4_long"
    DATA_PATH = "/projects/u5be/alex_data/Micropatterns/Timecourse_individual_images/*"
    data, aux, CHANNEL_NAMES, boundary_mask = load_micropattern_circle_8ch_individual(
        impath=DATA_PATH, 
        BATCHES=BATCHES, 
        DOWNSAMPLE=DOWNSAMPLE,
        TIMESTEPS=[0,12,24,36,48,60],
        PROCESSING_MODES=["map_to_0_1","downsample"],
    )
    print("Data shape = " + str(data.shape))
    print("Boundary mask shape = " + str(boundary_mask.shape))


    MASK = np.array([
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,0,0,0]])

    print("-----------------------------------------------------------------------------------------------------")
    # print(f"Training gNCA on with STEPS_BETWEEN_IMAGES: {STEPS_BETWEEN_IMAGES} CHANNELS: {CHANNELS}")
    # nca = gNCA(**NCA_hyperparameters)
    nca = gNCA(**NCA_hyperparameters)
    opt,optmode = build_opt(args)
    name = f"optimizer_test_"+optmode+"_micropattern_t{STEPS_BETWEEN_IMAGES}_ch{CHANNELS}_ds{DOWNSAMPLE}"
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
            # "contiguous_growth":1.0,
        },
        WARMUP=warmup_steps,
        optimiser=opt,
        WRITE_IMAGES=True,
        LOSS_FUNC_STR="euclidean",
        wandb_args={
            "project":"nca_optimizer_test",
            "name":name,
            "tags":["gNCA","micropattern","optimizer_test","long"],
            "group":"individual_8ch_micropattern",
            
        },
        LOG_EVERY=200,
        CLEAR_CACHE_EVERY=500
    )




def main():
    if args.task=="micropattern":
        micropattern_task()
    elif args.task=="emoji":
        emoji_task()
    else:
        raise ValueError(f"Task {args.task} not recognized. Available tasks: micropattern, emoji")

if __name__ == "__main__":
    main()
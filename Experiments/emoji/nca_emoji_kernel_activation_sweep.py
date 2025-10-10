from NCA.model.NCA_model import NCA
from NCA.trainer.NCA_trainer import NCA_Trainer
from Common.utils import load_emoji_sequence
from Common.eddie_indexer import index_to_activations_and_kernels
import time
import jax.random as jr
import optax
import sys



index=int(sys.argv[1])-1
key = jr.PRNGKey(int(time.time()))
key = jr.fold_in(key,index)
key_model,key_trainer = jr.split(key,2)
N_BATCHES = 4
TRAIN_ITERS = 8000
LEARN_RATE = 5e-3
N_CHANNELS = 16
SAMPLING = 64

KERNEL_STR,ACT_STR = index_to_activations_and_kernels(index)

KERNEL_STR_PRINT = "_".join(KERNEL_STR)
FILENAME = "model_exploration/emoji_"+str(N_CHANNELS)+"_channels_"+str(SAMPLING)+"_sampling_"+ACT_STR+"_activation_"+KERNEL_STR_PRINT+"_kernels_v4"


schedule = optax.exponential_decay(LEARN_RATE, transition_steps=TRAIN_ITERS, decay_rate=0.99)
optimiser = optax.chain(optax.scale_by_param_block_norm(),
                        optax.adamw(schedule))

data = load_emoji_sequence(["alien_monster.png","microbe.png","rooster_1f413.png","rooster_1f413.png"],downsample=2)

nca = NCA(N_CHANNELS=N_CHANNELS,
          KERNEL_STR=KERNEL_STR,
          ACTIVATION_STR=ACT_STR,
          PERIODIC=False,
          FIRE_RATE=0.5,
          key=key_model)
opt = NCA_Trainer(nca,data,model_filename=FILENAME)
opt.train(SAMPLING,
          TRAIN_ITERS,
          optimiser=optimiser,
          LOG_EVERY=40,
          key=key_trainer)
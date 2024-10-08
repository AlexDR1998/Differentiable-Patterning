import jax
import optax
import sys
from einops import repeat

from NCA.trainer.data_augmenter_nca import DataAugmenter
from NCA.model.NCA_gated_model import gNCA
from NCA.trainer.NCA_trainer import *
from Common.utils import load_micropattern_time_series_batch_average


index = int(sys.argv[1])-1
key = jax.random.PRNGKey(int(time.time()))
key = jax.random.fold_in(key,index)

STEPS_BETWEEN_IMAGES = 64
BATCHES = 4
DOWNSAMPLE = 12
TRAINING_ITERATIONS = 4000
NCA_hyperparameters = {"N_CHANNELS":16,
                       "KERNEL_STR":["ID","LAP","DIFF"],
                       "FIRE_RATE":0.5,
                       "PADDING":"circular",
                       "key":key}

impath = "../Data//Timecourse 60h June/S2 FOXA2_SOX17_TBXT_LMBR/Max Projections/*"
FILENAME = "timecourse_60h_june/S2_FOXA2_SOX17_TBXT_LMBR/average/gNCA_steps_between_images_"+str(STEPS_BETWEEN_IMAGES)+"_ch_"+str(NCA_hyperparameters["N_CHANNELS"])+"_instance_"+str(index)
schedule = optax.exponential_decay(1e-3, transition_steps=TRAINING_ITERATIONS, decay_rate=0.99)
optimiser = optax.chain(optax.scale_by_param_block_norm(),optax.nadam(schedule))



data = load_micropattern_time_series_batch_average(impath,downsample=DOWNSAMPLE)
data = list(repeat(data,"T C X Y -> B T C X Y", B=BATCHES))

nca = gNCA(**NCA_hyperparameters)
opt = NCA_Trainer(nca,
                  data,
                  model_filename=FILENAME)
opt.train(STEPS_BETWEEN_IMAGES,
          TRAINING_ITERATIONS,
          optimiser=optimiser,
          LOSS_FUNC_STR="euclidean",
          LOG_EVERY=100)
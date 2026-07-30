import jax
import equinox as eqx
import optax
import time
from ABM.model.neural_slime_angle import NeuralSlimeAngle
from ABM.model.neural_chemotaxis import NeuralChemotaxis
from ABM.model.neural_wavelet_chemotaxis import NeuralWaveletChemotaxis
from ABM.model.neural_slime import NeuralSlime
from ABM.trainer.ant_trainer import AntTrainer
from ABM.ABM_visualiser import my_animate_agents
from ABM.trainer.data_augmenter_abm_basic import DataAugmenter
from tqdm import tqdm
from Common.trainer.loss import vgg
from Common.utils import load_textures
from Common.utils import key_array_gen
import matplotlib.pyplot as plt


#data = load_textures(["grid/grid_0057.jpg"],downsample=4,crop_square=True)
#print(data.shape)
#imsize = data.shape[-1]


timesteps = 128
resolution = 64
warmup = 10
init_val = 1e-3
peak_val = 5e-3

cooldown = 60
#iters=5*warmup+5*cooldown
iters = 1000
N_agents = 4000
nslime = NeuralWaveletChemotaxis(N_AGENTS = N_agents, 
                                 GRID_SIZE = resolution, 
                                 N_CHANNELS = 16,
                                 LENGTH_SCALE = 2,
                                 dt = 1.0,
                                 decay_rate = 0.96,
                                 PERIODIC = False,
                                 gaussian_blur = 1)


schedule = optax.exponential_decay(1e-3, transition_steps=iters, decay_rate=0.99)
# schedule = optax.sgdr_schedule([{"init_value":init_val, "peak_value":peak_val, "decay_steps":cooldown, "warmup_steps":warmup, "end_value":init_val},
#                                 {"init_value":init_val, "peak_value":peak_val, "decay_steps":cooldown, "warmup_steps":warmup, "end_value":init_val/2.0},
#                                 {"init_value":init_val/2.0, "peak_value":peak_val/2.0, "decay_steps":cooldown, "warmup_steps":warmup, "end_value":init_val/4.0},
#                                 {"init_value":init_val/4.0, "peak_value":peak_val/4.0, "decay_steps":cooldown, "warmup_steps":warmup, "end_value":init_val/8.0},
#                                {"init_value":init_val/8.0, "peak_value":peak_val/8.0, "decay_steps":cooldown, "warmup_steps":warmup, "end_value":init_val/16.0},])
#optimiser = optax.adam(schedule)
optimiser = optax.chain(optax.scale_by_param_block_norm(),
                        optax.nadam(schedule))
#optimiser = optax.chain(optax.scale(),
#                        optax.adam(schedule))


trainer = AntTrainer(nslime = nslime,
                     int_list = [0,1],
                     BATCHES=4,
                     DATA_AUGMENTER=DataAugmenter,
                     N_agents=N_agents,
                     model_filename="ant_sinkhorn_basic_wavelet_checkpointed_6",
                     alpha=1.0)
trainer.train(timesteps,iters,WARMUP=warmup,optimiser=optimiser)
nslime = trainer.nslime




import jax
import jax.numpy as np
from einops import repeat
from PDE.model.reaction_diffusion_chemotaxis.update import F
from PDE.model.solver.semidiscrete_solver import PDE_solver
from PDE.trainer.PDE_trainer import PDE_Trainer
from NCA.model.NCA_model import NCA
from Common.utils import load_emoji_sequence
from NCA.trainer.data_augmenter_nca import DataAugmenter as DataAugmenterNCA
import time

key = jax.random.PRNGKey(int(time.time()))




CHANNELS = 16
CELL_CHANNELS = 3
SIGNAL_CHANNELS = CHANNELS-CELL_CHANNELS

# Load saved NCA model and data, then run to generate trajectory to train PDE to
data = load_emoji_sequence(["alien_monster.png","microbe.png","rooster_1f413.png","rooster_1f413.png"],downsample=2)
#da = DataAugmenterNCA(data,12)
#da.data_init()
#x0 = np.array(da.split_x_y()[0])[0,0]
#nca = NCA(CHANNELS,KERNEL_STR=["ID","LAP","DIFF"],FIRE_RATE=1.0,PERIODIC=False)
#nca = nca.load("models/model_exploration/emoji_16_channels_64_sampling_relu_activation_ID_DIFF_LAP_kernels_v4.eqx")

#nca_iters = 128
#NCA_trajectory = nca.run(nca_iters,x0)
#NCA_trajectory = repeat(NCA_trajectory,"T C X Y -> B T C X Y",B=1)
#print(NCA_trajectory.shape)

# Define PDE model
func = F(CELL_CHANNELS,SIGNAL_CHANNELS,PADDING="REPLICATE",dx=1.0,key=key)
pde = PDE_solver(func,dt=0.1)

trainer = PDE_Trainer(pde,
                      data,
                      model_filename="pde_chemreacdiff_to_emoji_nca")

trainer.train(nca_iters,100)
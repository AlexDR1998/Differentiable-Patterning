import jax
import jax.numpy as np
import optax
from PDE.trainer.optimiser import non_negative_diffusion_chemotaxis
from einops import repeat
from PDE.model.reaction_diffusion_chemotaxis.update import F
from PDE.model.solver.semidiscrete_solver import PDE_solver
from PDE.trainer.PDE_trajectory_trainer import PDE_Trainer
from NCA.model.NCA_model import NCA
from NCA.model.NCA_gated_model import gNCA
from Common.utils import load_emoji_sequence
from Common.eddie_indexer import index_to_data_nca_type
from NCA.trainer.data_augmenter_nca import DataAugmenter as DataAugmenterNCA
import time
import sys




CHANNELS = 32
CELL_CHANNELS = 3
SIGNAL_CHANNELS = CHANNELS-CELL_CHANNELS
DOWNSAMPLE=1
index=int(sys.argv[1])-1
data_index,nca_type_index = index_to_data_nca_type(index)
key = jax.random.PRNGKey(int(time.time()))
key = jax.random.fold_in(key,index)
# Load saved NCA model and data, then run to generate trajectory to train PDE to

if data_index == 0:
    data = load_emoji_sequence(["crab.png","microbe.png","alien_monster.png","alien_monster.png"],downsample=DOWNSAMPLE)
    data_filename = "cr_mi_al"
if data_index == 1:
    data = load_emoji_sequence(["microbe.png","avocado_1f951.png","alien_monster.png","alien_monster.png"],downsample=DOWNSAMPLE)
    data_filename = "mi_av_al"
if data_index == 2:
    data = load_emoji_sequence(["avocado_1f951.png","mushroom_1f344.png","lizard_1f98e.png","lizard_1f98e.png"],downsample=DOWNSAMPLE)
    data_filename = "av_mu_li"
if data_index == 3:
    data = load_emoji_sequence(["mushroom_1f344.png","alien_monster.png","rooster_1f413.png","rooster_1f413.png"],downsample=DOWNSAMPLE)
    data_filename = "mu_al_ro"


#data = load_emoji_sequence(["alien_monster.png","microbe.png","rooster_1f413.png","rooster_1f413.png"],downsample=2)
da = DataAugmenterNCA(data,28)
da.data_init()
x0 = np.array(da.split_x_y()[0])[0,0]



if nca_type_index==0:
    nca = NCA(CHANNELS,KERNEL_STR=["ID","LAP","GRAD"],KERNEL_SCALE=1,PADDING="REPLICATE")
    nca = nca.load("models/demo_lowres_stable_emoji_anisotropic_nca_"+data_filename+".eqx")
    nca_type="_pure_anisotropic_"
elif nca_type_index==1:
    nca = gNCA(CHANNELS,KERNEL_STR=["ID","LAP","GRAD"],KERNEL_SCALE=1,PADDING="REPLICATE")
    nca = nca.load("models/demo_lowres_stable_emoji_anisotropic_gated_nca_"+data_filename+".eqx")
    nca_type="_gated_anisotropic_"
elif nca_type_index==2:
    nca = NCA(CHANNELS,KERNEL_STR=["ID","LAP","DIFF"],KERNEL_SCALE=1,PADDING="REPLICATE")
    nca = nca.load("models/demo_lowres_stable_emoji_isotropic_nca_"+data_filename+".eqx")
    nca_type="_pure_isotropic_"
elif nca_type_index==3:
    nca = gNCA(CHANNELS,KERNEL_STR=["ID","LAP","DIFF"],KERNEL_SCALE=1,PADDING="REPLICATE")
    nca = nca.load("models/demo_lowres_stable_emoji_isotropic_gated_nca_"+data_filename+".eqx")
    nca_type="_gated_isotropic_"
#nca = NCA(CHANNELS,KERNEL_STR=["ID","LAP","DIFF"],FIRE_RATE=1.0,PERIODIC=False)
#nca = nca.load("models/model_exploration/emoji_16_channels_64_sampling_relu_activation_ID_DIFF_LAP_kernels_v4.eqx")



NCA_trajectory = nca.run(128,x0)
NCA_trajectory = repeat(NCA_trajectory,"T C X Y -> B T C X Y",B=4)
print(NCA_trajectory.shape)


# Define PDE model
func = F(CELL_CHANNELS,SIGNAL_CHANNELS,PADDING="REPLICATE",dx=1.0,key=key)
pde = PDE_solver(func,dt=0.1)


# Define optimiser and lr schedule
iters = 1000
schedule = optax.exponential_decay(1e-4, transition_steps=iters, decay_rate=0.99)
opt = non_negative_diffusion_chemotaxis(schedule)



trainer = PDE_Trainer(pde,
                      NCA_trajectory,
                      model_filename="pde_chemreacdiff_to_emoji"+nca_type+"nca_"+data_filename)

trainer.train(32,iters,optimiser=opt)
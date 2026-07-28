import jax
import jax.random as jr
import jax.numpy as np

from NCA.model.NCA_model import NCA
from NCA.model.NCA_gated_model import gNCA
from NCA.NCA_analyser import nca_fixed_point
from tqdm import tqdm
from Common.utils import load_emoji_sequence
from Common.eddie_indexer import index_to_data_nca_type
from NCA.trainer.data_augmenter_nca import DataAugmenter
import time
import optax
import sys
import optimistix as optx


index=int(sys.argv[1])-1

data_index,nca_type_index = index_to_data_nca_type(index)
CHANNELS=32
DOWNSAMPLE = 1
t=64
iters=8000
PATCH_SIZE=32

#@jax.jit
def sample_random_patch(trajectory,key,size=PATCH_SIZE):
    T = trajectory.shape[0]
    t =  jr.randint(key,(1,),0,T-1)[0]
    inds = jr.randint(key,(2,),0,trajectory.shape[2]-size)
    return trajectory[t,:,inds[0]:inds[0]+size,inds[1]:inds[1]+size]
#@jax.jit
def add_noise(x,key,sigma=0.1):
    return x + jr.normal(key,x.shape)*sigma



key = jax.random.PRNGKey(int(time.time()))
key = jax.random.fold_in(key,index)

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



if nca_type_index==0:
    nca = NCA(CHANNELS,
              KERNEL_STR=["ID","LAP","GRAD"],
              KERNEL_SCALE=1,
              FIRE_RATE=0.5,
              PADDING="REPLICATE",
              key=key)
    
    nca_deterministic = NCA(CHANNELS,
                            KERNEL_STR=["ID","LAP","GRAD"],
                            KERNEL_SCALE=1,
                            FIRE_RATE=1.0,
                            PADDING="REPLICATE",
                            key=key)
    filename = "demo_stable_sparse_emoji_anisotropic_nca_"+data_filename
    


if nca_type_index==1: 
    nca = gNCA(CHANNELS,
               KERNEL_STR=["ID","LAP","GRAD"],
               KERNEL_SCALE=1,
               FIRE_RATE=0.5,
               PADDING="REPLICATE",
               key=key)
    
    nca_deterministic = gNCA(CHANNELS,
                        KERNEL_STR=["ID","LAP","GRAD"],
                        KERNEL_SCALE=1,
                        FIRE_RATE=1.0,
                        PADDING="REPLICATE",
                        key=key)
    filename = "demo_stable_sparse_emoji_anisotropic_gated_nca_"+data_filename
    #nca = nca.load("demo_stable_sparse_emoji_anisotropic_gated_nca_"+data_filename)


if nca_type_index==2:
    nca = NCA(CHANNELS,
              KERNEL_STR=["ID","LAP","DIFF"],
              KERNEL_SCALE=1,
              FIRE_RATE=0.5,
              PADDING="REPLICATE",
              key=key)
    
    nca_deterministic = NCA(CHANNELS,
                        KERNEL_STR=["ID","LAP","DIFF"],
                        KERNEL_SCALE=1,
                        FIRE_RATE=1.0,
                        PADDING="REPLICATE",
                        key=key)
    filename = "demo_stable_sparse_emoji_isotropic_nca_"+data_filename
    #nca = nca.load("demo_stable_sparse_emoji_isotropic_nca_"+data_filename)


if nca_type_index==3: 
    
    nca = gNCA(CHANNELS,
               KERNEL_STR=["ID","LAP","DIFF"],
               KERNEL_SCALE=1,
               FIRE_RATE=0.5,
               PADDING="REPLICATE",
               key=key)
    nca_deterministic = gNCA(CHANNELS,
                        KERNEL_STR=["ID","LAP","DIFF"],
                        KERNEL_SCALE=1,
                        FIRE_RATE=1.0,
                        PADDING="REPLICATE",
                        key=key)
    filename = "demo_stable_sparse_emoji_isotropic_gated_nca_"+data_filename
    #nca = nca.load("demo_stable_sparse_emoji_isotropic_gated_nca_"+data_filename)


nca = nca.load("models/"+filename+".eqx")
nca_deterministic = nca_deterministic.load("models/"+filename+".eqx")

da = DataAugmenter(data,CHANNELS-4)


x,y = da.data_load()
x0 = x[0][0]
#print(x[0].shape)
trajectory = nca.run(t*4,x0)
print(trajectory.shape)

fps = []
for i in tqdm(range(iters)):
    key = jr.fold_in(key,i)
    x_patch = sample_random_patch(trajectory,key)
    x_patch = add_noise(x_patch,key)
    fp = nca_fixed_point(nca_deterministic,x_patch,key)
    fps.append(fp)
fps = np.array(fps)
np.save("data/fps_32_"+filename+".npy",fps)
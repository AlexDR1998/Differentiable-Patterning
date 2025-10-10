from NCA.model.NCA_gated_model import gNCA
from NCA.trainer.NCA_trainer import *
from Common.utils import load_micropattern_radii,load_micropattern_ellipse,load_micropattern_triangle
import matplotlib.pyplot as plt
from NCA.NCA_visualiser import *
#from Common.trainer.data_augmenter_tree_subsample import DataAugmenterSubsampleTexture
from NCA.trainer.data_augmenter_micropattern_texture import DataAugmenterSubsampleMicropatternTexture as DA
import optax
import einops
import numpy as np
import jax.numpy as jnp
import jax
import random
import sys




index=int(sys.argv[1])-1


CHANNELS = 16
t = 64
iters=8000
BATCHES = 1

# Select which subset of data to train on
#data,masks = load_micropattern_radii("../Data/micropattern_radii/Chir_Fgf_*")
data_disc,masks_disc,_ = load_micropattern_radii("../Data//micropattern_shapes/Max Projections */lowres_2/*Disc*")
data_triangle,masks_triangle,_ = load_micropattern_triangle("../Data//micropattern_shapes/Max Projections */lowres_2/*Triangle*")
data_ellipse,masks_ellipse,_ = load_micropattern_ellipse("../Data//micropattern_shapes/Max Projections */lowres_2/*Ellipse*")
print(data_ellipse)
print(len(data_ellipse))
print(data_ellipse[0].shape)
data = [
    data_ellipse[index][...,100:-100,100:-100], data_triangle[index][...,84:-32,84:-32], data_disc[index][...,64:-64,64:-64]
]
masks = masks_ellipse[index:index+1] + masks_triangle[index:index+1] + masks_disc[index:index+1]
#print(masks[0].shape)
for i in range(len(masks)):
    masks[i] = np.ones((1,32,32))
    #data[i] = data[i][...,100:-100,100:-100]
plt.imshow(einops.rearrange(data[1][1][:3],"c x y -> x y c"))
plt.show()
plt.imshow(einops.rearrange(data[0][1][:3],"c x y -> x y c"))
plt.show()
plt.imshow(einops.rearrange(data[2][1][:3],"c x y -> x y c"))
plt.show()
schedule = optax.exponential_decay(1e-2, transition_steps=iters, decay_rate=0.99)
#optimiser= optax.adamw(schedule)
optimiser = optax.chain(optax.scale_by_param_block_norm(),
                        optax.adam(schedule))
# Remove most of the data augmentation - don't need shifting or extra batches or intermediate propagation

    

nca = gNCA(CHANNELS,KERNEL_STR=["ID","LAP","DIFF"],FIRE_RATE=0.5,PERIODIC=False)
opt = NCA_Trainer(nca,
				  data,
				  model_filename="micropattern_shapes_gated_texture_fft_"+str(index),
				  BOUNDARY_MASK=masks,
				  DATA_AUGMENTER = DA)
x0,y0 = opt.DATA_AUGMENTER.data_load()
print(len(x0))
print(x0[0].shape)
print(y0[0].shape)
#plt.imshow(einops.rearrange(x0[0][0][:15],"(c w) x y -> (c x) y w",w=3))
#plt.imshow(einops.rearrange(x0[1][0][:3],"c x y -> x y c"))
#plt.show()

opt.train(t,iters,optimiser=optimiser,LOSS_FUNC_STR="spectral_full")
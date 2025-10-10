from NCA.model.NCA_gated_model import gNCA
from NCA.trainer.NCA_trainer import *
from Common.utils import load_micropattern_radii,load_micropattern_ellipse,load_micropattern_triangle
import matplotlib.pyplot as plt
from NCA.NCA_visualiser import *
import optax
#import einops
import numpy as np
import jax.numpy as jnp
import jax
import random
import sys




index=int(sys.argv[1])-1


CHANNELS = 16
t = 64
iters=4000
BATCHES = 1

# Select which subset of data to train on
#data,masks = load_micropattern_radii("../Data/micropattern_radii/Chir_Fgf_*")
data_disc,masks_disc,_ = load_micropattern_radii("../Data//micropattern_shapes/Max Projections */lowres_2/*Disc*")
data_triangle,masks_triangle,_ = load_micropattern_triangle("../Data//micropattern_shapes/Max Projections */lowres_2/*Triangle*")
data_ellipse,masks_ellipse,_ = load_micropattern_ellipse("../Data//micropattern_shapes/Max Projections */lowres_2/*Ellipse*")
data = data_ellipse[index:index+1] + data_triangle[index:index+1] + data_disc[index:index+1]
masks = masks_ellipse[index:index+1] + masks_triangle[index:index+1] + masks_disc[index:index+1]

#plt.imshow(einops.rearrange(data[1][1][:3],"c x y -> x y c"))
#plt.show()
#plt.imshow(einops.rearrange(data[1][0][:3],"c x y -> x y c"))
#plt.show()

schedule = optax.exponential_decay(1e-2, transition_steps=iters, decay_rate=0.99)
#optimiser= optax.adamw(schedule)
optimiser = optax.chain(optax.scale_by_param_block_norm(),
                        optax.adam(schedule))
# Remove most of the data augmentation - don't need shifting or extra batches or intermediate propagation
class data_augmenter_subclass(DataAugmenter):
    #Redefine how data is pre-processed before training
    def data_init(self,batches):
        data = self.return_saved_data()
        self.save_data(data)
        return None  
    def data_callback(self, x, y, i):
        x_true,_ =self.split_x_y(1)
        reset_x0 = lambda x,x_true:x.at[0].set(x_true[0])
        x = jax.tree_util.tree_map(reset_x0,x,x_true) # Keep first initial x correct
        return x,y

nca = gNCA(CHANNELS,KERNEL_STR=["ID","LAP","DIFF"],FIRE_RATE=0.5,PERIODIC=False)
opt = NCA_Trainer(nca,
				  data,
				  model_filename="micropattern_shapes_gated_mixed_l2_"+str(index),
				  BOUNDARY_MASK=masks,
				  DATA_AUGMENTER = data_augmenter_subclass)
x0,y0 = opt.DATA_AUGMENTER.data_load()
print(len(x0))
print(x0[0].shape)
print(y0[0].shape)
#plt.imshow(einops.rearrange(x0[0][0][:15],"(c w) x y -> (c x) y w",w=3))
#plt.imshow(einops.rearrange(x0[1][0][:3],"c x y -> x y c"))
#plt.show()

opt.train(t,iters,optimiser=optimiser,LOSS_FUNC_STR="l2")
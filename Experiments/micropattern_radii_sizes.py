from NCA.model.NCA_model import NCA
from NCA.trainer.NCA_trainer import *
from Common.utils import load_pickle,load_micropattern_radii
#from NCA_JAX.NCA_visualiser import *
import optax
import numpy as np
import jax.numpy as jnp
import jax
import random
import sys




index=int(sys.argv[1])-1
B = int(sys.argv[2])

CHANNELS = 16
t = 64
iters=21
#BATCHES = 2

# Select which subset of data to train on
# data,masks,shapes = load_micropattern_radii("../Data/micropattern_radii/Chir_Fgf_*")
#data,masks,shapes = load_micropattern_radii("../Data/micropattern_radii/Chir_Fgf_*/processed/*")
# #print(shapes)
#combined = list(zip(data,masks,range(len(data)),shapes))
#combined_sorted = sorted(combined,key=lambda pair:pair[3])

# #random.shuffle(combined)
#data,masks,inds,shapes = zip(*combined_sorted)
#data = list(data)
#masks = list(masks)

#shapes = np.array(list(shapes))
#print(shapes)

ns = np.array([0,13,22,15,21,22,23,10,9]) # Divide up micropatterns based on size A1-B4
ns_sum = np.cumsum(ns)

# data = data[index*B:(index+1)*B]
# masks= masks[index*B:(index+1)*B]
# inds = inds[index*B:(index+1)*B]
# shapes = shapes[index*B:(index+1)*B]
# print("Selected batches: "+str(inds))
# print("Size of selected images: "+str(shapes))

# data = load_pickle("../Data/micropattern_radii/micropattern_data_size_sorted.pickle")
# masks = load_pickle("../Data/micropattern_radii/micropattern_masks_size_sorted.pickle")

data = load_pickle("data/micropattern_data_size_sorted.pickle")
masks = load_pickle("data/micropattern_masks_size_sorted.pickle")

if B==0:
	data = data[ns_sum[index]:ns_sum[index+1]][:ns[index+1]//2]
	masks=masks[ns_sum[index]:ns_sum[index+1]][:ns[index+1]//2]
else:
	data = data[index:(index+B)]
	masks= masks[index:(index+B)]

schedule = optax.exponential_decay(1e-2, transition_steps=iters, decay_rate=0.99)
optimiser= optax.adamw(schedule)

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

nca = NCA(CHANNELS,KERNEL_STR=["ID","LAP","DIFF"],FIRE_RATE=0.5,PERIODIC=False)
opt = NCA_Trainer(nca,
				  data,
				  #model_filename="micropattern_radii_sized_b"+str(B)+"_r1e-2_v2_"+str(index),
				  model_filename="micropattern_radii_sized_b"+str(B)+"_r1e-2_save_test_"+str(index),
				  BOUNDARY_MASK=masks,
				  DATA_AUGMENTER = data_augmenter_subclass)

opt.train(t,iters,optimiser=optimiser,WARMUP=10)
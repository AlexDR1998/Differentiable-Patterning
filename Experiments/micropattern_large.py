import jax
import optax
import sys

from NCA.trainer.data_augmenter_nca import DataAugmenter
from NCA.model.NCA_gated_model import gNCA
from NCA.trainer.NCA_trainer import *
from Common.utils import load_micropattern_time_series
from einops import rearrange
import jax.numpy as np
import jax.random as jr

index = int(sys.argv[1])-1
key = jax.random.PRNGKey(int(time.time()))
key = jax.random.fold_in(key,index)

BATCHES = 3
STEPS_BETWEEN_IMAGES = 64
TRAINING_ITERATIONS = 4000
NCA_hyperparameters = {"N_CHANNELS":16,
                       "KERNEL_STR":["ID","LAP","DIFF"],
                       "FIRE_RATE":0.5,
                       "PADDING":"circular",
                       "key":key}

impath = "../Data//Timecourse 60h June/S2 FOXA2_SOX17_TBXT_LMBR/Max Projections/*"
FILENAME = "timecourse_60h_june/S2_FOXA2_SOX17_TBXT_LMBR/gNCA_steps_between_images_"+str(STEPS_BETWEEN_IMAGES)+"_ch_"+str(NCA_hyperparameters["N_CHANNELS"])+"_instance_"+str(index)
schedule = optax.exponential_decay(1e-3, transition_steps=TRAINING_ITERATIONS, decay_rate=0.99)
optimiser = optax.chain(optax.scale_by_param_block_norm(),optax.nadam(schedule))

class data_augmenter_subclass(DataAugmenter):
        #Redefine how data is pre-processed before training
    def data_init(self,SHARDING=None):
        data = self.return_saved_data()
        self.save_data(data)
        return None  
    @eqx.filter_jit
    def data_callback(self,x,y,i,key):
        """
        Called after every training iteration to perform data augmentation and processing		


        Parameters
        ----------
        x : PyTree [BATCHES] f32[N-N_steps,CHANNELS,WIDTH,HEIGHT]
            Initial conditions
        y : PyTree [BATCHES] f32[N-N_steps,CHANNELS,WIDTH,HEIGHT]
            Final states
        i : int
            Current training iteration - useful for scheduling mid-training data augmentation

        Returns
        -------
        x : PyTree [BATCHES] f32[N-N_steps,CHANNELS,WIDTH,HEIGHT]
            Initial conditions
        y : PyTree [BATCHES] f32[N-N_steps,CHANNELS,WIDTH,HEIGHT]
            Final states

        """
        
        x_true,_ =self.split_x_y(1)
                
        propagate_xn = lambda x:x.at[1:].set(x[:-1])
        reset_x0 = lambda x,x_true:x.at[0].set(x_true[0])
        
        x = jax.tree_util.tree_map(propagate_xn,x) # Set initial condition at each X[n] at next iteration to be final state from X[n-1] of this iteration
        x = jax.tree_util.tree_map(reset_x0,x,x_true) # Keep first initial x correct
        
                
        for b in range(len(x)//2):
            x[b*2] = x[b*2].at[:,:self.OBS_CHANNELS].set(x_true[b*2][:,:self.OBS_CHANNELS]) # Set every other batch of intermediate initial conditions to correct initial conditions
            
        #if i < 1000:
        #x = self.zero_random_circle(x,key=key)
        x = self.noise(x,0.005,key=key)
        
        return x,y
    


data = load_micropattern_time_series(impath,downsample=8)
key = jr.PRNGKey(int(time.time()))
data_subset = []
for i in range(len(data)):
    data_time_subset = []
    key = jr.fold_in(key,i)
    for j in list(jr.randint(key,(BATCHES,),0,len(data[i]))):
        data_time_subset.append(data[i][j])
    data_subset.append(data_time_subset)
#d_arr = np.array(data)
d_arr = np.array(data_subset)
d_arr = rearrange(d_arr,"T B C H W -> B T C H W")
nca = gNCA(**NCA_hyperparameters)
opt = NCA_Trainer(nca,
                  d_arr,
                  model_filename=FILENAME,
                  DATA_AUGMENTER=data_augmenter_subclass,
                  OBS_CHANNELS=4)
opt.train(STEPS_BETWEEN_IMAGES,
          TRAINING_ITERATIONS,
          optimiser=optimiser,
          LOSS_FUNC_STR="euclidean",
          LOG_EVERY=100)
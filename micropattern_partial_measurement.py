import jax
import jax.numpy as np
import jax.random as jr
import optax
import equinox as eqx
import sys
from einops import repeat
import glob
from NCA.trainer.data_augmenter_nca import DataAugmenter
from NCA.model.NCA_gated_model import gNCA
from NCA.trainer.NCA_trainer import *
from Common.utils import load_micropattern_time_series
from einops import rearrange
PVC_PATH = "/mnt/ceph_rbd/ar-dp/"

#index = int(sys.argv[1])-1



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
    


BATCHES = 4
DOWNSAMPLE = 2
TRAINING_ITERATIONS = 2000
STEPS_BETWEEN_IMAGES = 64
CHANNELS = 16
index = 8
key = jr.PRNGKey(int(time.time()))
key = jr.fold_in(key,index)



# for i in range(1,4):  
i=0
impath = PVC_PATH+"Data/Timecourse 60h June/S2 FOXA2_SOX17_TBXT_LMBR/Max Projections/*"
data = load_micropattern_time_series(impath,downsample=DOWNSAMPLE)


key = jr.fold_in(key,i)
data_subset = []
for i in range(len(data)):
    data_time_subset = []
    key = jr.fold_in(key,i)
    for j in list(jr.randint(key,(BATCHES,),0,len(data[i]))):
        data_time_subset.append(data[i][j])
    data_subset.append(data_time_subset)
#d_arr = np.array(data)
data_subset = np.array(data_subset)
data_subset = rearrange(data_subset,"T B C X Y -> B T C X Y")



#data = list(repeat(data,"T C X Y -> B T C X Y", B=BATCHES))
print(data_subset.shape)
data_subset = list(data_subset)






# #indices = np.unravel_index(i,(6,7))
MASK  =  np.array([[[1,1,1,1],                # Channels --->
                    [1,1,1,1],                # Time
                    [1,1,1,1],                #  |
                    [1,1,1,1]],##             #  |
                    [[1,1,1,1],               #  V
                     [1,1,1,1],                #
                     [0,0,0,1],                #
                     [0,0,0,1]],# remove channel           
                    [[1,1,1,1],
                     [0,0,0,1],
                     [1,1,1,1],
                     [0,0,0,1]],# remove timestep
                    [[0,0,1,1],
                     [0,1,0,1],
                     [1,0,0,1],
                     [1,0,0,1]]]) # Only LMBR at all timesteps
MASK = MASK[i]
#print("Training gNCA with STEPS_BETWEEN_IMAGES:",STEPS_BETWEEN_IMAGES,"CHANNELS:",CHANNELS)
NCA_hyperparameters = {"N_CHANNELS":CHANNELS,
                    "KERNEL_STR":["ID","LAP","DIFF"],
                    "FIRE_RATE":0.5,
                    "PADDING":"circular",
                    "key":key}

#print(glob.glob("/mnt/ceph_rbd/ar-dp/*"))
FILENAME = "S2_FOXA2_SOX17_TBXT_LMBR_random_gNCA_steps_between_images_"+str(STEPS_BETWEEN_IMAGES)+"_ch_"+str(NCA_hyperparameters["N_CHANNELS"])+"_masked_"+str(i)+"_instance_"+str(index)
schedule = optax.exponential_decay(1e-4, transition_steps=TRAINING_ITERATIONS, decay_rate=0.99)
optimiser = optax.chain(optax.scale_by_param_block_norm(),optax.nadam(schedule))
    

nca = gNCA(**NCA_hyperparameters)
opt = NCA_Trainer(nca,
                data_subset,
                model_filename=FILENAME,
                DATA_AUGMENTER=data_augmenter_subclass,
                MODEL_DIRECTORY=PVC_PATH+"models/",
                LOG_DIRECTORY=PVC_PATH+"logs/",
                LOSS_TIME_CHANNEL_MASK=MASK)
try:
    opt.train(t=STEPS_BETWEEN_IMAGES,
            iters=TRAINING_ITERATIONS,
            WARMUP=10,
            optimiser=optimiser,
            LOSS_FUNC_STR="euclidean",
            LOG_EVERY=100)
except:
    print("Training failed at step ",i)

#del opt
#del nca
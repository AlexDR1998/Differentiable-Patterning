import jax
import jax.numpy as np
import optax
import equinox as eqx
import sys
from einops import repeat,rearrange
import glob
from NCA.trainer.data_augmenter_nca import DataAugmenter
from NCA.model.NCA_gated_model import gNCA
from NCA.trainer.NCA_trainer import *
from Common.utils import load_micropattern_time_series_batch_average, adhesion_mask_convex_hull_circle, load_micropattern_smad23_lef1_batch_average

PVC_PATH = "/mnt/ceph_rbd/ar-dp/"


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
            #x[b*2] = x[b*2].at[:,:self.OBS_CHANNELS].set(x_true[b*2][:,:self.OBS_CHANNELS]) # Set every other batch of intermediate initial conditions to correct initial conditions
            #'' Do for every 2nd timestep, as every other timestep has no true data
            x[b*2] = x[b*2].at[::2,:self.OBS_CHANNELS].set(x_true[b*2][::2,:self.OBS_CHANNELS]) # Set every other batch of intermediate initial conditions to correct initial conditions
            
        #if i < 1000:
        #x = self.zero_random_circle(x,key=key)
        x = self.noise(x,0.005,key=key)
        
        return x,y
    


BATCHES = 4
DOWNSAMPLE = 2
TRAINING_ITERATIONS = 2000
STEPS_BETWEEN_IMAGES = 32
CHANNELS = 10
index = 3
key = jax.random.PRNGKey(int(time.time()))
key = jax.random.fold_in(key,index)

impath_fstl = PVC_PATH+"Timecourse 60h June/S2 FOXA2_SOX17_TBXT_LMBR/Max Projections/*"
impath_sll = PVC_PATH+"Timecourse 60h June/Smad23_LEF 48h/Max Projections/*"
data_fstl = load_micropattern_time_series_batch_average(impath_fstl,downsample=DOWNSAMPLE,VERBOSE=False)
data_sll = load_micropattern_smad23_lef1_batch_average(impath_sll,downsample=DOWNSAMPLE,VERBOSE=False)
data_fstl = np.array(data_fstl)
data_sll = np.array(data_sll)

data_fstl = np.concatenate([data_fstl[:1],np.zeros((1,*data_fstl.shape[1:])),data_fstl[1:]],axis=0)
data_sll = np.concatenate([data_sll,np.zeros((1,*data_sll.shape[1:]))],axis=0)
data_fstl = data_fstl.at[1,-1].set(data_sll[1,1])
data_full = np.concatenate([data_fstl,data_sll[:,0:1],data_sll[:,2:]],axis=1)
zero_slice = np.zeros((data_full[0:1].shape))
data_full = np.concatenate([data_full[:1],
                            data_full[1:2],
                            data_full[2:3],
                            zero_slice,
                            data_full[3:4],
                            zero_slice,
                            data_full[4:5],
                            zero_slice,
                            data_full[5:6],
                            zero_slice,
                            data_full[6:]],
                        axis=0)

#data = load_micropattern_time_series_batch_average(impath,downsample=DOWNSAMPLE)
boundary_mask = adhesion_mask_convex_hull_circle(rearrange(data_fstl[0],"C X Y -> X Y C"))[0]

boundary_mask = repeat(boundary_mask,"X Y -> B () X Y",B=BATCHES)
data = repeat(data_full,"T C X Y -> B T C X Y", B=BATCHES)

print("Data shape: ",data.shape)
print("Boundary mask shape: ",boundary_mask.shape)
data = list(data)


key = jax.random.fold_in(key,index)



#for M in range(4):
MASK = np.array([#[1,1,1,1,1,1],
                 [0,0,0,1,1,1],
                 [1,1,1,1,1,1],
                 [0,0,0,0,0,0],
                 [1,1,1,1,1,1],
                 [0,0,0,0,0,0],
                 [1,1,1,1,1,1],
                 [0,0,0,0,0,0],
                 [1,1,1,1,1,1],
                 [0,0,0,0,0,0],
                 [1,1,1,1,0,0]])
#M = 4
#MASK = MASKS[M]
NCA_hyperparameters = {"N_CHANNELS":CHANNELS,
                    "KERNEL_STR":["ID","LAP","DIFF"],
                    "FIRE_RATE":0.5,
                    "PADDING":"circular",
                    "key":key}

#print(glob.glob("/mnt/ceph_rbd/ar-dp/*"))
FILENAME = "S2_FOXA2_SOX17_TBXT_LMBR_SMAD23_LEF1_average_gNCA_boundary_mask_steps_between_images_"+str(STEPS_BETWEEN_IMAGES)+"_ch_"+str(NCA_hyperparameters["N_CHANNELS"])+"_instance_"+str(index)
schedule = optax.exponential_decay(1e-3, transition_steps=TRAINING_ITERATIONS, decay_rate=0.99)
optimiser = optax.chain(optax.scale_by_param_block_norm(),optax.nadam(schedule))
    

print("Training gNCA with STEPS_BETWEEN_IMAGES:",STEPS_BETWEEN_IMAGES,"CHANNELS:",CHANNELS)
nca = gNCA(**NCA_hyperparameters)
opt = NCA_Trainer(nca,
                data,
                model_filename=FILENAME,
                DATA_AUGMENTER=data_augmenter_subclass,
                MODEL_DIRECTORY=PVC_PATH+"models/",
                LOG_DIRECTORY=PVC_PATH+"logs/",
                BOUNDARY_MASK=boundary_mask,
                LOSS_TIME_CHANNEL_MASK=MASK)
opt.train(t=STEPS_BETWEEN_IMAGES,
        iters=TRAINING_ITERATIONS,
        WARMUP=10,
        optimiser=optimiser,
        WRITE_IMAGES=True,
        UPDATE_DATA_EVERY=1,
        LOSS_FUNC_STR="euclidean",
        LOG_EVERY=100)

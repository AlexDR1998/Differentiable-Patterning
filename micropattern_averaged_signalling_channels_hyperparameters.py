import jax
import jax.numpy as np
import optax
import equinox as eqx
import sys
from einops import repeat,rearrange
import glob
from NCA.trainer.data_augmenter_nca import DataAugmenter
from NCA.model.NCA_gated_model import gNCA
from NCA.trainer.NCA_trainer import NCA_Trainer
from Common.utils import load_micropattern_time_series, adhesion_mask_convex_hull_circle, load_micropattern_time_series_nodal_lef_cer, load_micropattern_smad23_lef1
import time
import itertools
PVC_PATH = "/mnt/ceph_rbd/ar-dp/"



index = int(sys.argv[1])
full_hyperparameters = {"CHANNELS": [12,16,24,32],
                       "BOUNDARY_REGULARISER": [0.01,0.1,0.5,1.0],
                       "USE_LR_WARMUP": [True,False],
                       "STEPS_BETWEEN_IMAGES": [32,64,128],}

def index_to_param_list(index,n_processes=7):
    """
    We have array of hyperparameters, and M parallel GPUs, so we divide up the set of hyperparameters onto the GPUs
    """
    #n_processes = jax.process_count()
    keys = list(full_hyperparameters.keys())
    values = [full_hyperparameters[k] for k in keys]
    all_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    return all_combinations[index::n_processes]

HYPERPARAMETER_LIST = index_to_param_list(index)    

@eqx.filter_jit
def callback_helper(x,x_true,y,i,key,OBS_CHANNELS):
    propagate_xn = lambda x:x.at[1:].set(x[:-1])
    reset_x0 = lambda x,x_true:x.at[0].set(x_true[0])
    
    x = jax.tree_util.tree_map(propagate_xn,x) # Set initial condition at each X[n] at next iteration to be final state from X[n-1] of this iteration
    x = jax.tree_util.tree_map(reset_x0,x,x_true) # Keep first initial x correct
            
    for b in range(len(x)//2):
        x[b*2] = x[b*2].at[:,:OBS_CHANNELS].set(x_true[b*2][:,:OBS_CHANNELS]) # Set every other batch of intermediate initial conditions to correct initial conditions
        
    #if i < 1000:
    #x = self.zero_random_circle(x,key=key)
    return x,y


class data_augmenter_subclass(DataAugmenter):
        #Redefine how data is pre-processed before training
    def data_init(self,SHARDING=None):
        data = self.return_saved_data()
        self.save_data(data)
        return None  
    #@eqx.filter_jit
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
                
        # propagate_xn = lambda x:x.at[1:].set(x[:-1])
        # reset_x0 = lambda x,x_true:x.at[0].set(x_true[0])
        
        # x = jax.tree_util.tree_map(propagate_xn,x) # Set initial condition at each X[n] at next iteration to be final state from X[n-1] of this iteration
        # x = jax.tree_util.tree_map(reset_x0,x,x_true) # Keep first initial x correct
        
                
        # for b in range(len(x)//2):
        #     x[b*2] = x[b*2].at[:,:self.OBS_CHANNELS].set(x_true[b*2][:,:self.OBS_CHANNELS]) # Set every other batch of intermediate initial conditions to correct initial conditions
        #     #'' Do for every 2nd timestep, as every other timestep has no true data
        #     #x[b*2] = x[b*2].at[::2,:self.OBS_CHANNELS].set(x_true[b*2][::2,:self.OBS_CHANNELS]) # Set every other batch of intermediate initial conditions to correct initial conditions
            
        x,y = callback_helper(x,x_true,y,i,key,self.OBS_CHANNELS)
        # #if i < 1000:
        # #x = self.zero_random_circle(x,key=key)
        x = self.noise(x,0.005,key=key)
        
        return x,y
    


BATCHES = 4
DOWNSAMPLE = 2
TRAINING_ITERATIONS = 2000


key = jax.random.PRNGKey(int(time.time()))
key = jax.random.fold_in(key,index)

impath_fstl = PVC_PATH+"Data/Timecourse 60h June/S2 FOXA2_SOX17_TBXT_LMBR/Max Projections/*"     # Foxa2, Sox17, TbxT, Lmbr
impath_nlc = PVC_PATH+"Data/Nodal_LEFTY_CER/**"                                                  # Lmbr, Cer Lefty, Nodal
impath_ls = PVC_PATH+"Data/Timecourse 60h June/Smad23_LEF 48h/Max Projections/*"            # Lef1, Lmbr, Smad23
data_fstl = load_micropattern_time_series(impath_fstl,downsample=DOWNSAMPLE,VERBOSE=False,BATCH_AVERAGE=True)               # 0h, 12h, 24h, 36h, 48h, 60h
data_nlc = load_micropattern_time_series_nodal_lef_cer(impath_nlc,downsample=DOWNSAMPLE,VERBOSE=False,BATCH_AVERAGE=True)   # 0h, 6h, 12h, 24h, 36h, 48h
data_ls = load_micropattern_smad23_lef1(impath_ls,downsample=DOWNSAMPLE,VERBOSE=False,BATCH_AVERAGE=True)                   # 0h, 6h, 12h, 24h, 36h, 48h
data_fstl = np.array(data_fstl)
data_nlc = np.array(data_nlc)[:,:,0] # select only condition 1 from data
data_ls = np.array(data_ls)



# print("---- Before removing 6h and duplicate LMBR ----")

# Try without the 6h data first - it makes the timestepping a lot simpler
data_nlc = np.concatenate([data_nlc[:1],data_nlc[2:],np.zeros((1,*data_nlc.shape[1:]))],axis=0)
data_ls = np.concatenate([data_ls[:1],data_ls[2:],np.zeros((1,*data_ls.shape[1:]))],axis=0)

# Remove duplicates of LMBR channel
data_nlc = data_nlc[:,:,1:]
data_ls = data_ls[:,:,:1] # Also remove smad23 channel as guillaume recommended



# Combine the datasets
data = np.concatenate([data_fstl,data_nlc,data_ls],axis=2)
boundary_mask = adhesion_mask_convex_hull_circle(rearrange(data_fstl[0,0],"C X Y -> X Y C"))[0]
boundary_mask = repeat(boundary_mask,"X Y -> B () X Y",B=BATCHES)
data = repeat(data,"T () C X Y -> B T C X Y", B=BATCHES)
data = data*rearrange(boundary_mask,"B () X Y -> B () () X Y")

key = jax.random.fold_in(key,index)
MODE = "full"
MASK = np.array([[1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,0,0,0,0]])


for hyperparameters in HYPERPARAMETER_LIST:
    try:
        STEPS_BETWEEN_IMAGES = hyperparameters["STEPS_BETWEEN_IMAGES"]
        key = jax.random.fold_in(key,index)
        NCA_hyperparameters = {"N_CHANNELS":hyperparameters["CHANNELS"],
                            "KERNEL_STR":["ID","LAP","DIFF"],
                            "FIRE_RATE":0.5,
                            "PADDING":"circular",
                            "key":key}


        if hyperparameters["USE_LR_WARMUP"]:    
            warmup_steps = 100  # number of steps for warmup
            init_lr = 1e-6      # starting learning rate
            target_lr = 1e-3    # learning rate after warmup

            warmup_fn = optax.linear_schedule(
                init_value=init_lr,
                end_value=target_lr,
                transition_steps=warmup_steps,
            )

            decay_fn = optax.exponential_decay(
                init_value=target_lr,
                transition_steps=TRAINING_ITERATIONS,
                decay_rate=0.98,
            )

            schedule = optax.join_schedules(
                schedules=[warmup_fn, decay_fn],
                boundaries=[warmup_steps],
            )
            TRAINING_ITERATIONS += warmup_steps
        else:
            warmup_steps = 10
            schedule = optax.exponential_decay(
                init_value=1e-3,
                transition_steps=TRAINING_ITERATIONS,
                decay_rate=0.98,
            )
        optimiser = optax.chain(optax.scale_by_param_block_norm(), optax.nadam(schedule))



        nca = gNCA(**NCA_hyperparameters)
        print("-----------------------------------------------------------------------------------------------------")
        print(f"Training gNCA with hyperparameters: {hyperparameters}")
        FILENAME = f"FOXA2_SOX17_TBXT_LMBR_CER_LEFTY_NODAL_LEF1_average_gNCA_boundary_regulariser_{hyperparameters['BOUNDARY_REGULARISER']}_steps_between_images_{STEPS_BETWEEN_IMAGES}_ch_{NCA_hyperparameters['N_CHANNELS']}_warmup_{hyperparameters['USE_LR_WARMUP']}_mode_{MODE}.eqx"
        opt = NCA_Trainer(nca,
                        data,
                        model_filename=FILENAME,
                        DATA_AUGMENTER=data_augmenter_subclass,
                        MODEL_DIRECTORY=PVC_PATH+"models/",
                        LOG_DIRECTORY=PVC_PATH+"logs/",
                        BOUNDARY_MASK=boundary_mask,
                        BOUNDARY_MODE="soft",
                        LOSS_TIME_CHANNEL_MASK=MASK)
        opt.train(t=STEPS_BETWEEN_IMAGES,
                iters=TRAINING_ITERATIONS,
                BOUNDARY_REGULARISER=hyperparameters["BOUNDARY_REGULARISER"],
                WARMUP=warmup_steps,
                optimiser=optimiser,
                WRITE_IMAGES=True,
                LOSS_FUNC_STR="euclidean",
                CLEAR_CACHE_EVERY=200,
                LOG_EVERY=200)
    except Exception as e:
        print(f"Error with hyperparameters {hyperparameters}: {e}")
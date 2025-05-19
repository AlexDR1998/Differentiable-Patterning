import jax
import jax.numpy as np
import optax
import equinox as eqx
import sys
from einops import repeat
import glob
from NCA.trainer.data_augmenter_nca import DataAugmenter
from NCA.model.NCA_gated_model import gNCA
from NCA.trainer.NCA_trainer import *
from Common.utils import load_micropattern_time_series_batch_average

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
    


BATCHES = 8
DOWNSAMPLE = 4
TRAINING_ITERATIONS = 2000

impath = PVC_PATH+"Data/Timecourse 60h June/S2 FOXA2_SOX17_TBXT_LMBR/Max Projections/*"
data = load_micropattern_time_series_batch_average(impath,downsample=DOWNSAMPLE)
data = list(repeat(data,"T C X Y -> B T C X Y", B=BATCHES))
print(data[0].shape)

index = 5
key = jax.random.PRNGKey(int(time.time()))
key = jax.random.fold_in(key,index)

for i in range(6*7):
    key = jax.random.fold_in(key,i)
    indices = np.unravel_index(i,(6,7))

    STEPS_BETWEEN_IMAGES = [4,8,16,32,64,128][indices[0]]
    CHANNELS = [4,6,8,12,16,24,32][indices[1]]

    print("Training gNCA with STEPS_BETWEEN_IMAGES:",STEPS_BETWEEN_IMAGES,"CHANNELS:",CHANNELS)
    NCA_hyperparameters = {"N_CHANNELS":CHANNELS,
                        "KERNEL_STR":["ID","LAP","DIFF"],
                        "FIRE_RATE":0.5,
                        "PADDING":"circular",
                        "key":key}

    #print(glob.glob("/mnt/ceph_rbd/ar-dp/*"))
    FILENAME = "S2_FOXA2_SOX17_TBXT_LMBR_average_gNCA_steps_between_images_"+str(STEPS_BETWEEN_IMAGES)+"_ch_"+str(NCA_hyperparameters["N_CHANNELS"])+"_instance_"+str(index)
    schedule = optax.exponential_decay(1e-3, transition_steps=TRAINING_ITERATIONS, decay_rate=0.99)
    optimiser = optax.chain(optax.scale_by_param_block_norm(),optax.nadam(schedule))
        

    nca = gNCA(**NCA_hyperparameters)
    opt = NCA_Trainer(nca,
                    data,
                    model_filename=FILENAME,
                    DATA_AUGMENTER=data_augmenter_subclass,
                    MODEL_DIRECTORY=PVC_PATH+"models/",
                    LOG_DIRECTORY=PVC_PATH+"logs/")
    try:
        opt.train(t=STEPS_BETWEEN_IMAGES,
                iters=TRAINING_ITERATIONS,
                WARMUP=10,
                optimiser=optimiser,
                LOSS_FUNC_STR="euclidean",
                LOG_EVERY=100)
    except:
        print("Training failed at STEPS_BETWEEN_IMAGES:",STEPS_BETWEEN_IMAGES,"CHANNELS:",CHANNELS)
        
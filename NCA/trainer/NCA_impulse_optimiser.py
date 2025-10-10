import jax
import jax.numpy as np
import jax.random as jr
import optax
import equinox as eqx
import datetime
import Common.trainer.loss as loss
import jaxpruner
from einops import repeat, rearrange
from Common.trainer.abstract_wandb_log import Train_log
from Common.utils import key_pytree_gen
from Common.model.boundary import model_boundary, hard_boundary, no_boundary

# from NCA.trainer.boundary_tensorboard_log import boundary_train_log
from tqdm import tqdm
from jaxtyping import Float,Array,Key

import time


class NCA_impulse_optimiser(object):
    """
        This class takes a pre-trained NCA and finds localised perturbations (impulses) that trigger the NCA to switch between its stable configurations.
    """
    def __init__(self,
                 NCA_model,
                 data,
                 DATA_AUGMENTER,
                 STEPS_TO_STABLE,
                 FILENAME,
                 BOUNDARY_MASK = None,
                 BOUNDARY_MODE = "soft",
				 OBS_CHANNELS = None,
                 LOG_DIRECTORY ="logs/",
                 MODEL_DIRECTORY = "models/"):
        
        self.NCA_model = NCA_model
        self.CHANNELS = self.NCA_model.N_CHANNELS
        if OBS_CHANNELS is None:    
           self.OBS_CHANNELS = data[0].shape[1]
        else:
            self.OBS_CHANNELS = OBS_CHANNELS
        
        
        

		# Set up data and data augmenter class
        self._data_raw = data
        self.DATA_AUGMENTER = DATA_AUGMENTER(data,self.CHANNELS-self.OBS_CHANNELS)
        self.DATA_AUGMENTER.data_init()
        self.data = self.DATA_AUGMENTER.return_saved_data() # Data with all the hidden channels set up
        self.BATCHES = len(self.data) 
        print("Batches = "+str(self.BATCHES))

        self.STEPS_TO_STABLE = STEPS_TO_STABLE
        self.setup_logging(self.data,{"name":FILENAME,"project":"NCA_impulse_optimiser"})
        
        # self.BOUNDARY_CALLBACK = []
        # for b in range(self.BATCHES):
        #     if BOUNDARY_MASK is not None:
        #         if BOUNDARY_MODE=="soft":
        #             self.BOUNDARY_CALLBACK.append(model_boundary(BOUNDARY_MASK[b]))
        #         elif BOUNDARY_MODE=="hard":
        #             self.BOUNDARY_CALLBACK.append(hard_boundary(BOUNDARY_MASK[b]))
        #     else:
        #         self.BOUNDARY_CALLBACK.append(no_boundary())
        state = self.DATA_AUGMENTER.data_load(key=jr.PRNGKey(0))[0][0]
        print("Initial state shape: "+str(state.shape))
        
        init_data = self.generate_stable_configurations(16,self.NCA_model,jr.PRNGKey(0)) # Generate a pool of stable configurations to train against

        data_for_log = rearrange(init_data,"POOL BATCHES CHANNELS W H -> (BATCHES POOL) W H CHANNELS")[:,:,:,:3]
        
        self.logger.log_image("Train/configurations",data_for_log,step=1)
        # print("Successfully generated stable configurations, shape = " + str(init_data.shape))
        # print(init_data.shape)
    def setup_logging(self,data,wandb_args):
        self.logger = Train_log(data,wandb_args)


    def generate_stable_configurations(self,POOL_SIZE,nca,key):
        """
        Generate a pool of stable configurations by running the NCA from initial conditions taken from the training data

        Returns
        -------
        stable_configurations : float32 array [POOL_SIZE,BATCH,CHANNELS,WIDTH,HEIGHT]
        """
        final_states = []
        for i in tqdm(range(POOL_SIZE)):
            # choose an initial state (wrap-around if POOL_SIZE > available data)
            key = jr.fold_in(key,i)
            state = self.DATA_AUGMENTER.data_load(key)[0][0] # Load x from x,y pair, and take first timestep of x
            # iterate the NCA for the required number of steps
            v_nca = jax.vmap(nca,in_axes=(0,None,None),out_axes=0,axis_name="B")
            def nca_step(carry,j):
                key,x = carry
                key = jr.fold_in(key,j)
                x = v_nca(x,lambda x:x,key)
                # x = v_nca(x, self.BOUNDARY_CALLBACK, jr.split(key,len(x)))
                return (key,x),None
            (key,state),_ = eqx.internal.scan(nca_step,(key,state),xs=np.arange(self.STEPS_TO_STABLE),kind="lax")
            final_states.append(state)
        # stack and return/save the final states only
        return np.stack(final_states)





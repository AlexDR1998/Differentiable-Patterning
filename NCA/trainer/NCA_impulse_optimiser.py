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
    def __init__(
            self,
            NCA_model,
            data,
            DATA_AUGMENTER,
            STEPS_TO_STABLE,
            FILENAME,
            BOUNDARY_MASK = None,
            BOUNDARY_MODE = "soft",
            OBS_CHANNELS = None,
            LOG_DIRECTORY ="logs/",
            MODEL_DIRECTORY = "models/"
        ):
        
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

        data_for_log = rearrange(init_data[:,:,:27],"POOL BATCHES (C1 C2 C3) W H -> (BATCHES C1 W) (POOL C2 H) C3",C3=3,C1=3,C2=3)
        # print(f"Logging {data_for_log.shape[0]} stable configurations for visualisation")
        # import matplotlib.pyplot as plt
        # plt.imshow(data_for_log[0])
        # plt.show()
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
        stable_configurations : float32 array [POOL_SIZE,CONDITION,CHANNELS,WIDTH,HEIGHT]
        """
        final_states = []
        for i in tqdm(range(POOL_SIZE)):
            # choose an initial state (wrap-around if POOL_SIZE > available data)
            key = jr.fold_in(key,i)
            state = np.array(self.DATA_AUGMENTER.data_load(key)[0])[:,0] # Load x from x,y pair, and take first timestep of x
            print(f"Initial condition {i} shape: "+str(state.shape))
            # iterate the NCA for the required number of steps
            v_nca = jax.vmap(nca,in_axes=(0,None,None),out_axes=0,axis_name="B")
            def nca_step(carry,j):
                key,x = carry
                key = jr.fold_in(key,j)
                x = v_nca(x,lambda x:x,key)
                # x = v_nca(x, self.BOUNDARY_CALLBACK, jr.split(key,len(x)))
                return (key,x),None
            (key,state),_ = eqx.internal.scan(nca_step,(key,state),xs=np.arange(self.STEPS_TO_STABLE),kind="lax")
            print(f"Final state {i} shape: "+str(state.shape))
            final_states.append(state)
        # stack and return/save the final states only
        return np.stack(final_states)


    def loss_func(self,x,y):
        """
            Handles observable channel selection for loss calculation.
            Parameters
            ----------
            x : float32 array [BATCHES,CHANNELS,WIDTH,HEIGHT]
                Predicted states from NCA
            y : float32 array [BATCHES,CHANNELS,WIDTH,HEIGHT]
                Target stable configurations
            Returns
            -------
            loss_value : float32 [BATCHES]
                Loss value reduced over channel and spatial axes
        """
        x_obs = x[:,:self.OBS_CHANNELS]
        y_obs = y[:,:self.OBS_CHANNELS]
        return self._loss_func(x_obs,y_obs,key=jr.PRNGKey(0),where=None)


    def train(
            self,
            iters,
            optimiser = optax.adam(1e-3),
            log_interval = 100,
            LOSS_FUNC_STR = "l2",
            RESAMPLE_EVERY = 100,
            key=jr.PRNGKey(int(time.time()))
        ):
        LOSS_FUNCS = {
			"l2":loss.l2,
			"l1":loss.l1,
			"vgg":loss.vgg_hyperspectral,#lambda x,y,key,where:loss.vgg_hyperspectral(x,y,key,where,experiment_groups=LOSS_ARGS["experiment_groups"]),
			"vgg_3ch":loss.vgg,
			"euclidean":loss.euclidean,
			"spectral":loss.spectral,
			"spectral_full":loss.spectral_weighted,
			# "rand_euclidean":lambda x,y,key:loss.random_sampled_euclidean(x,y,key=key)
		}
        self.OPTIMISER = optimiser
        self._loss_func = LOSS_FUNCS[LOSS_FUNC_STR]
        @eqx.filter_jit
        def makestep(dx,nca,x,y,t,opt_state,key):
            """
            Single optimisation step to find impulse dx that drives NCA from x to y in t steps
            We want to find dx such that NCA^t(x+dx) -> y 
            Parameters
            ----------
            dx : float32 array [BATCHES,CHANNELS,WIDTH,HEIGHT]
                Current impulse perturbation to be optimised
            nca : NCA model
                The NCA model to be optimised
            x : float32 array [BATCHES,CHANNELS,WIDTH,HEIGHT]
                Stable configuration initial conditions
            y : float32 array [BATCHES,CHANNELS,WIDTH,HEIGHT]
                Stable configuration target states
            t : int
                Number of steps to run NCA for
            opt_state : optax optimiser state
                Current optimiser state
            key : jr.PRNGKey
                Jax random key
            Returns
            -------
            dx : float32 array [BATCHES,CHANNELS,WIDTH,HEIGHT]
                Updated impulse perturbation
            opt_state : optax optimiser state
                Updated optimiser state
            loss_value : float32
                Current loss value
            """

            @eqx.filter_value_and_grad(has_aux=True)
            def compute_loss(dx,nca,x,y,t,key):
                v_nca = jax.vmap(nca,in_axes=(0,None,0),out_axes=0,axis_name="B")
                def nca_step(carry,j):
                    key,x = carry
                    key = jr.fold_in(key,j)
                    keys = jr.split(key,len(x))
                    x = v_nca(x,lambda x:x,keys)
                    # x = v_nca(x, self.BOUNDARY_CALLBACK, jr.split(key,len(x)))
                    return (key,x),None
                # Apply impulse at first step only
                x = x + dx
                (key,x),_ = eqx.internal.scan(nca_step,(key,x),xs=np.arange(t),kind="lax")
                loss_batches = self.loss_func(x,y)
                loss = np.mean(loss_batches)
                aux = (x,loss_batches)
                return loss,aux
            loss_aux,grads = compute_loss(dx,nca,x,y,t,key)
            
            updates, opt_state = self.OPTIMISER.update(grads, opt_state, dx)
            dx = eqx.apply_updates(dx, updates)
            aux = {
                "mean_loss":loss_aux[0],
                "loss_batches":loss_aux[1][1],
                "final_states":loss_aux[1][0],
            }
            return dx, opt_state, aux




        pbar = tqdm(range(iters))
        stable_pool = self.generate_stable_configurations(self.BATCHES,self.NCA_model,key)
        x = stable_pool[:,0]
        y = stable_pool[:,1]
        dx = np.zeros_like(x[:1]) # Initialise impulse perturbation to zero. single batch, so shape is [1,CHANNELS,WIDTH,HEIGHT]
        opt_state = self.OPTIMISER.init(dx)
        for i in pbar:
            key = jr.fold_in(key,i)
            dx, opt_state, aux = makestep(dx,self.NCA_model,x,y,self.STEPS_TO_STABLE,opt_state,key)
            pbar.set_description(f"Loss: {aux['mean_loss']:.6f}")
            if i % log_interval == 0:
                self.logger.log({"Train/loss":aux['mean_loss']},step=i)
                self.logger.log_image(
                    tag = "Train/output",
                    images = rearrange(aux['final_states'][:,:,:27],"POOL BATCHES (C1 C2 C3) W H -> (BATCHES C1 W) (POOL C2 H) C3",C3=3,C1=3,C2=3),
                    step=i
                )
                # self.logger.log_image("Train/output")
                # Log the impulse perturbation
                # impulse_for_log = rearrange(dx,"B (C1 C2 C3) W H -> (C1 W) (B C2 H) C3",C3=3,C1=4,C2=1)
                # self.logger.log_image("Train/impulse",impulse_for_log,step=i)
            if RESAMPLE_EVERY > 0 and (i+1) % RESAMPLE_EVERY == 0:
                stable_pool = self.generate_stable_configurations(self.BATCHES,self.NCA_model,key)
                x = stable_pool[:,0]
                y = stable_pool[:,1]

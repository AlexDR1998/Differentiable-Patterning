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
            MODEL_DIRECTORY = "models/",
            wandb_args = {
                "name":"switch_signal_hidden",
                "project":"NCA_impulse_optimiser",
                "group":"test_runs"}
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
        self.BATCHES = 16 
        print("Batches = "+str(self.BATCHES))

        self.STEPS_TO_STABLE = STEPS_TO_STABLE
        self.setup_logging(
            self.data,wandb_args)
        
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
        for i in range(POOL_SIZE):
            # choose an initial state (wrap-around if POOL_SIZE > available data)
            key = jr.fold_in(key,i)
            state = np.array(self.DATA_AUGMENTER.data_load(key)[0])[:,0] # Load x from x,y pair, and take first timestep of x
            # print(f"Initial condition {i} shape: "+str(state.shape))
            # iterate the NCA for the required number of steps
            v_nca = jax.vmap(nca,in_axes=(0,None,None),out_axes=0,axis_name="B")
            def nca_step(carry,j):
                key,x = carry
                key = jr.fold_in(key,j)
                x = v_nca(x,lambda x:x,key)
                # x = v_nca(x, self.BOUNDARY_CALLBACK, jr.split(key,len(x)))
                return (key,x),None
            steps = jr.randint(key, (), minval=self.STEPS_TO_STABLE[0], maxval=self.STEPS_TO_STABLE[1])
            (key,state),_ = eqx.internal.scan(nca_step,(key,state),xs=np.arange(steps),kind="lax")
            # print(f"Final state {i} shape: "+str(state.shape))
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


    def init_dx(self,x):
        """
        
        """
        
        CH = {
            "all":np.s_[:1,:self.CHANNELS],
            "obs":np.s_[:1,:self.OBS_CHANNELS],
            "hidden":np.s_[:1,self.OBS_CHANNELS:],
            "single":np.s_[:1,self.OBS_CHANNELS:self.OBS_CHANNELS+1]
        }[self.perturbation_mode['channel']]
        SP = {
            "full":np.s_[:,:],
            "pixel":np.s_[:1,:1],
            "patch":np.s_[:3,:3],
            "flat":np.s_[:1,:1],
        }[self.perturbation_mode['spatial']]
        inds = CH + SP
        dx = np.zeros_like(x[inds]) # Initialise impulse perturbation to zero
        return dx

    def update_x(self,x,dx):
        """
            Update the NCA state x with impulse perturbation dx
            Parameters
            ----------
            x : float32 array [BATCHES,CHANNELS,WIDTH,HEIGHT]
                Current NCA states
            dx : float32 array [BATCHES,CHANNELS,WIDTH,HEIGHT]
                Impulse perturbations to be applied
            Returns
            -------
            x_updated : float32 array [BATCHES,CHANNELS,WIDTH,HEIGHT]
                Updated NCA states
        """
        CH = {
            "all":np.s_[:,:self.CHANNELS],
            "obs":np.s_[:,:self.OBS_CHANNELS],
            "hidden":np.s_[:,self.OBS_CHANNELS:],
            "single":np.s_[:,self.OBS_CHANNELS:self.OBS_CHANNELS+1]
        }[self.perturbation_mode['channel']]
        l = [64,64]
        SP = {
            "full":np.s_[:,:],
            "pixel":np.s_[l[0]:l[0]+1,l[1]:l[1]+1],
            "patch":np.s_[l[0]:l[0]+3,l[1]:l[1]+3],
            "flat":np.s_[:,:],
        }[self.perturbation_mode['spatial']]
        inds = CH + SP
        x = x.at[inds].set(x[inds] + dx)
        return x

    def train(
            self,
            iters,
            optimiser,
            perturbation_mode = {"channel":"single","spatial":"patch"},
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
        self.perturbation_mode = perturbation_mode
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
                # x = x + dx
                x = self.update_x(x,dx)

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
        print(f"X shape: {x.shape}, Y shape: {y.shape}")
        # dx = np.zeros_like(x[:1,self.OBS_CHANNELS:self.OBS_CHANNELS+1,]) # Initialise impulse perturbation to zero. single batch, so shape is [1,CHANNELS,WIDTH,HEIGHT]
        dx = self.init_dx(x)
        print(f"Initial dx shape: {dx.shape}")
        opt_state = self.OPTIMISER.init(dx)
        for i in pbar:
            key = jr.fold_in(key,i)
            t = int(jr.randint(key, (), minval=self.STEPS_TO_STABLE[0], maxval=self.STEPS_TO_STABLE[1]))
            
            dx, opt_state, aux = makestep(dx,self.NCA_model,x,y,t,opt_state,key)
            
            pbar.set_description(f"Loss: {aux['mean_loss']:.6f}")
            
            
            
            if i % log_interval == 0:
                self.logger.log({"Train/loss":aux['mean_loss']},step=i)
                self.logger.log_image(
                    tag = "Train/output",
                    images = rearrange(aux['final_states'][:,:27],"POOL (C1 C2 C3) W H -> POOL (C1 W) (C2 H) C3",C3=3,C1=3,C2=3),
                    step=i
                )
                # self.logger.log_image(
                #     tag = "Train/impulse",
                #     images = rearrange(dx,"() C W H -> () (C W) H () "),
                #     step=i
                # )
                # self.logger.log_image("Train/output")
                # Log the impulse perturbation
                # impulse_for_log = rearrange(dx,"B (C1 C2 C3) W H -> (C1 W) (B C2 H) C3",C3=3,C1=4,C2=1)
                # self.logger.log_image("Train/impulse",impulse_for_log,step=i)
            if RESAMPLE_EVERY > 0 and (i+1) % RESAMPLE_EVERY == 0 and i>0:
                stable_pool = self.generate_stable_configurations(self.BATCHES,self.NCA_model,key)
                x = stable_pool[:,0]
                y = stable_pool[:,1]
                print(f"X shape: {x.shape}, Y shape: {y.shape}")


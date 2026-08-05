import jax
import jax.numpy as np
import jax.random as jr
import numpy as onp
import optax
import equinox as eqx
import datetime
import Common.trainer.loss as loss
from Common.model.abstract_model import AbstractModel
import jaxpruner
from einops import repeat, rearrange
# from Common.trainer.abstract_wandb_log import Train_log
from NCA.trainer.impulse_tensorboard_log import Impulse_Train_log
from Common.utils import key_pytree_gen
from Common.model.boundary import model_boundary, hard_boundary, no_boundary
from NCA.model.NCA_perturbation import perturbation
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
            STEPS_FROM_STABLE,
            FILENAME,
            BOUNDARY_MASK = None,
            BOUNDARY_MODE = "soft",
            OBS_CHANNELS = None,
            LOG_DIRECTORY ="logs/",
            MODEL_DIRECTORY = "models/",
            OUTPUT_DIRECTORY = "perturbations/",
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
        self.STEPS_TO_STABLE = STEPS_TO_STABLE
        self.STEPS_FROM_STABLE = STEPS_FROM_STABLE
        self.setup_logging(self.data,wandb_args)
        self.OUTPUT_DIRECTORY = OUTPUT_DIRECTORY
        self.FILENAME = FILENAME
        state = self.DATA_AUGMENTER.initialize_pool(key=jr.PRNGKey(0))[0][0]
        print("Initial state shape: "+str(state.shape))
        

    def setup_logging(self,data,wandb_args):
        self.logger = Impulse_Train_log(data,wandb_args)

    def run_nca_model_batch(self,x,t,key):
        """
            Runs the NCA model on a batch of inputs x for t steps
            Parameters
            ----------
            x : float32 array [BATCHES,CHANNELS,WIDTH,HEIGHT]
                Initial states to run NCA from
            t : int
                Number of steps to run NCA for
            key : jr.PRNGKey
                Jax random key
            Returns
            -------
            final_x : float32 array [BATCHES,CHANNELS,WIDTH,HEIGHT]
                Final state after running NCA for t steps
            xs : float32 array [BATCHES,t,CHANNELS,WIDTH,HEIGHT]
        """
        v_nca = jax.vmap(self.NCA_model,in_axes=(0,None,0),out_axes=0,axis_name="B")
        def nca_step(carry,j):
            key,x = carry
            key = jr.fold_in(key,j)
            keys = jr.split(key,len(x))
            x = v_nca(x,lambda x:x,keys)
            # x = v_nca(x, self.BOUNDARY_CALLBACK, jr.split(key,len(x)))
            return (key,x),x
        (key,final_x),xs = eqx.internal.scan(nca_step,(key,x),xs=np.arange(t),kind="lax")
        xs = rearrange(xs,"T B C W H -> B T C W H")
        return final_x,xs

    def generate_stable_configurations(self,key):
        """
        Generate a pool of stable configurations by running the NCA from initial conditions taken from the training data

        Returns
        -------
        stable_configurations : float32 array [POOL_SIZE,CONDITION,CHANNELS,WIDTH,HEIGHT]
        """
        final_states = []
        for i in range(self.BATCHES):
            # choose an initial state (wrap-around if POOL_SIZE > available data)
            key = jr.fold_in(key,i)
            state = np.array(self.DATA_AUGMENTER.initialize_pool(key)[0])[:,0] # Load x from x,y pair, and take first timestep of x
            steps = jr.randint(key, (), minval=self.STEPS_TO_STABLE[0], maxval=self.STEPS_TO_STABLE[1])
            state,_ = self.run_nca_model_batch(state,steps,key)
            final_states.append(state)
        # stack and return/save the final states only
        return np.stack(final_states)

    def generate_init_target_pairs(self,key):
        stable_pool = self.generate_stable_configurations(key)
        x = stable_pool[:,self.initial_index]
        y = stable_pool[:,self.target_index]
        return x,y

    def run_full_trajectory(self,key,steps,dx_func):
        """
            Computes the full NCA trajectory including before and after the impulse perturbation, for logging/evaluation
        """

        x0 = np.array(self.DATA_AUGMENTER.initialize_pool(key)[0])[self.initial_index,0] # Load x from x,y pair, and take first timestep of x
        x0 = rearrange(x0,"C W H -> () C W H")
        final_1,trajectory_1 = self.run_nca_model_batch(x0,steps,key)
        # x1 = self.update_x(final_1,dx)
        x1 = dx_func(final_1)
        final_2,trajectory_2 = self.run_nca_model_batch(x1,steps*4,key)
        trajectory_1 = onp.array(trajectory_1[:,:,:3])
        trajectory_2 = onp.array(trajectory_2[:,:,:3])
        full_trajectory = onp.concatenate([trajectory_1,trajectory_2],axis=1)
        return full_trajectory

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
            optimiser,
            BATCHES = 16,
            initial_index = 0,
            target_index = 1,
            perturbation_mode = {"channel":"single","spatial":"patch"},
            perturbation_reg_coeff = {
                "l2":0.0,
                "l1":0.0,
                "smooth":0.0,
                "max":0.0,
                "in_0_1":0.0
            },
            perturbation_width = 0.05,
            log_interval = 100,
            LOSS_FUNC_STR = "l2",
            RESAMPLE_EVERY = 200,
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
		}
        def _smooth_reg(x,dx):
            avg_dx = jax.lax.conv_general_dilated(
                lhs=dx,
                rhs=np.ones((1,1,3,3))/(3*3),
                window_strides=(1,1),
                padding='SAME',
            )
            return np.mean((dx - avg_dx)**2)
        REG_FUNCS = {
            "l2":lambda x,dx: np.mean(dx**2),
            "l1":lambda x,dx: np.mean(np.abs(dx)),
            "smooth":_smooth_reg,
            "max":lambda x,dx: np.max(np.abs(dx)),
            "in_0_1":lambda x,dx: np.mean(np.abs(x)+np.abs(x-1)-1),
		}
        

        # Filter REG_FUNCS to the same set (optional but keeps things consistent)
        perturbation_reg_coeff = {name:perturbation_reg_coeff[name] for name in perturbation_reg_coeff.keys() if perturbation_reg_coeff[name]!=0.0}
        REG_FUNCS = {name: REG_FUNCS[name] for name in perturbation_reg_coeff.keys()}
            # Multiply each regularization function by its coefficient
        REG_FUNCS = {name: lambda x, dx, f=func, c=perturbation_reg_coeff[name]: c * f(x, dx) 
                        for name, func in REG_FUNCS.items()}
        self.BATCHES = BATCHES
        print("Batches = "+str(self.BATCHES))

        self.initial_index = initial_index
        self.target_index = target_index
        self.OPTIMISER = optimiser
        # self.perturbation_mode = perturbation_mode
        # self.perturbation_location = perturbation_location
        self._loss_func = LOSS_FUNCS[LOSS_FUNC_STR]
        @eqx.filter_jit
        def makestep(dx_func,nca,x,y,t,opt_state,key):
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
            # def compute_reg_loss(x,dx):
            #     reg_loss = 0.0
            #     for name in perturbation_reg_coeff.keys():
            #         reg_loss = reg_loss + perturbation_reg_coeff[name]*REG_FUNCS[name](x,dx)
            #     return reg_loss

            @eqx.filter_value_and_grad(has_aux=True)
            def compute_loss(dx_func,nca,x,y,t,key):
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
                x_dx = dx_func(x)
                dx_reg = dx_func.regulariser(x,REG_FUNCS)
                # x_dx = self.update_x(x,dx)
                # dx_reg = compute_reg_loss(x_dx,dx)

                (key,x),_ = eqx.internal.scan(nca_step,(key,x_dx),xs=np.arange(t),kind="lax")
                loss_batches = self.loss_func(x,y)
                mean_loss = np.mean(loss_batches)
                aux = (x,loss_batches,mean_loss,dx_reg,x_dx)
                loss = mean_loss + dx_reg
                return loss,aux
            loss_aux,grads = compute_loss(dx_func,nca,x,y,t,key)
            dx_func_diff,dx_func_static = eqx.partition(dx_func, eqx.is_inexact_array)
            updates, opt_state = self.OPTIMISER.update(grads, opt_state, dx_func_diff)
            dx_func_diff = eqx.apply_updates(dx_func_diff, updates)
            dx_func = eqx.combine(dx_func_diff, dx_func_static)
            aux = {
                "total_loss":loss_aux[0],
                "final_states":loss_aux[1][0],
                "loss_batches":loss_aux[1][1],
                "mean_loss":loss_aux[1][2],
                "dx_reg":loss_aux[1][3],
                "x_dx":loss_aux[1][4]
            }
            return dx_func, opt_state, aux


        pbar = tqdm(range(iters))
        loss_best = 1e5
        dx_best = None
        x,y = self.generate_init_target_pairs(key)
        dx_func = perturbation(
            perturbation_mode,
            CHANNELS=self.CHANNELS,
            OBS_CHANNELS=self.OBS_CHANNELS,
            x=x,
            WIDTH=perturbation_width,
            key=key)
        dx_func_diff,_ = eqx.partition(dx_func, eqx.is_inexact_array)
        opt_state = self.OPTIMISER.init(dx_func_diff)

        print(f"X shape: {x.shape}, Y shape: {y.shape}")
        print("Perturbation module:")
        print(dx_func)
        print(f"Initial dx_func values shape: {dx_func.get_values().shape}")
        # dx = np.zeros_like(x[:1,self.OBS_CHANNELS:self.OBS_CHANNELS+1,]) # Initialise impulse perturbation to zero. single batch, so shape is [1,CHANNELS,WIDTH,HEIGHT]

        # print(f"Initial dx shape: {dx.shape}")
        # dx_func_diff,_ = eqx.partition(dx_func, eqx.is_inexact_array)
        for i in pbar:
            key = jr.fold_in(key,i)
            t = int(jr.randint(key, (), minval=self.STEPS_FROM_STABLE[0], maxval=self.STEPS_FROM_STABLE[1]))

            dx_func, opt_state, aux = makestep(dx_func,self.NCA_model,x,y,t,opt_state,key)
            pbar.set_description(f"Loss: {aux['total_loss']:.6f}, Best loss: {loss_best:.6f}, Reg: {aux['dx_reg']:.6f}")
            aux["dx"] = dx_func.get_values()
            aux["dx_location"] = dx_func.get_location()
            self.logger.log_training(aux,i,log_interval)

            if RESAMPLE_EVERY > 0 and (i+1) % RESAMPLE_EVERY == 0 and i>0:
                x, y = self.generate_init_target_pairs(key)

            if aux['mean_loss']<loss_best:
                loss_best = aux['mean_loss']
                dx_best = dx_func
        print("Saving best perturbation...")
        dx_best.save(self.OUTPUT_DIRECTORY+self.FILENAME,overwrite=True)
        print("Loading best perturbation and running final trajectory...")
        print(f"Saving as {self.OUTPUT_DIRECTORY+self.FILENAME}")
        dx_best = dx_best.load(self.OUTPUT_DIRECTORY+self.FILENAME)
        T = self.run_full_trajectory(key,128,dx_best)

        self.logger.log_final_trajectory(T)
        self.logger.finish()
        

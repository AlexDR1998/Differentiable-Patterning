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
            FILENAME,
            OBS_CHANNELS = None,
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
        

        self.data = data # Shape [T C W H]
        self.setup_logging(repeat(data,"T C W H -> B T C W H",B=1),wandb_args)
        self.OUTPUT_DIRECTORY = OUTPUT_DIRECTORY
        self.FILENAME = FILENAME
        
    def setup_logging(self,data,wandb_args):
        self.logger = Impulse_Train_log(data,wandb_args)
    
        
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


    def generate_init_target_pairs(self,key):
        x = self.data[:1] # Shape [1 Channels Width Height] 
        x = repeat(x,"1 C W H -> B C W H",B=self.BATCHES)
        # steps = jr.randint(key, (), minval=self.STEPS_TO_STABLE[0], maxval=self.STEPS_TO_STABLE[1])
        steps = self.STEPS_TO_TARGET
        y,_ = self.run_nca_model_batch(x,steps,key)
        
        return x,y

    def train(
            self,
            iters,
            optimiser,
            BATCHES = 16,
            STEPS_TO_TARGET = 64,
            perturbation_mode = {"channel":"obs","spatial":"global"},
            optimisation_mode = "maximal_preservative",
            perturbation_reg_coeff = {
                "l2":0.0,
                "l1":0.0,
                "smooth":0.0,
                "max":0.0,
                "in_0_1":0.0
            },
            log_interval = 100,
            LOSS_FUNC_STR = "l2",
            RESAMPLE_EVERY = 200,
            key=jr.PRNGKey(int(time.time()))
    ):
        self.optimisation_mode = optimisation_mode
        assert self.optimisation_mode in ["maximal_preservative","minimal_destructive"], "MODE must be either 'maximal_preservative' or 'minimal_destructive'"
        self.STEPS_TO_TARGET = STEPS_TO_TARGET
        self.BATCHES = BATCHES
        LOSS_FUNCS = {
			"l2":loss.l2,
			"l1":loss.l1,
			"vgg":loss.vgg_hyperspectral,#lambda x,y,key,where:loss.vgg_hyperspectral(x,y,key,where,experiment_groups=LOSS_ARGS["experiment_groups"]),
			"vgg_grouped":loss.vgg_hyperspectral_colony,
			"vgg_grouped_and_l2":loss.vgg_hyperspectral_colony_and_l2,
			"vgg_3ch":loss.vgg,
			"euclidean":loss.euclidean,
			"cosine":loss.cosine,
			"spectral":loss.spectral,
			"spectral_no_phase":loss.spectral_no_phase,
			"spectral_phase":loss.spectral_only_phase,
			# "sliced_wasserstein_spatial":lambda x,y,key,where:loss.sliced_wasserstein_spatial(x,y,key,where,aux={"samples":LOSS_ARGS["samples"]}),
			# "sliced_wasserstein_channel":lambda x,y,key,where:loss.sliced_wasserstein_channel(x,y,key,where,aux={"samples":LOSS_ARGS["samples"]}),
			# "bhattacharyya":loss.bhattacharyya_distance,
			# "kl_divergence":loss.kl_divergence,
			# "hellinger":loss.hellinger_distance,
			# "average_amplitude":loss.average_amplitude_distance,
			# "ott":lambda x,y,key,where:loss_ott.ott_loss(x,y,key,where,aux=_ott_aux),
			# "ott_chstack":lambda x,y,key,where:loss_ott.ott_channel_stack_loss(x,y,key,where,aux=_ott_aux),
			# "ott_grouped":lambda x,y,key,where:loss_ott.ott_grouped_loss(x,y,key,where,aux=_ott_aux),
			# "ott_grouped_and_l2":lambda x,y,key,where:loss_ott.ott_grouped_and_l2_loss(x,y,key,where,aux=_ott_aux)
			# "rand_euclidean":lambda x,y,key:loss.random_sampled_euclidean(x,y,key=key)
		}
        self._loss_func = LOSS_FUNCS[LOSS_FUNC_STR]
        self.OPTIMISER = optimiser
        # def _smooth_reg(x,dx):
        #     avg_dx = jax.lax.conv_general_dilated(
        #         lhs=dx,
        #         rhs=np.ones((1,1,3,3))/(3*3),
        #         window_strides=(1,1),
        #         padding='SAME',
        #     )
        #     return np.mean((dx - avg_dx)**2)
        REG_FUNCS = {
            "l2":lambda x,dx: np.mean(dx**2),
            "l1":lambda x,dx: np.mean(np.abs(dx)),
            # "smooth":_smooth_reg,
            "max":lambda x,dx: np.max(np.abs(dx)),
            "in_0_1":lambda x,dx: np.mean(np.abs(x)+np.abs(x-1)-1),
		}
        

        # Filter REG_FUNCS to the same set (optional but keeps things consistent)
        perturbation_reg_coeff = {name:perturbation_reg_coeff[name] for name in perturbation_reg_coeff.keys() if perturbation_reg_coeff[name]!=0.0}
        REG_FUNCS = {name: REG_FUNCS[name] for name in perturbation_reg_coeff.keys()}
            # Multiply each regularization function by its coefficient
        REG_FUNCS = {name: lambda x, dx, f=func, c=perturbation_reg_coeff[name]: c * f(x, dx) 
                        for name, func in REG_FUNCS.items()}
        def makestep(dx_func,nca,x,y,t,opt_state,key):
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
                x_dx = dx_func(x) # Apply perturbation to initial condition batches
                dx_reg = dx_func.regulariser(x,REG_FUNCS) # Calculate regulariser loss on initial condition and perturbation

                

                (key,x),_ = eqx.internal.scan(nca_step,(key,x_dx),xs=np.arange(t),kind="lax") # Propagate NCA forward to final state y
                loss_batches = self.loss_func(x,y) # Loss between final state and target
                mean_loss = np.mean(loss_batches)
                aux = (x,loss_batches,mean_loss,dx_reg,x_dx)
                if self.optimisation_mode=="minimal_destructive":
                    loss = -mean_loss + dx_reg
                elif self.optimisation_mode=="maximal_preservative":
                    loss = mean_loss - dx_reg
                
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
            mode=perturbation_mode,
            CHANNELS=self.CHANNELS,
            OBS_CHANNELS=self.OBS_CHANNELS,
            x=x,
            WIDTH=1,
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
            # t = int(jr.randint(key, (), minval=self.STEPS_FROM_STABLE[0], maxval=self.STEPS_FROM_STABLE[1]))
            t = self.STEPS_TO_TARGET

            dx_func, opt_state, aux = makestep(dx_func,self.NCA_model,x,y,t,opt_state,key)
            pbar.set_description(f"Loss: {aux['total_loss']:.6f}, Best loss: {loss_best:.6f}, Reg: {aux['dx_reg']:.6f}")
            aux["dx"] = dx_func.get_values()
            self.logger.log_training(aux,i,log_interval)
            if RESAMPLE_EVERY > 0 and (i+1) % RESAMPLE_EVERY == 0 and i>0:
                x, y = self.generate_init_target_pairs(key)

            if aux['total_loss']<loss_best:
                loss_best = aux['total_loss']
                dx_best = dx_func
        print("Saving best perturbation...")
        # dx_best.save(f"{self.OUTPUT_DIRECTORY}{self.FILENAME}.eqx",overwrite=True)
        print("Loading best perturbation and running final trajectory...")
        print(f"Saving as {self.OUTPUT_DIRECTORY}{self.FILENAME}.eqx")
        # dx_best = dx_best.load(f"{self.OUTPUT_DIRECTORY}{self.FILENAME}.eqx")
        # T = self.run_full_trajectory(key,128,dx_best)
        x,_ = self.generate_init_target_pairs(key)
        x_dx = dx_best(x[:1])
        _,T = self.run_nca_model_batch(x_dx,self.STEPS_TO_TARGET*2,key) # One batch
        self.logger.log_final_trajectory(T)
        T = T[0,:,:4] # Remove batch dimension, only observable channels
        
        def numpy_image_float_to_int(array):
            array = onp.clip(array,0.0,1.0)
            array = (array * 255.0).astype(onp.uint8)
            return array
        T = onp.array(T) # Convert to numpy for logging
        T = numpy_image_float_to_int(T)
        onp.save(f"{self.OUTPUT_DIRECTORY}{self.FILENAME}_trajectory.npy",T)
        self.logger.finish()
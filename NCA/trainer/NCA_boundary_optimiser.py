import jax
import jax.numpy as np
import jax.random as jr
import optax
import equinox as eqx
import datetime
import Common.trainer.loss as loss
import jaxpruner
from einops import repeat
from Common.utils import key_pytree_gen
from Common.model.boundary import trainable_boundary
from NCA.trainer.boundary_tensorboard_log import boundary_train_log
from tqdm import tqdm
from jaxtyping import Float,Array,Key
import time


class NCA_boundary_optimiser(object):
    def __init__(self,
                 NCA_model,
                 initial_mask,
                 FILENAME,
                 LOG_DIRECTORY ="logs/",
                 MODEL_DIRECTORY = "models/"):
        
        self.NCA_model = NCA_model
        self.Boundary = trainable_boundary(mask=initial_mask)
        self.logger = boundary_train_log(LOG_DIRECTORY+FILENAME)    


    def loss_func(self,
                  x:Float[Array,"{self.NCA_model.N_CHANNELS} x y"],
                  boundary):
        """ The interesting bit. Maximise or minimise given cell types, and regularise by micropattern size

        Args:
            x float32 array [N_CHANNELS,_,_]: final NCA state.
            boundary callable: callable object that forces intermediate NCA states to be fixed to boundary condition at specified channels.

        Returns:
            loss: float32 scalar
        """
        
        channels_to_max = 0
        channels_to_min = 1
        celltype_min = np.sum(np.abs(x[channels_to_min]))
        celltype_max =-np.sum(np.abs(x[channels_to_max]))
        micropattern_size = boundary.coverage()
        return celltype_min + celltype_max + micropattern_size, {"celltype_min":celltype_min,"celltype_max":celltype_max,"micropattern_size":micropattern_size}
    
    def train(self,
              timesteps,
              iters,
              optimiser = None,
              LOOP_AUTODIFF="lax",
              key=jr.PRNGKey(int(time.time()))):
        
        @eqx.filter_jit()
        def make_step(boundary,nca,opt_state,key):

            @eqx.filter_value_and_grad(has_aux=True)
            def compute_loss(boundary,nca,key):

                def nca_step(carry,j):
                    key,x = carry
                    key = jr.fold_in(key,j)
                    x = nca(x,boundary,key)
                    return (key,x),None
                (key,x),_ = eqx.internal.scan(nca_step,(key,x),xs=np.arange(timesteps),kind=LOOP_AUTODIFF)

                loss,loss_dict = self.loss_func(x)
                return loss,loss_dict
            
            (loss,loss_dict),grads = compute_loss(boundary,nca,key)
            updates, opt_state = optimiser.update(grads, opt_state, boundary)
            boundary = eqx.apply_updates(boundary, updates)
            return boundary, opt_state, loss, loss_dict
        


        if optimiser is None:
            schedule = optax.exponential_decay(1e-3, transition_steps=iters, decay_rate=0.99)
            optimiser = optax.chain(optax.scale_by_param_block_norm(),optax.nadam(schedule))
        
        boundary = self.Boundary
        boundary_diff,_ = eqx.partition(boundary,eqx.is_inexact_array)
        opt_state = optimiser.init(boundary_diff)
        


        for i in tqdm(range(iters)):
            key = jr.fold_in(key,i)
            boundary, opt_state, loss, loss_dict = make_step(boundary,self.NCA_model,opt_state,key)
            print(f"Loss: {loss}")


import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jt
import optax
import equinox as eqx
import datetime
from tqdm import tqdm
import Common.trainer.loss as loss
from Common.utils import key_pytree_gen,key_array_gen
from einops import rearrange

class NCA_channel_extractor(object):
    """ Class to find mappings between learned hidden channels and target channels.
    For example, NCA trained on some experimental data end up learning representations in the hidden channels.
    If these hidden channels can be mapped to other target channels (unseen at NCA train time, perhaps measured later in another experiment)
    We then have some interpretability of the NCA model and confidence in the learned dynamics
    """
    def __init__(self,
            NCA_model,
            BOUNDARY_CALLBACK,
            GATED=False):
        self.NCA_model = NCA_model
        self.BOUNDARY_CALLBACK = BOUNDARY_CALLBACK
        self.GATED = GATED
    

    def initial_condition(self,key):
        # Implement this in subclasses
        raise NotImplementedError("Initial condition not implemented")
    
    def _generate_hidden_channels(self,X0,STEPS_BETWEEN_IMAGES,STEPS,channels,key):
        """_summary_

        Args:
            X0 (float32 [BATCH,C,X,Y]): batch of initial conditions
            timesteps (int): timesteps to run for
            channels (list of int): list of channels to return
        """

        #print(nca)
        vcall = jax.vmap(self.NCA_model,in_axes=(0,None,0),out_axes=(0))
        T = []
        X = X0
        for i in tqdm(range(STEPS*STEPS_BETWEEN_IMAGES),desc="Generating hidden channels"):
            key = jr.fold_in(key,i)
            key_pytree = key_array_gen(key,(len(X),))
            X = vcall(X,lambda x:x,key_pytree)
            
            if i%STEPS_BETWEEN_IMAGES == 0:
                
                T.append(X)
            
        T = jnp.array(T)
        T = rearrange(T,"T B C X Y -> B T C X Y")
        return T[:,:, channels, :, :]
        

    def generate_data(self,STEPS_BETWEEN_IMAGES,true_data,channels,key):
        """_summary_

        Args:
            STEPS_BETWEEN_IMAGES (int): number of steps between images
            true_data (float32 [BATCH,C,X,Y]): batch of initial conditions
            channels (list of int): list of channels to return
        """
        # Generate data from the NCA model
        
        X0 = true_data[:,0]
        TIMESTEPS = true_data.shape[1]
        data = self._generate_hidden_channels(X0,STEPS_BETWEEN_IMAGES,TIMESTEPS,channels,key)
        return data
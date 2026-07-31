from jaxtyping import Float, Int, PyTree, Scalar
from Common.trainer.abstract_data_augmenter_tree import DataAugmenterAbstract
from Common.trainer.custom_functions import multi_channel_perlin_noise
import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import time
from jaxtyping import Float, Int, PyTree, Scalar, Array, Key



""" 
Use noise as initial condition, learn how to generate textures via LPIPS distance to images.
No fancy intermediate time step stuff, just learn textures as fixed points of the PDE.

"""


class DataAugmenter(DataAugmenterAbstract):
    """
        Inherits the methods of DataAugmenter, but overwrites the batch cloning in the init
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.OVERWRITE_OBS_CHANNELS = False
        self.NOISE_CUTOFF = 4


    def data_init(self,SHARDING = None,key=jax.random.PRNGKey(int(time.time()))):
        """
        Data must be normalised to [-1,1] range for LPIPS to work. Set the initial conditions to perlin noise

        """
        
        map_to_m1p1 = lambda x:2*(x - jnp.min(x)) / (jnp.max(x) - jnp.min(x)) -1
        data = self.return_saved_data()
        data = jax.tree_util.tree_map(map_to_m1p1, data)
        data = self.duplicate_batches(data, 4)
        keys = list(jr.split(key, len(data)))
        reset_initial = lambda value, item_key: value.at[0].set(
            multi_channel_perlin_noise(
                value.shape[2], value.shape[1], self.NOISE_CUTOFF, item_key
            )
        )
        data = jax.tree_util.tree_map(reset_initial, data, keys)
        self.save_data(data)
        return None

    def advance_pool(self,
                      x: PyTree[Float[Array, "N C W H"]], 
                      y: PyTree[Float[Array, "N C W H"]], 
                      i: Int[Scalar, ""],
                      key: Key):
        
        x = jittable_callback_helper(x,self.NOISE_CUTOFF,key)
        x = self.noise(x,0.005,key=key)
        return x,y

@eqx.filter_jit
def jittable_callback_helper(x,NOISE_CUTOFF,key):
    propagate_xn = lambda x:x.at[1:].set(x[:-1])
    reset_x0 = lambda x,key:x.at[0].set(multi_channel_perlin_noise(x.shape[2],x.shape[1],NOISE_CUTOFF,key))

    keys = jr.split(key, len(x))

    x = jax.tree_util.tree_map(propagate_xn,x) # Set initial condition at each X[n] at next iteration to be final state from X[n-1] of this iteration
    x = jax.tree_util.tree_map(reset_x0,x,list(keys)) # Reset initial conditions to noise
    return x

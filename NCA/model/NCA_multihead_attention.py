import jax 
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import time
from einops import rearrange,einsum
from jaxtyping import Float, Array, Key, Int, Scalar

from NCA.model.NCA_model import NCA, Ops

class aNCA(NCA):
    layers: list
    KERNEL_STR: list
    N_CHANNELS: int
    N_FEATURES: int
    FIRE_RATE: float
    op: Ops
    perception: callable

    def __init__(self, N_CHANNELS, KERNEL_STR=["ID","LAP"], ACTIVATION=jax.nn.relu, PADDING="CIRCULAR", FIRE_RATE=1.0, KERNEL_SCALE = 1, key=jax.random.PRNGKey(int(time.time()))):
        super().__init__(N_CHANNELS, KERNEL_STR, ACTIVATION, PADDING, FIRE_RATE, KERNEL_SCALE, key)
        key = jr.fold_in(key,1)
        keys = jr.split(key,3)
        self.params = {"QW":jr.normal(keys[0],(self.N_FEATURES,self.N_FEATURES)),
                       "KW":jr.normal(keys[1],(self.N_FEATURES,self.N_FEATURES)),
                       "VW":jr.normal(keys[2],(self.N_FEATURES,self.N_FEATURES))}
        def pixelwise_attention(x: Float[Array,"{self.N_FEATURES} x y"]):
            Q = einsum(x,self.params["QW"],"F x y, F T -> T x y")
            K = einsum(x,self.params["KW"],"F x y, F T -> T x y")
            V = einsum(x,self.params["VW"],"F x y, F T -> T x y")

            
            #return jax.nn.sigmoid(x)



        self.layers = [

        ]        
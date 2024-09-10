import jax
import equinox as eqx
import jax.numpy as jnp
import time
from Common.model.spatial_operators import Ops
from jaxtyping import Array, Float
import jax.random as jr

class D(eqx.Module):
    diffusion_constants: Float[Array, "{self.N_CHANNELS} 1 1"]
    ops: eqx.Module
    N_CHANNELS: int
    PADDING: str
    def __init__(self,
                 N_CHANNELS,
                 PADDING,
                 dx,
                 INIT_SCALE,
                 key):
        self.N_CHANNELS = N_CHANNELS
        self.PADDING = PADDING

        # self.diffusion_constants = eqx.nn.Conv2d(in_channels=self.N_CHANNELS,
        #                                     out_channels=self.N_CHANNELS,
        #                                     kernel_size=1,
        #                                     use_bias=False,
        #                                     key=key,
        #                                     groups=self.N_CHANNELS)
        self.diffusion_constants = jr.uniform(key=key,shape=(self.N_CHANNELS,1,1))*INIT_SCALE

        self.ops = Ops(PADDING=PADDING,dx=dx)
        where = lambda l: l.weight
        #self.diffusion_constants= eqx.tree_at(where,self.diffusion_constants,jax.numpy.abs(self.diffusion_constants.weight))
    @eqx.filter_jit
    def __call__(self,X: Float[Array, "{self.N_CHANNELS} x y"])->Float[Array, "{self.N_CHANNELS} x y"]:
        #return self.diffusion_constants(self.ops.Lap(X))
        return jax.nn.relu(self.diffusion_constants)*self.ops.Lap(X)

    def partition(self):
        total_diff,total_static = eqx.partition(self,eqx.is_array)
        op_diff,op_static = self.ops.partition()
        where_ops = lambda m: m.ops
        total_diff = eqx.tree_at(where_ops,total_diff,op_diff)
        total_static = eqx.tree_at(where_ops,total_static,op_static)
        return total_diff,total_static
    
    def combine(self,diff,static):
        self = eqx.combine(diff,static)

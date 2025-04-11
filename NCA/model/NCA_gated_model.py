import jax
import jax.numpy as jnp
import equinox as eqx
import time
#from Common.model.abstract_model import AbstractModel # Inherit model loading and saving
from NCA.model.NCA_model import NCA, Ops


class gNCA(NCA):
    layers: list
    KERNEL_STR: list
    N_CHANNELS: int
    N_FEATURES: int
    FIRE_RATE: float
    op: Ops
    perception: callable
    CONFIG: dict
    
    def __init__(self,
                N_CHANNELS,
                KERNEL_STR=["ID","LAP"], 
                ACTIVATION=jax.nn.relu, 
                PADDING="CIRCULAR", 
                FIRE_RATE=1.0, 
                KERNEL_SCALE = 1, 
                key=jax.random.PRNGKey(int(time.time()))):
        super().__init__(N_CHANNELS, KERNEL_STR, ACTIVATION, PADDING, FIRE_RATE, KERNEL_SCALE, key)
        #key1,key2 = jax.random.split(key,2)
        key = jax.random.fold_in(key,1)
        self.layers[-1] = eqx.nn.Conv2d(in_channels=self.N_FEATURES, 
						  out_channels=2*self.N_CHANNELS,
						  kernel_size=1,
						  use_bias=True,
						  key=key)
        gate_func = lambda x: jax.nn.glu(x,axis=0)
        self.layers.append(gate_func)


        # Initialise final convolution to zero
        w_zeros = jnp.zeros((2*self.N_CHANNELS,self.N_FEATURES,1,1))
        b_zeros = jnp.zeros((2*self.N_CHANNELS,1,1))
        w_where = lambda l: l.weight
        b_where = lambda l: l.bias
        self.layers[-2] = eqx.tree_at(w_where,self.layers[-2],w_zeros)
        self.layers[-2] = eqx.tree_at(b_where,self.layers[-2],b_zeros)

        # Dict of hyperparameters for logging
        self.CONFIG["MODEL"] = "gNCA"
import jax
import jax.numpy as jnp
import numpy as onp
import equinox as eqx
import time
from einops import repeat, reduce
#from Common.model.abstract_model import AbstractModel # Inherit model loading and saving
from NCA.model.NCA_model import NCA, Ops
from jaxtyping import Float, Array, Key, Int, Scalar

class sub_NCA(NCA):
    layers: list
    KERNEL_STR: list
    N_CHANNELS: int
    OUTPUT_CHANNELS: list
    OUTPUT_CHANNEL_NUMBER: int
    GATED: bool
    SCALE: int
    N_FEATURES: int
    FIRE_RATE: float
    op: Ops
    perception: callable
    def __init__(self,
                N_CHANNELS,
                OUTPUT_CHANNELS,
                GATED = False,
                SCALE = 1,
                KERNEL_STR=["ID","LAP"], 
                ACTIVATION=jax.nn.relu, 
                PADDING="CIRCULAR", 
                FIRE_RATE=1.0, 
                KERNEL_SCALE = 1, 
                key=jax.random.PRNGKey(int(time.time()))):
        super().__init__(N_CHANNELS, KERNEL_STR, ACTIVATION, PADDING, FIRE_RATE, KERNEL_SCALE, key)
        self.OUTPUT_CHANNELS = OUTPUT_CHANNELS
        self.OUTPUT_CHANNEL_NUMBER = onp.count_nonzero(OUTPUT_CHANNELS)
        self.GATED = GATED
        self.SCALE = SCALE
        key = jax.random.fold_in(key,1)
        if GATED:
            self.layers[-1] = eqx.nn.Conv2d(in_channels=self.N_FEATURES, 
                            out_channels=2*self.OUTPUT_CHANNEL_NUMBER,
                            kernel_size=1,
                            use_bias=True,
                            key=key)
            gate_func = lambda x: jax.nn.glu(x,axis=0)
            self.layers.append(gate_func)

            # Initialise final convolution to zero
            w_zeros = jnp.zeros((2*self.OUTPUT_CHANNEL_NUMBER,self.N_FEATURES,1,1))
            b_zeros = jnp.zeros((2*self.OUTPUT_CHANNEL_NUMBER,1,1))
            w_where = lambda l: l.weight
            b_where = lambda l: l.bias
            self.layers[-2] = eqx.tree_at(w_where,self.layers[-2],w_zeros)
            self.layers[-2] = eqx.tree_at(b_where,self.layers[-2],b_zeros)
        else:
            self.layers[-1] = eqx.nn.Conv2d(in_channels=self.N_FEATURES, 
                            out_channels=self.OUTPUT_CHANNEL_NUMBER,
                            kernel_size=1,
                            use_bias=True,
                            key=key)
            
            # Initialise final convolution to zero
            w_zeros = jnp.zeros((self.OUTPUT_CHANNEL_NUMBER,self.N_FEATURES,1,1))
            b_zeros = jnp.zeros((self.OUTPUT_CHANNEL_NUMBER,1,1))
            w_where = lambda l: l.weight
            b_where = lambda l: l.bias
            self.layers[-1] = eqx.tree_at(w_where,self.layers[-1],w_zeros)
            self.layers[-1] = eqx.tree_at(b_where,self.layers[-1],b_zeros)

    def __call__(self,
                    x: Float[Array,"{self.N_CHANNELS} x y"],
                    boundary_callback=lambda x:x,
                    key: Key=jax.random.PRNGKey(int(time.time())))->Float[Array, "{self.N_CHANNEL} x y"]:
        """
        

        Parameters
        ----------
        x : float32 [N_CHANNELS,_,_]
            input NCA lattice state.
        boundary_callback : callable (float32 [N_CHANNELS,_,_]) -> (float32 [N_CHANNELS,_,_]), optional
            function to augment intermediate NCA states i.e. imposing complex boundary conditions or external structure. Defaults to None
        key : jax.random.PRNGKey, optional
            Jax random number key. The default is jax.random.PRNGKey(int(time.time())).

        Returns
        -------
        x : float32 [N_CHANNELS,_,_]
            output NCA lattice state.

        """
        x = reduce(x, "C (h h2) (w w2) -> C h w ", "mean", h2=self.SCALE, w2=self.SCALE)
        #print(x.shape)
        dx = self.perception(x)
        for layer in self.layers:
            dx = layer(dx)
        sigma = jax.random.bernoulli(key,p=self.FIRE_RATE,shape=dx.shape)
        dx = dx*sigma
        dx = repeat(dx, "C h w -> C (h h2) (w w2)", h2=self.SCALE, w2=self.SCALE)
        return dx
        #x_new = x[self.OUTPUT_CHANNELS] + sigma*dx
        #x_new = x
        #x_new = jnp.where(self.OUTPUT_CHANNELS, x.at[self.OUTPUT_CHANNELS].set(x[self.OUTPUT_CHANNELS]+sigma*dx), 0)
        #print(x_new.shape)
        #x_new.at[self.OUTPUT_CHANNELS].set(x[self.OUTPUT_CHANNELS] + sigma*dx)
        #return boundary_callback(x_new)
    
    #def partition(self):
    #    diff_main,static_main = eqx.partition(self,eqx.is_inexact_array)
        #where = lambda s:s.OUTPUT_CHANNELS
        #diff_main = eqx.tree_at(where,diff_main,None)
        #static_main = eqx.tree_at(where,static_main,self.OUTPUT_CHANNELS)
    #    return diff_main,static_main
import jax
import equinox as eqx
import jax.numpy as jnp
import time
from Common.model.spatial_operators import Ops
from jaxtyping import Array, Float
from Common.model.custom_functions import construct_polynomials,set_layer_weights
import jax.random as jr
from einops import rearrange,repeat
class D(eqx.Module):
    layers: list
    diffusion_constants: Float[Array, "{self.N_CHANNELS} 1 1"]
    ops: eqx.Module
    N_CHANNELS: int
    PADDING: str
    ORDER: int
    polynomial_preprocess: callable
    def __init__(self,
                 N_CHANNELS,
                 PADDING,
                 dx,
                 INTERNAL_ACTIVATION,
                 OUTER_ACTIVATION,
                 INIT_SCALE,
                 INIT_SCALE_LINEAR,
                 INIT_TYPE,
                 USE_BIAS,
                 ORDER,
                 N_LAYERS,
                 ZERO_INIT,
                 key):
        self.N_CHANNELS = N_CHANNELS
        self.ORDER = ORDER

        # ----------------- Nonlinear part -----------------
        N_FEATURES = len(construct_polynomials(jnp.zeros((N_CHANNELS,)),self.ORDER))
        _v_poly = jax.vmap(lambda x: construct_polynomials(x,self.ORDER),in_axes=1,out_axes=1)
        self.polynomial_preprocess = jax.vmap(_v_poly,in_axes=1,out_axes=1)
        self.PADDING = PADDING
        keys = jr.split(key,2*(N_LAYERS+1))
        _inner_layers = [eqx.nn.Conv2d(in_channels=N_FEATURES,out_channels=self.N_CHANNELS,kernel_size=1,padding=0,use_bias=USE_BIAS,key=key) for key in keys[:N_LAYERS]]
        _inner_activations = [lambda x:INTERNAL_ACTIVATION(self.polynomial_preprocess(x)) for _ in range(N_LAYERS)]
        self.layers = _inner_layers + _inner_activations
        self.layers[::2] = _inner_layers
        self.layers[1::2] = _inner_activations
        self.layers.append(eqx.nn.Conv2d(in_channels=N_FEATURES,out_channels=self.N_CHANNELS,kernel_size=1,padding=0,use_bias=USE_BIAS,key=keys[N_LAYERS]))
        self.layers.append(OUTER_ACTIVATION)        
        w_where = lambda l: l.weight
        b_where = lambda l: l.bias
        for i in range(0,len(self.layers)//2):
            self.layers[2*i] = eqx.tree_at(w_where,
                                           self.layers[2*i],
                                           set_layer_weights(self.layers[2*i].weight.shape,keys[i],INIT_TYPE,INIT_SCALE))
            if USE_BIAS:
                self.layers[2*i] = eqx.tree_at(b_where,
                                               self.layers[2*i],
                                               INIT_SCALE*jr.normal(keys[i+len(self.layers)],self.layers[2*i].bias.shape))

        if ZERO_INIT:
            self.layers[-2] = eqx.tree_at(w_where,
                                         self.layers[-2],
                                         jnp.zeros(self.layers[-2].weight.shape))
            if USE_BIAS:
                self.layers[-2] = eqx.tree_at(b_where,
                                             self.layers[-2],
                                             jnp.zeros(self.layers[-2].bias.shape))
                
        

        # ----------------- Linear part -----------------
        self.diffusion_constants = jr.normal(key=key,shape=(self.N_CHANNELS,1,1))*INIT_SCALE_LINEAR
        
        # ----------------- Differential operators -----------------
        self.ops = Ops(PADDING=PADDING,dx=dx)


    @eqx.filter_jit
    def __call__(self,X: Float[Array, "{self.N_CHANNELS} x y"])->Float[Array, "{self.N_CHANNELS} x y"]:
        nonlin_x = self.polynomial_preprocess(X)
        for L in self.layers:
            nonlin_x = L(nonlin_x)
        linear = jax.nn.sparse_plus(self.diffusion_constants)*self.ops.Lap(X)
        nonlinear = self.ops.NonlinearDiffusion(nonlin_x,X)
        return linear + nonlinear
    
    def partition(self):
        total_diff,total_static = eqx.partition(self,eqx.is_array)
        op_diff,op_static = self.ops.partition()
        where_ops = lambda m: m.ops
        total_diff = eqx.tree_at(where_ops,total_diff,op_diff)
        total_static = eqx.tree_at(where_ops,total_static,op_static)
        return total_diff,total_static
    
    def combine(self,diff,static):
        self = eqx.combine(diff,static)

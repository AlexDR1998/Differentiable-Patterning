import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import time
from jaxtyping import Array, Float
from Common.model.custom_functions import construct_polynomials,set_layer_weights
from einops import rearrange,repeat

class R(eqx.Module):
    layers: list
    N_CHANNELS: int
    STABILITY_FACTOR: float
    ORDER: int
    polynomial_preprocess: callable
    def __init__(self,
                 N_CHANNELS,
                 INTERNAL_ACTIVATION,
                 OUTER_ACTIVATION,
                 INIT_SCALE,
                 INIT_TYPE,
                 USE_BIAS,
                 STABILITY_FACTOR,
                 ORDER,
                 N_LAYERS,
                 ZERO_INIT=True,
                 key=jax.random.PRNGKey(int(time.time()))):
        self.STABILITY_FACTOR = STABILITY_FACTOR
        keys = jax.random.split(key,2*(N_LAYERS+1))
        self.N_CHANNELS = N_CHANNELS
        self.ORDER = ORDER
        N_FEATURES = len(construct_polynomials(jnp.zeros((N_CHANNELS,)),self.ORDER))
        _v_poly = jax.vmap(lambda x: construct_polynomials(x,self.ORDER),in_axes=1,out_axes=1)
        self.polynomial_preprocess = jax.vmap(_v_poly,in_axes=1,out_axes=1)

        _inner_layers = [eqx.nn.Conv2d(in_channels=N_FEATURES,out_channels=self.N_CHANNELS,kernel_size=1,padding=0,use_bias=USE_BIAS,key=key) for key in keys[:N_LAYERS]]
        _inner_activations = [lambda x:INTERNAL_ACTIVATION(self.polynomial_preprocess(x)) for _ in range(N_LAYERS)]
        self.layers = _inner_layers+_inner_activations
        self.layers[::2] = _inner_layers
        self.layers[1::2] = _inner_activations
        self.layers.append(eqx.nn.Conv2d(in_channels=N_FEATURES,out_channels=self.N_CHANNELS,kernel_size=1,padding=0,use_bias=USE_BIAS,key=keys[2*N_LAYERS]))
        self.layers.append(OUTER_ACTIVATION)

        



        where = lambda l:l.weight
        where_b = lambda l:l.bias

        for i in range(0,len(self.layers)//2):
            self.layers[2*i] = eqx.tree_at(where,
                                           self.layers[2*i],
                                           set_layer_weights(self.layers[2*i].weight.shape,keys[i],INIT_TYPE,INIT_SCALE))
            
            if USE_BIAS:
                self.layers[2*i] = eqx.tree_at(where_b,
                                               self.layers[2*i],
                                               INIT_SCALE*jax.random.normal(key=keys[i+len(self.layers)],shape=self.layers[2*i].bias.shape))
                
        

        if ZERO_INIT:
            self.layers[-2] = eqx.tree_at(where,
                                          self.layers[-2],
                                          jnp.zeros(self.layers[-2].weight.shape))
            
            if USE_BIAS:
                self.layers[-2] = eqx.tree_at(where_b,
                                              self.layers[-2],
                                              jnp.zeros(self.layers[-2].bias.shape))
                



    def __call__(self,X: Float[Array, "{self.N_CHANNELS} x y"])->Float[Array,"{self.N_CHANNELS} x y"]:
        X_poly = self.polynomial_preprocess(X)
        #print(f"Reaction shape: {X_poly.shape}")

        for L in self.layers:
            X_poly = L(X_poly)
        
        return X_poly - self.STABILITY_FACTOR*X**3
    
    def partition(self):
        return eqx.partition(self,eqx.is_array)
    
    def combine(self,diff,static):
        self = eqx.combine(diff,static)
        
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from Common.model.abstract_model import AbstractModel
from jaxtyping import Float, Array, Key, Int, Scalar, PyTree
class SparseAutoencoder(AbstractModel):
    """
    A simple sparse autoencoder implemented using Equinox.
    """

    encoder: eqx.nn.Linear
    decoder: eqx.nn.Linear
    bias_params: list
    sparsity_param: int
    ACTIVATION: str
    TARGET_LAYER: str
    GATED: bool
    config: dict

    def __init__(self, 
                 TARGET_LAYER: str, 
                 N_CHANNELS: int, 
                 N_KERNELS: int,
                 ACTIVATION: str, 
                 GATED: bool, 
                 hidden_dim: int, 
                 sparsity_param: float, 
                 key):
        keys = jr.split(key,3)
        assert TARGET_LAYER in ["perception", "linear_hidden", "activation", "linear_output", "gate_func"], "Invalid target layer"
        assert ACTIVATION in ["topk", "relu", "identity"], "Invalid activation function"
        
        self.ACTIVATION = ACTIVATION
        self.TARGET_LAYER = TARGET_LAYER
        self.GATED = GATED
        if TARGET_LAYER in ["perception", "linear_hidden", "activation"]:
            input_dim = N_CHANNELS*N_KERNELS
        elif TARGET_LAYER == "linear_output":
            if GATED:
                input_dim = 2*N_CHANNELS
            else:
                input_dim = N_CHANNELS
        elif TARGET_LAYER == "gate_func":
            input_dim = N_CHANNELS
        
        self.encoder = eqx.nn.Linear(input_dim, hidden_dim, key=keys[0],use_bias=False)
        self.decoder = eqx.nn.Linear(hidden_dim, input_dim, key=keys[1],use_bias=False)
        w_where = lambda l: l.weight
        self.bias_params = [
            jr.normal(keys[2],(input_dim,)),
            jr.normal(keys[3],(hidden_dim,))
        ]

        #b_where = lambda l: l.bias
        self.decoder = eqx.tree_at(w_where,self.decoder,self.encoder.weight.T+1e-3*jax.random.normal(keys[4],self.decoder.weight.shape))
        self.sparsity_param = sparsity_param
        self.config = {
            "TARGET_LAYER": TARGET_LAYER,
            "N_CHANNELS": N_CHANNELS,
            "N_KERNELS": N_KERNELS,
            "ACTIVATION": ACTIVATION,
            "GATED": GATED,
            "hidden_dim": hidden_dim,
            "sparsity_param": sparsity_param
        }

    def encode(self,x: Float[Array,"{self.N_CHANNELS}"])->Float[Array,"{self.hidden_dim}"]:
        activation = {"topk":self.topk_activation, "relu":jax.nn.relu, "identity":lambda x:x}[self.ACTIVATION]
        return activation(self.encoder(x-self.bias_params[0])+self.bias_params[1])
    
    def decode(self,x: Float[Array,"{self.hidden_dim}"])->Float[Array,"{self.N_CHANNELS}"]:
        return self.decoder(x) + self.bias_params[0]
    
    def topk_activation(self,x: Float[Array,"{self.hidden_dim}"])->Float[Array,"{self.hidden_dim}"]:
        vals,inds = jax.lax.top_k(x,k=self.sparsity_param)
        output = jnp.zeros_like(x)
        output = output.at[inds].set(vals)
        return output
    
    @eqx.filter_jit
    def mult_kth_top_feature(self,
                             latents: Float[Array,"{self.hidden_dim}"],
                             K: Int,
                             val: Float)->Float[Array,"{self.hidden_dim}"]:
        # Multiples kth top latent feature by val
        K = K+1
        vals,inds = jax.lax.top_k(latents,k=self.sparsity_param)
        inds_2 = jnp.argpartition(vals,-K,axis=0)[-K]
        pos = inds[inds_2]
        #pos = jnp.argpartition(latents,-K,axis=0)[-K]
        latents = latents.at[pos].set(latents[pos]*val)
        return latents
    @eqx.filter_jit
    def mult_k_top_features(self,
                            latents: Float[Array,"{self.hidden_dim}"],
                            Ks,
                            mult)->Float[Array,"{self.hidden_dim}"]:
        # Multiples k top latent features by vals
        #print(Ks)
        vals,inds = jax.lax.top_k(latents,k=self.sparsity_param)
        inds_2 = jnp.argsort(vals,axis=0)[::-1][Ks]
        #print(inds_2.shape)
        pos = inds[inds_2]
        #print(pos.shape)
        #print(latents.shape)
        #print(latents[pos].shape)
        latents = latents.at[pos].set(latents[pos]*mult)
        return latents

    @eqx.filter_jit
    def get_top_k_feature_positions(self,
                           latents: Float[Array,"{self.hidden_dim}"])->Float[Array,"{self.sparsity_param}"]:
        # Returns the k top latent feature positions, sorted by feature activation
        vals,inds = jax.lax.top_k(latents,k=self.sparsity_param)
        inds_2 = jnp.argsort(vals,axis=0)[::-1]
        pos = inds[inds_2]
        vals = vals[inds_2]
        return vals,pos
    
    @eqx.filter_jit
    def set_features_at_positions(self,
                                  latents: Float[Array,"{self.hidden_dim}"],
                                  positions: Float[Array,"L"],
                                  values: Float[Array,"L"])->Float[Array,"{self.hidden_dim}"]:
        # Sets the features at positions to values
        latents = latents.at[positions].set(values)
        return latents
    
    
    
    @eqx.filter_jit
    def __call__(self, x):
        
        decoded = self.decode(self.encode(x))
        
        return decoded
    
    def partition(self):
        total_diff,total_static = eqx.partition(self,eqx.is_inexact_array)
        return total_diff,total_static
    
    


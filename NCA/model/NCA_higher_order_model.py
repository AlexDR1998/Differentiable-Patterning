import jax
import jax.numpy as jnp
import equinox as eqx
import time
#from Common.model.abstract_model import AbstractModel # Inherit model loading and saving
from NCA.model.NCA_model import NCA, Ops
from Common.model.custom_functions import construct_polynomials
from einops import rearrange
from jaxtyping import Float, Array, Key, Int, Scalar

class hNCA(NCA):
    layers: list
    KERNEL_STR: list
    N_CHANNELS: int
    N_FEATURES: int
    FIRE_RATE: float
    op: Ops
    perception: callable
    ORDER: int
    def __init__(self, N_CHANNELS, KERNEL_STR=["ID","LAP"], ACTIVATION=jax.nn.relu, PADDING="CIRCULAR", FIRE_RATE=1.0, KERNEL_SCALE = 1, ORDER = 2, key=jax.random.PRNGKey(int(time.time()))):
        key1,key2 = jax.random.split(key,2)
        self.N_CHANNELS = N_CHANNELS
        self.FIRE_RATE = FIRE_RATE
        self.KERNEL_STR = KERNEL_STR
        self.op = Ops(PADDING=PADDING,dx=1,KERNEL_SCALE=KERNEL_SCALE)

		
        self.ORDER = ORDER
        v_poly = jax.vmap(lambda X: construct_polynomials(X,self.ORDER),in_axes=1,out_axes=1)
        vv_poly = jax.vmap(v_poly,in_axes=1,out_axes=1)
        _n_features = len(construct_polynomials(jnp.zeros((N_CHANNELS,)),self.ORDER))
        _kernel_length = len(self.KERNEL_STR)
        if "GRAD" in KERNEL_STR:
            _kernel_length+=1
        self.N_FEATURES = _kernel_length*_n_features

        def spatial_layer(X: Float[Array,"{self.N_CHANNELS} x y"])-> Float[Array, "H x y"]:
            X = vv_poly(X)
            
            output = []
            if "ID" in KERNEL_STR:
                output.append(X)
            if "DIFF" in KERNEL_STR:
                gradnorm = self.op.GradNorm(X)
                output.append(gradnorm)
            if "GRAD" in KERNEL_STR:
                grad = self.op.Grad(X)
                output.append(grad[0])
                output.append(grad[1])
            if "AV" in KERNEL_STR:
                output.append(self.op.Average(X))
            if "LAP" in KERNEL_STR:
                output.append(self.op.Lap(X))
            output = rearrange(output,"b C x y -> (b C) x y")
            return output
        self.perception = lambda x:spatial_layer(x)

        self.layers = [
            eqx.nn.Conv2d(in_channels=self.N_FEATURES,
                            out_channels=self.N_FEATURES,
                            kernel_size=1,
                            use_bias=False,
                            key=key1),
            ACTIVATION,
            eqx.nn.Conv2d(in_channels=self.N_FEATURES, 
                            out_channels=self.N_CHANNELS,
                            kernel_size=1,
                            use_bias=True,
                            key=key2)
            ]
        
        
        # Initialise final layer to zero
        w_zeros = jnp.zeros((self.N_CHANNELS,self.N_FEATURES,1,1))
        b_zeros = jnp.zeros((self.N_CHANNELS,1,1))
        w_where = lambda l: l.weight
        b_where = lambda l: l.bias
        self.layers[-1] = eqx.tree_at(w_where,self.layers[-1],w_zeros)
        self.layers[-1] = eqx.tree_at(b_where,self.layers[-1],b_zeros)
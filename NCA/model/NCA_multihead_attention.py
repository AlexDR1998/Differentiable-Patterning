import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import time
from einops import rearrange, einsum
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
    attention_inputs: list

    def __init__(
        self,
        N_CHANNELS,
        KERNEL_STR=["ID", "LAP"],
        ACTIVATION=jax.nn.relu,
        PADDING="CIRCULAR",
        FIRE_RATE=1.0,
        KERNEL_SCALE=1,
        HEADS=4,
        key=jax.random.PRNGKey(int(time.time())),
    ):
        super().__init__(
            N_CHANNELS, KERNEL_STR, ACTIVATION, PADDING, FIRE_RATE, KERNEL_SCALE, key
        )
        key = jr.fold_in(key, 1)
        keys = jr.split(key, 5)
        heads = HEADS
        dseq = N_CHANNELS
        dquery = N_CHANNELS

        def reshape_pre_attention(x: Float[Array, "dseq*dquery X Y"]):
            size = x.shape[1:]
            return (
                rearrange(x, "(dseq dquery) X Y -> (X Y) dseq dquery", dseq=dseq),
                size,
            )

        def reshape_post_attention(input):
            x, size = input
            return rearrange(
                x,
                "(X Y) dseq dquery -> (dseq dquery) X Y",
                X=size[0],
                Y=size[1],
                dseq=dseq,
            )

        self.attention_inputs = [
            eqx.nn.Conv2d(
                in_channels=self.N_FEATURES,
                out_channels=dseq * dquery,
                kernel_size=1,
                use_bias=True,
                key=keys[0],
            ),
            eqx.nn.Conv2d(
                in_channels=self.N_FEATURES,
                out_channels=dseq * dquery,
                kernel_size=1,
                use_bias=True,
                key=keys[1],
            ),
            eqx.nn.Conv2d(
                in_channels=self.N_FEATURES,
                out_channels=dseq * dquery,
                kernel_size=1,
                use_bias=True,
                key=keys[2],
            ),
        ]
        self.layers = [
            reshape_pre_attention,
            eqx.nn.MultiheadAttention(
                num_heads=heads,
                query_size=dquery,
                key_size=dquery,
                value_size=dquery,
                key=keys[3],
            ),
            reshape_post_attention,
            ACTIVATION,
            eqx.nn.Conv2d(
                in_channels=dseq * dquery,
                out_channels=self.N_CHANNELS,
                kernel_size=1,
                use_bias=True,
                key=keys[4],
            ),
        ]
        w_zeros = jnp.zeros((self.N_CHANNELS,dseq*dquery,1,1))
        b_zeros = jnp.zeros((self.N_CHANNELS,1,1))
        w_where = lambda l: l.weight
        b_where = lambda l: l.bias
        self.layers[-1] = eqx.tree_at(w_where,self.layers[-1],w_zeros)
        self.layers[-1] = eqx.tree_at(b_where,self.layers[-1],b_zeros)
    def __call__(
        self,
        x: Float[Array, "{self.N_CHANNELS} x y"],
        boundary_callback=lambda x: x,
        key: Key = jax.random.PRNGKey(int(time.time())),
    ) -> Float[Array, "{self.N_CHANNEL} x y"]:
        px = self.perception(x)
        Qx = self.attention_inputs[0](px)  # (dseq dquery) X Y
        Kx = self.attention_inputs[1](px)
        Vx = self.attention_inputs[2](px)
        Qx, size = self.layers[0](Qx)  # Reshape to: (X Y) dseq dquery
        Kx, _ = self.layers[0](Kx)
        Vx, _ = self.layers[0](Vx)
        dx = eqx.filter_vmap(self.layers[1])(
            Qx, Kx, Vx
        )  # Multihead attention vmapped over pixels
        dx = self.layers[2]((dx, size))
        dx = self.layers[3](dx)
        dx = self.layers[4](dx)

        sigma = jax.random.bernoulli(key, p=self.FIRE_RATE, shape=dx.shape)
        x_new = x + sigma * dx
        return boundary_callback(x_new)

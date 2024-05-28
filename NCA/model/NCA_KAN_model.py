import jax
import jax.numpy as jnp
import equinox as eqx
import time
from Common.model.KAN import gaussKAN
from NCA.model.NCA_model import NCA, Ops


class kaNCA(NCA):
    layers: list
    KERNEL_STR: list
    N_CHANNELS: int
    N_FEATURES: int
    FIRE_RATE: float
    op: Ops
    perception: callable
    def __init__(self, N_CHANNELS, KERNEL_STR=["ID","LAP"], ACTIVATION=jax.nn.relu, PADDING="CIRCULAR", FIRE_RATE=1.0, KERNEL_SCALE = 1, key=jax.random.PRNGKey(int(time.time()))):
        super().__init__(N_CHANNELS, KERNEL_STR, ACTIVATION, PADDING, FIRE_RATE, KERNEL_SCALE, key)
        # gaussKAN hyperparameters
        bounds = 3 # The expected range of input parameters.
        ORDER = 11 # How many radial basis functions to use per edge?
        width = 4 # Width of radial basis functions
        scale = 4 # Initialised scale, 0 on second layer
        key1,key2 = jax.random.split(key,2)
        self.layers = [
            eqx.filter_pmap(
                eqx.filter_pmap(
                    gaussKAN(in_features=self.N_FEATURES,
                            out_features=self.N_FEATURES,
                            ORDER=ORDER,
                            scale=scale,
                            width=width,
                            bounds=bounds,
                            key=key1),
                    in_axes=1,
                    out_axes=1),
                in_axes=1,
                out_axes=1),
            eqx.filter_pmap(
                eqx.filter_pmap(
                    gaussKAN(in_features=self.N_FEATURES,
                            out_features=self.N_CHANNELS,
                            ORDER=ORDER,
                            scale=0,
                            width=width,
                            bounds=bounds,
                            key=key2),
                    in_axes=1,
                    out_axes=1),
                in_axes=1,
                out_axes=1)
        ]

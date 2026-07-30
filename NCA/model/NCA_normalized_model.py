"""Standard NCA with optional state normalisation around the update.

This is deliberately a small experimental subclass of :class:`NCA`.  The
baseline model is unchanged, and the public call signature remains compatible
with the NCA trainers.

The normalised state is updated in normalised coordinates and then transformed
back using the same statistics as the input state.  In particular, statistics
are not recomputed after applying the update, which makes the inverse explicit
and keeps the residual update easy to interpret.
"""

import time

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float

from NCA.model.NCA_model import NCA


class NormalizedNCA(NCA):
    """Drop-in NCA with selectable normalisation of the state before updating.

    Parameters
    ----------
    NORMALIZATION : str
        ``"none"``; ``"fixed"`` for fixed per-channel statistics;
        ``"instance"`` for per-channel spatial statistics; or ``"group"``
        for statistics over the complete state.  The latter two are useful
        quick experiments, but introduce spatial/global coupling.
    NORMALIZATION_MEAN, NORMALIZATION_STD : array-like, optional
        Per-channel statistics for ``"fixed"``.  They should have shape
        ``[N_CHANNELS]`` (or be broadcastable to ``[N_CHANNELS, H, W]``).
    NORMALIZATION_EPS : float
        Numerical floor used in standard deviations.
    """

    NORMALIZATION: str
    NORMALIZATION_MEAN: Array
    NORMALIZATION_STD: Array
    NORMALIZATION_EPS: float

    def __init__(
        self,
        N_CHANNELS,
        KERNEL_STR=["ID", "LAP"],
        ACTIVATION=jax.nn.relu,
        PADDING="CIRCULAR",
        FIRE_RATE=1.0,
        KERNEL_SCALE=1,
        key=None,
        NORMALIZATION="none",
        NORMALIZATION_MEAN=None,
        NORMALIZATION_STD=None,
        NORMALIZATION_EPS=1e-6,
    ):
        super().__init__(
            N_CHANNELS=N_CHANNELS,
            KERNEL_STR=KERNEL_STR,
            ACTIVATION=ACTIVATION,
            PADDING=PADDING,
            FIRE_RATE=FIRE_RATE,
            KERNEL_SCALE=KERNEL_SCALE,
            key=key,
        )

        normalization = NORMALIZATION.lower()
        valid = {"none", "fixed", "instance", "group"}
        if normalization not in valid:
            raise ValueError(f"NORMALIZATION must be one of {sorted(valid)}")
        if NORMALIZATION_EPS <= 0:
            raise ValueError("NORMALIZATION_EPS must be positive")

        if NORMALIZATION_MEAN is None:
            NORMALIZATION_MEAN = jnp.zeros(N_CHANNELS)
        if NORMALIZATION_STD is None:
            NORMALIZATION_STD = jnp.ones(N_CHANNELS)
        if normalization == "fixed" and jnp.any(jnp.asarray(NORMALIZATION_STD) <= 0):
            raise ValueError("NORMALIZATION_STD must be positive")

        self.NORMALIZATION = normalization
        self.NORMALIZATION_MEAN = jnp.asarray(NORMALIZATION_MEAN)
        self.NORMALIZATION_STD = jnp.asarray(NORMALIZATION_STD)
        self.NORMALIZATION_EPS = NORMALIZATION_EPS

    def _normalise(self, x):
        if self.NORMALIZATION == "none":
            mean, std = 0.0, 1.0
        elif self.NORMALIZATION == "fixed":
            # State tensors are [C, H, W]; make [C] statistics broadcastable.
            mean = self.NORMALIZATION_MEAN.reshape((-1, 1, 1))
            std = self.NORMALIZATION_STD.reshape((-1, 1, 1))
        elif self.NORMALIZATION == "instance":
            mean = jnp.mean(x, axis=(1, 2), keepdims=True)
            std = jnp.std(x, axis=(1, 2), keepdims=True)
        else:  # group
            mean = jnp.mean(x, keepdims=True)
            std = jnp.std(x, keepdims=True)
        std = jnp.maximum(std, self.NORMALIZATION_EPS)
        return (x - mean) / std, mean, std

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "MODEL": "NormalizedNCA",
                "NORMALIZATION": self.NORMALIZATION,
                "NORMALIZATION_EPS": self.NORMALIZATION_EPS,
            }
        )
        return config

    def partition(self):
        """Keep fixed statistics out of the trainable parameter PyTree.

        They remain ordinary array leaves, so Equinox serialization stores them
        in the checkpoint, but optimizers do not receive them as trainable
        leaves.
        """
        diff, static = super().partition()
        diff = eqx.tree_at(
            lambda model: (model.NORMALIZATION_MEAN, model.NORMALIZATION_STD),
            diff,
            (None, None),
        )
        static = eqx.tree_at(
            lambda model: (model.NORMALIZATION_MEAN, model.NORMALIZATION_STD),
            static,
            (self.NORMALIZATION_MEAN, self.NORMALIZATION_STD),
            is_leaf=lambda value: value is None,
        )
        return diff, static

    def __call__(self, x, boundary_callback=lambda x: x, key=None):
        if key is None:
            key = jax.random.PRNGKey(int(time.time()))
        x_normalised, mean, std = self._normalise(x)
        dx = self.perception(x_normalised)
        for layer in self.layers:
            dx = layer(dx)
        sigma = jax.random.bernoulli(key, p=self.FIRE_RATE, shape=dx.shape)
        x_new = (x_normalised + sigma * dx) * std + mean
        return boundary_callback(x_new)

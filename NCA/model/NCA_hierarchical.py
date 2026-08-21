"""Two-level Hierarchical Neural Cellular Automaton.

The implementation follows the Sensor/Actuator architecture described by
Pande and Grattarola.  To remain compatible with the existing trainers, which
carry a single ``[channels, height, width]`` array, the parent state is stored
after the child channels and repeated over its corresponding fine-grid
blocks.  It is reduced to its native resolution before every update.
"""

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Int, Scalar

from Common.model.abstract_model import AbstractModel
from NCA.model.NCA_gated_model import gNCA
from NCA.model.NCA_model import NCA


class HNCA(AbstractModel):
    """A child NCA coupled bidirectionally to a lower-resolution parent NCA.

    ``N_CHANNELS`` is the number of channels at *each* level.  The public
    state contains ``2 * N_CHANNELS`` channels: child channels first, followed
    by the blockwise-repeated parent channels.  ``self.N_CHANNELS`` therefore
    reports the packed channel count expected by the existing trainer.

    The Sensor average-pools every child channel and adds the result to the
    parent state.  The Actuator upsamples the parent state, projects it with a
    learned 1x1 convolution, and adds the signal only to the child's hidden
    channels.  Only the child is passed to ``boundary_callback``.
    """

    BATCHED_BOUNDARY_MODE = "internal"

    child_nca: NCA
    parent_nca: NCA
    actuator: eqx.nn.Conv2d
    N_CHANNELS: int
    LEVEL_CHANNELS: int
    OBS_CHANNELS: int
    SCALE: int
    PARENT_LEARNABLE_KERNELS: bool
    CHILD_GATED: bool
    PARENT_GATED: bool
    ACTUATOR_GATED: bool
    KERNEL_STR: list
    FIRE_RATE: float

    def __init__(
        self,
        N_CHANNELS,
        SCALE,
        OBS_CHANNELS=4,
        PARENT_LEARNABLE_KERNELS=False,
        CHILD_GATED=False,
        PARENT_GATED=False,
        ACTUATOR_GATED=False,
        KERNEL_STR=["ID", "GRAD", "LAP"],
        ACTIVATION=jax.nn.relu,
        PADDING="CIRCULAR",
        FIRE_RATE=1.0,
        KERNEL_SCALE=1,
        key=None,
    ):
        if not isinstance(N_CHANNELS, int) or N_CHANNELS < 1:
            raise ValueError("N_CHANNELS must be a positive integer")
        if not isinstance(SCALE, int) or SCALE < 1:
            raise ValueError("SCALE must be a positive integer")
        if not isinstance(OBS_CHANNELS, int) or not 0 <= OBS_CHANNELS < N_CHANNELS:
            raise ValueError("OBS_CHANNELS must be in [0, N_CHANNELS)")
        if not isinstance(PARENT_LEARNABLE_KERNELS, bool):
            raise TypeError("PARENT_LEARNABLE_KERNELS must be a boolean")
        for name, value in (
            ("CHILD_GATED", CHILD_GATED),
            ("PARENT_GATED", PARENT_GATED),
            ("ACTUATOR_GATED", ACTUATOR_GATED),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean")

        if key is None:
            key = jr.PRNGKey(int(time.time()))
        child_key, parent_key, actuator_key = jr.split(key, 3)

        self.LEVEL_CHANNELS = N_CHANNELS
        self.N_CHANNELS = 2 * N_CHANNELS
        self.OBS_CHANNELS = OBS_CHANNELS
        self.SCALE = SCALE
        self.PARENT_LEARNABLE_KERNELS = PARENT_LEARNABLE_KERNELS
        self.CHILD_GATED = CHILD_GATED
        self.PARENT_GATED = PARENT_GATED
        self.ACTUATOR_GATED = ACTUATOR_GATED
        self.KERNEL_STR = list(KERNEL_STR)
        self.FIRE_RATE = FIRE_RATE

        block_kwargs = dict(
            N_CHANNELS=N_CHANNELS,
            KERNEL_STR=KERNEL_STR,
            ACTIVATION=ACTIVATION,
            PADDING=PADDING,
            FIRE_RATE=FIRE_RATE,
            KERNEL_SCALE=KERNEL_SCALE,
        )
        child_type = gNCA if CHILD_GATED else NCA
        parent_type = gNCA if PARENT_GATED else NCA
        self.child_nca = child_type(**block_kwargs, key=child_key)
        self.parent_nca = parent_type(**block_kwargs, key=parent_key)

        hidden_channels = N_CHANNELS - OBS_CHANNELS
        self.actuator = eqx.nn.Conv2d(
            in_channels=N_CHANNELS,
            out_channels=hidden_channels * (2 if ACTUATOR_GATED else 1),
            kernel_size=1,
            use_bias=True,
            key=actuator_key,
        )
        # The child and parent NCA output layers begin at zero update.  The
        # recurrent cross-scale path must obey the same invariant; otherwise
        # a random actuator creates an unstable child-parent feedback loop
        # before the first optimiser step.
        self.actuator = eqx.tree_at(
            lambda layer: (layer.weight, layer.bias),
            self.actuator,
            (
                jnp.zeros_like(self.actuator.weight),
                jnp.zeros_like(self.actuator.bias),
            ),
        )

    def get_config(self):
        return {
            "MODEL": "HNCA",
            "N_CHANNELS": self.LEVEL_CHANNELS,
            "PACKED_CHANNELS": self.N_CHANNELS,
            "OBS_CHANNELS": self.OBS_CHANNELS,
            "SCALE": self.SCALE,
            "PARENT_LEARNABLE_KERNELS": self.PARENT_LEARNABLE_KERNELS,
            "CHILD_GATED": self.CHILD_GATED,
            "PARENT_GATED": self.PARENT_GATED,
            "ACTUATOR_GATED": self.ACTUATOR_GATED,
            "KERNEL_STR": self.KERNEL_STR,
            "PADDING": self.child_nca.op.PADDING,
            "FIRE_RATE": self.FIRE_RATE,
        }

    def _validate_state(self, x):
        if x.ndim != 3 or x.shape[0] != self.N_CHANNELS:
            raise ValueError(
                "HNCA state must have shape "
                f"({self.N_CHANNELS}, height, width); got {x.shape}"
            )
        if x.shape[-2] % self.SCALE or x.shape[-1] % self.SCALE:
            raise ValueError(
                f"Spatial dimensions {x.shape[-2:]} must be divisible by SCALE={self.SCALE}"
            )

    def initialise_state(self, child):
        """Pack a fine state with its average-pooled initial parent state."""
        if child.ndim != 3 or child.shape[0] != self.LEVEL_CHANNELS:
            raise ValueError(
                "Child state must have shape "
                f"({self.LEVEL_CHANNELS}, height, width); got {child.shape}"
            )
        if child.shape[-2] % self.SCALE or child.shape[-1] % self.SCALE:
            raise ValueError(
                f"Spatial dimensions {child.shape[-2:]} must be divisible by SCALE={self.SCALE}"
            )
        parent = self._pool(child)
        return jnp.concatenate((child, self._upsample(parent)), axis=0)

    # American spelling is convenient for callers outside this repository.
    initialize_state = initialise_state

    def boundary_regulariser_state(self, state):
        """Expose only the fine grid to generic boundary regularisation."""
        return state[..., : self.LEVEL_CHANNELS, :, :]

    def prepare_pool_state(self, state):
        """Rebuild the parent from the child at each training-pool boundary.

        The reference HNCA training loop pools child states and constructs a
        fresh parent in its model-specific input wrapper. The packed-array
        representation must do this explicitly or the additive Sensor signal
        accumulates in the parent across successive optimiser iterations.
        """
        if state.shape[-3] != self.N_CHANNELS:
            raise ValueError(
                "HNCA pool state must have "
                f"{self.N_CHANNELS} channels; got {state.shape[-3]}"
            )
        child = state[..., : self.LEVEL_CHANNELS, :, :]
        parent = self._pool(child)
        return jnp.concatenate((child, self._upsample(parent)), axis=-3)

    def _pool(self, x):
        height, width = x.shape[-2:]
        if height % self.SCALE or width % self.SCALE:
            raise ValueError(
                f"Spatial dimensions {(height, width)} must be divisible by "
                f"SCALE={self.SCALE}"
            )
        blocked_shape = (
            *x.shape[:-2],
            height // self.SCALE,
            self.SCALE,
            width // self.SCALE,
            self.SCALE,
        )
        return x.reshape(blocked_shape).mean(axis=(-3, -1))

    def _upsample(self, x):
        return jnp.repeat(
            jnp.repeat(x, self.SCALE, axis=-2),
            self.SCALE,
            axis=-1,
        )

    def __call__(
        self,
        x: Float[Array, "{self.N_CHANNELS} h w"],
        boundary_callback=lambda value: value,
        key=None,
    ) -> Float[Array, "{self.N_CHANNELS} h w"]:
        self._validate_state(x)
        if key is None:
            key = jr.PRNGKey(int(time.time()))
        child_key, parent_key = jr.split(key)

        child = x[: self.LEVEL_CHANNELS]
        parent = self._pool(x[self.LEVEL_CHANNELS :])

        # Sensor and multiplexer: summarize the child and add it to every
        # corresponding parent channel.
        parent_input = parent + self._pool(child)

        # Actuator and multiplexer: parent directives are projected and routed
        # exclusively into the child's hidden channels.
        actuator_signal = self.actuator(self._upsample(parent))
        if self.ACTUATOR_GATED:
            actuator_signal = jax.nn.glu(actuator_signal, axis=0)
        child_input = child.at[self.OBS_CHANNELS :].add(actuator_signal)

        identity = lambda value: value
        child_next = self.child_nca(child_input, identity, child_key)
        parent_next = self.parent_nca(parent_input, identity, parent_key)
        child_next = boundary_callback(child_next)

        return jnp.concatenate((child_next, self._upsample(parent_next)), axis=0)

    def run(
        self,
        iters: Int[Scalar, ""],
        x: Float[Array, "{self.N_CHANNELS} h w"],
        boundary_callback=lambda value: value,
        key=None,
    ):
        if key is None:
            key = jr.PRNGKey(int(time.time()))
        trajectory = [x]
        for i in range(iters):
            x = self(x, boundary_callback, jr.fold_in(key, i))
            trajectory.append(x)
        return jnp.stack(trajectory)

    def partition(self):
        """Keep the fixed perception kernels out of the differentiable tree."""
        diff, static = eqx.partition(self, eqx.is_inexact_array)
        child_diff, child_static = self.child_nca.partition()
        diff = eqx.tree_at(lambda model: model.child_nca, diff, child_diff)
        static = eqx.tree_at(lambda model: model.child_nca, static, child_static)
        if not self.PARENT_LEARNABLE_KERNELS:
            parent_diff, parent_static = self.parent_nca.partition()
            diff = eqx.tree_at(lambda model: model.parent_nca, diff, parent_diff)
            static = eqx.tree_at(lambda model: model.parent_nca, static, parent_static)
        return diff, static

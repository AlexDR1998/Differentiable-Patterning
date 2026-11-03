"""NCA with grown signalling structures ("wires", "nerves" and "hormones").

The last channels of the state are split into:

    [ ... | signal (SIGNAL_CHANNELS) | structure (1-3, depends on MODE) ]

The inner NCA updates every channel as usual, so it can both inject signals
and grow the structure. After each NCA update, a fixed rule moves the signal
channels through the structure. Signals can therefore travel far in one NCA
step, but only where (and how) the NCA has grown a structure for them.

Structure channels are read on [0, 1], the range the intermediate_state
regulariser keeps states in. A gate is its channel value clipped to [0, 1]
(or 1 minus that with OPEN_AT_REST, so that empty tissue conducts and the NCA
has to write insulation). Directions and offsets map 0..1 to -1..1. The edge
between two cells has conductance g_i g_j, so a closed cell (gate 0) lets no
signal through at all.

Modes and their structure channels:
- "cable" [conductance g]: SUBSTEPS diffusion steps along edges weighted by
  g_i g_j. Symmetric, like a dendrite network or gap-junction coupled cells.
- "axon" [g, direction (2)]: SUBSTEPS upwind transport steps along a learned
  direction. Directed; a signal piles up at the end of a wire.
- "steady" [g]: the steady state of the cable, with screening rate
  SCREENING, solved by conjugate gradients (at most SUBSTEPS iterations).
  Instant (non-local in one step) averaging along each connected wire.
- "excitable" [g]: signal channels are split into halves (voltage u, recovery
  w) with excitable (FitzHugh-Nagumo like) dynamics on conductive cells, for
  SUBSTEPS steps. Pulses travel along wires without fading out.
- "broadcast" [transmitter gate, receiver gate]: transmitters emit their
  signal, which is spread by a learned mix of Gaussian kernels (widths 1, 2,
  4, ... up to RANGE) and a global average. Receivers replace their signal
  with what they receive. Like hormones or cytonemes.
- "pointer" [gate, offset (2)]: each gated cell reads the signal at a learned
  offset of up to RANGE cells (bilinear interpolation). Like a dendrite that
  reaches a distant cell.

cable, axon and steady conserve the total signal. All modes are
differentiable everywhere.
"""

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Int, Scalar

from Common.model.abstract_model import AbstractModel
from NCA.model.NCA_model import NCA


# Number of structure channels written by the NCA for each mode
STRUCTURE_CHANNELS = {"cable": 1, "axon": 3, "steady": 1, "excitable": 1, "broadcast": 2, "pointer": 3}


# Excitable dynamics: du = gain u(1-u)(u-threshold) - w, dw = recovery_rate (u - w).
# Rest is u = w = 0, a pulse takes u to about 1 and travels ~0.3 cells per
# sub-step along a wire.
EXCITABLE_GAIN = 2.0
EXCITABLE_THRESHOLD = 0.1
EXCITABLE_RECOVERY_RATE = 0.02

_PAD_MODES = {"CIRCULAR": "wrap", "ZEROS": "constant", "REFLECT": "reflect", "REPLICATE": "edge"}
# jax.scipy.ndimage.map_coordinates names for the same boundary handling
_SAMPLE_MODES = {"CIRCULAR": "wrap", "ZEROS": "constant", "REFLECT": "mirror", "REPLICATE": "nearest"}


def _neighbours(x, padding, wall=False):
    """Values of the 4 neighbours of each cell, as a dict of arrays shaped like ``x``.

    With ``wall=True`` cells outside a non-periodic domain read 0. Used for the
    conductance, so that no signal crosses the domain edge.
    """
    mode = _PAD_MODES[padding]
    if wall and padding != "CIRCULAR":
        mode = "constant"
    p = jnp.pad(x, [(0, 0)] * (x.ndim - 2) + [(1, 1), (1, 1)], mode=mode)
    return {
        "up": p[..., :-2, 1:-1],
        "down": p[..., 2:, 1:-1],
        "left": p[..., 1:-1, :-2],
        "right": p[..., 1:-1, 2:],
    }


def conductance_from_channel(raw, open_at_rest=False):
    """Gate in [0, 1] from a raw structure channel: closed at 0 and open at 1 (reversed if ``open_at_rest``).

    A smooth (sigmoid) gate is never fully closed, and over many sub-steps
    most of a signal leaks out of a wire through the cells around it.
    """
    gate = jnp.clip(raw, 0.0, 1.0)
    return 1.0 - gate if open_at_rest else gate


def direction_from_channels(raw):
    """Direction [2, x, y] (row, column) from 2 raw channels on [0, 1], with |v_row| + |v_col| <= 1.

    A channel at 0.5 means no movement along that axis; 0 and 1 mean full speed
    up/left and down/right.
    """
    v = 2 * raw - 1
    return v / jnp.maximum(1.0, jnp.sum(jnp.abs(v), axis=0))


def offset_from_channels(raw, signal_range):
    """Pointer offset [2, x, y] in cells from 2 raw channels on [0, 1]: 0.5 is no offset."""
    return signal_range * (2 * raw - 1)


def _edges(g, padding):
    """Conductance g_i g_j of the edge to each of the 4 neighbours j, as a dict."""
    g_n = _neighbours(g, padding, wall=True)
    return {d: g * g_n[d] for d in g_n}


def broadcast_scales(signal_range):
    """Gaussian widths 1, 2, 4, ... up to ``signal_range`` used by broadcast mode."""
    scales = [1]
    while scales[-1] * 2 <= signal_range:
        scales.append(scales[-1] * 2)
    return tuple(scales)


def _graph_flux(s, g, padding):
    """sum_j c_ij (s_j - s_i) over the 4 neighbours j: minus the conductance-weighted graph Laplacian."""
    c = _edges(g, padding)
    s_n = _neighbours(s, padding)
    return sum(c[d] * (s_n[d] - s) for d in s_n)


def cable_step(s, g, padding, diffusion):
    """One explicit diffusion step of signals ``s`` [K, x, y] through conductance ``g`` [x, y].

    The flux between two neighbouring cells is weighted by g_i g_j. Stable for
    ``diffusion <= 0.25``.
    """
    return s + diffusion * _graph_flux(s, g, padding)


def axon_step(s, g, v, padding):
    """One upwind transport step of signals ``s`` [K, x, y] along direction ``v`` [2, x, y].

    Each cell sends the fraction |v| of its signal to the neighbour(s) that v
    points at, scaled by the edge conductance g_i g_j. With |v_row| + |v_col| <= 1
    a cell never sends more than it has, and v = (0, 1) on a wire moves a
    signal exactly one cell per step.
    """
    c = _edges(g, padding)
    s_n = _neighbours(s, padding)
    v_row_n = _neighbours(v[0], padding)
    v_col_n = _neighbours(v[1], padding)
    relu = jax.nn.relu
    sent = s * (
        c["down"] * relu(v[0]) + c["up"] * relu(-v[0])
        + c["right"] * relu(v[1]) + c["left"] * relu(-v[1])
    )
    received = (
        c["up"] * relu(v_row_n["up"]) * s_n["up"]
        + c["down"] * relu(-v_row_n["down"]) * s_n["down"]
        + c["left"] * relu(v_col_n["left"]) * s_n["left"]
        + c["right"] * relu(-v_col_n["right"]) * s_n["right"]
    )
    return s - sent + received


def steady_cable(s, g, padding, screening, iterations):
    """Steady state r of the screened cable: (screening + L_g) r = screening * s.

    L_g is the conductance-weighted graph Laplacian (see ``cable_step``). Cells
    with g = 0 keep their signal; along a connected wire the signal is spread
    out over a length of about sqrt(g / screening) cells. Solved by conjugate
    gradients; gradients use a second solve rather than going through the loop.
    """
    operator = lambda r: screening * r - _graph_flux(r, g, padding)
    r, _ = jax.scipy.sparse.linalg.cg(operator, screening * s, x0=s, maxiter=iterations)
    return r


def excitable_step(s, g, padding, diffusion):
    """One step of excitable dynamics on conductive cells.

    The first half of ``s`` is the voltage u, the second half the recovery w.
    u diffuses along the wire like ``cable_step``; the local dynamics are
    scaled by g, so cells off the wire are left unchanged.
    """
    u, w = jnp.split(s, 2)
    # The cubic is only applied on [0, 1], and values outside are pulled back
    # linearly, so large values written by the NCA can't blow up
    c = jnp.clip(u, 0.0, 1.0)
    du = EXCITABLE_GAIN * c * (1 - c) * (c - EXCITABLE_THRESHOLD) - 0.5 * (u - c) - w
    dw = EXCITABLE_RECOVERY_RATE * (u - w)
    u = u + g * du + diffusion * _graph_flux(u, g, padding)
    w = w + g * dw
    return jnp.concatenate((u, w))


def _gaussian_blur(x, sigma, padding):
    """Gaussian blur of ``x`` [K, x, y] with width ``sigma``, by FFT.

    Periodic for CIRCULAR padding; otherwise the field is zero padded, so
    signal that spreads past the edge is lost.
    """
    height, width = x.shape[-2:]
    if padding != "CIRCULAR":
        x = jnp.pad(x, [(0, 0), (0, height), (0, width)])
    fy = jnp.fft.fftfreq(x.shape[-2])[:, None]
    fx = jnp.fft.rfftfreq(x.shape[-1])[None, :]
    transfer = jnp.exp(-2 * jnp.pi**2 * sigma**2 * (fx**2 + fy**2))
    blurred = jnp.fft.irfft2(jnp.fft.rfft2(x) * transfer, s=x.shape[-2:])
    return blurred[..., :height, :width]


def broadcast(s, transmit, receive, kernel_logits, scales, padding):
    """Receivers take a learned blend of Gaussian blurs and the global mean of the transmitted signal.

    ``kernel_logits`` [K, len(scales) + 1] gives, per signal channel, the weight
    (softmax) of each Gaussian width and of the global mean (last entry).
    """
    emitted = transmit * s
    spread = [_gaussian_blur(emitted, sigma, padding) for sigma in scales]
    spread.append(jnp.broadcast_to(jnp.mean(emitted, axis=(-2, -1), keepdims=True), s.shape))
    weights = jax.nn.softmax(kernel_logits, axis=-1)
    received = jnp.einsum("kn,nkxy->kxy", weights, jnp.stack(spread))
    return (1 - receive) * s + receive * received


def pointer_read(s, g, offset, padding):
    """Each cell reads the signal at its own position plus ``offset`` [2, x, y], mixed in by gate ``g``."""
    height, width = s.shape[-2:]
    rows, cols = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
    coordinates = [rows + offset[0], cols + offset[1]]
    sample = lambda channel: jax.scipy.ndimage.map_coordinates(
        channel, coordinates, order=1, mode=_SAMPLE_MODES[padding]
    )
    return (1 - g) * s + g * jax.vmap(sample)(s)


def _repeat(step, s, count):
    return jax.lax.fori_loop(0, count, lambda _, s: step(s), s)


class SignalNCA(AbstractModel):
    """An NCA plus signal channels that move quickly along structures the NCA grows."""

    nca: NCA
    # Broadcast mode only (None otherwise): weight of each kernel width per signal channel
    kernel_logits: Array | None
    MODE: str = eqx.field(static=True)
    SIGNAL_CHANNELS: int = eqx.field(static=True)
    SUBSTEPS: int = eqx.field(static=True)
    DIFFUSION: float = eqx.field(static=True)
    SCREENING: float = eqx.field(static=True)
    RANGE: int = eqx.field(static=True)
    OPEN_AT_REST: bool = eqx.field(static=True)

    def __init__(self,
                 N_CHANNELS,
                 KERNEL_STR=["ID", "LAP"],
                 ACTIVATION=jax.nn.relu,
                 PADDING="CIRCULAR",
                 FIRE_RATE=1.0,
                 KERNEL_SCALE=1,
                 key=None,
                 GATED=False,
                 MODE="cable",
                 SIGNAL_CHANNELS=4,
                 SUBSTEPS=8,
                 DIFFUSION=0.25,
                 SCREENING=0.01,
                 RANGE=16,
                 OPEN_AT_REST=False):
        """
        Parameters
        ----------
        N_CHANNELS, KERNEL_STR, ACTIVATION, PADDING, FIRE_RATE, KERNEL_SCALE, key, GATED
            Passed to the inner NCA, which sees and updates all channels.
        MODE : str
            One of "cable", "axon", "steady", "excitable", "broadcast", "pointer"
            (see the module docstring).
        SIGNAL_CHANNELS : int
            Number of signal channels, placed just before the structure
            channels. Must be even for "excitable".
        SUBSTEPS : int
            Propagation sub-steps per NCA step (cable, axon, excitable), or the
            maximum number of solver iterations (steady).
        DIFFUSION : float
            Diffusion rate of each cable or excitable sub-step, in (0, 0.25].
        SCREENING : float
            Screening rate of the steady cable, > 0. Smaller values spread
            signals further along a wire.
        RANGE : int
            Widest broadcast kernel, or the furthest pointer reach, in cells.
        OPEN_AT_REST : bool
            If True, gates are open where their channel is 0 and closed at 1, so
            signals can cross empty tissue and the NCA learns where to block them.
        """
        if MODE not in STRUCTURE_CHANNELS:
            raise ValueError(f"MODE must be one of {tuple(STRUCTURE_CHANNELS)}, got {MODE!r}")
        if SIGNAL_CHANNELS < 1 or SIGNAL_CHANNELS + STRUCTURE_CHANNELS[MODE] >= N_CHANNELS:
            raise ValueError("Need 1 <= SIGNAL_CHANNELS and room for the structure channels within N_CHANNELS")
        if MODE == "excitable" and SIGNAL_CHANNELS % 2:
            raise ValueError("MODE='excitable' needs an even SIGNAL_CHANNELS (voltage and recovery halves)")
        if SUBSTEPS < 1:
            raise ValueError("SUBSTEPS must be a positive integer")
        if not 0 < DIFFUSION <= 0.25:
            raise ValueError("DIFFUSION must be in (0, 0.25] for a stable cable step")
        if SCREENING <= 0:
            raise ValueError("SCREENING must be positive")
        if RANGE < 1:
            raise ValueError("RANGE must be a positive integer")
        self.nca = NCA(
            N_CHANNELS=N_CHANNELS,
            KERNEL_STR=KERNEL_STR,
            ACTIVATION=ACTIVATION,
            PADDING=PADDING,
            FIRE_RATE=FIRE_RATE,
            KERNEL_SCALE=KERNEL_SCALE,
            key=key,
            GATED=GATED,
        )
        self.MODE = MODE
        self.SIGNAL_CHANNELS = SIGNAL_CHANNELS
        self.SUBSTEPS = SUBSTEPS
        self.DIFFUSION = DIFFUSION
        self.SCREENING = SCREENING
        self.RANGE = RANGE
        self.OPEN_AT_REST = OPEN_AT_REST
        # Starts as an equal blend of all kernel widths and the global mean
        self.kernel_logits = (
            jnp.zeros((SIGNAL_CHANNELS, len(broadcast_scales(RANGE)) + 1))
            if MODE == "broadcast" else None
        )

    # The trainer reads these from the model
    @property
    def N_CHANNELS(self):
        return self.nca.N_CHANNELS

    @property
    def KERNEL_STR(self):
        return self.nca.KERNEL_STR

    @property
    def FIRE_RATE(self):
        return self.nca.FIRE_RATE

    @property
    def perception(self):
        return self.nca.perception

    @property
    def structure_start(self):
        return self.N_CHANNELS - STRUCTURE_CHANNELS[self.MODE]

    @property
    def signal_start(self):
        return self.structure_start - self.SIGNAL_CHANNELS

    def get_config(self):
        config = self.nca.get_config()
        config["MODEL"] = "gsNCA" if self.nca.GATED else "sNCA"
        config.update({
            "SIGNAL_MODE": self.MODE,
            "SIGNAL_CHANNELS": self.SIGNAL_CHANNELS,
            "SIGNAL_SUBSTEPS": self.SUBSTEPS,
            "SIGNAL_DIFFUSION": self.DIFFUSION,
            "SIGNAL_SCREENING": self.SCREENING,
            "SIGNAL_RANGE": self.RANGE,
            "SIGNAL_OPEN_AT_REST": self.OPEN_AT_REST,
        })
        return config

    def conductance(self, state):
        """First structure gate of each cell, read from ``state`` [..., C, x, y].

        This is the conductance (cable, axon, steady, excitable), the
        transmitter gate (broadcast) or the reach gate (pointer).
        """
        return conductance_from_channel(state[..., self.structure_start, :, :], self.OPEN_AT_REST)

    def direction(self, state):
        """Axon direction [..., 2, x, y] of each cell (only for MODE="axon")."""
        if self.MODE != "axon":
            raise ValueError("Only MODE='axon' has a direction")
        start = self.structure_start + 1
        return direction_from_channels(state[..., start:start + 2, :, :])

    def propagate(self, x):
        """Move the signal channels of ``x`` [C, x, y] through its structure."""
        padding = self.nca.op.PADDING
        structure = x[self.structure_start:]
        g = conductance_from_channel(structure[0], self.OPEN_AT_REST)
        s = x[self.signal_start:self.structure_start]
        if self.MODE == "cable":
            s = _repeat(lambda s: cable_step(s, g, padding, self.DIFFUSION), s, self.SUBSTEPS)
        elif self.MODE == "axon":
            v = direction_from_channels(structure[1:3])
            s = _repeat(lambda s: axon_step(s, g, v, padding), s, self.SUBSTEPS)
        elif self.MODE == "steady":
            s = steady_cable(s, g, padding, self.SCREENING, self.SUBSTEPS)
        elif self.MODE == "excitable":
            s = _repeat(lambda s: excitable_step(s, g, padding, self.DIFFUSION), s, self.SUBSTEPS)
        elif self.MODE == "broadcast":
            receive = conductance_from_channel(structure[1], self.OPEN_AT_REST)
            s = broadcast(s, g, receive, self.kernel_logits, broadcast_scales(self.RANGE), padding)
        else:  # pointer
            offset = offset_from_channels(structure[1:3], self.RANGE)
            s = pointer_read(s, g, offset, padding)
        return x.at[self.signal_start:self.structure_start].set(s)

    def signal_movement(self, x):
        """Mean absolute change of the signal channels made by one ``propagate`` of ``x`` [C, x, y].

        0 means the signalling does nothing in this state.
        """
        moved = self.propagate(x) - x
        return jnp.mean(jnp.abs(moved[self.signal_start:self.structure_start]))

    def __call__(self,
                 x: Float[Array, "C x y"],
                 boundary_callback=lambda x: x,
                 key=None) -> Float[Array, "C x y"]:
        """One NCA update, then signal propagation, then ``boundary_callback``."""
        if key is None:
            key = jr.PRNGKey(int(time.time()))
        x = self.nca(x, key=key)
        return boundary_callback(self.propagate(x))

    def run(self,
            iters: Int[Scalar, ""],
            x: Float[Array, "C x y"],
            callback=lambda x: x,
            key=None) -> Float[Array, "{iters} C x y"]:
        if key is None:
            key = jr.PRNGKey(int(time.time()))
        trajectory = [x]
        for i in range(iters):
            key = jr.fold_in(key, i)
            x = self(x, callback, key=key)
            trajectory.append(x)
        return jnp.stack(trajectory)

    def partition(self):
        """Keep the fixed perception kernels of the inner NCA out of the differentiable tree."""
        diff, static = eqx.partition(self, eqx.is_inexact_array)
        nca_diff, nca_static = self.nca.partition()
        diff = eqx.tree_at(lambda m: m.nca, diff, nca_diff)
        static = eqx.tree_at(lambda m: m.nca, static, nca_static)
        return diff, static

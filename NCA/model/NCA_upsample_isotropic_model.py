import math

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import einsum, rearrange
from NCA.model.NCA_model import NCA, Ops
import time
from jaxtyping import Key
Array = jax.Array


def _wendland_c2(dist: Array, radius: float) -> Array:
    s = dist / radius
    t = jnp.clip(1.0 - s, 0.0, None)
    return (t**4) * (4.0 * s + 1.0)


class PointwiseConvNet(eqx.Module):
    """Pointwise MLP as stacked 1x1 convolutions."""
    layers: tuple

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        depth: int,
        *,
        key: Array,
    ):
        if depth < 1:
            raise ValueError("depth must be >= 1")

        dims = [in_channels] + [hidden_channels] * (depth - 1) + [out_channels]
        keys = jax.random.split(key, len(dims) - 1)

        self.layers = tuple(
            eqx.nn.Conv2d(
                in_channels=dims[i],
                out_channels=dims[i + 1],
                kernel_size=1,
                key=keys[i],
            )
            for i in range(len(dims) - 1)
        )

    def __call__(self, x: Array) -> Array:
        for layer in self.layers[:-1]:
            x = jax.nn.silu(layer(x))
        return self.layers[-1](x)


class IsotropicMLSUpsampler(eqx.Module):
    """
    Input:  latent (C, Xc, Yc)
    Output: image  (D, X, Y), where X = scale_x * Xc, Y = scale_y * Yc

    Learnable part:
        self.decoder

    Fixed part:
        isotropic radial MLS + M=2 harmonic basis
    """

    decoder: PointwiseConvNet

    latent_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    scale_x: int = eqx.field(static=True)
    scale_y: int = eqx.field(static=True)
    radius: float = eqx.field(static=True)
    mls_reg: float = eqx.field(static=True)
    stencil_radius: int = eqx.field(static=True)

    def __init__(
        self,
        latent_channels: int,
        out_channels: int,
        *,
        scale_x: int,
        scale_y: int | None = None,
        hidden_channels: int = 128,
        depth: int = 3,
        radius: float = 2.0,
        mls_reg: float = 1e-4,
        key: Array,
    ):
        if out_channels > latent_channels:
            raise ValueError("out_channels must be <= latent_channels")
        if scale_y is None:
            scale_y = scale_x

        self.latent_channels = latent_channels
        self.out_channels = out_channels
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.radius = radius
        self.mls_reg = mls_reg
        self.stencil_radius = int(math.ceil(radius))

        self.decoder = PointwiseConvNet(
            in_channels=latent_channels + 2,   # z(q) + [r, r^2]
            out_channels=5 * out_channels,     # a0, a1, b1, a2, b2
            hidden_channels=hidden_channels,
            depth=depth,
            key=key,
        )
        self.decoder = eqx.tree_at(
            lambda m: m.layers,
            self.decoder,
            self.decoder.layers[:-1] + (self._zero_conv(self.decoder.layers[-1]),),
        )

    @staticmethod
    def _zero_conv(layer: eqx.nn.Conv2d) -> eqx.nn.Conv2d:
        if layer.bias is None:
            return eqx.tree_at(lambda l: l.weight, layer, jnp.zeros_like(layer.weight))
        return eqx.tree_at(
            lambda l: (l.weight, l.bias),
            layer,
            (jnp.zeros_like(layer.weight), jnp.zeros_like(layer.bias)),
        )

    def _geometry(self, x_coarse: int, y_coarse: int, dtype) -> tuple[Array, Array, Array, Array]:
        x_fine = self.scale_x * x_coarse
        y_fine = self.scale_y * y_coarse

        # Fine pixel centres in coarse-grid coordinates
        xs = (jnp.arange(x_fine, dtype=dtype) + 0.5) / self.scale_x - 0.5
        ys = (jnp.arange(y_fine, dtype=dtype) + 0.5) / self.scale_y - 0.5
        qx, qy = jnp.meshgrid(xs, ys, indexing="ij")   # (X, Y)

        cx = jnp.rint(qx).astype(jnp.int32)
        cy = jnp.rint(qy).astype(jnp.int32)

        dx = qx - cx.astype(dtype)
        dy = qy - cy.astype(dtype)

        radial_feat = jnp.stack(
            [
                jnp.sqrt(dx * dx + dy * dy),
                dx * dx + dy * dy,
            ],
            axis=0,
        )  # (2, X, Y)

        # Real M=2 basis
        basis = jnp.stack(
            [
                dx,
                dy,
                dx * dx - dy * dy,
                2.0 * dx * dy,
            ],
            axis=0,
        )  # (4, X, Y)

        rr = self.stencil_radius
        ox, oy = jnp.meshgrid(
            jnp.arange(-rr, rr + 1),
            jnp.arange(-rr, rr + 1),
            indexing="ij",
        )
        offsets = jnp.stack([ox.reshape(-1), oy.reshape(-1)], axis=1)  # (N, 2)

        ix = cx[..., None] + offsets[None, None, :, 0]   # (X, Y, N)
        iy = cy[..., None] + offsets[None, None, :, 1]

        valid = (
            (ix >= 0) & (ix < x_coarse) &
            (iy >= 0) & (iy < y_coarse)
        )

        ix = jnp.clip(ix, 0, x_coarse - 1)
        iy = jnp.clip(iy, 0, y_coarse - 1)

        px = ix.astype(dtype)
        py = iy.astype(dtype)

        ddx = px - qx[..., None]
        ddy = py - qy[..., None]
        dist = jnp.sqrt(ddx * ddx + ddy * ddy)

        w = _wendland_c2(dist, self.radius) * valid.astype(dtype)   # (X, Y, N)

        A = jnp.stack(
            [
                jnp.ones_like(ddx),
                ddx,
                ddy,
            ],
            axis=-1,
        )  # (X, Y, N, 3)

        Aw = A * w[..., None]
        M = einsum(A, Aw, "x y n i, x y n j -> x y i j")
        M = M + self.mls_reg * jnp.eye(3, dtype=dtype)[None, None, :, :]

        ATW = einsum(A, w, "x y n i, x y n -> x y i n")   # (X, Y, 3, N)
        G = jnp.linalg.solve(M, ATW)               # (X, Y, 3, N)
        lam_mls = G[..., 0, :]                     # (X, Y, N)

        wsum = jnp.sum(w, axis=-1, keepdims=True)
        lam_avg = w / (wsum + 1e-8)
        lam = jnp.where(wsum > 1e-6, lam_mls, lam_avg)

        flat_idx = ix * y_coarse + iy              # (X, Y, N)
        return flat_idx, lam, radial_feat, basis

    def _interpolate_latent(self, latent: Array, flat_idx: Array, lam: Array) -> Array:
        # latent: (C, Xc, Yc)
        C, Xc, Yc = latent.shape
        latent_flat = rearrange(latent, "c x y -> c (x y)")
        vals = jnp.take(latent_flat, flat_idx, axis=1)   # (C, X, Y, N)
        return einsum(vals, lam, "c x y n, x y n -> c x y")

    def __call__(self, latent: Array) -> Array:
        # latent: (C, Xc, Yc)
        C, Xc, Yc = latent.shape
        if C != self.latent_channels:
            raise ValueError(f"Expected {self.latent_channels} latent channels, got {C}")

        dtype = latent.dtype
        flat_idx, lam, radial_feat, basis = self._geometry(Xc, Yc, dtype)

        z = self._interpolate_latent(latent, flat_idx, lam)   # (C, X, Y)
        feat = jnp.concatenate([z, radial_feat], axis=0)      # (C+2, X, Y)

        coeffs = self.decoder(feat)                           # (5D, X, Y)

        X = self.scale_x * Xc
        Y = self.scale_y * Yc
        coeffs = rearrange(
            coeffs,
            "(g d) x y -> g d x y",
            g=5,
            d=self.out_channels,
        )

        a0 = coeffs[0] + z[: self.out_channels]
        a1 = coeffs[1]
        b1 = coeffs[2]
        a2 = coeffs[3]
        b2 = coeffs[4]

        dx = basis[0][None, :, :]
        dy = basis[1][None, :, :]
        q2 = basis[2][None, :, :]
        qxy = basis[3][None, :, :]

        return a0 + a1 * dx + b1 * dy + a2 * q2 + b2 * qxy



class uNCA(NCA):
    layers: list
    KERNEL_STR: list
    N_CHANNELS: int
    # O_CHANNELS: int
    N_FEATURES: int
    SPATIAL_UPSAMPLE: int
    FIRE_RATE: float
    op: Ops
    perception: callable
    upsample: IsotropicMLSUpsampler
    #CONFIG: dict

    def __init__(self,
                N_CHANNELS,
                O_CHANNELS,
                KERNEL_STR=["ID","LAP"], 
                ACTIVATION=jax.nn.relu, 
                PADDING="CIRCULAR", 
                FIRE_RATE=1.0, 
                KERNEL_SCALE = 1, 
                SPATIAL_UPSAMPLE = 4,
                key=jax.random.PRNGKey(int(time.time()))):
        super().__init__(N_CHANNELS, KERNEL_STR, ACTIVATION, PADDING, FIRE_RATE, KERNEL_SCALE, key)
        #key1,key2 = jax.random.split(key,2)
        key = jax.random.fold_in(key,1234)
        self.SPATIAL_UPSAMPLE = SPATIAL_UPSAMPLE
        self.upsample = IsotropicMLSUpsampler(
            latent_channels=N_CHANNELS, 
            out_channels=O_CHANNELS, 
            scale_x=SPATIAL_UPSAMPLE, 
            scale_y=SPATIAL_UPSAMPLE, 
            key=key)

    def prepare_state(self, x):
        latent_shape = x.shape[:-2] + (
            max(1, x.shape[-2] // self.SPATIAL_UPSAMPLE),
            max(1, x.shape[-1] // self.SPATIAL_UPSAMPLE),
        )
        return jax.image.resize(x, latent_shape, method="linear")

    def process(self, x):
        if x.ndim == 3:
            return self.upsample(x)

        # x = rearrange(x, "b c h w -> b c h w")
        return jax.vmap(lambda x_i: self.upsample(x_i))(x)


    def __call__(self,
			  	 x,
				 boundary_callback=lambda x:x,
				 key: Key=jax.random.PRNGKey(int(time.time()))):
        dx = self.perception(x)
        for layer in self.layers:
            dx = layer(dx)
        sigma = jax.random.bernoulli(key,p=self.FIRE_RATE,shape=dx.shape)
        x_new = x + sigma*dx
        x_new = boundary_callback(x_new)
        return x_new

    def get_config(self):
        """
        Returns the model configuration as a dictionary.

        Returns
        -------
        dict
            dictionary of model hyperparameters

        """
        
        return {
            "MODEL":"uNCA",
            "N_CHANNELS":self.N_CHANNELS,
            "KERNEL_STR":self.KERNEL_STR,
            "ACTIVATION":self.layers[1].__name__,
            "PADDING":self.op.PADDING,
            "FIRE_RATE":self.FIRE_RATE
        }
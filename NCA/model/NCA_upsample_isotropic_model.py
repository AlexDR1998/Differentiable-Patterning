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
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)



class IsotropicKernelUpsampler(eqx.Module):
    """
    Fixed isotropic radial kernel upsampling, followed by an optional learnable 1x1 projection.

    Input:
        latent: (C, Xc, Yc)

    Output:
        image: (D, scale * Xc, scale * Yc)

    The spatial upsampling itself is fixed and isotropic.
    The learnable part is only a pointwise 1x1 channel projection/mixing applied after upsampling.
    """

    decoder: PointwiseConvNet
    op: Ops

    latent_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    scale: int = eqx.field(static=True)
    radius: float = eqx.field(static=True)
    stencil_radius: int = eqx.field(static=True)
    residual: bool = eqx.field(static=True)
    residual_scale: float = eqx.field(static=True)

    def __init__(
        self,
        latent_channels: int,
        out_channels: int,
        *,
        scale: int,
        radius: float = 2.0,
        residual: bool = True,
        decoder_depth: int = 3,
        residual_scale: float = 1.0,
        padding="CIRCULAR",
        key: Array,
    ):
        self.latent_channels = latent_channels
        self.out_channels = out_channels
        self.scale = scale
        self.radius = radius
        self.stencil_radius = int(math.ceil(radius))
        self.residual = residual
        self.residual_scale = residual_scale
        self.op = Ops(PADDING=padding,dx=1,KERNEL_SCALE=1,SMOOTHING=1) # type: ignore
        
        self.decoder = PointwiseConvNet(
            in_channels=3*latent_channels, # We concat laplacian and gradient magnitude to the input.
            out_channels=out_channels,
            hidden_channels=latent_channels,
            depth=decoder_depth,
            key=key,
        )
        # Helpful initialization:
        # start from either identity-like residual behaviour or near-zero learned correction.
        self.decoder = eqx.tree_at(
            lambda m: m.layers,
            self.decoder,
            self.decoder.layers[:-1] + (self._zero_conv(self.decoder.layers[-1]),),
        )

    def spatial_gradients(self,x):
        """
            [C X Y] -> [3C X Y]
            Returns identity, gradient magnitude and laplacian for each channel, concatenated along channel dimension
        """
        x_id = x
        x_diff = self.op.GradNorm(x)
        x_lap = self.op.Lap(x)
        output = rearrange([x_id,x_diff,x_lap],"b C x y -> (b C) x y")
        return output

    @staticmethod
    def _zero_conv(layer: eqx.nn.Conv2d) -> eqx.nn.Conv2d:
        if layer.bias is None:
            return eqx.tree_at(
                lambda l: l.weight,
                layer,
                jnp.zeros_like(layer.weight),
            )

        return eqx.tree_at(
            lambda l: (l.weight, l.bias),
            layer,
            (jnp.zeros_like(layer.weight), jnp.zeros_like(layer.bias)),
        )

    def _geometry(self, x_coarse: int, y_coarse: int, dtype):
        x_fine = self.scale * x_coarse
        y_fine = self.scale * y_coarse

        # Fine pixel centres in coarse-grid coordinates.
        xs = (jnp.arange(x_fine, dtype=dtype) + 0.5) / self.scale - 0.5
        ys = (jnp.arange(y_fine, dtype=dtype) + 0.5) / self.scale - 0.5

        qx, qy = jnp.meshgrid(xs, ys, indexing="ij")

        # Nearest coarse-grid point around which to form the local stencil.
        cx = jnp.rint(qx).astype(jnp.int32)
        cy = jnp.rint(qy).astype(jnp.int32)

        rr = self.stencil_radius

        ox, oy = jnp.meshgrid(
            jnp.arange(-rr, rr + 1),
            jnp.arange(-rr, rr + 1),
            indexing="ij",
        )

        offsets = jnp.stack(
            [ox.reshape(-1), oy.reshape(-1)],
            axis=1,
        )  # (N, 2)

        ix = cx[..., None] + offsets[None, None, :, 0]
        iy = cy[..., None] + offsets[None, None, :, 1]

        valid = (
            (ix >= 0)
            & (ix < x_coarse)
            & (iy >= 0)
            & (iy < y_coarse)
        )

        ix_clip = jnp.clip(ix, 0, x_coarse - 1)
        iy_clip = jnp.clip(iy, 0, y_coarse - 1)

        px = ix.astype(dtype)
        py = iy.astype(dtype)

        ddx = px - qx[..., None]
        ddy = py - qy[..., None]

        dist = jnp.sqrt(ddx * ddx + ddy * ddy)

        # Fixed isotropic radial weights.
        w = _wendland_c2(dist, self.radius) * valid.astype(dtype)

        # Nonnegative partition-of-unity weights.
        lam = w / (jnp.sum(w, axis=-1, keepdims=True) + 1e-8)

        flat_idx = ix_clip * y_coarse + iy_clip

        return flat_idx, lam

    def _interpolate(self, latent: Array, flat_idx: Array, lam: Array) -> Array:
        # latent: (C, Xc, Yc)
        latent_flat = rearrange(latent, "c x y -> c (x y)")

        # vals: (C, Xf, Yf, N)
        vals = jnp.take(latent_flat, flat_idx, axis=1)

        # z: (C, Xf, Yf)
        z = einsum(vals, lam, "c x y n, x y n -> c x y")

        return z

    def __call__(self, latent: Array) -> Array:
        """
        latent: (C, Xc, Yc)
        returns: (out_channels, scale * Xc, scale * Yc)
        """
        C, Xc, Yc = latent.shape # Shape C Xcourse Ycourse

        if C != self.latent_channels:
            raise ValueError(
                f"Expected {self.latent_channels} latent channels, got {C}"
            )

        flat_idx, lam = self._geometry(Xc, Yc, latent.dtype)
        z = self._interpolate(latent, flat_idx, lam) # Shape C Xfine Yfine
        z_spatial = self.spatial_gradients(z) # Shape 3C Xfine Yfine
        learned = self.decoder(z_spatial) # Shape out_channels Xfine Yfine

        if self.residual:
            if self.out_channels > self.latent_channels:
                raise ValueError(
                    "Residual mode requires out_channels <= latent_channels."
                )

            base = z[: self.out_channels]
            return base + self.residual_scale * learned

        return learned



class uNCA(NCA):
    layers: list
    KERNEL_STR: list
    N_CHANNELS: int
    # O_CHANNELS: int
    N_FEATURES: int
    UPSAMPLER_AUX: dict
    FIRE_RATE: float
    op: Ops
    perception: callable # type: ignore
    upsample: IsotropicKernelUpsampler
    #CONFIG: dict

    def __init__(self,
                N_CHANNELS,
                O_CHANNELS,
                KERNEL_STR=["ID","LAP"], 
                ACTIVATION=jax.nn.relu, 
                PADDING="CIRCULAR", 
                FIRE_RATE=1.0, 
                KERNEL_SCALE = 1, 
                UPSAMPLER_AUX = {
                    "depth": 3,
                    "width_factor": 1,
                    "radius": 2,
                    "upsample_factor": 4
                },
                key=None):
        if key is None:
            key = jax.random.PRNGKey(int(time.time()))
        super().__init__(N_CHANNELS, KERNEL_STR, ACTIVATION, PADDING, FIRE_RATE, KERNEL_SCALE, key)
        key = jax.random.fold_in(key,1234)
        self.UPSAMPLER_AUX = UPSAMPLER_AUX
        self.upsample = IsotropicKernelUpsampler(
            latent_channels=N_CHANNELS, 
            out_channels=O_CHANNELS, 
            scale=UPSAMPLER_AUX["upsample_factor"],
            radius=UPSAMPLER_AUX["radius"],
            decoder_depth=UPSAMPLER_AUX["depth"],
            residual=True,
            key=key)
            

    def real_to_latent(self, x):
        """
            Takes real image and downsamples to latent space using linear interpolation
            X: [...,W,H]
        """
        latent_shape = x.shape[:-2] + (
            max(1, x.shape[-2] // self.UPSAMPLER_AUX["upsample_factor"]),
            max(1, x.shape[-1] // self.UPSAMPLER_AUX["upsample_factor"]),
        )
        return jax.image.resize(x, latent_shape, method="bicubic",antialias=True)

    def latent_to_real(self, x):
        if x.ndim == 3:
            return self.upsample(x)

        # x = rearrange(x, "b c h w -> b c h w")
        return jax.vmap(lambda x_i: self.upsample(x_i))(x)


    def __call__(self,
			  	 x,
				 boundary_callback=lambda x:x,
				 key=None):
        if key is None:
            key = jax.random.PRNGKey(int(time.time()))
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
            "FIRE_RATE":self.FIRE_RATE,
            "UPSAMPLE_AUX":self.UPSAMPLER_AUX
        }
    def partition(self):
        """
        Behaves like eqx.partition, but moves the hard coded kernels (a jax array) from the "trainable" pytree to the "static" pytree

        Returns
        -------
        diff : PyTree
            PyTree of same structure as NCA, with all non trainable parameters set to None
        static : PyTree
            PyTree of same structure as NCA, with all trainable parameters set to None

        """
        
        total_diff,total_static = eqx.partition(self,eqx.is_inexact_array)
        ops_diff,ops_static = self.op.partition()
        up_ops_diff,up_ops_static = self.upsample.op.partition()
        where_ops = lambda m:m.op
        where_up_ops = lambda m:m.upsample.op
        total_diff = eqx.tree_at(where_ops,total_diff,ops_diff)
        total_diff = eqx.tree_at(where_up_ops,total_diff,up_ops_diff)
        
        total_static = eqx.tree_at(where_ops,total_static,ops_static)
        total_static = eqx.tree_at(where_up_ops,total_static,up_ops_static)
        return total_diff, total_static

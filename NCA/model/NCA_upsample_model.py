import jax
import jax.numpy as jnp
import equinox as eqx
import time
from jaxtyping import Float, Array, Key
from einops import rearrange, repeat
#from Common.model.abstract_model import AbstractModel # Inherit model loading and saving
from NCA.model.NCA_model import NCA, Ops




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

class local_upsample(eqx.Module):
    decoder: PointwiseConvNet
    op: Ops
    fourier_modes: int
    output_channels: int = eqx.field(static=True)
    
    def __init__(
            self, 
            channels: int, 
            output_channels: int, 
            hidden_channels: int,
            fourier_modes: int = 4,
            depth: int = 3, 
            padding: str = "CIRCULAR",
            key=jax.random.PRNGKey(0)):
        key1, key2 = jax.random.split(key, 2)

        if output_channels > channels:
            raise ValueError("output_channels must be <= channels")

        insize = channels + 4*fourier_modes # 4 because sin and cos, for x and y

        self.op = Ops(PADDING=padding,dx=1,KERNEL_SCALE=1,SMOOTHING=1) # type: ignore
        self.decoder = PointwiseConvNet(
            in_channels=3*insize,   # z(q) + [r, r^2]
            out_channels=output_channels,     # a0, a1, b1, a2, b2
            hidden_channels=hidden_channels,
            depth=depth,
            key=key,
        )
        self.decoder = eqx.tree_at(
            lambda m: m.layers,
            self.decoder,
            self.decoder.layers[:-1] + (self._zero_conv(self.decoder.layers[-1]),),
        )

        # self.layers = [
        #     eqx.nn.Conv2d(in_channels=insize,
		# 				  out_channels=insize*hidden_scale,
		# 				  kernel_size=1,
		# 				  use_bias=True,
		# 				  key=key1),
		# 	jax.nn.relu,
		# 	eqx.nn.Conv2d(in_channels=insize*hidden_scale, 
		# 				  out_channels=output_channels,
		# 				  kernel_size=1,
		# 				  use_bias=True,
		# 				  key=key2)
		# 	]
        self.fourier_modes = fourier_modes
        self.output_channels = output_channels

        

    @staticmethod
    def _zero_conv(layer: eqx.nn.Conv2d) -> eqx.nn.Conv2d:
        if layer.bias is None:
            return eqx.tree_at(lambda l: l.weight, layer, jnp.zeros_like(layer.weight))
        return eqx.tree_at(
            lambda l: (l.weight, l.bias),
            layer,
            (jnp.zeros_like(layer.weight), jnp.zeros_like(layer.bias)),
        )

    def local_interpolate(
            self, 
            x: Float[Array, "x y C"],
            resolution: int
            ):
        
        """
            Bilinearly interpolate coarse array to high resolution.
            
            Args:
                x: Coarse 2D array of shape (C, H, W)
                resolution: Upsampling factor
                modes: Number of fourier modes to encode local coordinates
                
            Returns:
                upsampled: High resolution array of shape (C, H*resolution, W*resolution)
                local_coords: Local coordinate fourier modes within each coarse grid cell, 
                shape (4*modes, H*resolution, W*resolution)
        """
        _, H, W = x.shape
        H_up, W_up = H * resolution, W * resolution

        # Create grid of coordinates in upsampled space
        y_up = jnp.arange(H_up, dtype=jnp.float32) / resolution
        x_up = jnp.arange(W_up, dtype=jnp.float32) / resolution
        xx_up, yy_up = jnp.meshgrid(x_up, y_up, indexing='xy')

        # Get integer and fractional parts
        x_int = jnp.floor(xx_up).astype(jnp.int32)
        y_int = jnp.floor(yy_up).astype(jnp.int32)
        x_frac = xx_up - jnp.floor(xx_up)
        y_frac = yy_up - jnp.floor(yy_up)

        # Clamp indices to valid range
        x_int = jnp.clip(x_int, 0, W - 2)
        y_int = jnp.clip(y_int, 0, H - 2)

        # Get the four corner values for bilinear interpolation
        v00 = x[:, y_int, x_int]
        v01 = x[:, y_int, x_int + 1]
        v10 = x[:, y_int + 1, x_int]
        v11 = x[:, y_int + 1, x_int + 1]

        x_frac = repeat(x_frac, "h w -> c h w", c=1)
        y_frac = repeat(y_frac, "h w -> c h w", c=1)
        
        # Bilinear interpolation
        v0 = v00 * (1 - x_frac) + v01 * x_frac
        v1 = v10 * (1 - x_frac) + v11 * x_frac
        upsampled = v0 * (1 - y_frac) + v1 * y_frac

        # Local coordinates within each coarse grid cell (0-1)
        x_frac = 2*x_frac - 1  # Scale to [-1, 1]
        y_frac = 2*y_frac - 1  # Scale to [-1, 1]
        modes = jnp.arange(1, self.fourier_modes + 1, dtype=jnp.float32)[:, None, None]
        x_modes = jnp.concatenate(
            [jnp.sin(modes * jnp.pi * x_frac), jnp.cos(modes * jnp.pi * x_frac)],
            axis=0,
        )
        y_modes = jnp.concatenate(
            [jnp.sin(modes * jnp.pi * y_frac), jnp.cos(modes * jnp.pi * y_frac)],
            axis=0,
        )

        local_coords = jnp.concatenate([x_modes, y_modes], axis=0)        
        # local_coords = rearrange(local_coords, "m () h w -> m h w")
        return upsampled, local_coords

    
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
    def __call__(self, x, resolution=4):
        """
        Args:
            x : float32 [CHANNELS,_,_]
                input array of shape (C, H, W)
            resolution : int
                upsampling factor
        Returns:
            x : float32 [OBD_CHANNELS,_,_]
                upsampled array of shape (C_out, H*resolution, W*resolution)
        """
        x_upsampled, local_coords = self.local_interpolate(
            x, 
            resolution=resolution, 
        )
        x = jnp.concatenate([x_upsampled, local_coords], axis=0)
        x_spatial = self.spatial_gradients(x)
        # for layer in self.layers:
            # x = layer(x)
        x = self.decoder(x_spatial)
        return x + x_upsampled[: self.output_channels]



class uNCA(NCA):
    layers: list
    KERNEL_STR: list
    N_CHANNELS: int
    # O_CHANNELS: int
    N_FEATURES: int
    # SPATIAL_UPSAMPLE: int
    UPSAMPLER_AUX: dict
    FIRE_RATE: float
    op: Ops
    perception: callable  # type: ignore
    upsample: local_upsample
    #CONFIG: dict

    def __init__(self,
                N_CHANNELS,
                O_CHANNELS,
                KERNEL_STR=["ID","LAP"], 
                ACTIVATION=jax.nn.relu, 
                PADDING="CIRCULAR", 
                FIRE_RATE=1.0, 
                KERNEL_SCALE = 1, 
                # SPATIAL_UPSAMPLE = 4,
                # fourier_modes = 4,
                UPSAMPLER_AUX = {
                    "depth": 3,
                    "width_factor": 1,
                    "fourier_modes": 2,
                    "upsample_factor": 4
                },
                key=jax.random.PRNGKey(int(time.time()))):
        super().__init__(N_CHANNELS, KERNEL_STR, ACTIVATION, PADDING, FIRE_RATE, KERNEL_SCALE, key)
        #key1,key2 = jax.random.split(key,2)
        key = jax.random.fold_in(key,1234)
        # self.SPATIAL_UPSAMPLE = UPSAMPLER_AUX["upsample_factor"]
        self.UPSAMPLER_AUX = UPSAMPLER_AUX
        self.upsample = local_upsample(
            channels=N_CHANNELS, 
            output_channels=O_CHANNELS, 
            fourier_modes=UPSAMPLER_AUX["fourier_modes"],
            hidden_channels=UPSAMPLER_AUX["width_factor"]*N_CHANNELS,
            depth=UPSAMPLER_AUX["depth"],
            key=key)

    def real_to_latent(self, x):
        # print("=== Real to latent call ===",flush=True)
        # print("     input real shape: ", x.shape,flush=True)
        latent_shape = x.shape[:-2] + (
            max(1, x.shape[-2] // self.UPSAMPLER_AUX["upsample_factor"]),
            max(1, x.shape[-1] // self.UPSAMPLER_AUX["upsample_factor"]),
        )
        x_lat = jax.image.resize(x, latent_shape, method="linear") 
        # print("     output latent shape: ", x_lat.shape,flush=True)
        return x_lat

    def latent_to_real(self, x):
        # print("=== Latent to real call ===",flush=True)
        # print("     input latent shape: ", x.shape,flush=True)
        if x.ndim == 3:
            x_real = self.upsample(x, resolution=self.UPSAMPLER_AUX["upsample_factor"])
        else:
            x_real = jax.vmap(lambda x_i: self.upsample(x_i, resolution=self.UPSAMPLER_AUX["upsample_factor"]))(x)

        # x = rearrange(x, "b c h w -> b c h w")
        # print("     output real shape: ", x_real.shape,flush=True)
        return x_real


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
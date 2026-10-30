import jax
import jax.numpy as np
import jax.random as jr
import equinox as eqx
import numpy as onp
from einops import repeat, rearrange

from jaxtyping import Float,Array,Key
from Common.model.abstract_model import AbstractModel


def sparse_grid(size,stride):
    """Indices of a sparse impulse along one lattice axis: every stride-th pixel, centred."""
    return range(((size - 1) % stride) // 2, size, stride)


class perturbation(AbstractModel):
    """Learned perturbation added to NCA states of shape (batch, C, X, Y).

    The perturbation can be local (Gaussian), patch (small squares), sparse
    (every STRIDE-th pixel in each direction), global or flat, and can act on
    all channels, the observed or hidden channels, or a single hidden channel.
    The same perturbation is applied to every batch element. It can last
    several NCA steps: values[s] is the delta for step s, all at the same
    centre (see ``rollout.run_with_intervention``).
    """
    location: Array
    values: Array
    mode: dict
    OBS_CHANNELS: int
    CHANNELS: int
    WIDTH: float
    # Side length in pixels of a patch perturbation. Static, so it is not
    # written to saved .eqx files and older files still load.
    PATCH_SIZE: int = eqx.field(static=True)
    # Pixel spacing of a sparse perturbation (1: every pixel, as global)
    STRIDE: int = eqx.field(static=True)
    # Number of separate patches, each with its own centre and values. With
    # one patch, location is (2,) and values (duration, C, k, k); with P
    # patches, (P, 2) and (duration, P, C, k, k).
    PATCHES: int = eqx.field(static=True)

    def __init__(self,mode,CHANNELS,OBS_CHANNELS,x,WIDTH,key,PATCH_SIZE=0,duration=1,init_scale=0.0,initial_location="random",STRIDE=1,PATCHES=1):
        # The centre is sigmoid(location) in normalised (0 to 1) coordinates.
        # Store the logit of a uniform draw so the starting centre is uniform
        # over the interior of the lattice, not biased towards one corner.
        # initial_location="centre" starts at the lattice centre instead.
        if PATCHES > 1 and mode['spatial'] != "patch":
            raise ValueError("Several patches need spatial mode 'patch'")
        self.WIDTH = WIDTH
        self.PATCH_SIZE = PATCH_SIZE
        self.STRIDE = STRIDE
        self.PATCHES = PATCHES
        centres = (2,) if PATCHES == 1 else (PATCHES,2)
        if initial_location == "random":
            self.location = jax.scipy.special.logit(jr.uniform(key,centres,minval=0.1,maxval=0.9))
        elif initial_location == "centre":
            self.location = np.zeros(centres)
        else:
            raise ValueError(f"initial_location must be 'random' or 'centre', got {initial_location!r}")
        self.mode=mode
        self.OBS_CHANNELS = OBS_CHANNELS
        self.CHANNELS = CHANNELS
        CH = {
            "all":np.s_[:1,:self.CHANNELS],
            "obs":np.s_[:1,:self.OBS_CHANNELS],
            "hidden":np.s_[:1,self.OBS_CHANNELS:],
            "single":np.s_[:1,self.OBS_CHANNELS:self.OBS_CHANNELS+1]
        }[self.mode['channel']]
        H,W = x.shape[-2:]
        SP = {
            "global":np.s_[:,:],
            "local":np.s_[:,:],
            "patch":np.s_[:PATCH_SIZE,:PATCH_SIZE],
            "sparse":np.s_[sparse_grid(H,STRIDE)[0]::STRIDE,sparse_grid(W,STRIDE)[0]::STRIDE],
            "flat":np.s_[:1,:1],
        }[self.mode['spatial']]
        inds = CH + SP
        
        # (duration, [patch,] C_sel, ...): one delta per NCA step. Zero, or
        # Gaussian of std init_scale: a zero start gives the centre no
        # gradient until the values have grown.
        patches = () if PATCHES == 1 else (PATCHES,)
        shape = (duration,*patches,*x[inds].shape[1:])
        self.values = init_scale*jr.normal(jr.fold_in(key,1),shape,dtype=x.dtype)

    @property
    def duration(self):
        """Number of NCA steps the perturbation is applied for."""
        return self.values.shape[0]

    def __call__(self,x):
        """Add the first step's delta to x."""
        return x + self.deltas(x)[0]

    def deltas(self,x):
        """Full-lattice delta for every step: (duration, 1, CHANNELS, H, W)."""
        CH = {
            "all":np.s_[:,:,:self.CHANNELS],
            "obs":np.s_[:,:,:self.OBS_CHANNELS],
            "hidden":np.s_[:,:,self.OBS_CHANNELS:],
            "single":np.s_[:,:,self.OBS_CHANNELS:self.OBS_CHANNELS+1]
        }[self.mode['channel']]
        H,W = x.shape[-2:]
        if self.mode['spatial'] == "patch":
            # Place each patch, then add them up (overlaps add)
            centres = jax.nn.sigmoid(self.location).reshape((-1,2))
            patches = self.values.reshape((self.duration,len(centres),*self.values.shape[-3:]))
            place = lambda centre,patch: self._place_patch(centre,patch,H,W)
            values = np.sum(jax.vmap(place,in_axes=(0,1))(centres,patches),axis=0)
        elif self.mode['spatial'] == "sparse":
            rows,cols = sparse_grid(H,self.STRIDE),sparse_grid(W,self.STRIDE)
            values = np.zeros((*self.values.shape[:2],H,W),dtype=x.dtype)
            values = values.at[:,:,rows.start::self.STRIDE,cols.start::self.STRIDE].set(self.values)
        else:
            values = self._spatial_mask(jax.nn.sigmoid(self.location),np.array([self.WIDTH,self.WIDTH]),self.values)
        deltas = np.zeros((self.duration,1,*x.shape[-3:]),dtype=x.dtype)
        return deltas.at[CH].set(np.broadcast_to(values[:,None],deltas[CH].shape))
    @eqx.filter_jit
    def _spatial_mask(self,centers,scales,x):
        if self.mode['spatial'] in ["global","flat"]:
            return x
        else:
            xs = np.linspace(0,1, x.shape[-2])
            ys = np.linspace(0,1, x.shape[-1])
            grid_x, grid_y = np.meshgrid(xs, ys, indexing='ij')
            grid = np.stack([grid_x, grid_y], axis=-1)  # Shape: (H, W, 2)
            centers = rearrange(centers, 'Dim -> 1 1 Dim')
            scales = rearrange(scales, 'Dim -> 1 1 Dim')
            m = np.exp(-0.5 * (((grid - centers)**2) / scales**2))
            m = m[:,:,0] * m[:,:,1]
            m = rearrange(m, 'H W -> 1 1 H W')
            return m*x

    def _place_patch(self,centre,values,H,W):
        """Place the (duration, C, k, k) patch on an H x W lattice, centred at ``centre``.

        Bilinear interpolation gives the centre a gradient: the patch covers
        exactly k x k pixels at whole-pixel centres and at most (k+1) x (k+1)
        in between. Pixel i is at i/(H-1), as for the Gaussian mask.
        """
        offset = (self.PATCH_SIZE - 1) / 2
        rows = np.arange(H) - centre[0] * (H - 1) + offset
        cols = np.arange(W) - centre[1] * (W - 1) + offset
        coordinates = np.meshgrid(rows, cols, indexing='ij')
        place = lambda patch: jax.scipy.ndimage.map_coordinates(patch, coordinates, order=1, mode='constant', cval=0.0)
        return jax.vmap(jax.vmap(place))(values)
    
    def get_values(self):
        return self.values
    
    def get_location(self):
        """Centre of a local perturbation in normalised (0 to 1) coordinates: (2,), or (P, 2) for P patches."""
        return jax.nn.sigmoid(self.location)
    
    def regulariser(self,x,REG_FUNCS):
        reg_loss = 0.0
        for name in REG_FUNCS.keys():
            reg_loss += REG_FUNCS[name](x,self.values)
        return reg_loss
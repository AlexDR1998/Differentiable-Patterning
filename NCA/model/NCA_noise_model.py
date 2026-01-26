import jax
import jax.numpy as jnp
import equinox as eqx
import time
#from Common.model.abstract_model import AbstractModel # Inherit model loading and saving
from NCA.model.NCA_model import NCA, Ops

class nNCA(NCA):
    layers: list
    KERNEL_STR: list
    N_CHANNELS: int
    N_FEATURES: int
    FIRE_RATE: float
    PARAMETER_NOISE_LEVEL: float
    op: Ops
    perception: callable

    #CONFIG: dict

    def __init__(self,
                N_CHANNELS,
                KERNEL_STR=["ID","LAP"], 
                ACTIVATION=jax.nn.relu, 
                PADDING="CIRCULAR", 
                FIRE_RATE=1.0, 
                KERNEL_SCALE = 1, 
                PARAMETER_NOISE_LEVEL=0.01,
                key=jax.random.PRNGKey(int(time.time()))):
        super().__init__(N_CHANNELS, KERNEL_STR, ACTIVATION, PADDING, FIRE_RATE, KERNEL_SCALE, key)
        self.PARAMETER_NOISE_LEVEL = PARAMETER_NOISE_LEVEL

 
    def __call__(self,
            x: jnp.ndarray,
            boundary_callback=lambda x: x,
            key: jax.random.PRNGKey = jax.random.PRNGKey(int(time.time()))):
        """
        Forward pass that perturbs model parameters with small additive noise
        before applying layers. Noise is applied to all array leaves in the
        perception and layers pytrees.

        Parameters
        ----------
        x : float32 [N_CHANNELS,_,_]
            input NCA lattice state.
        boundary_callback : callable, optional
            post-step boundary callback.
        key : jax.random.PRNGKey, optional
            PRNG key for parameter noise.

        Returns
        -------
        x : float32 [N_CHANNELS,_,_]
            output NCA lattice state.
        """
        # helper: add noise to array leaves of a pytree
        def _perturb_pytree(pytree, key):
            leaves, treedef = jax.tree.flatten(pytree)
            if not leaves:
                return pytree
            keys = jax.random.split(key, len(leaves))
            def _maybe_add_noise(leaf, k):
                if isinstance(leaf, jnp.ndarray) and jnp.issubdtype(leaf.dtype, jnp.floating):
                    noise = self.PARAMETER_NOISE_LEVEL * jax.random.normal(k, shape=leaf.shape)
                    return leaf + noise
                return leaf
            new_leaves = [ _maybe_add_noise(l, k) for l, k in zip(leaves, keys) ]
            return jax.tree.unflatten(treedef, new_leaves)

        # split keys for perception + each layer
        n_items = 1 + max(0, len(self.layers))
        keys = jax.random.split(key, n_items)

        # perturb perception and layers
        perception_perturbed = _perturb_pytree(self.perception, keys[0])
        layers_perturbed = []
        for i, layer in enumerate(self.layers):
            layers_perturbed.append(_perturb_pytree(layer, keys[i + 1]))

        # forward with perturbed parameters
        dx = perception_perturbed(x)
        for layer in layers_perturbed:
            dx = layer(dx)

        sigma = jax.random.bernoulli(key, p=self.FIRE_RATE, shape=dx.shape)
        x_new = x + sigma * dx
        return boundary_callback(x_new)
        #self.CONFIG["MODEL"] = "nNCA"

    def get_config(self):
        """
        Returns the model configuration as a dictionary.

        Returns
        -------
        dict
            dictionary of model hyperparameters

        """
        return {
            "MODEL":"nNCA",
            "N_CHANNELS":self.N_CHANNELS,
            "KERNEL_STR":self.KERNEL_STR,
            "PARAMETER_NOISE_LEVEL":self.PARAMETER_NOISE_LEVEL,
            "ACTIVATION":self.layers[1].__name__,
            "PADDING":self.op.PADDING,
            "FIRE_RATE":self.FIRE_RATE,
        }
import jax
import jax.numpy as jnp
import equinox as eqx
import time
#from Common.model.abstract_model import AbstractModel # Inherit model loading and saving
from NCA.model.NCA_model import NCA, Ops

class bNCA(NCA):
    layers: list
    KERNEL_STR: list
    N_CHANNELS: int
    N_FEATURES: int
    FIRE_RATE: float
    op: Ops
    perception: callable
    def __init__(self, N_CHANNELS, OBSERVABLE_CHANNELS,KERNEL_STR=["ID","LAP"], ACTIVATION=jax.nn.relu, PADDING="CIRCULAR", FIRE_RATE=1.0, KERNEL_SCALE=1, key=jax.random.PRNGKey(int(time.time()))):
        super().__init__(N_CHANNELS, KERNEL_STR, ACTIVATION, PADDING, FIRE_RATE, KERNEL_SCALE,key)
        self.OBSERVABLE_CHANNELS = OBSERVABLE_CHANNELS


    def __call__(self,x,boundary_callback=lambda x:x,key=jax.random.PRNGKey(int(time.time()))):
        """
        

        Parameters
        ----------
        x : float32 [N_CHANNELS,_,_]
            input NCA lattice state.
        boundary_callback : callable (float32 [N_CHANNELS,_,_]) -> (float32 [N_CHANNELS,_,_]), optional
            function to augment intermediate NCA states i.e. imposing complex boundary conditions or external structure. Defaults to None
        key : jax.random.PRNGKey, optional
            Jax random number key. The default is jax.random.PRNGKey(int(time.time())).

        Returns
        -------
        x : float32 [N_CHANNELS,_,_]
            output NCA lattice state.

        """
        
        dx = self.perception(x)
        for layer in self.layers:
            dx = layer(dx)
        sigma = jax.random.bernoulli(key,p=self.FIRE_RATE,shape=dx.shape)
        x_new = x + sigma*dx
        x_new = x_new.at[:self.OBSERVABLE_CHANNELS].set(jnp.clip(x_new[:self.OBSERVABLE_CHANNELS],a_min=0.0,a_max=1.0)) # Clip observable channels to [0,1]
        x_new = x_new.at[self.OBSERVABLE_CHANNELS:].set(jnp.clip(x_new[self.OBSERVABLE_CHANNELS:],a_min=-1.0,a_max=1.0)) # Clip hidden channels to [-1,1]

        return boundary_callback(x_new)
import jax
import jax.numpy as jnp
import equinox as eqx
from Common.model.abstract_model import AbstractModel # Inherit model 
from Common.dataloader.adhesion_mask import adhesion_mask_convex_hull_circle
from Common.model.boundary import model_boundary
from .NCA_channel_extractor import NCA_channel_extractor
from NCA.analysis.tensorboard_log import CM_Train_log
from jaxtyping import Float, Array
from einops import rearrange,repeat
from tqdm import tqdm
import time
import jax.random as jr

class NCA_channel_map_linear(AbstractModel):
    layers: list
    hyperparameters: dict
    def __init__(self,key,MEASURED_CHANNELS,TARGET_CHANNELS,LATENT=None):
        
        self.hyperparameters = {
            "MEASURED_CHANNELS":MEASURED_CHANNELS,
            "TARGET_CHANNELS":TARGET_CHANNELS,
        }
        self.layers = [
            eqx.nn.Linear(in_features=len(MEASURED_CHANNELS),
                          out_features=len(TARGET_CHANNELS),
                          key=key),
        ]

    
    def __call__(self,X:Float[Array,"B T c X Y"]):
        """
        Reshapes input from NCA trajectory to (K, C), performs transformation vmapped over K, 
        and reshapes back to (B, T, C, X, Y)
        Args:
            X (float32 [BATCH,TIME,C,X,Y]): batch of initial conditions
        
        """
        def _call(X):
            for layer in self.layers:
                X = layer(X)
            return X
        X_flat = rearrange(X,"B T C_INPUT X Y -> (B T X Y) C_INPUT")
        vfunc = eqx.filter_vmap(_call,in_axes=(0,),out_axes=(0))
        Y_flat = vfunc(X_flat)
        Y = rearrange(Y_flat,"(B T X Y) C_TARGET -> B T C_TARGET X Y",B=X.shape[0],T=X.shape[1],X=X.shape[3],Y=X.shape[4])
        return Y



class NCA_channel_map_fully_connected_local(AbstractModel):
    layers: list
    hyperparameters: dict
    def __init__(self,key,MEASURED_CHANNELS,TARGET_CHANNELS,LATENT=None):
        if LATENT is None:
            LATENT = len(MEASURED_CHANNELS)
        self.hyperparameters = {
            "MEASURED_CHANNELS":MEASURED_CHANNELS,
            "TARGET_CHANNELS":TARGET_CHANNELS,
            "LATENT":LATENT,
        }
        keys = jr.split(key,2)
        self.layers = [
            eqx.nn.Linear(in_features=len(MEASURED_CHANNELS),
                          out_features=LATENT,
                          key=keys[0]),
            jax.nn.relu,
            eqx.nn.Linear(in_features=LATENT,
                          out_features=len(TARGET_CHANNELS),
                          key=keys[1]),
            jax.nn.relu,
        ]

    
    def __call__(self,X:Float[Array,"B T c X Y"]):
        """
        Reshapes input from NCA trajectory to (K, C), performs transformation vmapped over K, 
        and reshapes back to (B, T, C, X, Y)
        Args:
            X (float32 [BATCH,TIME,C,X,Y]): batch of initial conditions
        
        """
        def _call(X):
            for layer in self.layers:
                X = layer(X)
            return X
        X_flat = rearrange(X,"B T C_INPUT X Y -> (B T X Y) C_INPUT")
        vfunc = eqx.filter_vmap(_call,in_axes=(0,),out_axes=(0))
        Y_flat = vfunc(X_flat)
        Y = rearrange(Y_flat,"(B T X Y) C_TARGET -> B T C_TARGET X Y",B=X.shape[0],T=X.shape[1],X=X.shape[3],Y=X.shape[4])
        return Y
    
class NCA_channel_map_conv(AbstractModel):
    layers: list
    hyperparameters: dict
    def __init__(
        self,
        key,
        MEASURED_CHANNELS,
        TARGET_CHANNELS,
        LATENT=None,
        kernel_size=5,
        PADDING="CIRCULAR"
    ):
        if LATENT is None:
            LATENT = len(MEASURED_CHANNELS)
        self.hyperparameters = {
            "MEASURED_CHANNELS":MEASURED_CHANNELS,
            "TARGET_CHANNELS":TARGET_CHANNELS,
            "LATENT":LATENT,
        }
        
        keys = jr.split(key,2)
        self.layers = [
            eqx.nn.Conv2d(in_channels=len(MEASURED_CHANNELS),
                          out_channels=LATENT,
                          kernel_size=kernel_size,
                          padding_mode=PADDING,
                          padding="SAME",
                          key=keys[0]),
            jax.nn.relu,
            eqx.nn.Conv2d(in_channels=LATENT,
                          out_channels=len(TARGET_CHANNELS),
                          kernel_size=kernel_size,
                          padding_mode=PADDING,
                          padding="SAME",
                          key=keys[1]),
            jax.nn.relu,    # We know output is non-negative
        ]
    
    def __call__(self,X:Float[Array,"B T c X Y"]):
        """
        Reshapes input from NCA trajectory to (K, C), performs transformation vmapped over K, 
        and reshapes back to (B, T, C, X, Y)
        Args:
            X (float32 [BATCH,TIME,C,X,Y]): batch of initial conditions
        
        """
        def _call(X):
            for layer in self.layers:
                X = layer(X)
            return X
        X_flat = rearrange(X,"B T C_INPUT X Y -> (B T) C_INPUT X Y")
        vfunc = eqx.filter_vmap(_call,in_axes=(0,),out_axes=(0))
        Y_flat = vfunc(X_flat)
        Y = rearrange(Y_flat,"(B T) C_TARGET X Y -> B T C_TARGET X Y",B=X.shape[0],T=X.shape[1],X=X.shape[3],Y=X.shape[4])
        return Y
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jt
import optax
import equinox as eqx
import datetime
import Common.trainer.loss as loss
import jaxpruner
from functools import partial
from NCA.trainer.tensorboard_log import NCA_Train_log, kaNCA_Train_log, mNCA_Train_log
from NCA.model.NCA_KAN_model import kaNCA
from NCA.model.NCA_multi_scale import mNCA
from NCA.trainer.data_augmenter_nca import DataAugmenter
from einops import repeat
from Common.utils import key_pytree_gen,key_array_gen
from Common.model.boundary import model_boundary
from tqdm import tqdm
from einops import rearrange
from jaxtyping import Float,Array,Key
import time
from Common.trainer.custom_functions import multi_channel_perlin_noise
class NCA_Feature_Extractor(object):
    """
    General class for extracting features from NCA models over trajectories
    """

    def __init__(self,
                NCA_models,
                BOUNDARY_MASKS,
                GATED=False):
        self.NCA_models = NCA_models
        self.BOUNDARY_MASKS = BOUNDARY_MASKS
        self.GATED = GATED

    def initial_condition(self,key):
        raise NotImplementedError("Initial condition not implemented")

    def extract_features(self,
                        t0,
                        t1,
                        BATCH_SIZE,
                        SIZE,
                        key
    ):
        

        """
        Extracts features from the NCA models
        X : [Models,Batch,Channels,Height,Width]
        As : [Layers] [Models,Timestep,Batch,Features,Height,Width]
        """
        def extract_features_single_model_multi_batch(nca,X,key):
            #print(nca)
            vcall = jax.vmap(nca.call_with_activations,in_axes=(0,None,0),out_axes=(0,0))
            activations = []
            for i in range(t1):
                key = jr.fold_in(key,i)
                key_pytree = key_array_gen(key,(BATCH_SIZE,))
                X,act = vcall(X,lambda x:x,key_pytree)
                if i >= t0:
                    activations.append(act)

            return X,activations
        
        
        
        X = self.initial_condition(SIZE,BATCH_SIZE,key)

        Xs = []
        As = []
        for nca in self.NCA_models:
            X,act = extract_features_single_model_multi_batch(nca,X,key)
            Xs.append(X)
            As.append(act)
        Xs = jnp.array(Xs)

        # Reshape activations from [Models,Timestep,Layer] [Batch,Features,Height,Width] to [Layers] [Models,Timestep,Batch,Features,Height,Width]
        A_new = [[list(i) for i in zip(*A)] for A in As]
        A_new = [list(i) for i in zip(*A_new)]
        A_new = [jnp.array(A) for A in A_new]
        if self.GATED:
            A_dict = {"perception":A_new[0],
                      "linear_hidden":A_new[1],
                      "activation":A_new[2],
                      "linear_output":A_new[3],
                      "gate_func":A_new[4]}
        else:
            A_dict = {"perception":A_new[0],
                      "linear_hidden":A_new[1],
                      "activation":A_new[2],
                      "linear_output":A_new[3],}
        return Xs,A_dict





    def flatten_activations(self,As):
        """
        Flattens the activations from [Layers] [Models,Timestep,Batch,Features,Height,Width] to [Layers] [Models*Timestep*Batch*Height*Width,Features]
        Args:
            As : [Layers] [Models,Timestep,Batch,Features,Height,Width]
        Returns : [Layers] [Models*Timestep*Batch*Height*Width,Features]
        """
        #As = [rearrange(A,"Models Timestep Batch Features H W -> (Models Timestep Batch H W) Features") for A in As]
        As = {k:rearrange(A,"Models Timestep Batch Features H W -> (Models Timestep Batch H W) Features") for k,A in As.items()}
        return As





class NCA_Feature_Extractor_Texture(NCA_Feature_Extractor):
    def initial_condition(self,SIZE,BATCH_SIZE,key):
        ic = []
        for i in range(BATCH_SIZE):
            ic.append(multi_channel_perlin_noise(SIZE,self.NCA_models[0].N_CHANNELS,4,jr.fold_in(key,i)))
        return jnp.array(ic)

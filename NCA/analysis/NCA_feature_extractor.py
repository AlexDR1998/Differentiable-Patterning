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
from Common.utils import key_pytree_gen,key_array_gen,load_emoji_sequence
from Common.model.boundary import model_boundary
from tqdm import tqdm
from einops import rearrange
from jaxtyping import Float,Array,Key
import random
import time
from Common.trainer.custom_functions import multi_channel_perlin_noise
import os
PVC_PATH = os.environ.get("PVC_PATH", "/mnt/ceph/ar-dp/")
class NCA_Feature_Extractor(object):
    """
    General class for extracting features from NCA models over trajectories
    """

    def __init__(self,
                NCA_models,
                BOUNDARY_MASKS,
                GATED=False,
                MODEL_BATCH_SIZE=1):
        self.NCA_models = NCA_models
        self.BOUNDARY_MASKS = BOUNDARY_MASKS
        self.GATED = GATED
        self.MODEL_BATCH_SIZE = MODEL_BATCH_SIZE

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
        @eqx.filter_jit
        def extract_features_single_model_multi_batch(nca,X,key):
            #print(nca)
            vcall = jax.vmap(nca.call_with_activations,in_axes=(0,None,0),out_axes=(0,0))
            
            
            def loop_body(carry,j):
                key,X = carry
                key = jr.fold_in(key,j)
                key_pytree = key_array_gen(key,(BATCH_SIZE,))
                X,act = vcall(X,lambda x:x,key_pytree)
                carry = (key,X)
                return carry, act
            
            carry,activations = jax.lax.scan(loop_body,(key,X),jnp.arange(t1))
            _,X = carry
            

            return X,activations
        
        
        
        X = self.initial_condition(SIZE,BATCH_SIZE,key)

        Xs = []
        As = []
        #inds = jr.choice(key,len(self.NCA_models),(self.MODEL_BATCH_SIZE,),replace=False)
        nca_sublist = random.choices(self.NCA_models,k=self.MODEL_BATCH_SIZE)
        for nca in nca_sublist:
            X,act = extract_features_single_model_multi_batch(nca,X,key)
            #print(act.shape)
   
            Xs.append(X)
            As.append(act)
        Xs = jnp.array(Xs)


        


        ### Reshape activations from [Models,Timestep,Layer] [Batch,Features,Height,Width] to [Layers] [Models,Timestep,Batch,Features,Height,Width]
        
        # Reshape activations from [Models,Layer,Timestep] [Batch,Features,Height,Width] to [Layers] [Models,Timestep,Batch,Features,Height,Width]

        #A_new = [[list(i) for i in zip(*A)] for A in As]
        #A_new = [list(i) for i in zip(*A_new)]
        
        A_new = [list(i) for i in zip(*As)]
        
        
        
        A_new = [jnp.array(A) for A in A_new]

        #print(len(A_new))
        #print(A_new[0].shape)
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
        #print(A_dict)
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

class NCA_Feature_Extractor_Emoji(NCA_Feature_Extractor):
    def initial_condition(self,SIZE,BATCH_SIZE,key):
        data = load_emoji_sequence([
        "crab.png",
        "microbe.png",
        "avocado.png",
        "alien_monster.png",
        "butterfly.png",
        "lizard.png",
        "mushroom.png",
        ],
        impath_emojis=PVC_PATH+"Data/Emojis/",
        downsample=1)


        # For the initial condition, take a small cropped square from the middle of the target image

        data = rearrange(data,"B T C W H -> T B C W H")
        data = repeat(data,"B T C W H -> (B b) T C W H",b=1)
        initial_condition = jnp.array(data)


        W = initial_condition.shape[-2]
        H = initial_condition.shape[-1]

        initial_condition = initial_condition.at[:,:,:,:W//2-6].set(0)
        initial_condition = initial_condition.at[:,:,:,W//2+5:].set(0)
        initial_condition = initial_condition.at[:,:,:,:,:H//2-6].set(0)
        initial_condition = initial_condition.at[:,:,:,:,H//2+5:].set(0)


        data = jnp.concatenate([initial_condition,data,data],axis=1) # Join initial condition and data along the time axis

        #print("(Batch, Time, Channels, Width, Height): "+str(data.shape))
        

        class data_augmenter_subclass(DataAugmenter):
            #Redefine how data is pre-processed before training
            def data_init(self,SHARDING=None):
                data = self.return_saved_data()
                data = self.duplicate_batches(data, 1)
                data = self.pad(data, 10) 		
                self.save_data(data)
                return None

        DA = data_augmenter_subclass(data,hidden_channels=self.NCA_models[0].N_CHANNELS-4)
        X0,_ = DA.split_x_y()
        X0 = jnp.array(X0)[:BATCH_SIZE,0]
        #print(X0.shape)
        return jnp.array(X0)
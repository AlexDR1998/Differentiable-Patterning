import jax
import jax.numpy as np
import numpy as onp
import jax.random as jr
import equinox as eqx
import time
from einops import repeat, reduce
from jaxtyping import Float, Array, Key, Int, Scalar

from Common.model.abstract_model import AbstractModel
from NCA.model.NCA_subchannels import sub_NCA, Ops



class mNCA(AbstractModel):
    subNCAs: list
    N_CHANNELS: int
    GATED: bool
    SCALES: list
    KERNEL_STR: list
    FIRE_RATE: float
    #op: Ops
    perception: callable

    def __init__(self,
                N_CHANNELS,
                SCALES,
                GATED = False,
                KERNEL_STR=["ID","LAP"], 
                ACTIVATION=jax.nn.relu, 
                PADDING="CIRCULAR", 
                FIRE_RATE=1.0, 
                KERNEL_SCALE = 1, 
                key=jr.PRNGKey(int(time.time()))):
        #assert len(SCALES)==N_CHANNELS
        assert N_CHANNELS%len(SCALES)==0

        
        OUTPUT_CHANNELS = onp.zeros((len(SCALES),N_CHANNELS))
        OUTPUT_CHANNEL_NUMBER = int(N_CHANNELS/len(SCALES))
        for i in range(len(SCALES)):
            #OUTPUT_CHANNELS = OUTPUT_CHANNELS.at[i,i*OUTPUT_CHANNEL_NUMBER:(i+1)*OUTPUT_CHANNEL_NUMBER].set(1)
            OUTPUT_CHANNELS[i,i*OUTPUT_CHANNEL_NUMBER:(i+1)*OUTPUT_CHANNEL_NUMBER] = 1
        OUTPUT_CHANNELS = OUTPUT_CHANNELS.astype(bool)
        #print(OUTPUT_CHANNELS)
        self.N_CHANNELS = N_CHANNELS
        self.SCALES = SCALES
        self.GATED = GATED
        self.KERNEL_STR = KERNEL_STR
        self.FIRE_RATE = FIRE_RATE
        self.subNCAs = [
            sub_NCA(N_CHANNELS=N_CHANNELS,
                    OUTPUT_CHANNELS=OUTPUT_CHANNELS[i],
                    GATED = GATED,
                    SCALE = SCALES[i],
                    KERNEL_STR=KERNEL_STR, 
                    ACTIVATION=ACTIVATION, 
                    PADDING=PADDING, 
                    FIRE_RATE=FIRE_RATE, 
                    KERNEL_SCALE=KERNEL_SCALE, 
                    key=jr.fold_in(key,i)

            )
            for i in range(len(SCALES))]
        
        self.perception = self.subNCAs[0].perception

    def __call__(self,
                 X: Float[Array,"{self.N_CHANNELS} x y"],
				 boundary_callback=lambda x:x,
				 key: Key=jr.PRNGKey(int(time.time())))->Float[Array, "{self.N_CHANNEL} x y"]:
        keys = jr.split(key,len(self.SCALES))
        dXS = [self.subNCAs[i](X,boundary_callback,keys[i]) for i in range(len(self.SCALES))]
        dX = np.concatenate(dXS,axis=0)
        return boundary_callback(X + dX)
    
    def partition(self):
        diff_main,static_main = eqx.partition(self,eqx.is_array)
        for i in range(len(self.SCALES)):
            diff_subnca,static_subnca = self.subNCAs[i].partition()
            where = lambda s:s.subNCAs[i]
            
            diff_main = eqx.tree_at(where,diff_main,diff_subnca)
            static_main = eqx.tree_at(where,static_main,static_subnca)
        
        where_scales = lambda s:s.SCALES
        diff_main = eqx.tree_at(where_scales,diff_main,None)
        static_main = eqx.tree_at(where_scales,static_main,self.SCALES)
        return diff_main,static_main
    
    def set_weights(self,weights):
        for i in range(len(self.SCALES)):
            self.subNCAs[i].set_weights(weights[i])
    
    def get_weights(self):
        return [self.subNCAs[i].get_weights() for i in range(len(self.SCALES))]
        #return super().get_weights()
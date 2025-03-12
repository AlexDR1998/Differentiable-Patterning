import jax
import time
import jax.numpy as np
import jax.tree_util as jt
import jax.random as jr
from jaxtyping import Float, Int, PyTree, Scalar, Array
from Common.trainer.abstract_data_augmenter_array import DataAugmenterAbstract
from einops import rearrange


class DataAugmenter(DataAugmenterAbstract):
    def __init__(self,
                 data_full,
                 hidden_channels=0,
                 BATCHES=4,
                 key=jr.PRNGKey(int(time.time()))):
        self.data_full = data_full
        self.BATCHES = BATCHES
        self.TIMESTEPS = len(self.data_full)
        self.hidden_channels = hidden_channels
        self.CALLBACK_PARAMS={"TRUE_INTERMEDIATES":True,
                                "RESAMPLE_FREQ":100,
                                "NOISE_FRACTION":0.005,
                                "ZERO_CIRCLE":True}
        self.OBS_CHANNELS = data_full[0][0].shape[0]
        #self.resample_data(key)
        self.key = key


    def data_init(self,SHARDING = None):
        self.resample_data(self.key)
        print("Sampled data size: ",self.data_true.shape)

    def resample_data(self,key):
        """Selects a new set of batches of trajectories from the full data set
        
        
        """
        data_subset = []
        for i in range(len(self.data_full)):
            data_time_subset = []
            key = jr.fold_in(key,i)
            for j in list(jr.randint(key,(self.BATCHES,),0,len(self.data_full[i]))):
                data_time_subset.append(self.data_full[i][j])
            data_subset.append(data_time_subset)
        #d_arr = np.array(data)
        d_arr = np.array(data_subset)
        d_arr = rearrange(d_arr,"T B C X Y -> B T C X Y")
        d_arr = np.pad(d_arr,((0,0),(0,0),(0,self.hidden_channels),(0,0),(0,0)))
        self.data_true = d_arr
        self.data_saved = d_arr
        
    def data_callback(self,x,y,i):
        
        self.key = jr.fold_in(self.key,i)   		
        
        x_true,_ =self.split_x_y(1)
        
        x = x.at[:,1:].set(x[:,:-1])
        x = x.at[:,0].set(x_true[:,0])
        
        x = self.noise(x,self.CALLBACK_PARAMS["NOISE_FRACTION"],key=self.key)

        if self.CALLBACK_PARAMS["TRUE_INTERMEDIATES"]:
            
            x = x.at[::2,:,:self.OBS_CHANNELS].set(x_true[::2,:,:self.OBS_CHANNELS])



        if self.CALLBACK_PARAMS["ZERO_CIRCLE"]:
            x = self.zero_random_circle(x,key=self.key)
        
        
        if i%self.CALLBACK_PARAMS["RESAMPLE_FREQ"] == 0:
            self.resample_data(self.key)
        return x,y
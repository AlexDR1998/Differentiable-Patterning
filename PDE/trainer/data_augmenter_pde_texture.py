import jax
import time
import jax.numpy as np
import jax.random as jr
from jaxtyping import Float, Int, PyTree, Scalar, Array
from Common.trainer.abstract_data_augmenter_tree import DataAugmenterAbstract
from Common.trainer.custom_functions import multi_channel_perlin_noise

""" 
Use noise as initial condition, learn how to generate textures via LPIPS distance to images.
No fancy intermediate time step stuff, just learn textures as fixed points of the PDE.

"""

class DataAugmenter(DataAugmenterAbstract):
    """
        Inherits the methods of DataAugmenter, but overwrites the batch cloning in the init
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.OVERWRITE_OBS_CHANNELS = False
        
    def data_init(self,SHARDING=None,key=jax.random.PRNGKey(int(time.time()))):
        data = self.return_saved_data()
        B = len(data)
        N = data[0].shape[0]
        C = data[0].shape[1]
        W = data[0].shape[2]
        H = data[0].shape[3]
        self.POOL_SIZE = 16*B 
        keys = jr.split(key,self.POOL_SIZE)
        self.INITIAL_CONDITION_POOL = [multi_channel_perlin_noise(W,C,4,key) for key in keys]
        scale_to_m1p1 = lambda x: 2*x-1
        for i in range(self.POOL_SIZE//4):
            self.INITIAL_CONDITION_POOL[i] = self.INITIAL_CONDITION_POOL[i].at[:3].set(0.8*scale_to_m1p1(data[i%B][0,:3]) +0.2*self.INITIAL_CONDITION_POOL[i][:3])
        #return None

    
    # def sub_trajectory_split(self,L,key=jax.random.PRNGKey(int(time.time()))):
    #     """
    #     Splits data into x (initial conditions) and y (following sub trajectory of length L).
    #     So that x[:,i]->y[:,i+1:i+1+L] is learned

    #     Parameters
    #     ----------
    #     L : Int
    #         Length of subdirectory

    #     Returns
    #     -------
    #     x : float32[BATCHES,CHANNELS,WIDTH,HEIGHT]
    #         Initial conditions
    #     y : float32[BATCHES,L,CHANNELS,WIDTH,HEIGHT]
    #         Following sub trajectories

    #     """
        
        
    #     #pos = list(jax.random.randint(key,shape=(len(self.data_saved),),minval=0,maxval=self.data_true[0].shape[0]-L-1))
    #     #x = jax.tree_util.tree_map(lambda data,p:data[p],self.data_saved,pos)
    #     #y = jax.tree_util.tree_map(lambda data,p:data[p+1:p+1+L],self.data_saved,pos)

        
        
    def advance_pool(self,
                    X: PyTree[Float[Array, "C W H"]], 
                    Y: PyTree[Float[Array, "N C W H"]], 
                    i: Int[Scalar, ""],
                    key):
        
        """ Preserves intermediate hidden channels after each training iteration

        Returns:
            x : PyTree [BATCHES] f32[1,CHANNELS,WIDTH,HEIGHT]
                Initial conditions
            y : PyTree [BATCHES] f32[L,CHANNELS,WIDTH,HEIGHT]
                True trajectories that follow the initial conditions
        """
        data = self.return_saved_data()
        B = len(data)
        N = data[0].shape[0]
        C = data[0].shape[1]
        W = data[0].shape[2]
        H = data[0].shape[3]
        key = jr.fold_in(key,i)
        keys = jr.split(key,B)
        pool_inds = jr.randint(key,shape=(3*B,),minval=0,maxval=self.POOL_SIZE)
        pool_inds_reset = pool_inds[:B]
        pool_inds_save = pool_inds[B:2*B]
        pool_inds_load = pool_inds[2*B:]
        scale_to_m1p1 = lambda x: 2*x-1
        for i in range(B):
            #self.INITIAL_CONDITION_POOL[pool_inds_reset[i]] = multi_channel_perlin_noise(W,C,4,keys[i])#jr.normal(keys[i],shape=(C,W,H))
            self.INITIAL_CONDITION_POOL[pool_inds_reset[i]] = self.INITIAL_CONDITION_POOL[pool_inds_reset[i]].at[:3].set(0.8*scale_to_m1p1(data[i%B][0,:3])+0.2*multi_channel_perlin_noise(W,3,4,keys[i]))
            self.INITIAL_CONDITION_POOL[pool_inds_save[i]] = Y[i][-1]
            if i<B//2:
                X[i] = self.INITIAL_CONDITION_POOL[pool_inds_load[i]]
            else:
                X[i] = multi_channel_perlin_noise(W,C,4,keys[i])

        #self.INITIAL_CONDITION_POOL = [0.5 + 0.1*jr.normal(key,shape=(C,W,H)) for key in keys]
        #self.INITIAL_CONDITION_POOL.append([y[-1] for y in Y])
        #x = [0.5 + 0.1*jr.normal(key,shape=(C,W,H)) for key in keys]
        Y = data

        return X,Y

        
    # def initialize_pool(self,L,key):
    #     data = self.return_saved_data()
        
    #     return x0,y0

        
    
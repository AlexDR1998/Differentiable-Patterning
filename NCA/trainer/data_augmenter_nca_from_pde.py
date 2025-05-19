import jax.numpy as jnp
import jax.random as jr
import jax
import time
import equinox as eqx
from jax.experimental import mesh_utils
from Common.utils import key_pytree_gen
from Common.trainer.abstract_data_augmenter_tree import DataAugmenterAbstract
import itertools


class DataAugmenter(DataAugmenterAbstract):

    def __init__(self,data_true,hidden_channels=0,MINIBATCHES=64):
        """
        Class for data augmentation, when training an NCA on PDE trajectories
        As PDE trajectories have much higher temporal resolution, we need an extra mini-batching step
        ----------
        data_true : PyTree [BATCHES] f32[N,CHANNELS,WIDTH,HEIGHT]
            true un-augmented data
        hidden_channels : int optional
            number of hidden channels to zero-pad to data. Defaults to zero
        """
        self.OBS_CHANNELS = data_true[0].shape[1]
        self.MINIBATCHES = MINIBATCHES
        print("Resampling subtrajectories every "+str(MINIBATCHES)+" steps")
        print("Observable channels: "+str(self.OBS_CHANNELS))
        data_tree = []
        try:
            for i in range(data_true.shape[0]): # if data is provided as big array, convert to list of arrays. If data is list of arrays, this will leave it unchanged
                data_tree.append(data_true[i])
        except:
            data_tree = data_true
        data_true = jax.tree_util.tree_map(lambda x: jnp.pad(x,((0,0),(0,hidden_channels),(0,0),(0,0))),data_tree) # Pad zeros onto hidden channels


        self.data_true = data_true
        self.data_saved = data_true
        self.key = jr.PRNGKey(int(time.time()*1000))
        
        
        
    def data_init(self,SHARDING = None):
        """
        Chain together various data augmentations to perform at intialisation of NCA training

        """
        
        
        return None

        
    #@eqx.filter_jit
    def data_callback(self,x,y,i):
        """
        Called after every training iteration to perform data augmentation and processing		


        Parameters
        ----------
        x : PyTree [BATCHES] f32[N-N_steps,CHANNELS,WIDTH,HEIGHT]
            Initial conditions
        y : PyTree [BATCHES] f32[N-N_steps,CHANNELS,WIDTH,HEIGHT]
            Final states
        i : int
            Current training iteration - useful for scheduling mid-training data augmentation

        Returns
        -------
        x : PyTree [BATCHES] f32[N-N_steps,CHANNELS,WIDTH,HEIGHT]
            Initial conditions
        y : PyTree [BATCHES] f32[N-N_steps,CHANNELS,WIDTH,HEIGHT]
            Final states

        """
        am=10
        if i%self.MINIBATCHES==0:
            self.key = jr.fold_in(self.key,i)
            x,y = self.sub_trajectory_split(self.MINIBATCHES,self.key)
            x = self.noise(x,0.1,mode="hidden",key=self.key)
        else:
            propagate_xn = lambda x:x.at[1:].set(x[:-1])
            reset_x0 = lambda x,x_true:x.at[0,:self.OBS_CHANNELS].set(x_true[0,:self.OBS_CHANNELS])
            x_true,_ = self.sub_trajectory_split(self.MINIBATCHES,self.key)
            x = jax.tree_util.tree_map(propagate_xn,x) # Set initial condition at each X[n] at next iteration to be final state from X[n-1] of this iteration
            x = jax.tree_util.tree_map(reset_x0,x,x_true) # Keep first initial x correct
        
        
        
                
        
        return x,y
		

    def sub_trajectory_split(self,L,key=jax.random.PRNGKey(int(time.time()))):
        """
        Splits data into shorter sub-trajectories x (initial conditions) and y (targets).
        So that x[:,i:i+L]->y[:,i:i+L] is learned

        Parameters
        ----------
        L : Int
            Length of sub - trajectory

        Returns
        -------
        x : float32[BATCHES,CHANNELS,WIDTH,HEIGHT]
            Initial conditions
        y : float32[BATCHES,L,CHANNELS,WIDTH,HEIGHT]
            Following sub trajectories

        """
        x,y = self.split_x_y(1)
        
        pos = list(jax.random.randint(key,shape=(len(x),),minval=0,maxval=x[0].shape[0]-L))
        x = jax.tree_util.tree_map(lambda data,p:data[p:p+L],x,pos)
        y = jax.tree_util.tree_map(lambda data,p:data[p:p+L],y,pos)
        
        return x,y
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
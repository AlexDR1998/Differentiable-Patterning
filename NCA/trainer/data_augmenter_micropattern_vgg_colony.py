import jax
import jax.numpy as np
import jax.random as jr
import equinox as eqx
import jax.tree_util as jtu

from NCA.trainer.data_augmenter_nca_basic import DataAugmenter as DataAugmenterBasic
from NCA.trainer.data_augmenter_nca_basic import jittable_callback_bit




class DataAugmenter(DataAugmenterBasic):

    def split_x_y(self,N_steps=1):
        """
        Splits data into x (initial conditions) and y (final states). 
        Offset by N_steps in N, so x[:,N]->y[:,N+N_steps] is learned

        Parameters
        ----------
        N_steps : int, optional
            How many steps along data trajectory to learn update rule for. The default is 1.

        Returns
        -------
        x : float32[BATCHES,N-N_steps,CHANNELS,WIDTH,HEIGHT]
            Initial conditions
        y : float32[BATCHES,N-N_steps,CHANNELS,WIDTH,HEIGHT]
            Final states

        """
        x = jtu.tree_map(lambda data:data[:-N_steps],self.data_saved)
        y = jtu.tree_map(lambda data:data[N_steps:],self.data_saved)
        # Need to have x be 8 channels and y be 11 channels for handling duplicate channels from different colonies
        def _reduce_to_8(data):
            x_obs = [data[:,:4],data[:,7:11]]
            x_obs = np.concatenate(x_obs,axis=1)
            return np.pad(x_obs,((0,0),(0,data.shape[1] - 8),(0,0),(0,0)))
        x = jtu.tree_map(_reduce_to_8,x)
        return x,y

    def data_callback(self,x,y,i,key):
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

        
        x_true,_ =self.split_x_y(1)
        x = jittable_callback_bit(x,x_true,self.OBS_CHANNELS,key)
        x = self.noise(x,0.01,key=key) # type: ignore
        self.PREVIOUS_KEY = key
        return x,y
		



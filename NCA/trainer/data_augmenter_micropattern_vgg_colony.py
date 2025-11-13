import jax
import jax.numpy as np
import jax.random as jr
import equinox as eqx
import jax.tree_util as jtu

from NCA.trainer.data_augmenter_nca_basic import DataAugmenter as DataAugmenterBasic





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
        x = jittable_callback_bit(x,x_true,self.OBS_CHANNELS)
        x = self.noise(x,0.01,key=key)
        self.PREVIOUS_KEY = key
        return x,y
		

@eqx.filter_jit
def jittable_callback_bit(x,x_true,OBS_CHANNELS):
	propagate_xn = lambda x:x.at[1:].set(x[:-1])
	reset_x0 = lambda x,x_true:x.at[0].set(x_true[0])
	
	x = jax.tree_util.tree_map(propagate_xn,x) # Set initial condition at each X[n] at next iteration to be final state from X[n-1] of this iteration
	x = jax.tree_util.tree_map(reset_x0,x,x_true) # Keep first initial x correct
			
	for b in range(len(x)//2):
		x[b*2] = x[b*2].at[:,:OBS_CHANNELS].set(x_true[b*2][:,:OBS_CHANNELS]) # Set every other batch of intermediate initial conditions to correct initial conditions
	return x


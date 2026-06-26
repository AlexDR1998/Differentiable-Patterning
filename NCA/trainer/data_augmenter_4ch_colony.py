import jax.tree_util as jtu
import equinox as eqx
from NCA.trainer.data_augmenter_nca_basic import DataAugmenter as DataAugmenterBasic


class DataAugmenter(DataAugmenterBasic):
    def split_x_y(self, N_steps=1):
        """
        Splits 4-channel group-A data into initial and target states.

        This keeps the simple 4-channel task direct, without the 12-to-9
        duplicate-colony reduction used by the full micropattern dataset.
        """
        x = jtu.tree_map(lambda data: data[:-N_steps], self.data_saved)
        y = jtu.tree_map(lambda data: data[N_steps:], self.data_saved)
        x = jtu.tree_map(lambda x: self.real_to_latent(x), x)
        return x, y
    
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
	
	x = jtu.tree_map(propagate_xn,x) # Set initial condition at each X[n] at next iteration to be final state from X[n-1] of this iteration
	x = jtu.tree_map(reset_x0,x,x_true) # Keep first initial x correct
			
	for b in range(len(x)//2):
		x[b*2] = x[b*2].at[:,:OBS_CHANNELS].set(x_true[b*2][:,:OBS_CHANNELS]) # Set every other batch of intermediate initial conditions to correct initial conditions
	return x

import jax.numpy as np
import jax.tree_util as jtu
from NCA.trainer.data_augmenter_4ch_colony import DataAugmenter as DataAugmenter4Ch

class DataAugmenter(DataAugmenter4Ch):

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
        if self.batch_mode == "array":
            x = self.data_saved[:, :-N_steps]
            y = self.data_saved[:, N_steps:]
        else:
            x = jtu.tree_map(lambda data:data[:-N_steps],self.data_saved)
            y = jtu.tree_map(lambda data:data[N_steps:],self.data_saved)
        # Need to have x be 9 channels and y be 12 channels for handling duplicate channels from different colonies
        def _reduce_to_9(data):
            x_obs = [data[:,:4],data[:,7:11],data[:,11:12]]
            x_obs = np.concatenate(x_obs,axis=1)
            return np.pad(x_obs,((0,0),(0,data.shape[1] - 9),(0,0),(0,0)))
        x = self.map_batches(_reduce_to_9, x)
        x = self.map_batches(self.real_to_latent, x)
        
        return x,y

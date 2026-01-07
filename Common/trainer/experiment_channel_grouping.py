import jax.numpy as np
import jax


def duplicate_x_channels_8ch(x):
    """
        Duplicates channels in x to match the individual colony experiment groups.
        X [N C H W] with C=8 -> [N 11 H W] with channels duplicated as needed.
        data_channels = ["lmbr","tbxt","sox17","sox2" - "lmbr","tbxt","sox17","foxa2" - "cer1","lefty2","nodal"]
        input_channels = ["lmbr","tbxt","sox17","sox2","foxa2","cer1","lefty2","nodal"]


    """
    x_dup = [x[:,0:4],x[:,0:3],x[:,4:8]]
    return np.concatenate(x_dup,axis=1)

def split_and_pad_by_experiment_groups_11ch(x): 
    """
        For VGG hyperspectral loss, sometimes we need to define which channels are aggregated together, as we compare corresponding blocks of 3 channels.
        
        Parameters
        ----------
        x : float32 [N,CHANNELS,WIDTH,HEIGHT]
            predictions or true data
        Returns
        -------
        x : float32 [N,(C_i groups),WIDTH,HEIGHT]
            x split into experiment groups and padded to multiples of 3 channels. groups=3
            
    """

    x_split = [x[:,0:4],x[:,4:8],x[:,8:11]]  # Hardcoded for jitting
    x_split = [np.pad(x,((0,0),(0,(3-x.shape[1]%3)%3),(0,0),(0,0)),mode="constant") for x in x_split]

    # Recombine
    x = np.concatenate(x_split,axis=1)
    return x


def duplicate_x_channels_9ch(x):
    """
        Duplicates channels in x to match the individual colony experiment groups.
        X [N C H W] with C=9 -> [N 12 H W] with channels duplicated as needed.
        data_channels = ["lmbr","tbxt","sox17","sox2" - "lmbr","tbxt","sox17","foxa2" - "cer1","lefty2","nodal" - "lef1" ]
        input_channels = ["lmbr","tbxt","sox17","sox2","foxa2","cer1","lefty2","nodal","lef1"]


    """
    x_dup = [x[:,0:4],x[:,0:3],x[:,4:8],x[:,8:9]]
    return np.concatenate(x_dup,axis=1)


def split_and_pad_by_experiment_groups_12ch(x):
    """
        For VGG hyperspectral loss, sometimes we need to define which channels are aggregated together, as we compare corresponding blocks of 3 channels.
        
        Parameters
        ----------
        x : float32 [N,CHANNELS,WIDTH,HEIGHT]
            predictions or true data - full 12 channels with duplicates
        Returns
        -------
        x : float32 [N,(C_i groups),WIDTH,HEIGHT]
            x split into experiment groups and padded to multiples of 3 channels. groups=3
            
    """

    x_split = [x[:,0:4],x[:,4:8],x[:,8:11],x[:,11:12]]  # Hardcoded for jitting
    x_split = [np.pad(x,((0,0),(0,(3-x.shape[1]%3)%3),(0,0),(0,0)),mode="constant") for x in x_split]

    # Recombine
    x = np.concatenate(x_split,axis=1)
    return x

def split_and_pad_by_experiment_groups_nodal_knockout(x):
    """
        For VGG hyperspectral loss, sometimes we need to define which channels are aggregated together, as we compare corresponding blocks of 3 channels.
        For nodal knockout experiments we use 9 channels, but only with data on TBXT-SOX17-SOX2-FOXA2 and LEF1 channels.

        Parameters
        ----------
        x : float32 [N,CHANNELS,WIDTH,HEIGHT]
            predictions or true data
        Returns
        -------
        x : float32 [N,(C_i groups),WIDTH,HEIGHT]
            x split into experiment groups and padded to multiples of 3 channels. groups=5
            
    """

    x_split = [x[:,1:5],x[:,8:9]]  # Hardcoded for jitting
    x_split = [np.pad(x,((0,0),(0,(3-x.shape[1]%3)%3),(0,0),(0,0)),mode="constant") for x in x_split]

    # Recombine
    x = np.concatenate(x_split,axis=1)
    return x






def pad_to_multiple_of_3_channels(x):
    """
        Pads x to have a multiple of 3 channels by adding dummy channels of zeros.
        
        Parameters
        ----------
        x : float32 [N,CHANNELS,WIDTH,HEIGHT]
            predictions or true data
        Returns
        -------
        x : float32 [N,CHANNELS_PADDED,WIDTH,HEIGHT]
            x padded to multiples of 3 channels.
            
    """

    x = np.pad(x,((0,0),(0,(3-x.shape[1]%3)%3),(0,0),(0,0)),mode="constant")
    return x
import jax.numpy as np
import jax

from Common.dataloader.channel_schema import ChannelSchema
from Common.dataloader.micropattern_schemas import (
    MICROPATTERN_GROUPED_11CH_SCHEMA,
    MICROPATTERN_GROUPED_12CH_SCHEMA,
)


def project_state_to_measurements(x, schema: ChannelSchema):
    """Project unique biological state channels into a target measurement layout.

    ``schema.target_to_state`` may select a state channel more than once when a
    marker was measured in multiple experiments.  The channel axis is fixed at
    one to match the ``[N, C, H, W]`` tensors used by the loss functions.
    """

    return np.take(x, np.asarray(schema.target_to_state), axis=1)


def split_and_pad_by_experiment_groups(
    x,
    schema: ChannelSchema,
    channel_multiple=3,
):
    """Keep co-measured channels together and pad each group independently."""

    if channel_multiple <= 0:
        raise ValueError("channel_multiple must be positive")
    schema.validate_measurement_channel_count(x.shape[1])

    padded_groups = []
    start = 0
    for group_size in schema.group_sizes:
        end = start + group_size
        group = x[:, start:end]
        padding = (channel_multiple - group_size % channel_multiple) % channel_multiple
        padded_groups.append(
            np.pad(
                group,
                ((0, 0), (0, padding), (0, 0), (0, 0)),
                mode="constant",
            )
        )
        start = end
    return np.concatenate(padded_groups, axis=1)


def duplicate_x_channels_8ch(x):
    """
        Duplicates channels in x to match the individual colony experiment groups.
        X [N C H W] with C=8 -> [N 11 H W] with channels duplicated as needed.
        data_channels = ["lmbr","tbxt","sox17","sox2" - "lmbr","tbxt","sox17","foxa2" - "cer1","lefty2","nodal"]
        input_channels = ["lmbr","tbxt","sox17","sox2","foxa2","cer1","lefty2","nodal"]


    """
    return project_state_to_measurements(x, MICROPATTERN_GROUPED_11CH_SCHEMA)

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

    return split_and_pad_by_experiment_groups(
        x[:, :MICROPATTERN_GROUPED_11CH_SCHEMA.n_measurement_channels],
        MICROPATTERN_GROUPED_11CH_SCHEMA,
    )


def duplicate_x_channels_9ch(x):
    """
        Duplicates channels in x to match the individual colony experiment groups.
        X [N C H W] with C=9 -> [N 12 H W] with channels duplicated as needed.
        data_channels = ["lmbr","tbxt","sox17","sox2" - "lmbr","tbxt","sox17","foxa2" - "cer1","lefty2","nodal" - "lef1" ]
        input_channels = ["lmbr","tbxt","sox17","sox2","foxa2","cer1","lefty2","nodal","lef1"]


    """
    return project_state_to_measurements(x, MICROPATTERN_GROUPED_12CH_SCHEMA)


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

    return split_and_pad_by_experiment_groups(
        x[:, :MICROPATTERN_GROUPED_12CH_SCHEMA.n_measurement_channels],
        MICROPATTERN_GROUPED_12CH_SCHEMA,
    )

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

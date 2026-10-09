import jax.numpy as np
import jax

from Common.dataloader.channel_schema import ChannelSchema
from Common.dataloader.micropattern_schemas import (
    MICROPATTERN_GROUPED_12CH_SCHEMA,
)


def project_state_to_measurements(x, schema: ChannelSchema):
    """Map state channels ``[N, C_state, H, W]`` to the measurement layout ``[N, C_target, H, W]``.

    A state channel appears more than once when its marker was measured in
    several experiment groups.
    """

    return np.take(x, np.asarray(schema.target_to_state), axis=1)


def split_and_pad_by_experiment_groups(
    x,
    schema: ChannelSchema,
    channel_multiple=3,
):
    """Zero-pad each experiment group of ``[N, C, H, W]`` to a multiple of ``channel_multiple`` channels."""

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


def duplicate_x_channels_9ch(x):
    """
        Duplicate channels of x [N 9 H W] -> [N 12 H W] to match the colony experiment groups.
        data_channels = ["lmbr","tbxt","sox17","sox2" - "lmbr","tbxt","sox17","foxa2" - "cer1","lefty2","nodal" - "lef1" ]
        input_channels = ["lmbr","tbxt","sox17","sox2","foxa2","cer1","lefty2","nodal","lef1"]
    """
    return project_state_to_measurements(x, MICROPATTERN_GROUPED_12CH_SCHEMA)


def split_and_pad_by_experiment_groups_12ch(x):
    """
        Split the 12 grouped channels into experiment groups, each padded to a multiple
        of 3 channels, so that VGG losses compare blocks of co-measured channels.

        Parameters
        ----------
        x : float32 [N,CHANNELS,WIDTH,HEIGHT]
            predictions or true data - full 12 channels with duplicates
        Returns
        -------
        x : float32 [N,(C_i groups),WIDTH,HEIGHT]
            x split into experiment groups and padded to multiples of 3 channels
    """

    return split_and_pad_by_experiment_groups(
        x[:, :MICROPATTERN_GROUPED_12CH_SCHEMA.n_measurement_channels],
        MICROPATTERN_GROUPED_12CH_SCHEMA,
    )

def pad_to_multiple_of_3_channels(x):
    """
        Zero-pad x to a multiple of 3 channels.

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

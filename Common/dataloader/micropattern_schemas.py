"""Channel schemas declared by the modern micropattern data loaders."""

from Common.dataloader.channel_schema import (
    ChannelSchema,
    ExperimentChannelGroup,
    MeasurementChannel,
)


def _channel(name, marker=None, source_index=None):
    return MeasurementChannel(
        name=name,
        marker=name if marker is None else marker,
        source_index=source_index,
    )


MICROPATTERN_4CH_SCHEMA = ChannelSchema(
    name="micropattern_4ch",
    state_channels=("LMBR", "TBXT", "SOX17", "SOX2"),
    experiment_groups=(
        ExperimentChannelGroup(
            "cell_fate_s1",
            tuple(
                _channel(name)
                for name in ("LMBR", "TBXT", "SOX17", "SOX2")
            ),
        ),
    ),
)


MICROPATTERN_NODAL_LEFTY_CER_SCHEMA = ChannelSchema(
    name="micropattern_nodal_lefty_cer",
    state_channels=("DAPI", "CER1", "LEFTY", "NODAL"),
    experiment_groups=(
        ExperimentChannelGroup(
            "rna_expression",
            (
                _channel("Dappi", "DAPI", 0),
                _channel("Cerberus", "CER1", 1),
                _channel("Lefty", "LEFTY", 2),
                _channel("Nodal", "NODAL", 3),
            ),
        ),
    ),
)


MICROPATTERN_SOX17_FOXA2_TBXT_LMBR_SCHEMA = ChannelSchema(
    name="micropattern_sox17_foxa2_tbxt_lmbr",
    state_channels=("SOX17", "FOXA2", "TBXT", "LMBR"),
    experiment_groups=(
        ExperimentChannelGroup(
            "cell_fate_s2",
            (
                _channel("Sox17", "SOX17", 0),
                _channel("Foxa2", "FOXA2", 1),
                _channel("TbxT", "TBXT", 2),
                _channel("Lmbr", "LMBR", 3),
            ),
        ),
    ),
)


MICROPATTERN_SMAD23_LEF1_SCHEMA = ChannelSchema(
    name="micropattern_smad23_lef1",
    state_channels=("LEF1", "LMBR", "SMAD23"),
    experiment_groups=(
        ExperimentChannelGroup(
            "protein_response",
            (
                _channel("Lef1", "LEF1", 0),
                _channel("Lmbr", "LMBR", 1),
                _channel("Smad23", "SMAD23", 2),
            ),
        ),
    ),
)


MICROPATTERN_AVERAGED_8CH_SCHEMA = ChannelSchema(
    name="micropattern_averaged_8ch",
    state_channels=(
        "SOX17",
        "FOXA2",
        "TBXT",
        "LMBR",
        "CER1",
        "LEFTY",
        "NODAL",
        "LEF1",
    ),
    experiment_groups=(
        ExperimentChannelGroup(
            "cell_fate_s2",
            (
                _channel("Sox17", "SOX17"),
                _channel("Foxa2", "FOXA2"),
                _channel("TbxT", "TBXT"),
                _channel("Lmbr", "LMBR"),
            ),
        ),
        ExperimentChannelGroup(
            "rna_expression",
            (
                _channel("Cerberus", "CER1"),
                _channel("Lefty", "LEFTY"),
                _channel("Nodal", "NODAL"),
            ),
        ),
        ExperimentChannelGroup(
            "protein_response",
            (_channel("Lef1", "LEF1"),),
        ),
    ),
)


MICROPATTERN_INDIVIDUAL_8CH_SCHEMA = ChannelSchema(
    name="micropattern_individual_8ch",
    state_channels=(
        "LMBR",
        "TBXT",
        "SOX17",
        "SOX2",
        "FOXA2",
        "CER1",
        "LEFTY",
        "NODAL",
    ),
    experiment_groups=(
        ExperimentChannelGroup(
            "cell_fate",
            (
                _channel("LMBR"),
                _channel("TBXT"),
                _channel("SOX17"),
                _channel("SOX2"),
                _channel("FOXA2"),
                _channel("Cer1", "CER1"),
                _channel("Lefty2", "LEFTY"),
                _channel("Nodal", "NODAL"),
            ),
        ),
    ),
)


MICROPATTERN_GROUPED_11CH_SCHEMA = ChannelSchema(
    name="micropattern_grouped_11ch_targets_8ch_state",
    state_channels=(
        "LMBR",
        "TBXT",
        "SOX17",
        "SOX2",
        "FOXA2",
        "CER1",
        "LEFTY",
        "NODAL",
    ),
    experiment_groups=(
        ExperimentChannelGroup(
            "A",
            (
                _channel("A-LMBR", "LMBR"),
                _channel("A-TBXT", "TBXT"),
                _channel("A-SOX17", "SOX17"),
                _channel("A-SOX2", "SOX2"),
            ),
        ),
        ExperimentChannelGroup(
            "B",
            (
                _channel("B-LMBR", "LMBR"),
                _channel("B-TBXT", "TBXT"),
                _channel("B-SOX17", "SOX17"),
                _channel("B-FOXA2", "FOXA2"),
            ),
        ),
        ExperimentChannelGroup(
            "C",
            (
                _channel("C-Cer1", "CER1"),
                _channel("C-Lefty2", "LEFTY"),
                _channel("C-Nodal", "NODAL"),
            ),
        ),
    ),
)


MICROPATTERN_GROUPED_12CH_SCHEMA = ChannelSchema(
    name="micropattern_grouped_12ch_targets_9ch_state",
    state_channels=(
        "LMBR",
        "TBXT",
        "SOX17",
        "SOX2",
        "FOXA2",
        "CER1",
        "LEFTY",
        "NODAL",
        "LEF1",
    ),
    experiment_groups=MICROPATTERN_GROUPED_11CH_SCHEMA.experiment_groups
    + (
        ExperimentChannelGroup(
            "D",
            (_channel("D-LEF1", "LEF1"),),
        ),
    ),
)


MICROPATTERN_260726_SCHEMA = ChannelSchema(
    name="micropattern_260726_14ch_targets_10ch_state",
    state_channels=(
        "LMBR",
        "TBXT",
        "SOX17",
        "SOX2",
        "FOXA2",
        "CER1",
        "LEFTY",
        "NODAL",
        "LEF1",
        "SMAD23",
    ),
    experiment_groups=(
        ExperimentChannelGroup(
            "cell_fate_s1",
            (
                _channel("cell_fate_s1/SOX17", "SOX17", 0),
                _channel("cell_fate_s1/SOX2", "SOX2", 1),
                _channel("cell_fate_s1/TBXT", "TBXT", 2),
                _channel("cell_fate_s1/LMBR", "LMBR", 3),
            ),
        ),
        ExperimentChannelGroup(
            "cell_fate_s2",
            (
                _channel("cell_fate_s2/SOX17", "SOX17", 0),
                _channel("cell_fate_s2/FOXA2", "FOXA2", 1),
                _channel("cell_fate_s2/TBXT", "TBXT", 2),
                _channel("cell_fate_s2/LMBR", "LMBR", 3),
            ),
        ),
        ExperimentChannelGroup(
            "rna_expression",
            (
                _channel("rna_expression/CER1", "CER1", 1),
                _channel("rna_expression/LEFTY", "LEFTY", 2),
                _channel("rna_expression/NODAL", "NODAL", 3),
            ),
        ),
        ExperimentChannelGroup(
            "protein_response",
            (
                _channel("protein_response/LEF1", "LEF1", 0),
                _channel("protein_response/LMBR", "LMBR", 1),
                _channel("protein_response/SMAD23", "SMAD23", 2),
            ),
        ),
    ),
)


def attach_channel_schema(aux, schema):
    """Attach schema metadata without changing an established return signature."""

    if aux is None:
        aux = {}
    aux["channel_schema"] = schema
    return aux

import pytest

from Common.dataloader.channel_schema import (
    ChannelSchema,
    ExperimentChannelGroup,
    MeasurementChannel,
)
from Common.dataloader.micropattern_schemas import (
    MICROPATTERN_260726_SCHEMA,
    MICROPATTERN_GROUPED_12CH_SCHEMA,
)


def test_legacy_grouped_schema_matches_existing_12_to_9_mapping():
    schema = MICROPATTERN_GROUPED_12CH_SCHEMA

    assert schema.n_state_channels == 9
    assert schema.n_measurement_channels == 12
    assert schema.group_sizes == (4, 4, 3, 1)
    assert schema.target_to_state == (0, 1, 2, 3, 0, 1, 2, 4, 5, 6, 7, 8)
    assert schema.measurement_weights == (
        0.5,
        0.5,
        0.5,
        1.0,
        0.5,
        0.5,
        0.5,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )


def test_new_schema_preserves_all_panels_and_extra_channels():
    schema = MICROPATTERN_260726_SCHEMA

    assert schema.n_state_channels == 10
    assert schema.n_measurement_channels == 14
    assert schema.group_sizes == (4, 4, 3, 3)
    assert schema.target_to_state == (
        2,
        3,
        1,
        0,
        2,
        4,
        1,
        0,
        5,
        6,
        7,
        8,
        0,
        9,
    )
    assert schema.duplicate_state_channels == ("LMBR", "TBXT", "SOX17")
    assert schema.measurement_weights[3] == pytest.approx(1.0 / 3.0)
    assert schema.measurement_weights[7] == pytest.approx(1.0 / 3.0)
    assert schema.measurement_weights[12] == pytest.approx(1.0 / 3.0)
    assert "DAPI" not in schema.state_channels
    assert all("DAPI" not in name for name in schema.measurement_names)


def test_co_measurement_pairs_never_cross_experiment_groups():
    schema = MICROPATTERN_260726_SCHEMA
    group_for_measurement = {
        measurement_index: group_index
        for group_index, group in enumerate(schema.group_measurement_indices)
        for measurement_index in group
    }

    assert all(
        group_for_measurement[left] == group_for_measurement[right]
        for left, right in schema.co_measurement_pairs
    )
    assert (1, 5) not in schema.co_measurement_pairs  # S1 SOX2 vs S2 FOXA2


def test_repeated_biological_pairs_share_total_correlation_weight():
    schema = MICROPATTERN_260726_SCHEMA
    state_pairs = [
        tuple(sorted((schema.target_to_state[left], schema.target_to_state[right])))
        for left, right in schema.co_measurement_pairs
    ]
    lmbr_tbxt = tuple(
        sorted((schema.state_channels.index("LMBR"), schema.state_channels.index("TBXT")))
    )
    weights = [
        weight
        for pair, weight in zip(state_pairs, schema.correlation_pair_weights)
        if pair == lmbr_tbxt
    ]

    assert weights == [0.5, 0.5]
    assert sum(weights) == 1.0


def test_schema_rejects_measurements_for_unknown_state_markers():
    with pytest.raises(ValueError, match="unknown state markers"):
        ChannelSchema(
            name="invalid",
            state_channels=("A",),
            experiment_groups=(
                ExperimentChannelGroup(
                    "experiment",
                    (MeasurementChannel("experiment/B", "B", 0),),
                ),
            ),
        )


def test_schema_validates_target_tensor_channel_count():
    schema = MICROPATTERN_260726_SCHEMA

    schema.validate_measurement_channel_count(14)
    with pytest.raises(ValueError, match="expects 14 measurement channels"):
        schema.validate_measurement_channel_count(11)


def test_schema_can_select_experiment_groups_and_recompute_state_layout():
    schema = MICROPATTERN_260726_SCHEMA.select_groups(
        ["cell_fate_s1", "rna_expression"]
    )

    assert schema.group_names == ("cell_fate_s1", "rna_expression")
    assert schema.state_channels == (
        "LMBR",
        "TBXT",
        "SOX17",
        "SOX2",
        "CER1",
        "LEFTY",
        "NODAL",
    )
    assert schema.n_measurement_channels == 7
    assert schema.target_to_state == (2, 3, 1, 0, 4, 5, 6)
    assert schema.primary_measurements == (3, 2, 0, 1, 4, 5, 6)


def test_schema_rejects_invalid_group_selection():
    with pytest.raises(ValueError, match="Unknown experiment groups"):
        MICROPATTERN_260726_SCHEMA.select_groups(["not_a_group"])

    with pytest.raises(ValueError, match="cannot contain duplicates"):
        MICROPATTERN_260726_SCHEMA.select_groups(["cell_fate_s1", "cell_fate_s1"])

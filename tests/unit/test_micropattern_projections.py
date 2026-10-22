import pytest

from Common.dataloader.micropattern_projections import image_key, keyed_by_image


@pytest.mark.parametrize(
    "path",
    [
        "cell_fate_markers/ctrl_s1_max/A2_F14_24h.ome.tif",
        "cell_fate_markers/ctrl_s1_sums/A2_F14_24h_sum.tif",
        "cell_fate_markers/ctrl_s1_slices/A2_F14_24h_slice.tif",
        # Paths from before the projections were added.
        "cell_fate_markers/ctrl_s1/A2_F14_24h.ome.tif",
        "cell_fate_markers/ctrl_s1/A2_F14_24h",
    ],
)
def test_every_projection_of_an_image_has_one_key(path):
    assert image_key(path) == "cell_fate_markers/ctrl_s1/A2_F14_24h"


def test_max_prefix_of_knockout_and_signalling_images_is_dropped():
    assert image_key("cell_fate_markers/sl0_s1_max/MAX_A2_F10_36h.ome.tif") == image_key(
        "cell_fate_markers/sl0_s1_slices/A2_F10_36h_slice.tif"
    )
    assert (
        image_key("signalling/protein_response/ctrl/MAX_A1_F1_CTRL_24h.ome.tif")
        == "signalling/protein_response/ctrl/A1_F1_24h"
    )
    assert image_key("signalling/rna_expression/sl0/12h_SL0h-1.tif") == "signalling/rna_expression/sl0/12h_SL0h-1"


@pytest.mark.parametrize(
    "path",
    [
        "signalling/protein_response/sl24_max/MAX_B1_F1_sl24_36h.ome.tif",
        "signalling/protein_response/sl24_sums/B1_F1_S24_36h_sum.tif",
        "signalling/protein_response/sl24_slices/B1_F1_SL24_36h_slice.tif",
    ],
)
def test_condition_spelling_in_protein_names_is_ignored(path):
    assert image_key(path) == "signalling/protein_response/sl24/B1_F1_36h"


def test_two_projections_of_one_image_cannot_both_be_listed():
    with pytest.raises(ValueError, match="more than once"):
        keyed_by_image(
            {
                "cell_fate_markers/ctrl_s1/A2_F2_12h.ome.tif": 1,
                "cell_fate_markers/ctrl_s1_max/A2_F2_12h.ome.tif": 2,
            },
            "Alignment file",
        )


def test_config_accepts_only_known_projections():
    from Experiments.micropatterns.config import MicropatternDataConfig

    assert MicropatternDataConfig().projection == "max"
    assert MicropatternDataConfig(projection="sum").projection == "sum"
    with pytest.raises(ValueError, match="Unknown projection"):
        MicropatternDataConfig(projection="mean")

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
import yaml
from matplotlib.colors import to_hex

from Common.dataloader.micropattern_schemas import MICROPATTERN_260726_SCHEMA
from Experiments.micropatterns.figures import style

CELL_TYPE_FILES = sorted(
    (Path(__file__).resolve().parents[2] / "Experiments/micropatterns/conf/cell_types").glob("*.yaml")
)


@pytest.mark.parametrize("path", CELL_TYPE_FILES, ids=lambda p: p.name)
def test_every_cell_type_has_a_colour(path):
    labels = set(yaml.safe_load(path.read_text())["cell_types"]) | {"Several", "Other"}
    assert labels <= set(style.CELL_TYPE_COLOURS)


def test_every_marker_has_a_channel_colour():
    for marker in MICROPATTERN_260726_SCHEMA.state_channels:
        assert style.MARKER_COLOURS[marker] in style.CHANNEL_COLOURS


def test_figure_channels_name_real_channels():
    groups = {g.name: {c.marker for c in g.channels} for g in MICROPATTERN_260726_SCHEMA.experiment_groups}
    assert set(style.FIGURE_CHANNELS) == set(style.EXPERIMENT_GROUPS) == set(groups)
    for group, markers in style.FIGURE_CHANNELS.items():
        assert set(markers) <= groups[group]
    for triplet in style.CMY_TRIPLETS:
        assert set(triplet) <= set(MICROPATTERN_260726_SCHEMA.state_channels)


def test_group_colours_show_only_figure_channels():
    names = ["cell_fate_s1/SOX17", "cell_fate_s1/SOX2", "cell_fate_s1/LMBR", "rna_expression/NODAL"]
    colours = style.group_colours(names, "cell_fate_s1")
    assert colours[0] == style.CHANNEL_COLOURS["yellow"]
    assert colours[1] == style.CHANNEL_COLOURS["cyan"]
    assert colours[2] is None and colours[3] is None


def test_paper_style_sets_fonts_and_colour_cycle():
    with style.paper_style():
        assert plt.rcParams["font.size"] == style.FONT["label"]
        assert plt.rcParams["legend.fontsize"] == style.FONT["legend"]
        cycle = [to_hex(c) for c in plt.rcParams["axes.prop_cycle"].by_key()["color"]]
        assert cycle == [c.lower() for c in style.SERIES]
    with style.paper_style("talk"):
        assert plt.rcParams["font.size"] == pytest.approx(11)


def test_label_panel_and_save_figure(tmp_path):
    with style.paper_style():
        fig = style.new_figure("single", aspect=0.5)
        ax = fig.add_subplot()
        style.label_panel(ax, "a", "Title", "subtitle")
        ax.legend(handles=[style.legend_swatch("red"), style.unmeasured_patch()], labels=["x", "y"],
                  **style.legend_options("white"))
    assert fig.get_size_inches()[0] == pytest.approx(89 / 25.4)
    paths = style.save_figure(fig, "test", "unit", svg=True, root=tmp_path)
    plt.close(fig)
    assert [p.suffix for p in paths] == [".pdf", ".png", ".svg"]
    assert all(p.is_file() for p in paths)
    assert "<text" in (tmp_path / "unit" / "test.svg").read_text()

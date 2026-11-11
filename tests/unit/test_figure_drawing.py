import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from Experiments.micropatterns.figures import drawing, style
from Experiments.micropatterns.figures.data import cell_type_rgba, masked_rgba
from Experiments.micropatterns.figures.toy import toy_patterning, toy_rgba


def _rollout(times=(0, 24, 48), size=12):
    names = ("cell_fate_s1/SOX17", "cell_fate_s1/SOX2", "cell_fate_s1/TBXT", "rna_expression/NODAL")
    rows, cols = np.indices((size, size))
    boundary = np.hypot(rows - size / 2, cols - size / 2) < size / 2 - 1
    images = np.random.default_rng(0).uniform(size=(len(times), len(names), size, size))
    measured = np.ones((len(times), len(names)), dtype=bool)
    measured[1, 3] = False
    return {"target": images, "prediction": images[::-1], "measured": measured, "channel_names": names,
            "time_labels": tuple(f"{t}h" for t in times), "boundary": boundary}


def test_trajectory_figure_draws_rows_and_legend():
    rollouts = {"a": _rollout()}
    blocks = [("Block", style.INK, [("a", "target", "True data", style.DATA), ("a", "prediction", "NCA", style.MODEL)])]
    with style.paper_style():
        fig = drawing.trajectory_figure(rollouts, blocks, ["cell_fate_s1", "rna_expression"], 1.0, 9.0, True,
                                        zoom={"time": 2, "centres": [(0.5, 0.5)], "width": 0.3,
                                              "colours": style.ZOOM_COLOURS, "title": "zoom"})
    assert fig.legends and fig.get_size_inches()[0] == style.WIDTH["double"]
    plt.close(fig)


def test_stack_panels_with_and_without_measured_scale():
    def panel(ax, pt_per_unit=None):
        drawing.draw_card(ax, (0, 0), 1.0, 10, drawing.placeholder_rgba(), unmeasured=(True, False))
        drawing.draw_arrow(ax, (1.1, 0.5), (1.6, 0.5))
        drawing.rounded_box(ax, (1.7, 0), 1.0, 1.0, style.ACCENT)
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 1)

    with style.paper_style():
        fig = drawing.stack_panels({"a": panel, "b": panel}, ["a", "b"], "single", {"a": "Title"})
        measured = drawing.stack_panels({"a": lambda ax, pt: panel(ax)}, ["a"], measure_scale=True)
    assert len(fig.axes) == 2 and len(measured.axes) == 1
    plt.close("all")


def test_toy_and_cell_type_images():
    frames, mask = toy_patterning(12, 3, 5, seed=0)
    assert toy_rgba(frames[-1], mask).shape == (12, 12, 4)
    shares = {"Mesoderm": mask * 0.5, "Endoderm": mask * 0.5}
    rgba = cell_type_rgba(shares, style.CELL_TYPE_COLOURS, mask)
    assert rgba.shape[-1] == 4 and np.all(rgba[..., 3][~mask[drawing.crop_box(mask)]] == 0)
    assert masked_rgba(np.zeros(mask.shape + (3,)), mask).shape == rgba.shape

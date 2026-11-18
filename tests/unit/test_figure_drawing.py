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


def test_zoom_boxes_are_striped_by_default():
    rollouts = {"a": _rollout()}
    blocks = [("Block", style.INK, [("a", "target", "True data", style.DATA)])]
    with style.paper_style():
        fig = drawing.trajectory_figure(rollouts, blocks, ["cell_fate_s1"], 1.0, 9.0, True,
                                        zoom={"time": 2, "centres": [(0.5, 0.5), (0.3, 0.3)], "width": 0.3,
                                              "title": "zoom"})
    edges = {matplotlib.colors.to_hex(p.get_edgecolor()) for p in fig.axes[0].patches if not p.get_fill()}
    assert {"#ffffff", "#000000"} <= edges
    plt.close(fig)

    fig, ax = plt.subplots()
    drawing.frame_zoom_axes(ax)
    assert not any(spine.get_visible() for spine in ax.spines.values())
    assert [p.get_linestyle() for p in ax.patches] == ["-", (0, style.ZOOM_OUTLINE["dashes"])]
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


def test_pixel_zoom_boxes_cover_the_same_region_at_every_resolution():
    # A 4x finer rollout whose colony crop starts at a different place than the
    # coarse one, as happens when the coarse mask is block-averaged
    fine = _rollout(size=32)
    fine["boundary"] = np.zeros((32, 32), dtype=bool)
    fine["boundary"][5:27, 6:30] = True
    coarse = _rollout(size=8)
    coarse["boundary"] = np.zeros((8, 8), dtype=bool)
    coarse["boundary"][1:7, 1:8] = True
    rollouts = {4: coarse, 1: fine}
    blocks = [("Block", style.INK, [(4, "target", "coarse", style.DATA), (1, "target", "fine", style.DATA)])]
    with style.paper_style():
        fig = drawing.trajectory_figure(rollouts, blocks, ["cell_fate_s1"], 1.0, 9.0, True,
                                        zoom={"time": 2, "centres": [(0.5, 0.5)], "width": 0.3, "title": "zoom",
                                              "pixels": {4: 3, 1: 12}})
    ax = fig.axes[0]
    zooms = [im for im in ax.images if im.get_interpolation() == "nearest"]
    # Full-resolution pixel under the top-left corner of each zoom square
    corners = []
    for im, (top, left), downsample in zip(zooms, [(1, 1), (5, 6)], [4, 1]):
        x0, x1, y0, y1 = im.get_extent()
        rows, cols = im.get_array().shape[:2]
        square = im.clipbox.transformed(ax.transData.inverted())
        corners.append((((square.x0 - x0) / ((x1 - x0) / cols) + left) * downsample,
                        ((y1 - square.y1) / ((y1 - y0) / rows) + top) * downsample))
    assert np.allclose(corners[0], corners[1])
    plt.close(fig)


def test_grid_figure_draws_one_timepoint_and_marks_missing_models():
    rollouts = {("slice", 1): _rollout(), ("slice", 4): _rollout(size=6), ("max", 1): _rollout()}
    blocks = [(p, style.INK, [(p, src, src, style.INK) for src in ("target", "prediction")]) for p in ("slice", "max")]
    with style.paper_style():
        fig = drawing.grid_figure(rollouts, blocks, [(1, "Δ = 1"), (4, "Δ = 4")], ["cell_fate_s1"], 1.0, 9.0, True,
                                  "48h", title="Downsampling ratio")
    ax = fig.axes[0]
    assert len(ax.images) == 6  # 3 models x (true data, NCA)
    assert sum(t.get_text() == "no model" for t in ax.texts) == 2
    assert fig.legends
    plt.close(fig)


def test_snapshot_loss_figure_puts_bars_under_their_columns():
    rollouts = {"max": _rollout(), "sum": _rollout(), "slice": _rollout()}
    columns = [("slice", "Slice"), ("max", "Max"), ("sum", "Sum")]
    losses = {"slice": {"train": [1.0, 2.0], "validation": [3.0, 5.0]},
              "max": {"train": [1.0], "validation": [2.0]}}  # "sum" has no losses
    with style.paper_style():
        fig = drawing.snapshot_loss_figure(rollouts, columns, ["cell_fate_s1"], 1.0, 9.0, True, "48h", losses,
                                           "Loss")
    grid_ax, bar_ax = fig.axes
    assert len(grid_ax.images) == 6  # 3 columns x (true data, NCA)
    bars = bar_ax.patches
    assert sorted(b.get_height() for b in bars) == [1.0, 1.5, 2.0, 4.0]
    # The bars of each column are centred under its discs
    centres = sorted(b.get_x() + b.get_width() / 2 for b in bars)
    assert np.isclose((centres[0] + centres[1]) / 2, 0.0)
    assert [t.get_text() for t in bar_ax.get_xticklabels()] == ["Slice", "Max", "Sum"]
    plt.close(fig)

import numpy as np

from Experiments.micropatterns.figures.drawing import colony_rgba, sliced_rgba


def test_colony_rgba_crops_to_mask_and_is_transparent_outside():
    image = np.zeros((2, 6, 6))
    image[0, 2:4, 2:5] = 1.0
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:5, 1:5] = True
    mask[1, 1] = False
    rgba = colony_rgba(image, mask, [(1.0, 0.0, 0.0), None])
    assert rgba.shape == (4, 4, 4)
    np.testing.assert_array_equal(rgba[..., 3], mask[1:5, 1:5])
    np.testing.assert_allclose(rgba[1, 1], [1.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(rgba[0, 1, :3], 0.0)


def test_sliced_rgba_two_groups_are_left_and_right_halves():
    left, right = np.zeros((4, 4, 4)), np.ones((4, 4, 4))
    out = sliced_rgba([left, right])
    np.testing.assert_array_equal(out[:, :2], 0.0)
    np.testing.assert_array_equal(out[:, 2:], 1.0)


def _ko_rollouts(times=(0, 12, 24, 36, 48), size=12):
    names = ("cell_fate_s1/SOX17", "cell_fate_s1/SOX2", "cell_fate_s1/TBXT", "cell_fate_s1/LMBR",
             "rna_expression/NODAL")
    rows, cols = np.indices((size, size))
    boundary = np.hypot(rows - size / 2, cols - size / 2) < size / 2 - 1
    rng = np.random.default_rng(0)
    rollouts = {}
    for condition in ("baseline", "ko_24h", "ko_0h"):
        measured = np.ones((len(times), len(names)), dtype=bool)
        measured[3, :4] = False  # no fate staining at 36h
        images = rng.uniform(size=(len(times), len(names), size, size))
        rollouts[condition] = {"target": images, "prediction": images[::-1], "measured": measured,
                               "channel_names": names, "time_labels": tuple(f"{t}h" for t in times),
                               "boundary": boundary}
    return rollouts


def test_ko_trajectory_channel_colours_leave_out_lmbr_and_other_stainings():
    from Experiments.micropatterns.figures.ko_trajectory import channel_colours

    names = ["cell_fate_s2/SOX17", "cell_fate_s2/LMBR", "rna_expression/NODAL"]
    colours = channel_colours(names, ["cell_fate_s2"])["cell_fate_s2"]
    assert colours[0] is not None and colours[1] is None and colours[2] is None
    assert channel_colours(names, ["cell_fate_s2"], lmbr=True)["cell_fate_s2"][1] is not None
    # Sector per condition: cyan, then magenta, in channel order
    cmy = channel_colours(names, ["cell_fate_s2"], by_condition=True, lmbr=True)["cell_fate_s2"]
    assert cmy[:2] == [(0.0, 1.0, 1.0), (1.0, 0.0, 1.0)]


def test_ko_trajectory_figure_draws_every_arrangement():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from Common.dataloader.cell_types import CellTypeRules
    from Experiments.micropatterns.figures.ko_trajectory import KoTrajectoryOptions, ko_trajectory_figure
    from Experiments.micropatterns.figures.style import paper_style

    rules = CellTypeRules(
        channels={"SOX2": "cell_fate_s1/SOX2", "TBXT": "cell_fate_s1/TBXT"},
        thresholds={"SOX2": 0.5, "TBXT": 0.5},
        rules={"Mesoderm": {"TBXT": "high", "SOX2": "low"}},
    )
    rollouts = _ko_rollouts()
    groups = ("cell_fate_s1", "rna_expression")
    cases = [
        KoTrajectoryOptions(groups=groups),
        KoTrajectoryOptions(groups=groups, branch=True, panels="stacked", hatch=True, connectors="both",
                            legend_style="white"),
        KoTrajectoryOptions(groups=groups, branch=True, panels="panels", legend="gap", ko_band=None),
        KoTrajectoryOptions(groups=groups, layout="conditions", cell_type_rules=rules, legend="bottom"),
        KoTrajectoryOptions(groups=groups, cell_type_rules=rules, show_true=False, legend="top"),
    ]
    with paper_style():
        for options in cases:
            figure = ko_trajectory_figure(rollouts, options)
            assert figure.legends and figure.axes
            plt.close(figure)

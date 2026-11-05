"""W&B logging for micropattern geometry optimisation."""

import numpy as np
import wandb
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from Common.dataloader.cell_types import label_names
from Common.trainer.wandb_logger import WandbLogger
from Experiments.micropatterns.rendering import (
    CELL_TYPE_COLOURS,
    cell_type_prevalences,
    render_cell_types,
    render_fate_cmy,
)


def _grey(image):
    return np.repeat(np.clip(image, 0.0, 1.0)[..., None], 3, axis=-1)


def evaluation_panel(evaluation, share_maps, state_channel_names, cell_type_rules, rows=4):
    """One row per rollout: mask, SOX2/TBXT/SOX17 composite, cell types, target share.

    ``share_maps`` ``[seeds, H, W]`` holds each rollout's per-pixel target
    share. Returns an ``[rows * H, 4 * W, 3]`` image in [0, 1].
    """

    mask = np.asarray(evaluation.mask)
    states = np.asarray(evaluation.final_states)[:rows]
    fates = render_fate_cmy(states, list(state_channel_names))
    cell_types = render_cell_types(states, state_channel_names, cell_type_rules, mask)
    panels = [
        np.concatenate([_grey(mask), fate, cell_type, _grey(share * mask)], axis=1)
        for fate, cell_type, share in zip(fates, cell_types, np.asarray(share_maps)[:rows])
    ]
    return np.concatenate(panels, axis=0)


def trajectory_video(mask, trajectories, state_channel_names, cell_type_rules):
    """``[T, 3, seeds * H, 2 * W]`` video: SOX2/TBXT/SOX17 composite and cell types, one row per rollout.

    ``trajectories`` is ``[seeds, T, C, H, W]``. The cell type rules hold
    only at the hour they were tuned on, so earlier frames are a rough guide.
    """

    rows = []
    for trajectory in np.asarray(trajectories):
        fates = render_fate_cmy(trajectory, list(state_channel_names)) * np.asarray(mask)[..., None]
        cell_types = render_cell_types(trajectory, state_channel_names, cell_type_rules, mask)
        rows.append(np.concatenate([fates, cell_types], axis=2))
    video = np.concatenate(rows, axis=1)
    return np.moveaxis(video, -1, 1)


def cell_type_legend(cell_type_rules):
    """RGB image ``[H, W, 3]`` with the colour of each label of the cell type map."""

    labels = label_names(cell_type_rules)
    figure = Figure(figsize=(1.3 * len(labels), 0.4), dpi=100)
    canvas = FigureCanvasAgg(figure)
    figure.legend(
        handles=[
            Patch(color=CELL_TYPE_COLOURS.get(label, (0.0, 0.0, 0.0)), label=label)
            for label in labels
        ],
        loc="center",
        ncol=len(labels),
        frameon=False,
    )
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3]


class GeometryLogger(WandbLogger):
    """Logs optimisation metrics, binary-mask evaluations and baselines.

    Scores are written to the history as well as the run summary, so W&B
    line plots of them have data.
    """

    def __init__(self, wandb_config, state_channel_names, cell_type_rules):
        self.state_channel_names = tuple(state_channel_names)
        self.cell_type_rules = cell_type_rules
        self.circle_share = None
        super().__init__(None, wandb_config)

    def log_data_at_init(self, data):
        """Logs the cell type colour legend."""

        self.log_image_single("CellTypes/legend", cell_type_legend(self.cell_type_rules), step=0)

    def _prevalences(self, prefix, evaluation):
        """Mean colony share of each label over the rollouts of an evaluation."""

        shares = cell_type_prevalences(
            np.asarray(evaluation.final_states),
            self.state_channel_names,
            self.cell_type_rules,
            np.asarray(evaluation.mask),
        )
        return {f"{prefix}/{label}": float(np.mean(share)) for label, share in shares.items()}

    def _panel(self, evaluation, share_maps):
        return evaluation_panel(evaluation, share_maps, self.state_channel_names, self.cell_type_rules)

    def log_baselines(self, baselines, share_maps):
        """``baselines`` maps a name to ``(geometry, evaluation)``."""

        self.circle_share = baselines["circle"][1].mean
        scalars = {}
        for name, (_, evaluation) in baselines.items():
            scalars[f"Baseline/{name}/target_share"] = evaluation.mean
            scalars[f"Baseline/{name}/target_share_std"] = evaluation.std
            scalars.update(self._prevalences(f"Baseline/{name}/cell_types", evaluation))
            self.log_image_single(f"Baseline/{name}", self._panel(evaluation, share_maps[name]), step=0)
        self.log_scalars(scalars, step=0)

    def log_iteration(self, iteration, metrics, evaluation=None, share_maps=None):
        scalars = {f"Optimisation/{name}": value for name, value in metrics.items()}
        if evaluation is not None:
            scalars["Binary/target_share"] = evaluation.mean
            scalars["Binary/target_share_std"] = evaluation.std
            scalars["Binary/border_fraction"] = evaluation.border_fraction
            if self.circle_share is not None:
                scalars["Binary/gain_over_circle"] = evaluation.mean - self.circle_share
            scalars.update(self._prevalences("CellTypes", evaluation))
        self.log_scalars(scalars, step=iteration)
        if evaluation is not None and share_maps is not None:
            self.log_image_single("Binary/rollouts", self._panel(evaluation, share_maps), step=iteration)

    def log_result(self, result, step):
        """Best scores, at ``step`` (the last logged iteration) and in the summary."""

        scalars = {
            "Optimised/best_iteration": result.best_iteration,
            "Optimised/target_share": result.best_evaluation.mean,
            "Optimised/target_share_std": result.best_evaluation.std,
            **self._prevalences("Optimised/cell_types", result.best_evaluation),
        }
        self.log_scalars(scalars, step=step)
        for name, value in scalars.items():
            self.run.summary[name] = value

    def log_trajectories(self, mask, trajectories, step, fps=10):
        """Video of the best geometry's rollouts on its binary mask (``binary_trajectories``)."""

        video = trajectory_video(mask, trajectories, self.state_channel_names, self.cell_type_rules)
        video = np.clip(video * 255, 0, 255).astype(np.uint8)
        wandb.log(
            {"Optimised/rollout_video": wandb.Video(video, fps=fps, format="mp4")},
            step=step,
        )

    def finish(self):
        self.run.finish()

"""W&B logging for micropattern geometry optimisation."""

import numpy as np

from Common.trainer.wandb_logger import WandbLogger
from Experiments.micropatterns.rendering import render_fate_cmy


def evaluation_panel(evaluation, share_maps, state_channel_names, rows=4):
    """One row per rollout: mask, SOX2/TBXT/SOX17 composite, target share.

    ``share_maps`` ``[seeds, H, W]`` holds each rollout's per-pixel target
    share. Returns an ``[rows * H, 3 * W, 3]`` image in [0, 1].
    """

    def grey(image):
        return np.repeat(np.clip(image, 0.0, 1.0)[..., None], 3, axis=-1)

    mask = np.asarray(evaluation.mask)
    states = np.asarray(evaluation.final_states)[:rows]
    fates = render_fate_cmy(states, list(state_channel_names))
    panels = [
        np.concatenate([grey(mask), fate, grey(share * mask)], axis=1)
        for fate, share in zip(fates, np.asarray(share_maps)[:rows])
    ]
    return np.concatenate(panels, axis=0)


class GeometryLogger(WandbLogger):
    """Logs optimisation metrics, binary-mask evaluations and baselines."""

    def __init__(self, wandb_config, state_channel_names):
        self.state_channel_names = tuple(state_channel_names)
        super().__init__(None, wandb_config)

    def log_data_at_init(self, data):
        """Nothing to log before the baselines."""

    def log_baselines(self, baselines, share_maps):
        """``baselines`` maps a name to ``(geometry, evaluation)``."""

        for name, (_, evaluation) in baselines.items():
            self.run.summary[f"Baseline/{name}/target_share"] = evaluation.mean
            self.run.summary[f"Baseline/{name}/target_share_std"] = evaluation.std
            self.log_image_single(
                f"Baseline/{name}",
                evaluation_panel(evaluation, share_maps[name], self.state_channel_names),
                step=0,
            )

    def log_iteration(self, iteration, metrics, evaluation=None, share_maps=None):
        scalars = {f"Optimisation/{name}": value for name, value in metrics.items()}
        if evaluation is not None:
            scalars["Binary/target_share"] = evaluation.mean
            scalars["Binary/target_share_std"] = evaluation.std
            scalars["Binary/border_fraction"] = evaluation.border_fraction
        self.log_scalars(scalars, step=iteration)
        if evaluation is not None and share_maps is not None:
            self.log_image_single(
                "Binary/rollouts",
                evaluation_panel(evaluation, share_maps, self.state_channel_names),
                step=iteration,
            )

    def log_result(self, result):
        self.run.summary["Optimised/best_iteration"] = result.best_iteration
        self.run.summary["Optimised/target_share"] = result.best_evaluation.mean
        self.run.summary["Optimised/target_share_std"] = result.best_evaluation.std

    def finish(self):
        self.run.finish()

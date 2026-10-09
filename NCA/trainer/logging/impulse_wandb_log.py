import numpy as np
from einops import rearrange

from Common.trainer.wandb_logger import WandbLogger


def _rgb(states):
    """First three channels of [..., C, H, W] states, clipped to [0, 1]."""

    return np.clip(np.asarray(states)[..., :3, :, :], 0.0, 1.0)


class ImpulseLogger(WandbLogger):
    """W&B logging for impulse optimisation runs.

    ``data`` is the [batch, time, channels, H, W] trajectory array the pair
    source was built from. It is logged once at the start.
    """

    def __init__(self, data, observed_channels, wandb_config):
        self.observed_channels = observed_channels
        super().__init__(np.asarray(data)[:, :, :observed_channels], wandb_config)

    def log_training(self, metrics, step, log_every):
        """Log the scalar losses and intervention size of one optimisation step."""

        scalars = {
            "Loss/total": float(metrics["total_loss"]),
            "Loss/target": float(metrics["target_loss"]),
            "Loss/regulariser": float(metrics["regulariser"]),
        }
        for name, value in metrics["intervention_metrics"].items():
            scalars[f"Intervention/{name}"] = float(value)
        self.log_scalars(scalars, step=step)

    def log_result(self, result):
        """Log the best intervention: summary losses, states and trajectories."""

        self.log_scalars(
            {
                "Result/best_loss": result.best_loss,
                "Result/best_step": result.best_step,
                "Result/evaluation_loss": result.evaluation_loss,
                "Result/baseline_evaluation_loss": result.baseline_evaluation_loss,
            }
        )
        # One row per batch element: source, perturbed source, final state, target
        states = np.stack(
            [
                result.initial_states,
                result.perturbed_initial_states,
                result.final_states,
                result.target_states,
            ],
            axis=1,
        )
        self.log_image(
            "Result/source | perturbed | final | target",
            rearrange(_rgb(states), "B S C H W -> (B H) (S W) C"),
        )
        # Size of the applied change, summed over channels, scaled to [0, 1]
        delta = np.abs(
            np.asarray(result.perturbed_initial_states) - np.asarray(result.initial_states)
        ).sum(axis=1)
        delta = delta / max(float(delta.max()), 1e-12)
        self.log_image(
            "Result/intervention magnitude",
            rearrange(delta, "B H W -> (B H) W")[..., None].repeat(3, axis=-1),
        )
        for name, trajectory in (
            ("baseline", result.baseline_trajectory),
            ("perturbed", result.perturbed_trajectory),
        ):
            if trajectory.shape[1] > 0:
                self.log_video(f"Result/{name} trajectory", _rgb(trajectory[0]))

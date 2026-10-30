import numpy as np
from einops import rearrange

from Common.trainer.wandb_logger import WandbLogger
from NCA.trainer.logging.diagnostics import plot_to_image


def _rgb(states):
    """First three channels of [..., C, H, W] states, clipped to [0, 1]."""

    return np.clip(np.asarray(states)[..., :3, :, :], 0.0, 1.0)


def plot_location_path(location_history, best_step, source, target):
    """Path of the impulse centre during optimisation, drawn over the source and target.

    ``location_history`` is [steps, 2] (row, column) in 0 to 1 coordinates;
    ``source`` and ``target`` are [C, H, W] states. Returns an image array for W&B.
    """
    import matplotlib.pyplot as plt

    height, width = source.shape[-2:]
    rows = np.asarray(location_history)[:, 0] * (height - 1)
    cols = np.asarray(location_history)[:, 1] * (width - 1)
    steps = np.arange(len(rows))
    best = min(max(best_step, 0), len(rows) - 1)

    figure, axes = plt.subplots(1, 2, figsize=(10, 5.4), dpi=120)
    for axis, state, title in ((axes[0], source, "source"), (axes[1], target, "target")):
        axis.imshow(rearrange(_rgb(state), "C H W -> H W C"))
        axis.plot(cols, rows, color="white", linewidth=0.8, alpha=0.6)
        path = axis.scatter(cols, rows, c=steps, cmap="viridis", s=6, zorder=2)
        axis.scatter(cols[0], rows[0], marker="o", s=60, facecolor="none", edgecolor="black", label="start", zorder=3)
        axis.scatter(cols[best], rows[best], marker="*", s=120, color="red", edgecolor="black", label=f"best (step {best})", zorder=3)
        axis.scatter(cols[-1], rows[-1], marker="X", s=60, color="white", edgecolor="black", label="final", zorder=3)
        axis.set_xlim(-0.5, width - 0.5)
        axis.set_ylim(height - 0.5, -0.5)
        axis.set_title(title)
    axes[1].legend(loc="lower right", fontsize=8)
    figure.colorbar(path, ax=axes, label="optimisation step", shrink=0.8)
    return plot_to_image(figure)[0]


def plot_outcome_heatmap(history, names):
    """Where validation rollouts end up over optimisation: one row per outcome
    (attractors, then other, died, diverged), one column per validation step.

    ``history`` is a list of (step, fractions). Returns an image array for W&B.
    """
    import matplotlib.pyplot as plt

    steps = [step for step, _ in history]
    fractions = np.stack([values for _, values in history], axis=1)
    figure, axis = plt.subplots(figsize=(max(6.0, 0.3 * len(steps) + 3), 0.45 * len(names) + 1.5), dpi=120)
    image = axis.imshow(fractions, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
    axis.set_yticks(range(len(names)), names)
    ticks = np.linspace(0, len(steps) - 1, min(len(steps), 10)).round().astype(int)
    axis.set_xticks(ticks, [steps[index] for index in ticks])
    axis.set_xlabel("optimisation step")
    axis.set_title("Validation rollouts ending at each outcome")
    figure.colorbar(image, ax=axis, label="fraction of rollouts")
    figure.tight_layout()
    return plot_to_image(figure)[0]


def plot_loss_curve(perturbed, baseline, frame_every, rollout_steps=None, switch_steps=None):
    """Mean target loss over the evaluation rollout, with and without the impulse.

    The training rollout length and the mean switch time are marked when given.
    Returns an image array for W&B.
    """
    import matplotlib.pyplot as plt

    steps = (np.arange(len(perturbed)) + 1) * frame_every
    figure, axis = plt.subplots(figsize=(7, 4), dpi=120)
    axis.plot(steps, perturbed, label="with impulse")
    axis.plot(steps, baseline, label="baseline", linestyle="--")
    if rollout_steps is not None:
        axis.axvline(rollout_steps, color="grey", linewidth=0.8, label="training rollout")
    if switch_steps is not None and np.isfinite(switch_steps).any():
        axis.axvline(np.nanmean(switch_steps), color="red", linewidth=0.8, linestyle=":", label="mean switch time")
    axis.set_yscale("log")
    axis.set_xlabel("NCA step")
    axis.set_ylabel("target loss")
    axis.legend()
    axis.grid(True, which="both", alpha=0.25)
    figure.tight_layout()
    return plot_to_image(figure)[0]


class ImpulseLogger(WandbLogger):
    """W&B logging for impulse optimisation runs.

    ``data`` is the [batch, time, channels, H, W] trajectory array the pair
    source was built from; its targets are logged once at the start.
    ``attractor_names`` labels the validation outcomes of a multi-attractor
    model. Batch elements share one intervention, so images show the first only.
    """

    def __init__(self, data, observed_channels, wandb_config, attractor_names=None):
        self.observed_channels = observed_channels
        self.outcome_names = None if attractor_names is None else [*attractor_names, "other", "died", "diverged"]
        # (step, outcome fractions) of each validation, for the heatmap at the end
        self.outcome_history = []
        super().__init__(np.asarray(data)[:, :, :observed_channels], wandb_config)

    def log_data_at_init(self, data):
        """Log the target of every data pair, side by side."""

        self.log_image("Data/targets", rearrange(_rgb(data[:, -1]), "B C H W -> H (B W) C"))

    def log_training(self, metrics, step, log_every):
        """Log the losses, intervention size, gradient norms and switch rate of one optimisation step."""

        scalars = {
            "Loss/total": float(metrics["total_loss"]),
            "Loss/target": float(metrics["target_loss"]),
            "Loss/regulariser": float(metrics["regulariser"]),
        }
        for name, value in metrics["intervention_metrics"].items():
            scalars[f"Intervention/{name}"] = float(value)
        for name, value in metrics.get("gradient_norms", {}).items():
            scalars[f"Gradient/{name}"] = float(value)
        if "contrast" in metrics:
            scalars["Loss/contrast"] = float(metrics["contrast"])
        if "switch_rate" in metrics:
            scalars["Training/switch_rate"] = float(metrics["switch_rate"])
        # The curriculum stage: Loss/target jumps when it lengthens
        if "rollout_steps" in metrics:
            scalars["Training/rollout_steps"] = int(metrics["rollout_steps"])
        self.log_scalars(scalars, step=step)

    def log_validation(self, values, step):
        """Log the switch rate, target loss and outcomes of the current impulse on the validation batch."""

        values = dict(values)
        outcomes = values.pop("outcomes", None)
        scalars = {f"Validation/{name}": float(value) for name, value in values.items()}
        if outcomes is not None and self.outcome_names is not None:
            self.outcome_history.append((step, np.asarray(outcomes)))
            for name, fraction in zip(self.outcome_names, outcomes):
                scalars[f"Validation/ends at/{name}"] = float(fraction)
        self.log_scalars(scalars, step=step)

    def log_result(self, result, pixel_steps=None, frame_every=1, rollout_steps=None):
        """Log the best intervention: summary losses, states and trajectories.

        ``pixel_steps`` (k * k * duration of a patch impulse) groups runs by budget.
        ``frame_every`` is the step between saved evaluation frames and
        ``rollout_steps`` the training rollout length, marked on the loss curve.
        """

        scalars = {
            "Result/best_loss": result.best_loss,
            "Result/best_step": result.best_step,
            "Result/evaluation_loss": result.evaluation_loss,
            "Result/baseline_evaluation_loss": result.baseline_evaluation_loss,
            "Result/switch_rate": result.evaluation_switch_rate,
            "Result/baseline_switch_rate": result.baseline_switch_rate,
        }
        if result.best_validation_objective is not None:
            scalars["Result/best_validation_objective"] = result.best_validation_objective
        if result.restart_validation_objectives is not None:
            for index, value in enumerate(result.restart_validation_objectives):
                scalars[f"Result/restart {index} validation_objective"] = float(value)
        if pixel_steps is not None:
            scalars["Result/pixel_steps"] = pixel_steps
        if result.switch_steps is not None and np.isfinite(result.switch_steps).any():
            # Mean over the rollouts that switched
            scalars["Result/switch_time"] = float(np.nanmean(result.switch_steps))
        self.log_scalars(scalars)
        states = np.stack(
            [
                result.initial_states[0],
                result.perturbed_initial_states[0],
                result.final_states[0],
                result.target_states[0],
            ]
        )
        self.log_image(
            "Result/source | perturbed | final | target",
            rearrange(_rgb(states), "S C H W -> H (S W) C"),
        )
        # Size of the applied change, summed over channels, scaled to [0, 1].
        # The intervention is the same for every batch element, so show the first.
        delta = np.abs(
            np.asarray(result.perturbed_initial_states[0]) - np.asarray(result.initial_states[0])
        ).sum(axis=0)
        delta = delta / max(float(delta.max()), 1e-12)
        self.log_image(
            "Result/intervention magnitude",
            delta[..., None].repeat(3, axis=-1),
        )
        if result.location_history is not None:
            self.log_image(
                "Result/impulse centre path",
                plot_location_path(
                    result.location_history,
                    result.best_step,
                    result.initial_states[0],
                    result.target_states[0],
                ),
            )
        if self.outcome_history:
            self.log_image(
                "Result/validation outcomes",
                plot_outcome_heatmap(self.outcome_history, self.outcome_names),
            )
        if result.evaluation_loss_curve is not None and len(result.evaluation_loss_curve):
            self.log_image(
                "Result/target loss over evaluation",
                plot_loss_curve(
                    result.evaluation_loss_curve,
                    result.baseline_loss_curve,
                    frame_every,
                    rollout_steps,
                    result.switch_steps,
                ),
            )
        for name, trajectory in (
            ("baseline", result.baseline_trajectory),
            ("perturbed", result.perturbed_trajectory),
        ):
            if trajectory.shape[1] > 0:
                self.log_video(f"Result/{name} trajectory", _rgb(trajectory[0]))

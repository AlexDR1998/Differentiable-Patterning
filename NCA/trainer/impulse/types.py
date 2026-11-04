from dataclasses import dataclass, field
from typing import Any

from jaxtyping import Array


@dataclass
class ImpulseBatch:
    """Initial and target states used for one intervention optimisation batch."""

    initial_states: Array
    target_states: Array
    metadata: dict[str, Any] = field(default_factory=dict)
    # Source state a switch is scored against (None: initial_states). Set when
    # the impulse acts before the source pattern has fully grown.
    source_states: Array | None = None

    @property
    def switch_sources(self):
        """States a rollout must move away from to count as switched."""

        return self.initial_states if self.source_states is None else self.source_states


@dataclass
class ImpulseResult:
    """Best intervention and evaluation arrays returned by an optimisation run."""

    best_intervention: Any
    best_step: int
    best_loss: float
    metrics: dict[str, Any]
    initial_states: Array
    target_states: Array
    perturbed_initial_states: Array
    final_states: Array
    baseline_trajectory: Array
    perturbed_trajectory: Array
    # Mean target loss at the end of the evaluation rollout.
    evaluation_loss: float
    baseline_evaluation_loss: float
    # Fraction of evaluation rollouts that end nearer the target than the
    # source, with and without the intervention.
    evaluation_switch_rate: float
    baseline_switch_rate: float
    # [iterations + 1, 2] centre (row, column, 0 to 1) of local and patch
    # interventions: row i is the centre used at step i, the last row the
    # final centre. None for global and flat interventions.
    location_history: Any = None
    # [batch] NCA step from which each evaluation rollout stays nearer the
    # target than the source; NaN for rollouts that did not switch.
    switch_steps: Any = None
    # [frames] mean target loss after every evaluation_frame_every steps of
    # the evaluation rollout, with and without the intervention. None when the
    # loss uses channels the saved trajectories do not keep.
    evaluation_loss_curve: Any = None
    baseline_loss_curve: Any = None
    # Objective of the best intervention on the validation batch, after the
    # evaluation rollout (how the best intervention is chosen).
    best_validation_objective: float | None = None
    # [restarts] validation objective of each start after scouting (None
    # without restarts)
    restart_validation_objectives: Any = None
    # Mature source states the switch is scored against, when the impulse
    # acts during development (None: initial_states)
    source_states: Any = None


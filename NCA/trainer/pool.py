"""Recurrent training-pool admission policy."""

from dataclasses import dataclass


@dataclass
class PoolAdmissionState:
    loss_ema: float | None = None
    previous_admitted_loss: float | None = None
    admitted: int = 0
    rejected: int = 0


@dataclass(frozen=True)
class PoolDecision:
    admit: bool
    reject_relative: bool
    reject_previous_relative: bool
    reject_absolute: bool
    loss_reference: float
    loss_ratio: float
    previous_loss_reference: float
    previous_loss_ratio: float


class PoolAdmissionController:
    def __init__(self, config, default_warmup: int):
        self.config = config
        self.warmup = default_warmup if config.warmup is None else config.warmup
        self.state = PoolAdmissionState()

    def decide(self, loss: float, iteration: int) -> PoolDecision:
        state = self.state
        reference = loss if state.loss_ema is None else state.loss_ema
        previous = (
            loss
            if state.previous_admitted_loss is None
            else state.previous_admitted_loss
        )
        ratio = loss / max(reference, 1e-12)
        previous_ratio = loss / max(previous, 1e-12)
        compare_ema = (
            self.config.enabled
            and state.loss_ema is not None
            and iteration >= self.warmup
        )
        reject_relative = compare_ema and ratio > self.config.relative_threshold
        reject_previous = (
            self.config.enabled
            and state.previous_admitted_loss is not None
            and iteration >= self.warmup
            and previous_ratio > self.config.previous_relative_threshold
        )
        reject_absolute = (
            compare_ema
            and self.config.absolute_threshold is not None
            and loss > reference + self.config.absolute_threshold
        )
        return PoolDecision(
            admit=not (reject_relative or reject_previous or reject_absolute),
            reject_relative=reject_relative,
            reject_previous_relative=reject_previous,
            reject_absolute=reject_absolute,
            loss_reference=reference,
            loss_ratio=ratio,
            previous_loss_reference=previous,
            previous_loss_ratio=previous_ratio,
        )

    def update(self, decision: PoolDecision, loss: float) -> None:
        if not decision.admit:
            self.state.rejected += 1
            return
        state = self.state
        state.admitted += 1
        state.previous_admitted_loss = loss
        if not self.config.enabled:
            return
        state.loss_ema = (
            loss
            if state.loss_ema is None
            else self.config.ema_decay * state.loss_ema
            + (1 - self.config.ema_decay) * loss
        )

    def metrics(self, decision: PoolDecision) -> dict[str, float | int]:
        return {
            "pool/admit": int(decision.admit),
            "pool/reject": int(not decision.admit),
            "pool/reject_relative": int(decision.reject_relative),
            "pool/reject_previous_relative": int(decision.reject_previous_relative),
            "pool/reject_absolute": int(decision.reject_absolute),
            "pool/loss_ref": decision.loss_reference,
            "pool/loss_ratio": decision.loss_ratio,
            "pool/previous_loss_ref": decision.previous_loss_reference,
            "pool/previous_loss_ratio": decision.previous_loss_ratio,
            "pool/admit_count": self.state.admitted,
            "pool/reject_count": self.state.rejected,
        }

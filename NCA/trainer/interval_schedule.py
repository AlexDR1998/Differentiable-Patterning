"""Static per-transition timing for NCA rollouts between observed images.

Training rolls every transition ``k -> k+1`` out in parallel as one slot along
the vmapped time axis. An :class:`IntervalSchedule` says how much NCA time each
slot represents. It is resolved once from the observation times and holds only
Python scalars/tuples, so the training step still compiles exactly once.

Modes
-----
``uniform``
    Every slot runs ``t`` steps (the historical behaviour; observation times
    are ignored).
``steps``
    Slot ``i`` runs ``max(1, round(t * dt_i / dt_ref))`` steps, so ``t`` is the
    step count of the reference interval (median interval by default).
``fire_rate`` / ``dt``
    Every slot runs ``t`` steps with a per-slot multiplier ``scale`` on the
    stochastic fire rate (``dt_i / max dt``) or on the update size
    (``dt_i / dt_ref``). Resolved here; not yet supported by the trainer.
"""

from dataclasses import dataclass
from itertools import accumulate
import bisect
import statistics

INTERVAL_MODES = ("uniform", "steps", "fire_rate", "dt")


@dataclass(frozen=True)
class IntervalSchedule:
    mode: str
    steps: tuple[int, ...]
    scale: tuple[float, ...]
    times: tuple[float, ...] | None = None

    def __post_init__(self):
        if self.mode not in INTERVAL_MODES:
            raise ValueError(f"interval mode must be one of {INTERVAL_MODES}, got {self.mode!r}")
        if not self.steps or any(int(step) < 1 for step in self.steps):
            raise ValueError(f"interval steps must be positive integers, got {self.steps}")
        if len(self.scale) != len(self.steps):
            raise ValueError("interval scale must have one entry per slot")

    @property
    def n_slots(self):
        return len(self.steps)

    @property
    def scan_length(self):
        """Number of scan iterations needed to finish every slot."""
        return max(self.steps)

    @property
    def is_uniform(self):
        """True when every slot runs the same unscaled rollout."""
        return len(set(self.steps)) == 1 and all(value == 1.0 for value in self.scale)

    @property
    def observation_steps(self):
        """Cumulative NCA step at which each image is observed in a sequential rollout."""
        return (0, *accumulate(self.steps))

    @property
    def total_steps(self):
        return self.observation_steps[-1]

    def slot_for_time(self, time):
        """Index of the transition slot that starts at or before ``time``.

        On a uniform 12 h grid this equals ``time // 12`` (the convention used by
        knockout masks). Times after the last observation map to ``n_slots``.
        """
        if self.times is None:
            raise ValueError("slot_for_time requires a schedule built with observation times")
        return min(max(bisect.bisect_right(self.times, time) - 1, 0), self.n_slots)

    def step_for_time(self, time):
        """Cumulative step at the start of the slot containing ``time``."""
        return self.observation_steps[self.slot_for_time(time)]

    def step_at_time(self, time):
        """Continuous rollout step at ``time``, piecewise linear between images.

        On a uniform grid with spacing ``dt`` this is ``time * t / dt`` (e.g.
        ``hours * t / 12``). Times beyond the last image extrapolate at the final
        slot's rate.
        """
        if self.times is None:
            raise ValueError("step_at_time requires a schedule built with observation times")
        slot = min(max(bisect.bisect_right(self.times, time) - 1, 0), len(self.times) - 2, self.n_slots - 1)
        span = self.times[slot + 1] - self.times[slot]
        return self.observation_steps[slot] + (time - self.times[slot]) * self.steps[slot] / span

    def time_at_step(self, step):
        """Inverse of :meth:`step_at_time`."""
        if self.times is None:
            raise ValueError("time_at_step requires a schedule built with observation times")
        slot = min(max(bisect.bisect_right(self.observation_steps, step) - 1, 0), len(self.times) - 2, self.n_slots - 1)
        span = self.times[slot + 1] - self.times[slot]
        return self.times[slot] + (step - self.observation_steps[slot]) * span / self.steps[slot]

    def for_slot(self, slot):
        """Single-slot schedule, e.g. for sequential rollouts one transition at a time."""
        return IntervalSchedule(self.mode, (self.steps[slot],), (self.scale[slot],))


def uniform_schedule(t, n_slots, times=None):
    return IntervalSchedule(
        "uniform", (int(t),) * n_slots, (1.0,) * n_slots,
        None if times is None else tuple(float(value) for value in times),
    )


def build_interval_schedule(
    t,
    n_slots,
    mode="uniform",
    times=None,
    reference_interval=None,
    explicit_steps=None,
):
    """Resolve the per-slot schedule for ``n_slots`` transition slots.

    ``times`` are observation times (any unit) for the ``T`` images, giving
    ``T - 1`` intervals. ``n_slots`` may be one larger than that, for the
    duplicated "steady" final slot; that slot gets the reference interval.
    ``explicit_steps`` overrides the proportional rule in ``steps`` mode.
    """
    t = int(t)
    if t < 1:
        raise ValueError("t must be positive")
    if mode not in INTERVAL_MODES:
        raise ValueError(f"interval mode must be one of {INTERVAL_MODES}, got {mode!r}")
    if mode == "uniform":
        if explicit_steps is not None:
            raise ValueError("interval_steps requires a non-uniform interval mode")
        return uniform_schedule(t, n_slots, times)

    if explicit_steps is not None:
        if mode != "steps":
            raise ValueError("interval_steps is only supported with interval mode 'steps'")
        steps = tuple(int(step) for step in explicit_steps)
        if len(steps) != n_slots:
            raise ValueError(f"interval_steps has {len(steps)} entries for {n_slots} slots")
        return IntervalSchedule(mode, steps, (1.0,) * n_slots, None if times is None else tuple(times))

    if times is None:
        raise ValueError(f"interval mode {mode!r} requires observation times")
    times = tuple(float(value) for value in times)
    intervals = [later - earlier for earlier, later in zip(times, times[1:])]
    if not intervals or any(interval <= 0 for interval in intervals):
        raise ValueError(f"observation times must be strictly increasing, got {times}")
    if reference_interval is None:
        reference_interval = statistics.median(intervals)
    if reference_interval <= 0:
        raise ValueError("reference_interval must be positive")
    if n_slots == len(intervals) + 1:
        intervals.append(reference_interval)  # duplicated steady final slot
    elif n_slots != len(intervals):
        raise ValueError(
            f"{len(times)} observation times give {len(intervals)} intervals, "
            f"but the rollout has {n_slots} slots"
        )

    if mode == "steps":
        steps = tuple(max(1, round(t * interval / reference_interval)) for interval in intervals)
        scale = (1.0,) * n_slots
    elif mode == "fire_rate":
        steps = (t,) * n_slots
        scale = tuple(interval / max(intervals) for interval in intervals)
    else:  # dt
        steps = (t,) * n_slots
        scale = tuple(interval / reference_interval for interval in intervals)
    return IntervalSchedule(mode, steps, scale, times)


def interval_schedule_from_config(config, n_slots, observation_times=None):
    """Rebuild a training run's schedule from its saved experiment config.

    Configs saved before interval modes existed resolve to ``uniform``, so
    evaluation code reproduces historical ``k * t`` indexing.
    """
    loop = config.training.loop
    return build_interval_schedule(
        loop.t,
        n_slots,
        mode=getattr(loop, "interval_mode", "uniform"),
        times=observation_times,
        reference_interval=getattr(loop, "reference_interval", None),
        explicit_steps=getattr(loop, "interval_steps", None),
    )


__all__ = [
    "INTERVAL_MODES",
    "IntervalSchedule",
    "build_interval_schedule",
    "interval_schedule_from_config",
    "uniform_schedule",
]

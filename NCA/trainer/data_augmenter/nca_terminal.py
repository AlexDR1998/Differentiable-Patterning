import jax

from NCA.trainer.data_augmenter import scheduled_probability, terminal_carry
from NCA.trainer.data_augmenter.nca_basic import DataAugmenter as BasicDataAugmenter
from NCA.trainer.data_augmenter.nca_basic import jittable_callback_bit


class TerminalCarryDataAugmenter(BasicDataAugmenter):
    """Basic NCA augmenter that can recycle terminal predictions over long rollouts."""

    TERMINAL_CARRY_ENABLED = False
    TERMINAL_CARRY_START = 0
    TERMINAL_CARRY_SCHEDULE = 0
    TERMINAL_CARRY_INITIAL = 0.0
    TERMINAL_CARRY_FINAL = 0.0

    @staticmethod
    def scheduled_probability(i, start, schedule, initial, final):
        """Linearly move from ``initial`` to ``final`` after ``start``."""

        return scheduled_probability(i, start, schedule, initial, final)

    def terminal_carry_probability(self, i):
        """Return the configured terminal carry probability at iteration ``i``."""

        if not self.TERMINAL_CARRY_ENABLED:
            return 0.0
        return self.scheduled_probability(
            i,
            self.TERMINAL_CARRY_START,
            self.TERMINAL_CARRY_SCHEDULE,
            self.TERMINAL_CARRY_INITIAL,
            self.TERMINAL_CARRY_FINAL,
        )

    def propagate_with_terminal_carry(self, x, x_true, i, key):
        """Propagate the training pool and optionally retain each terminal state."""

        terminal_states = [trajectory[-1] for trajectory in x]
        x = jittable_callback_bit(x, x_true, self.OBS_CHANNELS, key)
        return terminal_carry(
            x,
            terminal_states,
            self.terminal_carry_probability(i),
            jax.random.fold_in(key, 1),
        )

    def advance_pool(self, x, y, i, key):
        """Apply pool propagation, terminal carry, and the standard small noise."""

        x_true, _ = self.split_x_y(1)
        x = self.propagate_with_terminal_carry(x, x_true, i, key)
        x = self.noise(x, 0.005, key=key)
        self.PREVIOUS_KEY = key
        return x, y

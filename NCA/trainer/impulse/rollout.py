import equinox as eqx
import jax
import jax.numpy as jnp


def identity_boundary(x):
    """Return an NCA state unchanged."""

    return x


def run_nca_batch(
    model,
    initial_states,
    steps,
    key,
    boundary_callback=identity_boundary,
    return_trajectory=False,
    scan_kind="lax",
):
    """Run an NCA over a batch with independent random keys.

    Parameters
    ----------
    model
        NCA callable with signature ``model(state, boundary_callback, key)``.
    initial_states
        Array with shape ``[batch, channels, height, width]``.
    steps
        Number of NCA updates to apply.
    key
        JAX random key used to derive timestep and batch keys.
    boundary_callback
        Function applied by the model after each update.
    return_trajectory
        If true, also return states with shape ``[batch, steps, channels, height, width]``.
    scan_kind
        Equinox scan mode, normally ``"lax"`` or ``"checkpointed"``.
    """

    if steps < 0:
        raise ValueError("steps must be non-negative")
    if steps == 0:
        empty = jnp.empty((len(initial_states), 0, *initial_states.shape[1:]), dtype=initial_states.dtype)
        return (initial_states, empty) if return_trajectory else initial_states

    batched_model = jax.vmap(model, in_axes=(0, None, 0), out_axes=0)

    def step(carry, step_index):
        step_key, states = carry
        step_key = jax.random.fold_in(step_key, step_index)
        batch_keys = jax.random.split(step_key, len(states))
        states = batched_model(states, boundary_callback, batch_keys)
        return (step_key, states), states

    (_, final_states), trajectory = eqx.internal.scan(
        step,
        (key, initial_states),
        xs=jnp.arange(steps),
        kind=scan_kind,
    )
    if not return_trajectory:
        return final_states
    trajectory = jnp.swapaxes(trajectory, 0, 1)
    return final_states, trajectory



@eqx.filter_jit
def run_nca_batch_to_steps(
    model,
    initial_states,
    steps,
    max_steps,
    key,
    boundary_callback=identity_boundary,
    scan_kind="lax",
):
    """Run each batch element for its own number of NCA updates.

    ``steps`` is an integer array with one entry per batch element, each at
    most ``max_steps``. All elements run in one compiled rollout of
    ``max_steps`` updates. An element stops changing once it has had its
    number of updates, so ``max_steps`` (a Python int) sets the compiled
    length, not the sampled step counts.
    """

    batched_model = jax.vmap(model, in_axes=(0, None, 0), out_axes=0)

    def step(states, step_index):
        step_key = jax.random.fold_in(key, step_index)
        batch_keys = jax.random.split(step_key, len(states))
        updated = batched_model(states, boundary_callback, batch_keys)
        running = (step_index < steps)[:, None, None, None]
        return jnp.where(running, updated, states), None

    final_states, _ = eqx.internal.scan(
        step,
        initial_states,
        xs=jnp.arange(max_steps),
        kind=scan_kind,
    )
    return final_states

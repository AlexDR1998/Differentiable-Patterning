"""Local inference for inspecting impulse results on multi-attractor NCA models.

Used by ``Experiments/impulse/impulse_explorer.py`` (impulse records) and
``impulse_sweep_explorer.py`` (old flat result files, see below). Every function runs the frozen NCA forward
only, so it is cheap enough for a laptop. States are [batch, channels, H, W];
reference attractors are [pattern, channels, H, W].
"""

import json
import warnings
import zipfile
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from Experiments.config import load_experiment_config
from Experiments.config_helpers import open_registry_bundle
from Experiments.impulse.config_helpers import intervention_from_config, load_impulse_data, pixel_steps
from NCA.trainer.impulse.rollout import run_nca_batch, run_with_intervention

# steps and return_trajectory are Python values, so they are static here
rollout = eqx.filter_jit(run_nca_batch)
rollout_with_intervention = eqx.filter_jit(run_with_intervention)


def _rollout(model, intervention, states, steps, key, **kwargs):
    """Roll out ``states``, with a (possibly multi-step) intervention unless it is None."""

    if intervention is None:
        return rollout(model, states, steps, key, **kwargs)
    return rollout_with_intervention(model, intervention, states, steps, key, **kwargs)


def load_model_and_data(cfg, store_root=None, key=None):
    """Model, [pattern, time, C, H, W] data and observed channel count for one run config."""

    bundle = open_registry_bundle(cfg.checkpoint.model_id, store_root or cfg.checkpoint.store_root)
    model = bundle.load_model(key=key)
    data_config = bundle.config.data if cfg.data is None else cfg.data
    trajectories = load_impulse_data(data_config, model)
    return model, trajectories, data_config.emoji.observed_channels


# ---------------------------------------------------------------------------
# Old flat result files (<name>.eqx/.json/.npz in one folder), written before
# impulse records. Only impulse_sweep_explorer.py reads them.
# ---------------------------------------------------------------------------

def load_impulse_results(directory):
    """Read every old-format impulse run (``*.json`` with its ``.eqx``) below ``directory``."""

    results = []
    for path in sorted(Path(directory).expanduser().rglob("*.json")):
        summary = json.loads(path.read_text())
        if "config" not in summary or not path.with_suffix(".eqx").is_file():
            continue
        try:
            cfg = load_experiment_config(summary["config"])
        except ValueError as error:
            # e.g. a run saved with an older impulse config layout
            warnings.warn(f"Skipping {path.name}: {error}")
            continue
        results.append(
            {
                "name": path.stem,
                "path": path.with_suffix(".eqx"),
                "model_id": summary["model_id"],
                "source": cfg.impulse.pair_source.source_index,
                "target": cfg.impulse.pair_source.target_index,
                "summary": summary,
                "config": cfg,
            }
        )
    return results


def _saved_shape(npz_path, name):
    """Shape of one array in an ``.npz`` file, read from its header without loading it."""

    with zipfile.ZipFile(npz_path) as archive, archive.open(f"{name}.npy") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            return np.lib.format.read_array_header_1_0(handle)[0]
        return np.lib.format.read_array_header_2_0(handle)[0]


def load_saved_intervention(result):
    """Rebuild an old-format saved intervention without its model or data.

    The lattice and channel counts come from the run's saved states (``.npz``).
    Returns the intervention and the run's [batch, C, H, W] start states.
    """

    npz_path = result["path"].with_suffix(".npz")
    with np.load(npz_path) as arrays:
        initial_states = jnp.asarray(arrays["initial_states"])
    # The saved trajectories keep the observed channels only
    observed_channels = _saved_shape(npz_path, "perturbed_trajectory")[2]
    skeleton = intervention_from_config(
        result["config"].impulse.intervention,
        observed_channels,
        initial_states.shape[1:],
        jax.random.PRNGKey(0),
    )
    return eqx.tree_deserialise_leaves(result["path"], skeleton), initial_states


def summary_row(result):
    """A run's settings and scores as one flat row, for comparing the runs of sweeps."""

    cfg = result["config"]
    impulse = cfg.impulse
    intervention = impulse.intervention
    summary = result["summary"]
    return {
        "run": result["name"],
        "sweep": cfg.experiment.name,
        "model_id": result["model_id"],
        "transition": f"{result['source']} -> {result['target']}",
        "objective": impulse.objective.type,
        "channels": intervention.channels,
        "spatial": intervention.spatial,
        "width": intervention.width if intervention.spatial == "local" else None,
        "patch_size": intervention.patch_size if intervention.spatial == "patch" else None,
        "duration": intervention.duration,
        "budget": pixel_steps(intervention),
        "l2": dict(impulse.loss.regularisers).get("l2", 0.0),
        "learn_rate": impulse.optimiser.learn_rate,
        "rollout_steps": impulse.rollout.steps,
        "loss_window": impulse.rollout.loss_window,
        "iterations": impulse.iterations,
        "repeat": impulse.repeat,
        "switch_rate": summary.get("evaluation_switch_rate"),
        "baseline_switch_rate": summary.get("baseline_switch_rate"),
        "evaluation_loss": summary.get("evaluation_loss"),
        "baseline_evaluation_loss": summary.get("baseline_evaluation_loss"),
        "best_loss": summary.get("best_loss"),
        "best_step": summary.get("best_step"),
    }


def scale_intervention(intervention, factor):
    """The same intervention with its values multiplied by ``factor``."""

    return eqx.tree_at(lambda item: item.values, intervention, intervention.values * factor)


def move_intervention(intervention, centre):
    """A local intervention moved to ``centre`` (normalised (y, x) in (0, 1))."""

    location = jax.scipy.special.logit(jnp.clip(jnp.asarray(centre, dtype=float), 1e-4, 1 - 1e-4))
    return eqx.tree_at(lambda item: item.location, intervention, location)


def grow_attractors(model, initial_states, steps, samples, key):
    """Grow each pattern ``samples`` times: returns [pattern, sample, C, H, W]."""

    patterns = len(initial_states)
    batch = jnp.repeat(initial_states, samples, axis=0)
    grown = rollout(model, batch, steps, key)
    return grown.reshape((patterns, samples, *grown.shape[1:]))


def attractor_distances(states, references, observed_channels):
    """RMS distance over observed channels: [batch, pattern]."""

    difference = states[:, None, :observed_channels] - references[None, :, :observed_channels]
    return jnp.sqrt(jnp.mean(difference**2, axis=(-3, -2, -1)))


def switch_matrix(model, intervention, attractors, references, steps, observed_channels, key):
    """Fraction of rollouts from each pattern that end nearest each pattern.

    ``attractors`` is [pattern, sample, C, H, W], as from ``grow_attractors``.
    Row k is the source pattern, column j the pattern its final state is
    nearest to. ``intervention=None`` gives the unperturbed baseline.
    """

    patterns, samples = attractors.shape[:2]
    states = attractors.reshape((-1, *attractors.shape[2:]))
    final = _rollout(model, intervention, states, steps, key)
    labels = jnp.argmin(attractor_distances(final, references, observed_channels), axis=1)
    one_hot = jax.nn.one_hot(labels.reshape((patterns, samples)), len(references))
    return jnp.mean(one_hot, axis=1)


def distance_over_time(model, intervention, states, references, steps, observed_channels, key):
    """Distances to every reference along the rollout: [batch, time, pattern], plus the trajectory."""

    _, trajectory = _rollout(model, intervention, states, steps, key, return_trajectory=True)
    batch, time = trajectory.shape[:2]
    flat = trajectory.reshape((-1, *trajectory.shape[2:]))
    distances = attractor_distances(flat, references, observed_channels)
    return distances.reshape((batch, time, len(references))), trajectory


def success_rate(model, interventions, states, references, target, steps, observed_channels, key):
    """For each intervention, the fraction of ``states`` that end nearest ``target``."""

    rates = []
    for index, intervention in enumerate(interventions):
        final = _rollout(model, intervention, states, steps, jax.random.fold_in(key, index))
        labels = jnp.argmin(attractor_distances(final, references, observed_channels), axis=1)
        rates.append(float(jnp.mean(labels == target)))
    return jnp.asarray(rates)


def add_state_noise(states, strength, key):
    """Gaussian noise of standard deviation ``strength`` on every channel."""

    return states + strength * jax.random.normal(key, states.shape, states.dtype)

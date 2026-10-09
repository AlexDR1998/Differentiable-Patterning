# /// script
# dependencies = [
#   "marimo",
#   "matplotlib",
#   "numpy",
# ]
# ///

"""Inspect impulse results on a multi-attractor emoji NCA by local inference.

Run from the repository root with:

    marimo edit Experiments/impulse/impulse_explorer.py

Reads the ``.json``/``.eqx`` pairs written by ``Experiments/impulse/optimise.py``
(``IMPULSE_OUTPUT_PATH``) and the model bundles they name (``MODEL_STORE_ROOT``,
default ``models/``). Emoji images are read from ``DATA_PATH_BASE``. The
computations are in ``Experiments/impulse/analysis.py``.
"""

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")

with app.setup(hide_code=True):
    import os

    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from Experiments.impulse.analysis import (
        add_state_noise,
        distance_over_time,
        grow_attractors,
        load_impulse_results,
        load_intervention,
        load_model_and_data,
        move_intervention,
        scale_intervention,
        success_rate,
        switch_matrix,
    )

    plt.style.use("default")

    def rgb(state):
        """[C, H, W] state -> [H, W, 3] image of its first three channels."""
        return np.clip(np.moveaxis(np.asarray(state[:3]), 0, -1), 0.0, 1.0)

    def show_matrix(axis, matrix, title):
        axis.imshow(np.asarray(matrix), vmin=0, vmax=1, cmap="Blues")
        size = len(matrix)
        axis.set_xticks(range(size))
        axis.set_yticks(range(size))
        axis.set_xlabel("ends nearest")
        axis.set_ylabel("source")
        axis.set_title(title, fontsize=9)
        for row in range(size):
            for column in range(size):
                axis.text(column, row, f"{float(matrix[row, column]):.2f}", ha="center", va="center", fontsize=8)


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Impulse explorer
    - Pick a model: its patterns are grown several times to give reference attractors
    - Switch matrices: where every pattern ends after each saved impulse (rows: source)
    - Then one impulse in detail: distances over time, and robustness to scale, noise and position
    """)
    return


@app.cell
def _():
    results_dir = mo.ui.text(
        value=os.environ.get("IMPULSE_OUTPUT_PATH", "impulse_runs"), label="Impulse results", full_width=True
    )
    store_root = mo.ui.text(
        value=os.environ.get("MODEL_STORE_ROOT", "models"), label="Model store", full_width=True
    )
    mo.vstack([results_dir, store_root])
    return results_dir, store_root


@app.cell
def _(results_dir):
    results = load_impulse_results(results_dir.value)
    model_ids = sorted({result["model_id"] for result in results})
    model_choice = mo.ui.dropdown(
        model_ids, value=model_ids[0] if model_ids else None, label="Model"
    )
    _rows = [
        {
            "run": result["name"],
            "model_id": result["model_id"],
            "switch": f"{result['source']} -> {result['target']}",
            "objective": result["config"].impulse.objective.type,
            "evaluation_loss": result["summary"].get("evaluation_loss"),
            "baseline_evaluation_loss": result["summary"].get("baseline_evaluation_loss"),
        }
        for result in results
    ]
    mo.vstack([mo.md(f"{len(results)} impulse runs"), mo.ui.table(_rows, selection=None), model_choice])
    return model_choice, results


@app.cell
def _(model_choice, results, store_root):
    mo.stop(model_choice.value is None, mo.md("No impulse results found"))
    model_results = [result for result in results if result["model_id"] == model_choice.value]
    model, trajectories, observed_channels = load_model_and_data(
        model_results[0]["config"], store_root=store_root.value or None
    )
    interventions = {
        result["name"]: load_intervention(result, model, trajectories, observed_channels)
        for result in model_results
    }
    pattern_count = len(trajectories)
    mo.md(f"{len(model_results)} runs, {pattern_count} patterns, {model.N_CHANNELS} channels")
    return interventions, model, model_results, observed_channels, pattern_count, trajectories


@app.cell
def _():
    grow_steps = mo.ui.number(start=0, stop=4096, step=32, value=256, label="Steps to grow each pattern")
    samples = mo.ui.number(start=1, stop=64, value=8, label="Samples per pattern")
    evaluation_steps = mo.ui.number(start=1, stop=4096, step=32, value=512, label="Steps after the impulse")
    seed = mo.ui.number(start=0, stop=10_000, value=0, label="Seed")
    run_inference = mo.ui.run_button(label="Grow patterns and compute switch matrices")
    mo.vstack([mo.hstack([grow_steps, samples, evaluation_steps, seed]), run_inference])
    return evaluation_steps, grow_steps, run_inference, samples, seed


@app.cell
def _(grow_steps, model, pattern_count, run_inference, samples, seed, trajectories):
    mo.stop(not run_inference.value, mo.md("Press the button to run"))
    key = jax.random.PRNGKey(seed.value)
    attractors = grow_attractors(
        model, trajectories[:, 0], int(grow_steps.value), int(samples.value), jax.random.fold_in(key, 0)
    )
    # Reference attractor of each pattern: the mean over its samples
    references = jnp.mean(attractors, axis=1)
    _fig, _axes = plt.subplots(1, pattern_count, figsize=(2.5 * pattern_count, 2.8), squeeze=False)
    for _k, _axis in enumerate(_axes[0]):
        _axis.imshow(rgb(references[_k]))
        _axis.set_title(f"pattern {_k}")
        _axis.axis("off")
    _fig.tight_layout()
    _fig
    return attractors, key, references


@app.cell
def _(attractors, evaluation_steps, interventions, key, model, model_results, observed_channels, references):
    _steps = int(evaluation_steps.value)
    _panels = [("no impulse", None)] + [
        (f"{result['source']} -> {result['target']}\n{result['name'][-12:]}", interventions[result["name"]])
        for result in model_results
    ]
    _columns = min(4, len(_panels))
    _rows = -(-len(_panels) // _columns)
    _fig, _axes = plt.subplots(_rows, _columns, figsize=(3.2 * _columns, 3.2 * _rows), squeeze=False)
    for _index, (_title, _intervention) in enumerate(_panels):
        _matrix = switch_matrix(
            model, _intervention, attractors, references, _steps, observed_channels, jax.random.fold_in(key, 1)
        )
        show_matrix(_axes.flat[_index], _matrix, _title)
    for _axis in list(_axes.flat)[len(_panels):]:
        _axis.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(model_results):
    run_choice = mo.ui.dropdown([result["name"] for result in model_results], value=model_results[0]["name"], label="Impulse")
    frame_count = mo.ui.slider(4, 16, value=8, label="Frames shown")
    mo.hstack([run_choice, frame_count])
    return frame_count, run_choice


@app.cell
def _(
    attractors,
    evaluation_steps,
    frame_count,
    interventions,
    key,
    model,
    model_results,
    observed_channels,
    pattern_count,
    references,
    run_choice,
):
    chosen = next(result for result in model_results if result["name"] == run_choice.value)
    chosen_intervention = interventions[chosen["name"]]
    start_states = attractors[chosen["source"]]
    _distances, _trajectory = distance_over_time(
        model,
        chosen_intervention,
        start_states,
        references,
        int(evaluation_steps.value),
        observed_channels,
        jax.random.fold_in(key, 2),
    )
    _fig, _axis = plt.subplots(figsize=(7, 3))
    _time = np.arange(1, _distances.shape[1] + 1)
    for _k in range(pattern_count):
        _mean = np.asarray(_distances[:, :, _k].mean(axis=0))
        _spread = np.asarray(_distances[:, :, _k].std(axis=0))
        _axis.plot(_time, _mean, label=f"pattern {_k}")
        _axis.fill_between(_time, _mean - _spread, _mean + _spread, alpha=0.2)
    _axis.set_xlabel("steps after impulse")
    _axis.set_ylabel("RMS distance (observed)")
    _axis.set_title(f"{chosen['source']} -> {chosen['target']}: mean ± sd over samples")
    _axis.legend()
    _fig.tight_layout()

    _frames = np.linspace(0, _trajectory.shape[1] - 1, frame_count.value).astype(int)
    _strip, _strip_axes = plt.subplots(1, len(_frames) + 1, figsize=(1.6 * (len(_frames) + 1), 1.9))
    _strip_axes[0].imshow(rgb(chosen_intervention(start_states[:1])[0]))
    _strip_axes[0].set_title("perturbed", fontsize=8)
    for _axis_frame, _t in zip(_strip_axes[1:], _frames):
        _axis_frame.imshow(rgb(_trajectory[0, _t]))
        _axis_frame.set_title(f"t={_t + 1}", fontsize=8)
    for _axis_frame in _strip_axes:
        _axis_frame.axis("off")
    _strip.tight_layout()
    mo.vstack([_fig, _strip])
    return chosen, chosen_intervention, start_states


@app.cell
def _(chosen):
    scales = mo.ui.text(value="0, 0.25, 0.5, 0.75, 1, 1.5, 2", label="Impulse scales")
    noise_levels = mo.ui.text(value="0, 0.01, 0.05, 0.1, 0.2", label="State noise (sd)")
    location_grid = mo.ui.number(start=2, stop=12, value=5, label="Position grid size (local impulses)")
    run_robustness = mo.ui.run_button(label="Compute robustness")
    _local = chosen["config"].impulse.intervention.spatial == "local"
    mo.vstack([mo.hstack([scales, noise_levels] + ([location_grid] if _local else [])), run_robustness])
    return location_grid, noise_levels, run_robustness, scales


@app.cell
def _(
    chosen,
    chosen_intervention,
    evaluation_steps,
    key,
    location_grid,
    model,
    noise_levels,
    observed_channels,
    references,
    run_robustness,
    scales,
    start_states,
):
    mo.stop(not run_robustness.value, mo.md("Press the button to run"))
    _steps = int(evaluation_steps.value)
    _target = chosen["target"]
    _scales = [float(value) for value in scales.value.split(",")]
    _noise = [float(value) for value in noise_levels.value.split(",")]
    _scale_rates = success_rate(
        model,
        [scale_intervention(chosen_intervention, value) for value in _scales],
        start_states,
        references,
        _target,
        _steps,
        observed_channels,
        jax.random.fold_in(key, 3),
    )
    _noise_rates = [
        float(
            success_rate(
                model,
                [chosen_intervention],
                add_state_noise(start_states, value, jax.random.fold_in(key, 10 + _index)),
                references,
                _target,
                _steps,
                observed_channels,
                jax.random.fold_in(key, 4),
            )[0]
        )
        for _index, value in enumerate(_noise)
    ]
    _local = chosen["config"].impulse.intervention.spatial == "local"
    _fig, _axes = plt.subplots(1, 3 if _local else 2, figsize=(13 if _local else 9, 3.2))
    _axes[0].plot(_scales, np.asarray(_scale_rates), marker="o")
    _axes[0].set_xlabel("impulse scale")
    _axes[1].plot(_noise, _noise_rates, marker="o")
    _axes[1].set_xlabel("noise sd on source state")
    for _axis in _axes[:2]:
        _axis.set_ylabel(f"fraction ending at {_target}")
        _axis.set_ylim(-0.05, 1.05)
    if _local:
        _n = int(location_grid.value)
        _centres = (np.arange(_n) + 0.5) / _n
        _rates = success_rate(
            model,
            [move_intervention(chosen_intervention, (y, x)) for y in _centres for x in _centres],
            start_states,
            references,
            _target,
            _steps,
            observed_channels,
            jax.random.fold_in(key, 5),
        ).reshape((_n, _n))
        _image = _axes[2].imshow(np.asarray(_rates), vmin=0, vmax=1, cmap="Blues", extent=(0, 1, 1, 0))
        _trained = np.asarray(chosen_intervention.get_location())
        _axes[2].plot(_trained[1], _trained[0], "rx", label="trained centre")
        _axes[2].set_title("success by impulse centre", fontsize=9)
        _axes[2].legend(fontsize=7)
        _fig.colorbar(_image, ax=_axes[2])
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()

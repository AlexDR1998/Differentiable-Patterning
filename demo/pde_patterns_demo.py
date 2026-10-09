# /// script
# dependencies = ["marimo", "jax", "diffrax", "matplotlib", "numpy", "pyyaml"]
# ///

"""Simulate the pattern-forming PDEs used as NCA training tasks.

Run from the repository root (after ``pip install -e .``) with:

    marimo edit demo/pde_patterns_demo.py
"""

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import jax
    import marimo as mo
    import matplotlib.pyplot as plt
    import yaml

    from PDE.catalogue import PDE_MODELS
    from PDE.dataset import simulate_trajectories

    BENCHMARK = Path("Experiments/pde/conf/experiments/nca_pde_benchmark.yaml")


@app.cell
def _():
    mo.md("""
    - Each PDE in `PDE/catalogue.py`, simulated with the settings of `nca_pde_benchmark.yaml`
    - All observed channels are shown (hidden PDE channels are not)
    """)
    return


@app.cell
def _():
    # Per-PDE settings: one row of the benchmark sweep's zipped group per model.
    _group = yaml.safe_load(BENCHMARK.read_text())["groups"][0]
    _models = _group["data.pde.model"]
    settings = {
        _model: {key.removeprefix("data.pde."): values[_i] for key, values in _group.items()}
        for _i, _model in enumerate(_models)
    }
    model = mo.ui.dropdown(list(PDE_MODELS), value="gray_scott", label="PDE")
    size = mo.ui.slider(32, 128, step=32, value=64, label="Grid size")
    mo.hstack([model, size])
    return model, settings, size


@app.cell
def _(model, settings, size):
    _options = dict(settings[model.value])
    _options.pop("model")
    trajectory = simulate_trajectories(
        model.value, jax.random.PRNGKey(0), batches=2, size=size.value, frames=12, **_options
    )
    frame = mo.ui.slider(0, trajectory.data.shape[1] - 1, value=0, label="Frame")
    frame
    return frame, trajectory


@app.cell
def _(frame, trajectory):
    _batches, _, _channels = trajectory.data.shape[:3]
    _fig, _axes = plt.subplots(_batches, _channels, figsize=(3 * _channels, 3 * _batches), squeeze=False)
    for _b in range(_batches):
        for _c in range(_channels):
            _axes[_b, _c].imshow(trajectory.data[_b, frame.value, _c], vmin=0, vmax=1)
            _axes[_b, _c].set_title(f"{trajectory.channel_names[_c]}, t={trajectory.observation_times[frame.value]:g}")
            _axes[_b, _c].axis("off")
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()

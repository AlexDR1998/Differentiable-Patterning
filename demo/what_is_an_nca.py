# /// script
# dependencies = ["marimo", "jax", "equinox", "matplotlib", "numpy"]
# ///

"""Tutorial: what a neural cellular automaton (NCA) is, built up step by step.

Run from the repository root (after ``pip install -e .``) with:

    marimo edit demo/what_is_an_nca.py

Runs on a laptop CPU in a few seconds. No training.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from Common.dataloader.emoji import load_emoji_sequence
    from Common.model.spatial_operators import Ops
    from NCA.model.config import ModelConfig
    from NCA.model.factory import build_model
    from NCA.trainer.intervention import rollout_model

    REPO_ROOT = Path(__file__).resolve().parents[1]
    EMOJI_FOLDER = REPO_ROOT / "demo" / "demo_data" / "Emojis"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # What is an NCA?

    - A grid of cells; each cell holds a vector of numbers (its **channels**), so the state is `[channels, H, W]`
    - Every step, each cell looks at its neighbours (**perception**), a small network turns that into a change (**update**), and each cell applies the change at random (**firing**)
    - The same network is used in every cell, so all structure has to come from local interactions
    - This notebook builds one up by hand, then runs it. Background: `docs/concepts.md`
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The data: one emoji

    - Observed channels are what we compare with data; for an emoji: red, green, blue, alpha (RGBA)
    - `load_emoji_sequence` returns `[batch, time, channels, H, W]`, as every loader does
    """)
    return


@app.cell
def _():
    # [batch=1, time=1, channels=4, H, W] -> one [4, H, W] image
    emoji = load_emoji_sequence(["crab.png"], impath_emojis=EMOJI_FOLDER, downsample=3).data[0, 0]

    _fig, _axes = plt.subplots(1, 5, figsize=(10, 2.2))
    _axes[0].imshow(np.moveaxis(emoji, 0, -1))
    _axes[0].set_title("RGBA image")
    for _channel, _name in enumerate(["red", "green", "blue", "alpha"]):
        _axes[_channel + 1].imshow(emoji[_channel], vmin=0, vmax=1, cmap="gray")
        _axes[_channel + 1].set_title(f"channel {_channel}: {_name}")
    for _axis in _axes:
        _axis.set_axis_off()
    mo.vstack([mo.md(f"Shape: `{emoji.shape}`"), _fig])
    return (emoji,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Perception: fixed kernels

    - `Ops` (`Common/model/spatial_operators.py`) holds fixed 3×3 convolution kernels; they are never trained
    - `LAP` (Laplacian): how different a cell is from its neighbours; large at edges
    - `GRAD`: change in x and in y; tells a cell which way the pattern is going
    - `AV`: local average; `ID`: the cell itself
    - Below: the kernels, then each applied to the alpha channel
    """)
    return


@app.cell
def _(emoji):
    ops = Ops(PADDING="CIRCULAR", dx=1.0)

    _kernels = {
        "LAP": ops.laplacian.weight[0, 0],
        "GRAD x": ops.grad_x.weight[0, 0],
        "GRAD y": ops.grad_y.weight[0, 0],
        "AV": ops.average.weight[0, 0],
    }
    _fig_kernels, _axes = plt.subplots(1, 4, figsize=(8, 2.2))
    for _axis, (_name, _kernel) in zip(_axes, _kernels.items()):
        _kernel = np.asarray(_kernel)
        _scale = np.abs(_kernel).max()
        _axis.imshow(_kernel, cmap="coolwarm", vmin=-_scale, vmax=_scale)
        for (_i, _j), _value in np.ndenumerate(_kernel):
            _axis.text(_j, _i, f"{_value:.2f}", ha="center", va="center", fontsize=7)
        _axis.set_title(_name)
        _axis.set_axis_off()

    # Each operator takes [C, H, W]; Grad returns [2, C, H, W] (x and y)
    _alpha = jnp.asarray(emoji[3:4])
    _gradient = ops.Grad(_alpha)
    _responses = {
        "ID": _alpha[0],
        "LAP": ops.Lap(_alpha)[0],
        "GRAD x": _gradient[0, 0],
        "GRAD y": _gradient[1, 0],
        "AV": ops.Average(_alpha)[0],
    }
    _fig_responses, _axes = plt.subplots(1, 5, figsize=(10, 2.2))
    for _axis, (_name, _image) in zip(_axes, _responses.items()):
        _axis.imshow(_image, cmap="coolwarm" if _name != "ID" else "gray")
        _axis.set_title(_name)
        _axis.set_axis_off()
    mo.vstack([_fig_kernels, _fig_responses])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. The update network

    - Built from a `ModelConfig` by `build_model` (`NCA/model/factory.py`), exactly as training does
    - Perception of every channel is stacked: `channels × kernels` numbers per cell (`GRAD` counts twice, x and y)
    - Two 1×1 convolutions (a small network applied to each cell separately) map that to one change per channel
    - Channels beyond the 4 observed ones are **hidden**: free for the model to use as internal signals
    """)
    return


@app.cell
def _():
    channels = mo.ui.slider(8, 32, step=4, value=16, label="Channels (4 observed + hidden)")
    fire_rate = mo.ui.slider(0.1, 1.0, step=0.1, value=0.5, label="Fire rate")
    mo.hstack([channels, fire_rate])
    return channels, fire_rate


@app.cell
def _(channels, fire_rate):
    model_config = ModelConfig(
        family="NCA",
        channels=channels.value,
        kernel_str=("ID", "GRAD", "LAP"),
        fire_rate=fire_rate.value,
        padding="CIRCULAR",
    )
    model, _run_name = build_model(model_config, key=jax.random.PRNGKey(0))

    # model.partition() splits trainable weights from everything else
    _trainable, _ = model.partition()
    _parameter_count = sum(_leaf.size for _leaf in jax.tree_util.tree_leaves(_trainable))
    mo.md(
        f"""
    - Model `{_run_name}`: {model.N_CHANNELS} channels, perceiving {model.N_FEATURES} numbers per cell
    - Layer 1: `{tuple(model.layers[0].weight.shape)}` (out, in, 1, 1), then `{model.layers[1].__name__}`
    - Layer 2: `{tuple(model.layers[2].weight.shape)}` → one change per channel
    - {_parameter_count:,} trainable numbers in total; the whole model is tiny
    """
    )
    return (model,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. One step, by hand

    - Start from a single **seed** cell: alpha and hidden channels set to 1, the classic "grow from one cell" start
    - A freshly built NCA changes nothing: its last layer starts at zero (so training starts from "do nothing")
    - To see it act, we give the last layer random weights; an untrained NCA, as at the start of training
    - The by-hand step below is checked against `model(x, key=key)`
    """)
    return


@app.cell
def _(model):
    _size = 40
    seed_state = jnp.zeros((model.N_CHANNELS, _size, _size)).at[3:, _size // 2, _size // 2].set(1.0)

    _fresh_change = model.layers[2](model.layers[1](model.layers[0](model.perception(seed_state))))

    _key = jax.random.PRNGKey(1)
    random_model = eqx.tree_at(
        lambda m: m.layers[2].weight,
        model,
        0.3 * jax.random.normal(_key, model.layers[2].weight.shape),
    )

    # The step, written out (the same as NCA.__call__ for the plain NCA family)
    _perceived = random_model.perception(seed_state)  # [N_FEATURES, H, W]
    _hidden = random_model.layers[1](random_model.layers[0](_perceived))
    _change = random_model.layers[2](_hidden)  # [channels, H, W]
    _step_key = jax.random.PRNGKey(2)
    _fires = jax.random.bernoulli(_step_key, p=random_model.FIRE_RATE, shape=_change.shape)
    _by_hand = seed_state + _fires * _change

    _by_model = random_model(seed_state, key=_step_key)
    mo.md(
        f"""
    - Perception shape: `{_perceived.shape}`; change shape: `{_change.shape}`
    - Largest change from the fresh model: `{float(jnp.abs(_fresh_change).max()):.1f}`
    - Largest change from the random model: `{float(jnp.abs(_change).max()):.3f}`, only next to the seed (one step reaches one cell away)
    - By hand vs `model(x, key=key)`: largest difference `{float(jnp.abs(_by_hand - _by_model).max()):.1e}`
    """
    )
    return random_model, seed_state


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Running it: a rollout

    - A **rollout** applies the step many times; `rollout_model` (`NCA/trainer/intervention.py`) does this in compiled JAX code and returns `[steps + 1, channels, H, W]`
    - The first run compiles (a second or two); changing only the seed or boundary is then fast
    - A **boundary** function is applied after every step; it can keep everything inside a disc, as for a micropattern colony
    - Try "none": an untrained NCA's values grow without limit. Training keeps them in range with the `intermediate_state` regulariser (`docs/losses.md`); here we simply clip to [0, 1]
    - The random NCA only makes noise; **training** searches for weights that grow the emoji instead (next tutorial: `demo/nca_training_config_walkthrough.py`)
    """)
    return


@app.cell
def _():
    steps = mo.ui.slider(10, 200, step=10, value=60, label="Steps")
    rollout_seed = mo.ui.number(0, 1000, value=0, label="Random seed")
    boundary = mo.ui.radio(
        options=["none", "clip to [0, 1]", "clip to [0, 1] inside a disc"],
        value="clip to [0, 1]",
        label="Boundary function",
    )
    mo.hstack([mo.vstack([steps, rollout_seed]), boundary])
    return boundary, rollout_seed, steps


@app.cell
def _(boundary, random_model, rollout_seed, seed_state, steps):
    _size = seed_state.shape[-1]
    _y, _x = jnp.mgrid[:_size, :_size]
    _disc = ((_y - _size / 2) ** 2 + (_x - _size / 2) ** 2 < (0.35 * _size) ** 2).astype(jnp.float32)

    # A boundary function maps a state [channels, H, W] to a state of the same shape
    _boundary_functions = {
        "none": lambda state: state,
        "clip to [0, 1]": lambda state: jnp.clip(state, 0.0, 1.0),
        "clip to [0, 1] inside a disc": lambda state: jnp.clip(state, 0.0, 1.0) * _disc,
    }

    trajectory = np.asarray(
        rollout_model(
            random_model,
            seed_state,
            _boundary_functions[boundary.value],
            jax.random.PRNGKey(int(rollout_seed.value)),
            int(steps.value),
        )
    )
    mo.md(f"Trajectory shape: `{trajectory.shape}` (steps + 1, channels, H, W)")
    return (trajectory,)


@app.cell
def _(trajectory):
    _frames = np.linspace(0, len(trajectory) - 1, 6).astype(int)
    _hidden_scale = max(float(np.percentile(np.abs(trajectory[:, 4:6]), 99)), 1e-6)
    _fig, _axes = plt.subplots(3, len(_frames), figsize=(2 * len(_frames), 6.2))
    for _column, _frame in enumerate(_frames):
        _state = trajectory[_frame]
        _axes[0, _column].imshow(np.clip(np.moveaxis(_state[:4], 0, -1), 0, 1))
        _axes[0, _column].set_title(f"step {_frame}")
        for _row, _channel in ((1, 4), (2, 5)):
            _axes[_row, _column].imshow(_state[_channel], cmap="coolwarm", vmin=-_hidden_scale, vmax=_hidden_scale)
    _axes[0, 0].set_ylabel("observed (RGBA)")
    _axes[1, 0].set_ylabel("hidden channel 4")
    _axes[2, 0].set_ylabel("hidden channel 5")
    for _axis in _axes.flat:
        _axis.set_xticks([])
        _axis.set_yticks([])
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Where next

    - **Train one:** `demo/nca_training_config_walkthrough.py` trains a small NCA on emojis, on a laptop
    - **What training optimises:** `demo/nca_objectives_and_pool_dynamics.py`
    - **Load and run trained models:** `demo/bundles_and_inference.py`
    - **Other model families** (gated, noisy, signalling, hierarchical, KAN): `docs/concepts.md`
    """)
    return


if __name__ == "__main__":
    app.run()

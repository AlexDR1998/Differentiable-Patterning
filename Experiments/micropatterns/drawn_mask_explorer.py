# /// script
# dependencies = [
#   "marimo",
#   "anywidget",
#   "equinox",
#   "jax",
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "python-dotenv",
#   "scipy",
#   "opencv-python",
#   "einops",
#   "tqdm",
# ]
# ///

"""Draw an adhesion mask and run a trained micropattern NCA on it.

Run from the repository root with:

    marimo edit Experiments/micropatterns/drawn_mask_explorer.py

Model bundles come from the model store (``MODEL_STORE_ROOT``, default
``models/``) and the reference colony from ``DATA_PATH_BASE/260726_nca_dataset``.
The initial biology is a texture synthesised from a measured 0 h colony over
the whole grid and then cut out by the drawn mask, as the planned web demo
will do.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import io
    import os
    import base64
    from dataclasses import replace
    from pathlib import Path

    repository_root = Path(__file__).resolve().parents[2]

    import jax.numpy as jnp
    import jax.random as jr
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from dotenv import load_dotenv

    from Experiments.model_registry import ModelRegistry, load_selection
    from Experiments.micropatterns.mask_canvas import MaskCanvas, decode_mask, encode_mask
    from Experiments.micropatterns.rendering import render_cmy_composite, render_fate_cmy
    from Experiments.micropatterns.texture_synthesis import (
        normalise_histograms_and_covariance,
        synthesise_patch_texture,
        synthesise_reference_texture,
    )
    from NCA.inverse_design.initial_state import build_initial_state
    from NCA.trainer.intervention import rollout_model_sampled

    load_dotenv(repository_root / ".env", override=False)

    DATASET = "micropatterns_260726"
    # Loaded reference colonies, keyed by data settings and path, so that
    # switching between models trained on the same data does not reload.
    _REFERENCE_CACHE = {}

    def load_reference(bundle, data_root):
        """0 h measurements of the model's training replicates, scaled as in training.

        Returns measurements ``[B, C, H, W]``, which measurements were taken
        ``[B, C]``, colony masks ``[B, H, W]``, the channel schema and the
        replicate labels.
        """

        from Experiments.micropatterns.config_helpers import load_data

        data_config = replace(
            bundle.config.data,
            knockout=replace(bundle.config.data.knockout, curriculum=("baseline",)),
        )
        cache_key = (repr(data_config), str(data_root))
        if cache_key not in _REFERENCE_CACHE:
            train_replicates = data_config.micropattern.train_replicates
            if train_replicates is None:
                train_replicates = tuple(range(1, data_config.batches + 1))
            data, aux, _, boundary, _, _ = load_data(
                data_config,
                impath=str(data_root),
                replicate_indices=tuple(index - 1 for index in train_replicates),
                pool_copies_override=1,
            )
            _REFERENCE_CACHE[cache_key] = {
                "measurements": np.asarray(data)[:, 0],
                "available": np.asarray(aux["measurement_mask"])[:, 0],
                "support": np.asarray(boundary)[:, 0] > 0.5,
                "schema": aux["channel_schema"],
                "replicates": tuple(aux.get("batch_replicates", train_replicates)),
            }
        return _REFERENCE_CACHE[cache_key]

    def combine_reference(reference, mode, sample, seed):
        """One reference colony ``[C, H, W]`` and its mask ``[H, W]``.

        ``"fixed"`` uses one replicate. ``"sample_groups"`` takes each staining
        panel from a random replicate that measured it, as in
        ``boundary_mask_explorer.py``.
        """

        measurements = reference["measurements"]
        if mode == "fixed":
            index = int(sample) % measurements.shape[0]
            return measurements[index], reference["support"][index]
        rng = np.random.default_rng(int(seed))
        combined = np.zeros_like(measurements[0])
        for name, indices in zip(
            reference["schema"].group_names,
            reference["schema"].group_measurement_indices,
        ):
            indices = np.asarray(indices, dtype=int)
            eligible = np.flatnonzero(np.all(reference["available"][:, indices], axis=1))
            if not len(eligible):
                raise ValueError(f"No training replicate measures {name!r} at 0 h")
            combined[indices] = measurements[int(rng.choice(eligible)), indices]
        support = reference["support"][int(rng.choice(measurements.shape[0]))]
        return combined, support

    def fate_rgb(states, names):
        """SOX2/TBXT/SOX17 composite ``[T, H, W, 3]``, or the first channel in grey."""

        if {"SOX2", "TBXT", "SOX17"}.issubset(names):
            return render_fate_cmy(states, names)
        return np.repeat(np.clip(states[:, :1], 0.0, 1.0).transpose(0, 2, 3, 1), 3, axis=-1)

    def png_data_url(rgb):
        """``[H, W, 3]`` array in [0, 1] to a PNG data URL."""

        buffer = io.BytesIO()
        plt.imsave(buffer, np.clip(rgb, 0.0, 1.0), format="png")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Drawn-mask micropattern NCA

    - Pick a trained model, draw the adhesive region, run the NCA.
    - The last state channel holds the mask (soft boundary) or the mask multiplies every channel (hard boundary).
    - The initial biology is a synthetic 0 h texture over the whole grid, cut out by the mask.
    """)
    return


@app.cell(hide_code=True)
def _():
    _data_base = os.environ.get("DATA_PATH_BASE")
    store_root = mo.ui.text(
        os.environ.get("MODEL_STORE_ROOT", str(repository_root / "models")),
        label="Model store root",
        full_width=True,
    )
    selection_path = mo.ui.text(
        "",
        label="Optional model selection YAML (from model_registry_explorer.py)",
        full_width=True,
    )
    data_root = mo.ui.text(
        str(Path(_data_base) / "260726_nca_dataset") if _data_base else "../Data/260726_nca_dataset",
        label="260726 micropattern dataset root",
        full_width=True,
    )
    reindex_button = mo.ui.button(
        label="Rebuild registry index", value=0, on_click=lambda count: count + 1
    )
    mo.vstack([store_root, selection_path, data_root, reindex_button])
    return data_root, reindex_button, selection_path, store_root


@app.cell(hide_code=True)
def _(reindex_button, selection_path, store_root):
    registry = ModelRegistry(Path(store_root.value).expanduser())
    if reindex_button.value:
        registry.reindex()
    _models = registry.models_df()
    _models = _models[_models["dataset"] == DATASET]
    _annotations = registry.annotations_df()
    _models = _models.merge(_annotations, on="model_id", how="left")
    if selection_path.value.strip():
        _selected = {
            str(_record["model_id"])
            for _record in load_selection(Path(selection_path.value).expanduser())
        }
        _models = _models[_models["model_id"].isin(_selected)]
    _columns = [
        "model_id", "alias", "family", "experiment", "wandb_group",
        "best_loss", "created_at", "notes",
    ]
    mo.stop(
        not len(_models),
        mo.callout(f"No {DATASET} bundles found in `{registry.root}`.", kind="warn"),
    )
    model_table = mo.ui.table(
        _models[[_c for _c in _columns if _c in _models.columns]].reset_index(drop=True),
        selection="single",
        page_size=8,
        label="Model",
    )
    model_table
    return model_table, registry


@app.cell(hide_code=True)
def _(model_table, registry):
    mo.stop(
        not len(model_table.value),
        mo.callout("Select a model in the table.", kind="info"),
    )
    bundle = registry.get(str(model_table.value["model_id"].iloc[0]))
    model = bundle.load_model(key=jr.PRNGKey(0), implementation="portable")
    boundary_mode = bundle.config.trainer.boundary_mode
    training_hours = tuple(int(_hour) for _hour in bundle.config.data.micropattern.timesteps)
    _family = bundle.config.model.family
    _summary = mo.md(
        f"""
        - **{bundle.id}** · family `{_family}` · {int(model.N_CHANNELS)} channels
        - boundary `{boundary_mode}` · {bundle.config.run.t} steps per 12 h · downsample ×{bundle.config.data.downsample}
        - training times: {", ".join(f"{_h} h" for _h in training_hours)}
        """
    )
    _notes = []
    if "HNCA" in _family:
        _notes.append(mo.callout("HNCA runs here but not in the WebGL demo.", kind="warn"))
    if boundary_mode == "none":
        _notes.append(mo.callout("This model was trained without a boundary, so the mask only shapes the initial state.", kind="warn"))
    mo.vstack([_summary, *_notes])
    return boundary_mode, bundle, model, training_hours


@app.cell(hide_code=True)
def _(bundle, data_root):
    with mo.status.spinner("Loading the 0 h reference colonies"):
        reference = load_reference(bundle, Path(data_root.value).expanduser())
    state_channel_names = tuple(reference["schema"].state_channels)
    grid_shape = reference["measurements"].shape[-2:]
    mo.md(
        f"- {reference['measurements'].shape[0]} reference colonies, "
        f"grid {grid_shape[0]} × {grid_shape[1]}"
    )
    return grid_shape, reference, state_channel_names


@app.cell(hide_code=True)
def _():
    texture_mode = mo.ui.dropdown(
        options={
            "Multichannel patch resampling": "patch",
            "IAAFT spectrum + histogram": "iaaft",
            "Uniform (reference mean)": "uniform",
        },
        value="Multichannel patch resampling",
        label="Texture",
    )
    reference_mode = mo.ui.dropdown(
        options={"Panel from random replicate": "sample_groups", "Fixed replicate": "fixed"},
        value="Panel from random replicate",
        label="Reference",
    )
    reference_sample = mo.ui.number(0, 32, value=0, step=1, label="Fixed replicate")
    reference_seed = mo.ui.number(0, 100000, value=0, step=1, label="Replicate seed")
    patch_size = mo.ui.slider(3, 31, value=11, step=2, label="Patch width (px)")
    patch_overlap = mo.ui.slider(0, 30, value=7, step=1, label="Patch overlap (px)")
    patch_jitter = mo.ui.slider(0, 12, value=3, step=1, label="Patch jitter (px)")
    patch_interior = mo.ui.slider(0.4, 1.0, value=0.8, step=0.01, label="Patch source radius fraction")
    patch_candidates = mo.ui.slider(1, 64, value=24, step=1, label="Quilting candidates")
    iaaft_iterations = mo.ui.slider(0, 100, value=25, step=5, label="IAAFT iterations")
    normalise = mo.ui.switch(value=False, label="Match histograms + covariance")
    normalise_iterations = mo.ui.slider(1, 12, value=4, step=1, label="Normalisation iterations")
    texture_seed = mo.ui.number(0, 100000, value=0, step=1, label="Texture seed")
    noise_strength = mo.ui.slider(0.0, 0.05, value=0.005, step=0.001, label="Initial noise (training: 0.005)")
    mo.vstack([
        mo.md("### Initial biology"),
        mo.hstack([texture_mode, reference_mode, reference_sample, reference_seed], justify="start"),
        mo.hstack([patch_size, patch_overlap, patch_jitter], justify="start"),
        mo.hstack([patch_interior, patch_candidates, iaaft_iterations], justify="start"),
        mo.hstack([normalise, normalise_iterations, texture_seed, noise_strength], justify="start"),
    ])
    return (
        iaaft_iterations,
        noise_strength,
        normalise,
        normalise_iterations,
        patch_candidates,
        patch_interior,
        patch_jitter,
        patch_overlap,
        patch_size,
        reference_mode,
        reference_sample,
        reference_seed,
        texture_mode,
        texture_seed,
    )


@app.cell(hide_code=True)
def _(
    grid_shape,
    iaaft_iterations,
    normalise,
    normalise_iterations,
    patch_candidates,
    patch_interior,
    patch_jitter,
    patch_overlap,
    patch_size,
    reference,
    reference_mode,
    reference_sample,
    reference_seed,
    state_channel_names,
    texture_mode,
    texture_seed,
):
    # The texture covers the whole grid and does not depend on the drawn mask,
    # so redrawing is instant and the browser demo can ship it as one tile.
    _measurements, reference_support = combine_reference(
        reference, reference_mode.value, reference_sample.value, reference_seed.value
    )
    _schema = reference["schema"]
    _groups = tuple(tuple(_g) for _g in _schema.group_measurement_indices)
    _full = np.ones(grid_shape, dtype=bool)
    with mo.status.spinner("Synthesising the 0 h texture"):
        if texture_mode.value == "patch":
            _tile, *_ = synthesise_patch_texture(
                _measurements, reference_support, grid_shape, _groups,
                int(patch_size.value), int(patch_overlap.value),
                float(patch_interior.value), int(patch_jitter.value),
                int(patch_candidates.value), int(texture_seed.value),
            )
        elif texture_mode.value == "iaaft":
            _tile = synthesise_reference_texture(
                _measurements, reference_support, grid_shape, _full,
                int(iaaft_iterations.value), int(texture_seed.value),
            )
        else:
            _tile = np.broadcast_to(
                _measurements[:, reference_support].mean(axis=1)[:, None, None],
                _measurements.shape,
            ).astype(np.float32)
        if normalise.value and texture_mode.value != "uniform":
            _tile = normalise_histograms_and_covariance(
                _tile, _measurements, reference_support, _full, _groups,
                int(normalise_iterations.value),
            )
    biology_tile = np.asarray(_tile)[np.asarray(_schema.primary_measurements)]
    texture_background = png_data_url(fate_rgb(biology_tile[None], state_channel_names)[0])
    return biology_tile, reference_support, texture_background


@app.cell(hide_code=True)
def _(grid_shape, reference_support):
    # Rebuilt only when the grid or the reference colony changes, so a
    # drawing survives texture changes.
    mask_canvas = MaskCanvas(
        height=int(grid_shape[0]),
        width=int(grid_shape[1]),
        mask=encode_mask(reference_support),
        reset_mask=encode_mask(reference_support),
        brush_radius=max(2.0, round(min(grid_shape) / 30)),
    )
    mask_view = mo.ui.anywidget(mask_canvas)
    return mask_canvas, mask_view


@app.cell(hide_code=True)
def _(mask_canvas, texture_background):
    mask_canvas.background = texture_background
    return


@app.cell(hide_code=True)
def _(grid_shape, mask_view):
    drawn_mask = decode_mask(mask_view.value["mask"], *grid_shape)
    mo.vstack([
        mo.md(
            "### Adhesive region\n\n"
            "- Drag to paint; **d** / **e** or the mode button switch draw and erase.\n"
            "- Reset restores the reference colony mask. Background: SOX2 cyan, TBXT magenta, SOX17 yellow."
        ),
        mask_view,
    ])
    return (drawn_mask,)


@app.cell(hide_code=True)
def _(training_hours):
    _last = max(training_hours)
    _options = sorted({*training_hours[1:], _last + 24, _last + 48, _last + 96})
    rollout_hours = mo.ui.dropdown(
        options={f"{_h} h": _h for _h in _options}, value=f"{_last} h", label="Run until"
    )
    frames_per_12h = mo.ui.dropdown(
        options=[1, 2, 4, 8, 16], value=4, label="Saved frames per 12 h"
    )
    rollout_seed = mo.ui.number(0, 100000, value=0, step=1, label="Rollout seed")
    run_button = mo.ui.run_button(label="Run NCA")
    mo.hstack([rollout_hours, frames_per_12h, rollout_seed, run_button], justify="start")
    return frames_per_12h, rollout_hours, rollout_seed, run_button


@app.cell(hide_code=True)
def _(
    biology_tile,
    boundary_mode,
    bundle,
    drawn_mask,
    frames_per_12h,
    model,
    noise_strength,
    rollout_hours,
    rollout_seed,
    run_button,
):
    mo.stop(
        not run_button.value,
        mo.callout("Draw a mask, then press **Run NCA**.", kind="info"),
    )
    mo.stop(
        not drawn_mask.any(),
        mo.callout("The mask is empty.", kind="warn"),
    )
    _steps_per_12h = int(bundle.config.run.t)
    _stride = max(1, _steps_per_12h // int(frames_per_12h.value))
    _total_steps = int(rollout_hours.value) // 12 * _steps_per_12h
    observation_steps = np.arange(0, _total_steps + 1, _stride)
    _mask = jnp.asarray(drawn_mask, dtype=jnp.float32)
    _noise_key, _rollout_key = jr.split(jr.PRNGKey(int(rollout_seed.value)))
    _state = build_initial_state(
        jnp.asarray(biology_tile), _mask, int(model.N_CHANNELS), boundary_mode=boundary_mode
    )
    # Blend in noise as MicropatternAugmenter does; the boundary resets the
    # mask channel after every step.
    _amount = float(noise_strength.value)
    _state = (1.0 - _amount) * _state + _amount * jr.normal(_noise_key, _state.shape)
    with mo.status.spinner(f"Running {_total_steps} NCA steps"):
        rollout_frames = np.asarray(rollout_model_sampled(
            model, _state, _mask[None], boundary_mode, _rollout_key,
            _total_steps, jnp.asarray(observation_steps),
        ))
    rollout_mask = np.asarray(drawn_mask)
    return observation_steps, rollout_frames, rollout_mask


@app.cell(hide_code=True)
def _(bundle, observation_steps, rollout_frames):
    frame_hours = observation_steps / int(bundle.config.run.t) * 12
    frame_slider = mo.ui.slider(
        0, len(rollout_frames) - 1, value=len(rollout_frames) - 1, step=1,
        label="Frame", show_value=False,
    )
    frame_slider
    return frame_hours, frame_slider


@app.cell(hide_code=True)
def _(frame_hours, frame_slider, rollout_frames, rollout_mask, state_channel_names):
    _index = int(frame_slider.value)
    _n_state = len(state_channel_names)
    _frame = rollout_frames[_index : _index + 1, :_n_state]
    try:
        _composite = render_cmy_composite(_frame, state_channel_names)[0]
        _panels = 3
        _title = "SOX2/TBXT/SOX17 · CER1/LEFTY/NODAL · FOXA2/LEF1/SMAD23 (cyan/magenta/yellow)"
    except ValueError:
        _composite = fate_rgb(_frame, state_channel_names)[0]
        _panels = 1
        _title = "SOX2/TBXT/SOX17 (cyan/magenta/yellow)"
    _figure, _axis = plt.subplots(figsize=(4 * _panels, 4.2))
    _axis.imshow(_composite)
    _width = rollout_mask.shape[1]
    for _offset in range(_panels):
        _axis.contour(
            np.arange(_width) + _offset * _width, np.arange(rollout_mask.shape[0]),
            rollout_mask, levels=[0.5], colors="white", linewidths=0.6,
        )
    _axis.set_title(f"{frame_hours[_index]:.1f} h · {_title}", fontsize=9)
    _axis.axis("off")
    _figure.tight_layout()

    _columns = 5
    _rows = int(np.ceil(_n_state / _columns))
    _channels_figure, _channel_axes = plt.subplots(
        _rows, _columns, figsize=(2.6 * _columns, 2.6 * _rows), squeeze=False
    )
    for _channel, _channel_axis in enumerate(_channel_axes.flat):
        _channel_axis.axis("off")
        if _channel < _n_state:
            _channel_axis.imshow(_frame[0, _channel], cmap="magma", vmin=0.0, vmax=1.0)
            _channel_axis.set_title(state_channel_names[_channel], fontsize=9)
    _channels_figure.tight_layout()
    mo.vstack([_figure, _channels_figure])
    return


@app.cell(hide_code=True)
def _():
    save_video_button = mo.ui.run_button(label="Save fate video (MP4)")
    save_video_button
    return (save_video_button,)


@app.cell(hide_code=True)
def _(bundle, rollout_frames, save_video_button, state_channel_names):
    mo.stop(not save_video_button.value)
    from Common.save_to_video import save_to_video_rgb

    _path = repository_root / "output" / "drawn_mask" / f"{bundle.id}.mp4"
    _path.parent.mkdir(parents=True, exist_ok=True)
    _rgb = fate_rgb(rollout_frames[:, : len(state_channel_names)], state_channel_names)
    save_to_video_rgb(_rgb, str(_path), fps=10, duration=max(2, len(_rgb) // 10), SCALE_UP=2)
    mo.md(f"- Saved `{_path}`")
    return


if __name__ == "__main__":
    app.run()

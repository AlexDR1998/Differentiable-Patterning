"""Interactively inspect the 260726 micropattern data augmenter.

Run with:
    marimo edit NCA/trainer/inspect_micropattern_augmenter.py
"""

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path as _Path
    import sys as _sys

    _repo_root = str(_Path(__file__).resolve().parents[2])
    if _repo_root not in _sys.path:
        _sys.path.insert(0, _repo_root)
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from Common.dataloader.micropattern import load_micropattern_260726
    from NCA.trainer.data_augmenter.micropattern import DataAugmenter

    return DataAugmenter, jax, jnp, load_micropattern_260726, mo, np, plt


@app.cell
def _(mo):
    mo.md("""
    # Micropattern augmenter inspection

    Load the 260726 dataset, construct the same snapshot augmenter used by
    training, and inspect one deterministic pool update without running an NCA.
    Loading and augmentation are gated by separate buttons.
    """)
    return


@app.cell
def _(mo):
    dataset_root = mo.ui.text(
        value="../Data/260726_nca_dataset",
        label="Dataset root",
        full_width=True,
    )
    dataset_timesteps = mo.ui.text(
        value="0,12,24,36,48", label="Timesteps (hours)"
    )
    dataset_groups = mo.ui.text(
        value="",
        label="Experiment groups (comma separated; blank loads all)",
        full_width=True,
    )
    dataset_downsample = mo.ui.slider(1, 16, value=4, label="Downsample")
    dataset_replicates = mo.ui.slider(
        1, 8, value=1, label="Physical replicates"
    )
    dataset_pool_copies = mo.ui.slider(1, 16, value=1, label="Pool copies")
    dataset_align = mo.ui.checkbox(value=True, label="Align images")
    dataset_percentile_low = mo.ui.number(
        0.0, 99.0, value=0.5, step=0.1, label="Histogram low percentile"
    )
    dataset_percentile_high = mo.ui.number(
        1.0, 100.0, value=99.95, step=0.05, label="Histogram high percentile"
    )
    dataset_load = mo.ui.run_button(label="Load dataset")
    mo.vstack(
        [
            mo.md("## 1. Load data"),
            dataset_root,
            mo.hstack(
                [
                    dataset_downsample,
                    dataset_replicates,
                    dataset_pool_copies,
                    dataset_align,
                ]
            ),
            dataset_timesteps,
            dataset_groups,
            mo.hstack([dataset_percentile_low, dataset_percentile_high]),
            dataset_load,
        ]
    )
    return (
        dataset_align,
        dataset_downsample,
        dataset_groups,
        dataset_load,
        dataset_percentile_high,
        dataset_percentile_low,
        dataset_pool_copies,
        dataset_replicates,
        dataset_root,
        dataset_timesteps,
    )


@app.cell
def _(
    dataset_align,
    dataset_downsample,
    dataset_groups,
    dataset_load,
    dataset_percentile_high,
    dataset_percentile_low,
    dataset_pool_copies,
    dataset_replicates,
    dataset_root,
    dataset_timesteps,
    load_micropattern_260726,
):
    dataset_load
    _times = tuple(
        int(_value.strip())
        for _value in dataset_timesteps.value.split(",")
        if _value.strip()
    )
    _groups = tuple(
        _value.strip()
        for _value in dataset_groups.value.split(",")
        if _value.strip()
    )
    loaded_dataset = load_micropattern_260726(
        dataset_root.value,
        timesteps=_times,
        downsample=dataset_downsample.value,
        replicate_count=dataset_replicates.value,
        pool_copies=dataset_pool_copies.value,
        experiment_groups=_groups or None,
        align=dataset_align.value,
        hist_eqs=(
            dataset_percentile_low.value,
            dataset_percentile_high.value,
        ),
    )
    return (loaded_dataset,)


@app.cell
def _(loaded_dataset, mo, np):
    loaded_array = np.asarray(loaded_dataset.data)
    loaded_boundary = np.asarray(loaded_dataset.boundary_mask)
    loaded_measurement_mask = np.asarray(loaded_dataset.measurement_mask)
    loaded_names = tuple(loaded_dataset.channel_names)
    loaded_schema = loaded_dataset.schema
    loaded_summary = [
        {
            "component": "data",
            "shape": str(tuple(loaded_array.shape)),
            "batch count": loaded_array.shape[0],
        },
        {
            "component": "boundary mask",
            "shape": str(tuple(loaded_boundary.shape)),
            "batch count": loaded_boundary.shape[0],
        },
        {
            "component": "measurement mask",
            "shape": str(tuple(loaded_measurement_mask.shape)),
            "batch count": loaded_measurement_mask.shape[0],
        },
    ]
    _counts = {int(_row["batch count"]) for _row in loaded_summary}
    loaded_alignment_message = (
        mo.callout("All loader outputs have aligned batch counts.", kind="success")
        if len(_counts) == 1
        else mo.callout("Loader batch counts are not aligned.", kind="danger")
    )
    mo.vstack(
        [
            mo.md("## 2. Loader contract"),
            mo.ui.table(loaded_summary),
            loaded_alignment_message,
            mo.md(
                f"Schema **{loaded_schema.name}**: "
                f"{loaded_schema.n_measurement_channels} measurement channels → "
                f"{loaded_schema.n_state_channels} state channels."
            ),
        ]
    )
    return loaded_array, loaded_boundary, loaded_names, loaded_schema


@app.cell
def _(loaded_array, loaded_names, mo):
    inspect_batch = mo.ui.slider(
        0, loaded_array.shape[0] - 1, value=0, label="Batch"
    )
    inspect_time = mo.ui.slider(
        0, loaded_array.shape[1] - 1, value=0, label="Timestep index"
    )
    inspect_channel = mo.ui.dropdown(
        options={_name: _index for _index, _name in enumerate(loaded_names)},
        value=loaded_names[0],
        label="Measurement channel",
    )
    inspect_zero_nan = mo.ui.checkbox(
        value=True, label="Render zero background as NaN"
    )
    mo.hstack([inspect_batch, inspect_time, inspect_channel, inspect_zero_nan])
    return inspect_batch, inspect_channel, inspect_time, inspect_zero_nan


@app.cell
def _(
    inspect_batch,
    inspect_channel,
    inspect_time,
    inspect_zero_nan,
    loaded_array,
    loaded_boundary,
    loaded_names,
    mo,
    np,
    plt,
):
    _channel = int(inspect_channel.value)
    _image = loaded_array[inspect_batch.value, inspect_time.value, _channel]
    _display_image = np.where(_image == 0, np.nan, _image) if inspect_zero_nan.value else _image
    _figure, _axes = plt.subplots(1, 2, figsize=(10, 4))
    _axes[0].imshow(_display_image, cmap="viridis")
    _axes[0].contour(
        loaded_boundary[inspect_batch.value, 0],
        levels=[0.5],
        colors="white",
        linewidths=0.7,
    )
    _axes[0].set_title(
        f"batch {inspect_batch.value}, t={inspect_time.value}, {loaded_names[_channel]}"
    )
    _axes[0].axis("off")
    _finite = _display_image[np.isfinite(_display_image)]
    if _finite.size:
        _axes[1].hist(_finite, bins=80, density=True)
    _axes[1].set_title("Selected image distribution")
    _axes[1].set_xlabel("intensity")
    _axes[1].grid(alpha=0.2)
    _figure.tight_layout()
    mo.vstack([mo.md("### Raw measurement"), _figure])
    return


@app.cell
def _(mo):
    augmenter_input_mode = mo.ui.dropdown(
        options={
            "Perfect rollout proxy": "perfect_rollout",
            "initialize_pool": "initialization",
        },
        value="Perfect rollout proxy",
        label="Pool operation",
    )
    augmenter_state_channels = mo.ui.number(
        1, 256, value=32, step=1, label="Total NCA state channels"
    )
    augmenter_noise = mo.ui.number(
        0.0, 1.0, value=0.005, step=0.001, label="Noise strength"
    )
    augmenter_probability_start = mo.ui.number(
        0.0, 1.0, value=0.5, step=0.05, label="Reinjection probability (start)"
    )
    augmenter_probability_end = mo.ui.number(
        0.0, 1.0, value=0.5, step=0.05, label="Reinjection probability (end)"
    )
    augmenter_decay_start = mo.ui.number(
        0.0, 0.99, value=0.25, step=0.05, label="Decay start fraction"
    )
    augmenter_total_iterations = mo.ui.number(
        1, 10_000_000, value=100_000, step=1000, label="Total iterations"
    )
    augmenter_iteration = mo.ui.number(
        0, 10_000_000, value=0, step=100, label="Preview iteration"
    )
    augmenter_seed = mo.ui.number(0, 2**31 - 1, value=0, step=1, label="PRNG seed")
    augmenter_run = mo.ui.run_button(label="Run augmentation preview")
    mo.vstack(
        [
            mo.md("## 3. Augmenter configuration"),
            mo.hstack(
                [augmenter_input_mode, augmenter_state_channels, augmenter_noise]
            ),
            mo.hstack(
                [
                    augmenter_probability_start,
                    augmenter_probability_end,
                    augmenter_decay_start,
                ]
            ),
            mo.hstack(
                [augmenter_total_iterations, augmenter_iteration, augmenter_seed]
            ),
            augmenter_run,
        ]
    )
    return (
        augmenter_decay_start,
        augmenter_input_mode,
        augmenter_iteration,
        augmenter_noise,
        augmenter_probability_end,
        augmenter_probability_start,
        augmenter_run,
        augmenter_seed,
        augmenter_state_channels,
        augmenter_total_iterations,
    )


@app.cell
def _(DataAugmenter, jax, jnp):
    def run_augmentation_preview(
        data,
        schema,
        state_channels,
        noise_strength,
        probability_start,
        probability_end,
        decay_start_fraction,
        total_iterations,
        iteration,
        seed,
        input_mode,
    ):
        if int(state_channels) < schema.n_state_channels:
            raise ValueError(
                "Total NCA state channels cannot be smaller than the schema state count"
            )

        _augmenter = DataAugmenter(
            data_true=jnp.asarray(data),
            hidden_channels=0,
            schema=schema,
            intermediate_reinjection_probability=float(probability_start),
            intermediate_reinjection_probability_end=float(probability_end),
            intermediate_reinjection_decay_start_fraction=float(decay_start_fraction),
            intermediate_reinjection_total_iterations=int(total_iterations),
        )
        _augmenter.noise_strength = float(noise_strength)
        _augmenter.data_init()
        _x_initial, _y_before = _augmenter.split_x_y(1)
        if input_mode == "perfect_rollout":
            # A perfect NCA rollout maps each transition slot onto its target
            # state. advance_pool can then shift t_(n+1) into the following
            # transition slot, exactly as it does with x_new during training.
            _x_before = [
                _augmenter._to_state(_trajectory)[1:]
                for _trajectory in _augmenter.data_saved
            ]
        elif input_mode == "initialization":
            # The raw snapshots are the input to initialize_pool.
            _x_before = _x_initial
        else:
            raise ValueError(f"Unknown pool operation: {input_mode!r}")
        _key = jax.random.PRNGKey(int(seed))

        # Reproduce advance_pool's random decisions for exact diagnostics. This
        # follows MicropatternDataAugmenter._group_reinject without changing it.
        _donor_count = len(_x_before)
        _time_count = _x_before[0].shape[0]
        _global_key, _reset_key = jax.random.split(_key)
        _reset_donors = jax.random.permutation(_reset_key, _donor_count)
        _probability = float(_augmenter.reinjection_probability(int(iteration)))
        _inject = jnp.zeros((_donor_count, max(_time_count - 1, 0)), dtype=bool)
        _choices = jnp.zeros_like(_inject, dtype=jnp.int32)
        _group_donors = jnp.zeros(
            (_time_count, len(schema.experiment_groups), _donor_count),
            dtype=jnp.int32,
        )
        if _time_count > 1:
            _global_key, _mask_key, _group_key = jax.random.split(_global_key, 3)
            _decision_shape = (_donor_count, _time_count - 1)
            _inject = jax.random.bernoulli(
                _mask_key, _probability, _decision_shape
            )
            _choices = jax.random.randint(
                _group_key,
                _decision_shape,
                0,
                len(schema.experiment_groups),
            )
            _donor_rows = []
            for _time_index in range(1, _time_count):
                _time_donors = []
                for _group_index in range(len(schema.experiment_groups)):
                    _global_key, _donor_key = jax.random.split(_global_key)
                    _time_donors.append(
                        jax.random.permutation(_donor_key, _donor_count)
                    )
                _donor_rows.append(jnp.stack(_time_donors))
            _group_donors = _group_donors.at[1:].set(jnp.stack(_donor_rows))

        if input_mode == "perfect_rollout":
            _x_after, _y_after = _augmenter.advance_pool(
                _x_before, _y_before, int(iteration), _key
            )
            _augmenter.noise_strength = 0.0
            _x_reinjected, _ = _augmenter.advance_pool(
                _x_before, _y_before, int(iteration), _key
            )
        else:
            _x_after, _y_after = _augmenter.initialize_pool(_key)
            _augmenter.noise_strength = 0.0
            _x_reinjected, _ = _augmenter.initialize_pool(_key)
            _inject = jnp.zeros_like(_inject)
            _choices = jnp.zeros_like(_choices)
            _reset_donors = jnp.arange(_donor_count)
            _group_donors = jnp.zeros_like(_group_donors)
        return {
            "augmenter": _augmenter,
            "x_before": jnp.stack(_x_before),
            "x_after": jnp.stack(_x_after),
            "x_reinjected": jnp.stack(_x_reinjected),
            "y_before": jnp.stack(_y_before),
            "y_after": jnp.stack(_y_after),
            "probability": _probability,
            "inject": _inject,
            "group_choices": _choices,
            "reset_donors": _reset_donors,
            "group_donors": _group_donors,
            "batch_count_before": len(_x_before),
            "batch_count_after": len(_x_after),
            "input_mode": input_mode,
        }

    return (run_augmentation_preview,)


@app.cell
def _(
    augmenter_decay_start,
    augmenter_input_mode,
    augmenter_iteration,
    augmenter_noise,
    augmenter_probability_end,
    augmenter_probability_start,
    augmenter_run,
    augmenter_seed,
    augmenter_state_channels,
    augmenter_total_iterations,
    loaded_array,
    loaded_schema,
    run_augmentation_preview,
):
    augmenter_run
    augmentation_preview = run_augmentation_preview(
        loaded_array,
        loaded_schema,
        augmenter_state_channels.value,
        augmenter_noise.value,
        augmenter_probability_start.value,
        augmenter_probability_end.value,
        augmenter_decay_start.value,
        augmenter_total_iterations.value,
        augmenter_iteration.value,
        augmenter_seed.value,
        augmenter_input_mode.value,
    )
    return (augmentation_preview,)


@app.cell
def _(augmentation_preview, loaded_schema, mo, np):
    preview_x_before = np.asarray(augmentation_preview["x_before"])
    preview_x_after = np.asarray(augmentation_preview["x_after"])
    preview_x_reinjected = np.asarray(augmentation_preview["x_reinjected"])
    preview_y_before = np.asarray(augmentation_preview["y_before"])
    preview_y_after = np.asarray(augmentation_preview["y_after"])
    preview_inject = np.asarray(augmentation_preview["inject"])
    preview_group_choices = np.asarray(augmentation_preview["group_choices"])
    preview_reset_donors = np.asarray(augmentation_preview["reset_donors"])
    preview_group_donors = np.asarray(augmentation_preview["group_donors"])
    preview_input_mode = augmentation_preview["input_mode"]
    preview_state_names = tuple(loaded_schema.state_channels) + tuple(
        f"hidden {index}"
        for index in range(
            preview_x_before.shape[2] - loaded_schema.n_state_channels
        )
    )
    _target_unchanged = bool(np.array_equal(preview_y_before, preview_y_after))
    _cardinality_unchanged = (
        augmentation_preview["batch_count_before"]
        == augmentation_preview["batch_count_after"]
    )
    _summary = [
        {
            "pool operation": augmentation_preview["input_mode"],
            "effective reinjection probability": augmentation_preview["probability"],
            "observed reinjection fraction": float(preview_inject.mean())
            if preview_inject.size
            else 0.0,
            "batch count before": augmentation_preview["batch_count_before"],
            "batch count after": augmentation_preview["batch_count_after"],
            "targets unchanged": _target_unchanged,
        }
    ]
    _checks = mo.hstack(
        [
            mo.callout(
                "Augmenter preserved batch cardinality.",
                kind="success" if _cardinality_unchanged else "danger",
            ),
            mo.callout(
                "Targets remained unchanged.",
                kind="success" if _target_unchanged else "danger",
            ),
        ]
    )
    mo.vstack([mo.md("## 4. Pool lifecycle contract"), mo.ui.table(_summary), _checks])
    return (
        preview_group_choices,
        preview_group_donors,
        preview_input_mode,
        preview_inject,
        preview_reset_donors,
        preview_state_names,
        preview_x_after,
        preview_x_before,
        preview_x_reinjected,
        preview_y_after,
        preview_y_before,
    )


@app.cell
def _(loaded_names, loaded_schema, mo, preview_state_names, preview_x_before):
    preview_batch = mo.ui.slider(
        0, preview_x_before.shape[0] - 1, value=0, label="Batch"
    )
    preview_time = mo.ui.slider(
        0, preview_x_before.shape[1] - 1, value=0, label="Input timestep"
    )
    preview_channel = mo.ui.dropdown(
        options={_name: _index for _index, _name in enumerate(preview_state_names)},
        value=preview_state_names[0],
        label="State channel",
    )
    preview_group = mo.ui.dropdown(
        options={_name: _index for _index, _name in enumerate(loaded_schema.group_names)},
        value=loaded_schema.group_names[0],
        label="Experiment group",
    )
    preview_target_channel = mo.ui.dropdown(
        options={_name: _index for _index, _name in enumerate(loaded_names)},
        value=loaded_names[0],
        label="Target measurement channel",
    )
    mo.vstack(
        [
            mo.hstack([preview_batch, preview_time]),
            mo.hstack([preview_channel, preview_target_channel, preview_group]),
        ]
    )
    return (
        preview_batch,
        preview_channel,
        preview_group,
        preview_target_channel,
        preview_time,
    )


@app.cell
def _(
    mo,
    np,
    plt,
    preview_batch,
    preview_channel,
    preview_input_mode,
    preview_state_names,
    preview_time,
    preview_x_after,
    preview_x_before,
):
    _batch = preview_batch.value
    _time = preview_time.value
    _channel = int(preview_channel.value)
    _before = preview_x_before[_batch, _time, _channel]
    _after = preview_x_after[_batch, _time, _channel]
    _difference = _after - _before
    _limit = max(float(np.nanmax(np.abs(_difference))), 1e-8)
    _figure, _axes = plt.subplots(1, 4, figsize=(15, 3.7))
    _axes[0].imshow(_before, cmap="viridis")
    _before_title = (
        "advance_pool input: perfect rollout output t_(n+1)"
        if preview_input_mode == "perfect_rollout"
        else "initialize_pool input: snapshot t_n"
    )
    _axes[0].set_title(_before_title)
    _axes[1].imshow(_after, cmap="viridis")
    _axes[1].set_title("Next pool input: transition starts at t_n")
    _axes[2].imshow(_difference, cmap="coolwarm", vmin=-_limit, vmax=_limit)
    _axes[2].set_title("Signed difference")
    _axes[3].hist(_difference.ravel(), bins=80)
    _axes[3].set_title("Difference distribution")
    for _axis in _axes[:3]:
        _axis.axis("off")
    _figure.suptitle(
        f"batch {_batch}, input t={_time}, {preview_state_names[_channel]}"
    )
    _figure.tight_layout()
    mo.vstack([mo.md("## 5. Before / after augmentation"), _figure])
    return


@app.cell
def _(
    loaded_measurement_mask,
    loaded_names,
    mo,
    np,
    plt,
    preview_batch,
    preview_target_channel,
    preview_time,
    preview_y_after,
    preview_y_before,
):
    _batch = preview_batch.value
    _time = preview_time.value
    _channel = int(preview_target_channel.value)
    _before = preview_y_before[_batch, _time, _channel]
    _after = preview_y_after[_batch, _time, _channel]
    _difference = _after - _before
    _available = bool(loaded_measurement_mask[_batch, _time + 1, _channel])
    _limit = max(float(np.nanmax(np.abs(_difference))), 1e-8)
    _figure, _axes = plt.subplots(1, 3, figsize=(12, 3.7))
    _axes[0].imshow(_before, cmap="viridis")
    _axes[0].set_title("y before pool operation")
    _axes[1].imshow(_after, cmap="viridis")
    _axes[1].set_title("y after pool operation")
    _axes[2].imshow(_difference, cmap="coolwarm", vmin=-_limit, vmax=_limit)
    _axes[2].set_title("y difference")
    for _axis in _axes:
        _axis.axis("off")
    _figure.suptitle(
        f"batch {_batch}, target slot {_time} (snapshot index {_time + 1}), "
        f"{loaded_names[_channel]}; available={_available}"
    )
    _figure.tight_layout()
    mo.vstack(
        [
            mo.md("### Target pool for the same batch/time selection"),
            _figure,
        ]
    )
    return


@app.cell
def _(
    loaded_schema,
    mo,
    np,
    plt,
    preview_batch,
    preview_group,
    preview_group_choices,
    preview_group_donors,
    preview_inject,
    preview_reset_donors,
):
    _batch = preview_batch.value
    _group = int(preview_group.value)
    _decision_rows = []
    for _time_offset in range(preview_inject.shape[1]):
        _chosen_group = int(preview_group_choices[_batch, _time_offset])
        _decision_rows.append(
            {
                "input timestep": _time_offset + 1,
                "reinject": bool(preview_inject[_batch, _time_offset]),
                "chosen group": loaded_schema.group_names[_chosen_group],
                "chosen donor": int(
                    preview_group_donors[
                        _time_offset + 1, _chosen_group, _batch
                    ]
                ),
            }
        )
    _figure, _axis = plt.subplots(figsize=(9, 3.5))
    _choice_map = np.where(preview_inject, preview_group_choices + 1, 0)
    _image = _axis.imshow(_choice_map, aspect="auto", cmap="tab20")
    _axis.set_xlabel("input timestep after initial state")
    _axis.set_ylabel("batch")
    _axis.set_title("Reinjection decisions (0 means no reinjection)")
    _figure.colorbar(_image, ax=_axis, label="group index + 1")
    _figure.tight_layout()
    _selected_donors = preview_group_donors[:, _group]
    mo.vstack(
        [
            mo.md("## 6. Exact random decisions"),
            mo.md(
                f"Initial-state donor for batch {_batch}: "
                f"**{int(preview_reset_donors[_batch])}**"
            ),
            mo.ui.table(_decision_rows),
            _figure,
            mo.md(
                f"Selected-group donor matrix for **{loaded_schema.group_names[_group]}**: "
                f"shape `{_selected_donors.shape}`."
            ),
        ]
    )
    return


@app.cell
def _(mo, np, plt, preview_state_names, preview_x_after, preview_x_before):
    _observed_channels = len(preview_state_names)
    _delta = preview_x_after[:, :, :_observed_channels] - preview_x_before[:, :, :_observed_channels]
    _stats = []
    for _channel in range(_observed_channels):
        _values = _delta[:, :, _channel].reshape(-1)
        _stats.append(
            {
                "state channel": preview_state_names[_channel],
                "mean change": float(np.mean(_values)),
                "std change": float(np.std(_values)),
                "mean absolute change": float(np.mean(np.abs(_values))),
                "max absolute change": float(np.max(np.abs(_values))),
            }
        )
    _figure, _axis = plt.subplots(figsize=(max(8, _observed_channels * 0.7), 3.8))
    _axis.boxplot(
        [_delta[:, :, _channel].reshape(-1) for _channel in range(_observed_channels)],
        showfliers=False,
    )
    _axis.set_xticks(
        range(1, _observed_channels + 1),
        preview_state_names,
        rotation=60,
        ha="right",
    )
    _axis.set_ylabel("after − before")
    _figure.tight_layout()
    mo.vstack([mo.md("## 7. Per-channel change statistics"), mo.ui.table(_stats), _figure])
    return


@app.cell
def _(mo, np, plt, preview_x_after, preview_x_reinjected):
    _noise = preview_x_after - preview_x_reinjected
    _noise_values = _noise.reshape(-1)
    _noise_summary = [
        {
            "mean": float(np.mean(_noise_values)),
            "standard deviation": float(np.std(_noise_values)),
            "mean absolute magnitude": float(np.mean(np.abs(_noise_values))),
            "maximum absolute magnitude": float(np.max(np.abs(_noise_values))),
        }
    ]
    _figure, _axis = plt.subplots(figsize=(8, 3.5))
    _axis.hist(_noise_values, bins=100, density=True)
    _axis.set_title("Noise added after reinjection")
    _axis.set_xlabel("noise value")
    _axis.grid(alpha=0.2)
    _figure.tight_layout()
    mo.vstack(
        [
            mo.md("## 8. Isolated noise diagnostics"),
            mo.ui.table(_noise_summary),
            _figure,
        ]
    )
    return


if __name__ == "__main__":
    app.run()

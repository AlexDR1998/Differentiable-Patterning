"""NODAL knockout evaluation of trained micropattern NCAs.

Rollouts under a knockout, cell-type shares of model and measured colonies,
and small helpers for picking models. Used by ``nodal_ko_eval.py`` and the
figure notebooks (through ``figures/data.py``).
"""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from omegaconf import OmegaConf

from Common.dataloader.cell_type_fit import radial_rings
from Common.dataloader.cell_type_shares import (
    colony_label_shares,
    pattern_counts,
    rule_stains,
    stain_counts_from_patterns,
)
from Common.dataloader.cell_types import marker_values
from Experiments.micropatterns.density_ko_sweep import density_converter, model_pattern_counts, record_label
from Experiments.model_registry import open_model_bundle
from NCA.trainer.intervention import rollout_model_sampled, rollout_model_with_blocked_channel_sampled

# Knockout hour of each evaluation condition (None: no knockout)
CONDITION_KO_HOURS = {"baseline": None, "ko_24h": 24, "ko_0h": 0}


def model_option_label(record):
    """Model picker label, with the projection and a short ID."""
    return (
        f"{record_label(record)} · {record['projection']} "
        f"ds{record['downsample']} [{record['model_id'][-8:]}]"
    )

def read_model_settings(registry, model_ids):
    """Projection and downsample of each model, read from its bundle config.

    The manifests are scanned once; IDs not found (e.g. aliases) fall
    back to ``registry.get``.
    """
    wanted = set(model_ids)
    bundles = {}
    for path in registry.bundle_paths():
        model_id = str(OmegaConf.load(path / "manifest.yaml").get("id", ""))
        if model_id in wanted:
            bundles[model_id] = open_model_bundle(path)
    settings = {}
    for model_id in model_ids:
        bundle = bundles.get(model_id) or registry.get(model_id)
        micropattern = bundle.config.data.micropattern
        settings[model_id] = {
            "projection": micropattern.projection if micropattern else "n/a",
            "downsample": int(bundle.config.data.downsample),
        }
    return settings

def rules_projection(cell_type_rules):
    """Projection a cell type file was tuned on (older files used max)."""
    return str(cell_type_rules.preprocessing.get("projection", "max"))

def marker_channel_indices(channel_names, cell_type_rules):
    """Index of each rule marker in ``channel_names`` (by channel, else by
    marker name as in model states), in the order of the rule markers."""
    channel_names = list(channel_names)
    return [
        channel_names.index(channel) if channel in channel_names else channel_names.index(marker)
        for marker, channel in cell_type_rules.channels.items()
    ]

def model_stain_counts(full_counts, cell_type_rules):
    """Per-stain counts from counts over every rule marker (model output)."""
    stains = [tuple(_channels) for _channels in rule_stains(cell_type_rules).values()]
    return stain_counts_from_patterns(full_counts, cell_type_rules.markers, stains)

def model_label_shares(
    image, channel_names, cell_type_rules, colony, n_rings, resolve_several=0.0
):
    """Label shares (``label_names`` order) of one model state ``[C, H, W]``.

    Every marker is in every pixel, but shares are estimated per stain and
    ring as for the measured data, so the two are comparable.
    """
    counts = model_pattern_counts(image, channel_names, cell_type_rules, colony, n_rings)
    return colony_label_shares(
        model_stain_counts(counts, cell_type_rules),
        cell_type_rules,
        resolve_several=resolve_several,
    )

def data_label_shares(
    image, channel_names, cell_type_rules, colony, n_rings, resolve_several=0.0
):
    """Label shares of one measured image ``[C, H, W]``, combining the
    stains of the rule file (e.g. SOX2 in stain 1, FOXA2 in stain 2)."""
    rings = radial_rings(colony, n_rings).ravel()
    stain_counts = []
    for _channels in rule_stains(cell_type_rules).values():
        values = marker_values(image, channel_names, cell_type_rules, axis=0, channels=_channels)
        markers = tuple(_channels)
        stain_counts.append((
            markers,
            pattern_counts(
                np.stack([np.ravel(values[_m]) for _m in markers]),
                [cell_type_rules.thresholds[_m] for _m in markers],
                rings,
                n_rings,
            ),
        ))
    return colony_label_shares(stain_counts, cell_type_rules, resolve_several=resolve_several)

def evaluate_condition(model, bundle, condition_data, condition, seed):
    """Roll out one model, blocking recurrent NODAL reads when requested."""

    initial = condition_data["initial_state"]
    channels = int(model.N_CHANNELS)
    if initial.shape[0] > channels:
        raise ValueError(
            f"{bundle.id} has {channels} state channels but its input has "
            f"{initial.shape[0]}."
        )
    state = jnp.pad(
        jnp.asarray(initial),
        ((0, channels - initial.shape[0]), (0, 0), (0, 0)),
    )

    boundary_state = jnp.asarray(condition_data["boundary"])
    boundary_mode = bundle.config.trainer.boundary_mode

    time_labels = tuple(
        f"{int(hour)}h" for hour in bundle.config.data.micropattern.timesteps
    )
    steps_per_observation = int(bundle.config.run.t)
    frame_indices = tuple(
        time_index * steps_per_observation
        for time_index in range(len(time_labels))
    )
    total_steps = frame_indices[-1]
    observation_steps = jnp.asarray(frame_indices, dtype=jnp.int32)
    rollout_key = jr.PRNGKey(seed)

    if condition == "baseline":
        selected_states = rollout_model_sampled(
            model,
            state,
            boundary_state,
            boundary_mode,
            rollout_key,
            total_steps,
            observation_steps,
        )
    else:
        knockout_hour = 0 if condition == "ko_0h" else 24
        knockout_step = (knockout_hour // 12) * steps_per_observation
        selected_states = rollout_model_with_blocked_channel_sampled(
            model,
            state,
            boundary_state,
            boundary_mode,
            rollout_key,
            total_steps,
            condition_data["nodal_channel"],
            jnp.asarray(knockout_step, dtype=jnp.int32),
            observation_steps,
        )

    prediction = np.asarray(selected_states)[
        :, condition_data["output_channels"]
    ]
    return prediction, time_labels

def make_render_initial_condition(initial, boundary, shape, radius, key):
    """Resample the measured 0h state inside a requested colony shape."""

    source_mask = np.asarray(boundary).squeeze().astype(bool)
    height, width = source_mask.shape
    y, x = np.mgrid[-1.0:1.0:complex(height), -1.0:1.0:complex(width)]
    if shape == "colony":
        render_mask = source_mask
    elif shape == "circle":
        render_mask = x**2 + y**2 <= float(radius) ** 2
    elif shape == "ellipse":
        render_mask = (x / float(radius)) ** 2 + (y / (0.62 * float(radius))) ** 2 <= 1.0
    elif shape == "triangle":
        scaled_y = y / float(radius)
        scaled_x = x / float(radius)
        render_mask = (
            (scaled_y >= -0.72)
            & (scaled_y <= 0.85)
            & (np.abs(scaled_x) <= (0.85 - scaled_y) / 1.57)
        )
    elif shape == "full":
        render_mask = np.ones((height, width), dtype=bool)
    else:
        raise ValueError(f"Unknown render shape {shape!r}.")

    source_values = jnp.asarray(initial)[:, source_mask]
    if source_values.shape[1] == 0:
        raise ValueError("The source colony mask contains no pixels.")
    sample_indices = jr.randint(
        key,
        (height, width),
        0,
        source_values.shape[1],
    )
    sampled = source_values[:, sample_indices.reshape(-1)]
    sampled = sampled.reshape(initial.shape[0], height, width)
    shaped_initial = jnp.where(jnp.asarray(render_mask)[None], sampled, 0.0)
    shaped_boundary = jnp.broadcast_to(
        jnp.asarray(render_mask, dtype=jnp.asarray(boundary).dtype),
        jnp.asarray(boundary).shape,
    )
    return shaped_initial, shaped_boundary, render_mask

def ko_time_label(hour, measurement_hour, prefix=""):
    """Label of a KO time; a KO at (or after) the measurement is the baseline."""
    if float(hour) >= float(measurement_hour):
        return "Baseline"
    return f"{prefix}{float(hour):g}h"

def ko_time_ticks(measurement_hour):
    """Ticks every 12h up to the measurement, which is labelled baseline."""
    ticks = [*np.arange(0.0, float(measurement_hour), 12.0), float(measurement_hour)]
    return ticks, [ko_time_label(tick, measurement_hour) for tick in ticks]

def parse_one_based_indices(specification):
    """Parse comma-separated indices and inclusive ranges such as 1,3-5."""

    indices = []
    for part in specification.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, stop_text = part.split("-", 1)
            start, stop = int(start_text), int(stop_text)
            if stop < start:
                raise ValueError(f"Invalid descending replicate range {part!r}.")
            indices.extend(range(start, stop + 1))
        else:
            indices.append(int(part))
    indices = list(dict.fromkeys(indices))
    if not indices or any(index < 1 for index in indices):
        raise ValueError("Replicates must contain positive one-based indices.")
    return tuple(indices)

# How the density × KO sweep sets the density (old results: covered share)
DENSITY_UNITS = {
    "per_mm2": "Initial cells per mm²",
    "per_colony": "Initial total cells in the colony",
    "share": "Initial pixel coverage (covered share of the colony)",
}
# Short axis labels, and the columns of each density measure in the density × KO tables
DENSITY_AXIS_LABELS = {"per_mm2": "Cells per mm²", "per_colony": "Cells per colony", "share": "Pixel coverage"}
DENSITY_COLUMNS = {"per_mm2": "cells_per_mm2", "per_colony": "cells_per_colony", "share": "covered_share"}


# References rolled out beside the density sweeps (result "source"), and how plots draw them
REFERENCE_LABELS = {"true_data": "True data", "measured_density": "Synthetic at measured density"}
REFERENCE_MARKERS = {"true_data": "*", "measured_density": "s"}
REFERENCE_LINESTYLES = {"true_data": (0, (5, 2)), "measured_density": (0, (1, 1.5))}


REFERENCE_SHORT_LABELS = {"true_data": "True data", "measured_density": "Measured\ndensity"}


def draw_reference_strip(axis, positions, rows, sources, image, vertical=False):
    """A thin heatmap of the references beside a main heatmap.

    ``rows`` ``[references, positions]`` are the references' values at
    ``positions`` (the main heatmap's columns, or its rows when
    ``vertical``), one per entry of ``sources``, drawn with the colour map
    and scale of ``image`` (the main heatmap's mesh). NaN shows as empty.
    """
    rows = np.ma.masked_invalid(np.asarray(rows, dtype=float))
    labels = [REFERENCE_SHORT_LABELS[_source] for _source in sources]
    ticks = np.arange(len(sources)) + 0.5
    edges = cell_edges(np.asarray(positions, dtype=float))
    common = {"shading": "flat", "cmap": image.get_cmap(), "norm": image.norm}
    if vertical:
        axis.pcolormesh(np.arange(len(sources) + 1), edges, rows.T, **common)
        axis.set_xticks(ticks, labels, rotation=90)
        axis.tick_params(axis="x", labelsize=7, length=0)
    else:
        axis.pcolormesh(edges, np.arange(len(sources) + 1), rows, **common)
        axis.set_yticks(ticks, labels)
        axis.tick_params(axis="y", labelsize=7, length=0)


def reference_legend_handles(sources):
    """Grey legend entries for the reference ``sources`` (their marker and line style)."""
    import matplotlib.pyplot as plt

    return [
        plt.Line2D(
            [], [], color="0.35", marker=REFERENCE_MARKERS[_source], linestyle=REFERENCE_LINESTYLES[_source],
            markersize=6, label=REFERENCE_LABELS[_source],
        )
        for _source in sources
    ]


def density_column(unit, sweep_unit):
    """Column of the density × KO tables holding ``unit``: the sweep's own
    measure is ``actual_density`` (the density each run used)."""
    return "actual_density" if unit == sweep_unit else DENSITY_COLUMNS[unit]


def main_density_unit(summary, sweep_unit, shown_units):
    """The density measure a plot is drawn linear in: the sweep's own when
    it is among ``shown_units`` (or nothing is shown), else the first shown
    measure ``summary`` has at every density (old results lack cell counts)."""
    shown = [_unit for _unit in DENSITY_AXIS_LABELS if _unit in shown_units]
    if not shown or sweep_unit in shown:
        return sweep_unit
    for unit in shown:
        if np.isfinite(summary[DENSITY_COLUMNS[unit]].to_numpy(dtype=float)).all():
            return unit
    return sweep_unit


def density_conversions(summary, unit, extra_units, main_unit=None):
    """``{extra unit: (forward, inverse)}`` from the plotted density (in
    ``main_unit``, default the sweep's ``unit``) to each of ``extra_units``,
    through the mean of each measure at every requested density in
    ``summary``. Units that cannot be converted (e.g. no cell counts in old
    results) are left out."""
    main_unit = main_unit or unit
    main = density_column(main_unit, unit)
    extras = [_unit for _unit in extra_units if _unit != main_unit]
    columns = list(dict.fromkeys([main, *(density_column(_unit, unit) for _unit in extras)]))
    levels = summary.groupby("density")[columns].mean()
    conversions = {}
    for extra in extras:
        functions = density_converter(levels[main], levels[density_column(extra, unit)])
        if functions is not None:
            conversions[extra] = functions
    return conversions


def add_density_axes(axis, which, main_unit, conversions, gap):
    """Label the density axis of ``axis`` (``which``: "x" or "y"), drawn
    linear in ``main_unit``, and add one axis per other measure in
    ``conversions`` (``density_conversions``), stacked below the x axis or
    left of the y axis, ``gap`` apart in axes coordinates."""
    axis.set(**{f"{which}label": DENSITY_AXIS_LABELS[main_unit]})
    for offset, (extra, functions) in enumerate(conversions.items(), start=1):
        if which == "x":
            axis.secondary_xaxis(-gap * offset, functions=functions).set_xlabel(DENSITY_AXIS_LABELS[extra])
        else:
            axis.secondary_yaxis(-gap * offset, functions=functions).set_ylabel(DENSITY_AXIS_LABELS[extra])


def measured_density(results, unit):
    """Mean measured density of the real 0 h colonies of ``results`` in
    ``unit``, or None when the results did not record it."""
    values = []
    for result in results:
        if unit == result["density_unit"]:
            value = result.get("measured_density")
        elif unit == "share":
            value = result.get("measured_covered_share")
        elif "measured_cells_per_mm2" not in result:
            value = None
        elif unit == "per_mm2":
            value = result["measured_cells_per_mm2"]
        else:
            value = result["measured_cells_per_mm2"] * result["colony_mm2"] if "colony_mm2" in result else None
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return float(np.mean(values)) if values else None


def uncertainty_band(samples, mode, clip=True):
    """Mean and band of ``samples`` (``std``, ``sem``, ``percentile`` or
    ``none``), the band clipped to [0, 1] unless ``clip`` is False."""
    samples = np.asarray(samples, dtype=float)
    mean = float(np.mean(samples))
    spread = float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0
    if mode == "std":
        lower, upper = mean - spread, mean + spread
    elif mode == "sem":
        spread /= np.sqrt(len(samples))
        lower, upper = mean - spread, mean + spread
    elif mode == "percentile":
        lower, upper = np.percentile(samples, [2.5, 97.5])
    else:
        lower = upper = mean
    if not clip:
        return mean, float(lower), float(upper)
    return mean, max(0.0, float(lower)), min(1.0, float(upper))


def density_rows(levels, tolerance=0.005):
    """Heatmap rows of a density sweep on a linear axis: ``(row values, row of level)``.

    ``levels`` maps each requested density to the density actually used
    (a pandas Series, sorted by requested density). Requested densities
    cut down to about the same value (within ``tolerance`` of the range,
    e.g. by the cell layers) share one row at their mean.
    """
    span = float(levels.max() - levels.min()) or 1.0
    merged, row_of = [], {}
    for density, actual in levels.items():
        if merged and actual - merged[-1][-1] <= tolerance * span:
            merged[-1].append(actual)
        else:
            merged.append([actual])
        row_of[density] = len(merged) - 1
    return np.array([np.mean(values) for values in merged]), row_of


def cell_edges(centres):
    """Edges of heatmap cells around ``centres``: halfway between neighbours,
    the end cells as wide as their neighbour (a lone value gets an
    arbitrary width)."""
    centres = np.asarray(centres, dtype=float)
    if len(centres) == 1:
        half = 0.05 * abs(centres[0]) or 0.5
        return np.array([centres[0] - half, centres[0] + half])
    middle = 0.5 * (centres[1:] + centres[:-1])
    return np.concatenate((
        [centres[0] - (middle[0] - centres[0])], middle, [centres[-1] + (centres[-1] - middle[-1])],
    ))

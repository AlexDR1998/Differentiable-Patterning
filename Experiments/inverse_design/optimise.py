"""Optimise the colony geometry of a trained micropattern NCA for a cell type.

Loads a model bundle and its 0 h training data, synthesises initial textures
over a grid ``geometry.grid_scale`` times larger than the training one, and
optimises a fixed-area Fourier-radial shape so that the model's share of the
target cell type (rules from a cell type file) after ``rollout.hours`` is as
high as possible. The circle and a few ellipses of the same area are scored
first with the same random numbers. The result is saved as a record in the
model store (``Experiments/inverse_design/records.py``).

With ``density.mode`` fixed or optimise, the textures are made of whole
cells cut out of the training colonies and placed at random, and each
rollout keeps enough of them for a cell density (the share of the colony
covered by cells; ``NCA/inverse_design/density.py``). With optimise the
density is learned together with the shape.
"""

from dataclasses import asdict, replace
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from Common.dataloader.cell_types import load_cell_type_rules
from Experiments.config import config_to_dict
from Experiments.config_helpers import build_wandb_tags, open_registry_bundle, resolve_store_root
from Experiments.inverse_design.records import create_run_id, publish_inverse_design_record
from Experiments.micropatterns.reference import cell_library, load_reference
from Experiments.micropatterns.synthetic_ic import synthesise_texture
from Experiments.micropatterns.texture_settings import (
    TEXTURE_SETTING_KEYS,
    load_texture_settings,
    scaled_patch_geometry,
)
from Experiments.micropatterns.texture_synthesis import place_cells
from NCA.inverse_design import (
    CellDensity,
    CellGaps,
    CellTypeShareObjective,
    FourierRadialGeometry,
    GeometryPenalties,
    OptimisationConfig,
    binary_trajectories,
    evaluate_binary,
    optimise_geometry,
    pack_cells,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RADIUS_RANGE = (0.1, 0.95)


def check_projection(cell_type_rules, data_config):
    """The thresholds only hold for images projected as they were tuned on."""

    rules_projection = str(cell_type_rules.preprocessing.get("projection", "max"))
    data_projection = str(data_config.micropattern.projection)
    if rules_projection != data_projection:
        raise ValueError(
            f"The cell type file was tuned on {rules_projection!r} projections, "
            f"but the model was trained on {data_projection!r}"
        )


def rollout_steps(hours, steps_per_12h):
    steps = hours * steps_per_12h / 12
    if steps != int(steps):
        raise ValueError(f"{hours} h is not a whole number of NCA steps ({steps_per_12h} per 12 h)")
    return int(steps)


def scaled_grid_shape(shape, grid_scale):
    """Training grid shape times ``grid_scale``, rounded to odd sizes so the grid has a centre pixel."""

    return tuple(int(2 * round(size * grid_scale / 2) + 1) for size in shape)


def reference_radius(support, grid_shape):
    """Mean radius of the reference colonies (pixels), in ``coordinate_grid`` units."""

    pixels = np.sqrt(np.mean(np.sum(support, axis=(-2, -1))) / np.pi)
    return float(pixels * 2.0 / (np.mean(grid_shape) - 1.0))


def resolve_texture(texture, projection):
    """The texture settings to use: from ``texture.file`` for ``projection`` if set."""

    if texture.file is None:
        return texture
    path = Path(texture.file)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return replace(texture, file=None, **load_texture_settings(path, projection))


def texture_bank(reference, texture, grid_shape, seed, downsample):
    """``[K, state channels, H, W]`` 0 h textures over the whole grid.

    Each texture takes every staining panel from a random training replicate
    (``synthetic_ic.synthesise_texture``). ``texture`` is resolved
    (``resolve_texture``); its patch sizes are rescaled to the data's
    ``downsample``.
    """

    settings = {key: getattr(texture, key) for key in TEXTURE_SETTING_KEYS}
    primary = np.asarray(reference["schema"].primary_measurements)
    return np.stack([
        synthesise_texture(reference, settings, grid_shape, seed * 100003 + index, downsample)[primary]
        for index in range(texture.bank_size)
    ])


def texture_summary(texture_config, texture, projection, downsample):
    """The texture settings a run used, for its record and W&B config."""

    patch_size, patch_overlap, patch_jitter = scaled_patch_geometry(
        texture.patch_size, texture.patch_overlap, texture.patch_jitter,
        texture.patch_downsample, downsample,
    )
    return {
        "file": texture_config.file,
        "projection": projection,
        **asdict(texture),
        "used_patch_pixels": {"size": patch_size, "overlap": patch_overlap, "jitter": patch_jitter},
    }


def baseline_geometries(radius, amplitudes):
    """The circle, and harmonic-2 ellipses at 0 and 45 degrees, all of area pi R^2."""

    baselines = {"circle": FourierRadialGeometry(1, radius=radius, radius_range=RADIUS_RANGE)}
    for amplitude in amplitudes:
        for degrees in (0, 45):
            geometry = FourierRadialGeometry(
                2, radius=radius, radius_range=RADIUS_RANGE, coefficient_limit=1.0,
                angle=np.deg2rad(degrees),
            )
            geometry = eqx.tree_at(
                lambda g: g.raw_cos, geometry, jnp.asarray([0.0, np.arctanh(amplitude)])
            )
            baselines[f"ellipse_{amplitude:g}_{degrees}deg"] = geometry
    return baselines


def density_model(reference, density_config, grid_shape, bank_size, seed):
    """``(texture bank, CellGaps, starting density, coverage rows)`` for the cell density textures.

    Whole cells are cut out of the training colonies (``cell_library``) and
    each of the ``bank_size`` textures places them at random over the grid
    (``place_cells``), packed as ``NCA/inverse_design/density.py`` expects.
    """

    library = cell_library(
        reference, density_config.markers, density_config.split_distance,
        density_config.interior_fraction, density_config.min_cell_area,
    )
    schema = reference["schema"]
    primary = np.asarray(schema.primary_measurements, dtype=int)
    panel_of = {
        int(index): panel for panel, (indices, _) in enumerate(library["panels"]) for index in indices
    }
    bank, coverages = [], []
    for index in range(bank_size):
        textures, ranks, coverage = place_cells(
            library["panels"], reference["measurements"].shape[1], grid_shape,
            density_config.layers, density_config.attempts, seed * 100003 + index,
        )
        bank.append(np.asarray(pack_cells(jnp.asarray(textures[:, primary]), jnp.asarray(ranks))))
        coverages.append(coverage)
    cell_gaps = CellGaps(
        gap_values=jnp.asarray(library["gap_values"]),
        channel_panels=jnp.asarray([panel_of[int(index)] for index in primary]),
        layer_coverage=jnp.asarray(np.mean(coverages, axis=0)),
        panels=len(library["panels"]),
        softness=density_config.softness,
    )
    initial = density_config.initial if density_config.initial is not None else library["density"]
    return np.stack(bank), cell_gaps, float(initial), library["coverage"]


def baseline_designs(radius, amplitudes, initial_density, density_multiples):
    """Baseline ``{name: (geometry, density or None)}``: circle and ellipses at the
    starting density (None: no density model), and circles at other densities."""

    density = None if initial_density is None else CellDensity(initial_density)
    designs = {
        name: (geometry, density)
        for name, geometry in baseline_geometries(radius, amplitudes).items()
    }
    if initial_density is not None:
        for multiple in density_multiples:
            value = float(np.clip(multiple * initial_density, 0.01, 0.99))
            designs[f"circle_density_{multiple:g}x"] = (designs["circle"][0], CellDensity(value))
    return designs


def run(cfg):
    """Optimise a colony geometry for a trained micropattern model and save the record."""

    design = cfg.inverse_design
    key = jax.random.PRNGKey(cfg.seed)
    model_key, geometry_key, optimise_key, evaluation_key = jax.random.split(key, 4)
    bundle = open_registry_bundle(cfg.checkpoint.model_id, cfg.checkpoint.store_root)
    model = bundle.load_model(key=model_key)
    data_config = bundle.config.data
    if data_config.dataset != "micropatterns_260726":
        raise ValueError("Inverse design needs a model trained on micropatterns_260726")

    cell_types_path = Path(design.cell_types_file)
    if not cell_types_path.is_absolute():
        cell_types_path = REPOSITORY_ROOT / cell_types_path
    cell_type_rules = load_cell_type_rules(cell_types_path)
    check_projection(cell_type_rules, data_config)
    if design.rollout.hours != cell_type_rules.hour:
        print(
            f"Warning: the cell type thresholds were tuned at {cell_type_rules.hour} h, "
            f"the rollout is scored at {design.rollout.hours} h"
        )

    reference = load_reference(data_config)
    state_channel_names = tuple(reference["schema"].state_channels)
    training_shape = reference["measurements"].shape[-2:]
    grid_shape = scaled_grid_shape(training_shape, design.geometry.grid_scale)
    gaps = initial_density = coverage_rows = texture_settings = None
    if design.density.mode == "none":
        projection = str(data_config.micropattern.projection)
        texture = resolve_texture(design.texture, projection)
        texture_settings = texture_summary(design.texture, texture, projection, data_config.downsample)
        print(f"Texture settings ({design.texture.file or 'config'}, {projection}): {texture_settings}")
        biology_bank = texture_bank(reference, texture, grid_shape, cfg.seed, data_config.downsample)
    else:
        # Whole measured cells placed at random; the patch texture settings are not used
        biology_bank, gaps, initial_density, coverage_rows = density_model(
            reference, design.density, grid_shape, design.texture.bank_size, cfg.seed
        )
        for row in coverage_rows:
            print(
                f"  replicate {row['replicate']} {row['panel']}: covered {row['coverage']:.3f}, "
                f"{row['cells']} whole cells"
            )
        layer_coverage = ", ".join(f"{value:.3f}" for value in np.asarray(gaps.layer_coverage))
        print(
            f"Cell density: starting at {initial_density:.3f} of the colony covered; "
            f"layers cover {layer_coverage}, at most {float(gaps.max_density):.3f} together"
        )
        if initial_density > float(gaps.max_density):
            print("Warning: the starting density is above what the layers can cover; add layers")
    radius = reference_radius(reference["support"], grid_shape) * design.geometry.radius_scale
    print(f"Grid {grid_shape} (training {tuple(training_shape)}), colony radius {radius:.3f}")

    boundary_mode = bundle.config.trainer.boundary_mode
    total_steps = rollout_steps(design.rollout.hours, bundle.config.run.t)
    objective = CellTypeShareObjective.from_rules(
        cell_type_rules,
        state_channel_names,
        design.objective.target,
        temperature=design.objective.temperature,
        amount_weight=design.objective.amount_weight,
        maximise=design.objective.maximise,
    )
    penalties = GeometryPenalties(
        perimeter_weight=design.penalties.perimeter_weight,
        coefficient_weight=design.penalties.coefficient_weight,
        border_weight=design.penalties.border_weight,
    )
    optimisation_config = OptimisationConfig(
        iterations=design.iterations,
        learning_rate=design.optimiser.learn_rate,
        total_steps=total_steps,
        batch_size=design.batch_size,
        mask_softness=design.mask_softness.start,
        final_mask_softness=design.mask_softness.end,
        noise_strength=data_config.micropattern.noise_strength,
        boundary_mode=boundary_mode,
        gradient_clip=design.optimiser.gradient_clip_norm,
        fixed_radius=design.geometry.fixed_radius,
        optimise_geometry=design.geometry.optimise,
        optimise_density=design.density.mode == "optimise",
        log_every=design.log_every,
        evaluation_seeds=design.evaluation_seeds,
        scan_kind=design.rollout.scan_kind,
        checkpoints=design.rollout.checkpoints,
    )
    geometry = FourierRadialGeometry(
        design.geometry.harmonics,
        radius=radius,
        radius_range=RADIUS_RANGE,
        coefficient_limit=design.geometry.coefficient_limit,
        key=geometry_key,
        initial_scale=design.geometry.initial_scale,
    )
    geometry_settings = {
        "harmonics": design.geometry.harmonics,
        "radius_range": list(RADIUS_RANGE),
        "coefficient_limit": design.geometry.coefficient_limit,
    }
    share_maps = jax.jit(jax.vmap(lambda state: objective.probability(state, hard=True)))

    # Every evaluation uses evaluation_key, so the baselines and the optimised
    # shapes see the same textures, cell choices, noise and NCA updates.
    baselines = {}
    designs = baseline_designs(
        radius, design.baseline_ellipses, initial_density, design.density.baselines
    )
    for name, (baseline, density) in designs.items():
        evaluation = evaluate_binary(
            model, baseline, biology_bank, evaluation_key, objective, optimisation_config,
            None if density is None else gaps, density,
        )
        baselines[name] = (baseline, evaluation)
        print(f"{name}: {design.objective.target} share {evaluation.mean:.4f} ± {evaluation.std:.4f}")

    run_id = create_run_id(cfg)
    logger = None
    if cfg.logging.backend == "wandb":
        from Experiments.inverse_design.wandb_log import GeometryLogger

        logger = GeometryLogger(
            wandb_config={
                "project": cfg.logging.wandb.project,
                "group": cfg.logging.wandb.group,
                # The model's resolution, tagged as in its training run
                "tags": [*build_wandb_tags(cfg), f"data.downsample:{data_config.downsample}"],
                "name": run_id,
                "config": {**config_to_dict(cfg), "texture_used": texture_settings},
            },
            state_channel_names=state_channel_names,
            cell_type_rules=cell_type_rules,
        )
        if coverage_rows is not None:
            logger.log_density_reference(coverage_rows, initial_density)
        logger.log_baselines(
            baselines,
            {name: share_maps(evaluation.final_states) for name, (_, evaluation) in baselines.items()},
        )

    def log_fn(iteration, metrics, evaluation):
        if evaluation is not None:
            density_text = "" if evaluation.density is None else f", density {evaluation.density:.3f}"
            print(
                f"[{iteration}] binary {design.objective.target} share "
                f"{evaluation.mean:.4f} ± {evaluation.std:.4f}, border {evaluation.border_fraction:.3f}"
                f"{density_text}"
            )
        if logger is not None:
            maps = None if evaluation is None else share_maps(evaluation.final_states)
            logger.log_iteration(iteration, metrics, evaluation, maps)

    result = optimise_geometry(
        model,
        geometry,
        biology_bank,
        optimise_key,
        objective,
        optimisation_config,
        penalties,
        log_fn=log_fn,
        evaluation_key=evaluation_key,
        gaps=gaps,
        density=None if initial_density is None else CellDensity(initial_density),
    )

    wandb_run_id = None
    if logger is not None:
        logger.log_result(result, step=design.iterations)
        if design.video_seeds > 0:
            # Same random numbers as the evaluations, so the last frames are
            # the best evaluation's final states
            mask, trajectories = binary_trajectories(
                model, result.best_geometry, biology_bank, evaluation_key,
                optimisation_config, min(design.video_seeds, design.evaluation_seeds),
                gaps, result.best_density,
            )
            logger.log_trajectories(mask, trajectories, step=design.iterations)
        wandb_run_id = getattr(logger.run, "id", None)
        logger.finish()

    path = publish_inverse_design_record(
        store_root=resolve_store_root(cfg.checkpoint.store_root),
        run_id=run_id,
        cfg=cfg,
        bundle=bundle,
        cell_types_path=cell_types_path,
        cell_type_rules=cell_type_rules,
        state_channel_names=state_channel_names,
        geometry_settings=geometry_settings,
        total_steps=total_steps,
        boundary_mode=boundary_mode,
        baselines=baselines,
        result=result,
        biology_bank=biology_bank,
        wandb_run_id=wandb_run_id,
        gaps=gaps,
        coverage_rows=coverage_rows,
        texture_settings=texture_settings,
    )
    density_text = (
        "" if result.best_density is None else f", density {float(result.best_density.value):.3f}"
    )
    print(
        f"Best {design.objective.target} share {result.best_evaluation.mean:.4f} "
        f"(iteration {result.best_iteration}{density_text}), circle {baselines['circle'][1].mean:.4f}"
    )
    print(f"Saved inverse design record to {path}")
    return result

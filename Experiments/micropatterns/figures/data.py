"""Data loading shared by the micropattern figure notebooks.

Everything here returns plain numpy arrays and dicts. Heavy imports (JAX, the
dataloader, the model registry) happen inside the functions, so importing this
module is cheap.
"""

import pickle
import re
from pathlib import Path

import numpy as np
import tifffile
from matplotlib.colors import to_rgb

from Experiments.micropatterns.figures.style import FIGURE_ROOT, UNKNOWN_CELL_TYPE

# Knockout hour of each condition (None: control). Before the knockout the loader
# substitutes the control images (see source_condition in Common/dataloader/micropattern_260726.py).
KO_HOURS = {"ctrl": None, "sl24": 24, "sl0": 0}
TIMEPOINTS = (0, 12, 24, 36, 48)
DATASET_FOLDER = "260726_nca_dataset"
CACHE_DIRECTORY = FIGURE_ROOT / "cache"

# Per-projection cleaning of the projection training runs, copied from
# conf/experiments/micropattern_projections_training_sweep.yaml
PROJECTION_CLEANING = {
    "max": {"horizontal_stripe_widths": {}, "stripe_block": 0},
    "sum": {"horizontal_stripe_widths": {"cell_fate_s1/TBXT": 10, "cell_fate_s2/TBXT": 10}, "stripe_block": 270},
    "slice": {"horizontal_stripe_widths": {}, "stripe_block": 0},
}


def cached(name, compute, refresh=False, directory=CACHE_DIRECTORY):
    """``compute()``, read from (or written to) the pickle ``directory/name``."""
    path = Path(directory) / name
    if path.is_file() and not refresh:
        with open(path, "rb") as file:
            return pickle.load(file)
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(value, file)
    return value


def cache_name(*parts):
    """File-system safe cache name from ``parts``, e.g. ``cache_name("data", "max", 4)``."""
    return "_".join(re.sub(r"[^A-Za-z0-9.-]", "-", str(part)) for part in parts) + ".pkl"


# --- Data configs --------------------------------------------------------------


def _base_config():
    from omegaconf import OmegaConf

    from Experiments.micropatterns.config_helpers import REPOSITORY_ROOT

    return OmegaConf.load(REPOSITORY_ROOT / "Experiments" / "micropatterns" / "conf" / "base_config.yaml")


def knockout_data_config(groups, projection, downsample, normalisation=None):
    """Typed data config: the micropattern base config (the standard preprocessing) with the
    given experiment groups, projection and downsampling. ``normalisation`` None keeps the
    base config's."""
    from omegaconf import OmegaConf

    from Experiments.config import load_experiment_config
    from Experiments.micropatterns.config_helpers import measurement_count

    base = _base_config()
    channels = measurement_count(groups, base.data.micropattern.excluded_measurements)
    overrides = {"data": {
        "dataset": "micropatterns_260726",
        "downsample": downsample,
        "micropattern": {
            "data_channels": channels,
            "experiment_groups": list(groups),
            "projection": projection,
            "cleaning": {**PROJECTION_CLEANING[projection],
                         **({} if normalisation is None else {"normalisation": normalisation})},
        },
    }}
    return load_experiment_config(OmegaConf.merge(base, overrides)).data


def projection_data_config(projection, downsample):
    """Typed data config of a projection training run: the base config plus the sweep's data settings."""
    from omegaconf import OmegaConf

    from Experiments.config import load_experiment_config
    from Experiments.micropatterns.config_helpers import measurement_count

    base = _base_config()
    overrides = {"data": {
        "dataset": "micropatterns_260726",
        "downsample": downsample,
        "micropattern": {
            "task": "micropattern_multi_replicate",
            "data_channels": measurement_count(None, base.data.micropattern.excluded_measurements),
            "experiment_groups": ["cell_fate_s1", "cell_fate_s2", "rna_expression", "protein_response"],
            "projection": projection,
            "cleaning": PROJECTION_CLEANING[projection],
        },
    }}
    return load_experiment_config(OmegaConf.merge(base, overrides)).data


def run_data_config(path, index=0):
    """Data config of a training run: a model bundle folder, a generated
    ``manifest.yaml`` (entry ``index``, as ``run_config.py --index``) or a single config file."""
    from omegaconf import OmegaConf

    from Experiments.config import load_experiment_config
    from Experiments.config_workflow import load_config_from_entry, resolve_manifest_index
    from Experiments.model_registry import open_model_bundle

    path = Path(path).expanduser()
    if path.is_dir():
        return open_model_bundle(path).config.data
    cfg = OmegaConf.load(path)
    if "configs" in cfg:
        manifest = OmegaConf.to_container(cfg, resolve=True)
        _, entry = resolve_manifest_index(manifest, index=index, worker_index=0, worker_count=1)
        cfg = load_config_from_entry(entry, manifest_dir=path.parent)
    return load_experiment_config(cfg).data


def with_groups(data_config, groups):
    """``data_config`` loading only the experiment groups ``groups``.

    Bounds are per channel, so this changes no channel's preprocessing.
    """
    from dataclasses import replace

    from Experiments.micropatterns.config_helpers import measurement_count

    channels = measurement_count(groups, data_config.micropattern.excluded_measurements)
    return replace(data_config, micropattern=replace(
        data_config.micropattern, experiment_groups=tuple(groups), data_channels=channels))


def training_bounds(data_config, root):
    """Clipping bounds ``{channel: (low, high)}`` that training fits, or None when
    every trajectory sets its own (per_replicate normalisation).

    Runs the training data loading (``load_train_validation_data``) on the
    training replicates, so the bounds match training exactly.
    """
    from dataclasses import replace

    from Experiments.micropatterns.config_helpers import load_train_validation_data

    micropattern = data_config.micropattern
    if micropattern.cleaning.enabled and not micropattern.cleaning.shares_bounds:
        return None
    config = replace(data_config, micropattern=replace(micropattern, validation_replicates=None, pool_copies=1))
    (_, aux, names, *_), _ = load_train_validation_data(config, impath=str(root))
    return dict(zip(names, np.asarray(aux["histogram_bins"]).tolist()))


# --- Measured data ---------------------------------------------------------------


def load_dataset(root, data_config, conditions, replicate_indices, timesteps=TIMEPOINTS, histogram_bins=None,
                 **loader_options):
    """``load_micropattern_260726`` with the preprocessing of ``data_config`` (zero-based replicates).

    ``loader_options`` go to the loader as they are (e.g. ``replicate_count``, ``pool_copies``).
    """
    from Common.dataloader.disk_cache import cache_dir_from_environment
    from Common.dataloader.micropattern_260726 import load_micropattern_260726
    from Experiments.micropatterns.config_helpers import _cleaning_for_loader, _excluded_images

    micropattern = data_config.micropattern
    return load_micropattern_260726(
        Path(root).expanduser(),
        conditions=tuple(conditions),
        projection=micropattern.projection,
        timesteps=tuple(timesteps),
        downsample=data_config.downsample,
        replicate_indices=tuple(replicate_indices),
        histogram_bins=histogram_bins,
        experiment_groups=tuple(micropattern.experiment_groups),
        excluded_measurements=micropattern.excluded_measurements,
        hist_eqs=micropattern.histogram_percentiles,
        intensity_factors=micropattern.intensity_factors,
        cleaning=_cleaning_for_loader(micropattern.cleaning),
        excluded_images=_excluded_images(micropattern),
        cache_dir=cache_dir_from_environment(),
        **loader_options,
    )


def load_colonies(root, conditions, groups, projection, downsample, replicates, normalisation=None):
    """All timepoints of the given conditions and one-based ``replicates``, with the standard
    preprocessing (``normalisation`` "pooled": one intensity scale per channel for every condition).

    Returns images [B, T, C, H, W], measured [B, T, C], masks [B, H, W], the
    (condition, replicate) of each batch entry, the channel names and the channel layout
    (markers, groups and their channel indices, state channels, excluded measurements).
    """
    data_config = knockout_data_config(groups, projection, int(downsample), normalisation)
    dataset = load_dataset(root, data_config, conditions, [int(r) - 1 for r in replicates])
    schema = dataset.aux["channel_schema"]
    return {
        "images": np.asarray(dataset.data),
        "measured": np.asarray(dataset.measurement_mask),
        "masks": np.asarray(dataset.boundary_mask[:, 0]).astype(bool),
        "keys": list(zip(dataset.aux["batch_conditions"], (int(r) for r in dataset.aux["batch_replicates"]))),
        "replicates": tuple(int(r) for r in dataset.aux["batch_replicates"]),
        "channel_names": tuple(dataset.channel_names),
        "markers": tuple(channel.marker for channel in schema.measurement_channels),
        "groups": tuple(schema.group_names),
        "group_indices": tuple(schema.group_measurement_indices),
        "state_channels": tuple(schema.state_channels),
        "excluded": tuple(data_config.micropattern.excluded_measurements),
    }


def load_control_trajectory(root, replicate, projection, downsample):
    """Measured control trajectory of one replicate (one-based), every experiment group,
    with the same keys as a rollout (without the prediction)."""
    from Experiments.micropatterns.figures.style import EXPERIMENT_GROUPS

    data = load_colonies(root, ("ctrl",), EXPERIMENT_GROUPS, projection, downsample, (replicate,))
    return {
        "target": data["images"][0],
        "measured": data["measured"][0],
        "channel_names": data["channel_names"],
        "time_labels": tuple(f"{t}h" for t in TIMEPOINTS),
        "boundary": data["masks"][0],
    }


def load_knockout_trajectories(data_config, root, replicates, histogram_bins=None):
    """One trajectory per condition, ``replicates`` mapping condition → zero-based replicate.

    Returns ``{condition: {"images": [T, C, H, W], "measured": [T, G], "substituted": [T, G],
    "mask": [H, W]}}``, the channel names and the experiment group of each channel. The
    groups share one colony mask. ``histogram_bins`` (``{channel: (low, high)}``) fixes the
    clipping bounds; without it they are fitted to the loaded trajectories.
    """
    conditions = tuple(KO_HOURS)
    indices = tuple(sorted(set(replicates.values())))
    dataset = load_dataset(root, data_config, conditions, indices, histogram_bins=histogram_bins)
    channel_groups = [name.split("/")[0] for name in dataset.channel_names]
    groups = list(dict.fromkeys(channel_groups))
    trajectories = {}
    for condition_index, condition in enumerate(conditions):
        batch = condition_index * len(indices) + indices.index(replicates[condition])
        measured = np.asarray(dataset.measurement_mask[batch])
        trajectories[condition] = {
            "images": np.asarray(dataset.data[batch]),
            "measured": np.stack([measured[:, [g == group for g in channel_groups]].any(axis=-1)
                                  for group in groups], axis=-1),
            "substituted": np.asarray(dataset.aux["is_substituted"][batch]),
            "mask": np.asarray(dataset.boundary_mask[batch, 0]),
        }
    return trajectories, dataset.channel_names, channel_groups


def load_training_projection(data_config, root, condition, group, replicate):
    """One replicate (zero-based) of one stain group, loaded with the same settings as ``config_helpers.load_data``."""
    from dataclasses import replace

    config = replace(data_config, micropattern=replace(data_config.micropattern, experiment_groups=(group,)))
    return load_dataset(root, config, (condition,), (replicate,), timesteps=data_config.micropattern.timesteps,
                        replicate_count=1, pool_copies=1)


def locate_image(source_files, root, key):
    """(batch, time, group) index of the image with projection-free ``key`` in [B, T, G] source paths, or None."""
    from Common.dataloader.micropattern_projections import image_key

    root = Path(root).resolve()
    for index in np.ndindex(source_files.shape):
        path = source_files[index]
        if path and image_key(Path(path).resolve().relative_to(root).as_posix()) == key:
            return index
    return None


# --- 3D stacks -----------------------------------------------------------------


def read_stack(path, level=1):
    """[Z, C, Y, X] stack of one OME-TIFF pyramid level, its voxel size (µm) and channel names."""
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        if series.axes != "ZCYX":
            raise ValueError(f"Expected ZCYX axes, got {series.axes}")
        stack = series.levels[level].asarray()
        ome = tif.ome_metadata or ""

    def size(name):
        match = re.search(rf'{name}="([^"]+)"', ome)
        return float(match.group(1)) if match else 1.0

    dx = size("PhysicalSizeX") * series.shape[-1] / stack.shape[-1]
    dz = size("PhysicalSizeZ")
    names = re.findall(r'<Channel [^>]*Name="([^"]+)"', ome)
    if len(names) != stack.shape[1]:
        names = [f"ch{i}" for i in range(stack.shape[1])]
    return stack, dx, dz, names


def project_stack(stack, mode, z_slice, z_range):
    """[C, Y, X] image from a [Z, C, Y, X] stack: one slice, or the max or sum over z in z_range."""
    z0, z1 = z_range
    if mode == "slice":
        return stack[z_slice].astype(np.float32)
    if mode == "max":
        return stack[z0:z1].max(axis=0).astype(np.float32)
    if mode == "sum":
        return stack[z0:z1].sum(axis=0, dtype=np.float32)
    raise ValueError(f"Unknown projection {mode!r}")


def normalise(image, lo, hi, gamma=1.0):
    """Scale channel-first [C, ...] data to [0, 1] between per-channel limits."""
    shape = (-1,) + (1,) * (image.ndim - 1)
    scaled = (image - lo.reshape(shape)) / np.maximum(hi - lo, 1e-6).reshape(shape)
    return np.clip(scaled, 0.0, 1.0) ** gamma


# --- Models --------------------------------------------------------------------


def load_selection_records(path, store_root):
    """Models of a selection exported by ``model_registry_explorer.py``, with each model's
    projection and downsample (from its bundle config) and a picker label."""
    from Experiments.micropatterns.ko_evaluation import model_option_label, read_model_settings
    from Experiments.model_registry import ModelRegistry, load_selection

    records = load_selection(path)
    settings = read_model_settings(ModelRegistry(Path(store_root).expanduser()), [r["model_id"] for r in records])
    records = [{**r, **settings[r["model_id"]]} for r in records]
    return [{**r, "option_label": model_option_label(r)} for r in records]


def model_rollouts(store_root, model_id, data_root, replicate, seed, conditions=("baseline",)):
    """Data and NCA trajectories of a registry model for each condition ("baseline", "ko_24h",
    "ko_0h"), from the 0h state of ``replicate`` (one-based), loaded and scaled as in training."""
    import jax.random as jr

    from Experiments.micropatterns.density_ko_sweep import load_condition_data, load_training_histogram_bins
    from Experiments.micropatterns.ko_evaluation import evaluate_condition
    from Experiments.model_registry import ModelRegistry

    bundle = ModelRegistry(Path(store_root).expanduser()).get(model_id)
    model = bundle.load_model(key=jr.PRNGKey(seed), implementation="portable")
    root = Path(data_root).expanduser()
    bins = load_training_histogram_bins(bundle, root)
    rollouts = {}
    for k, condition in enumerate(conditions):
        data = load_condition_data(bundle, root, condition, replicate, bins)
        prediction, time_labels = evaluate_condition(model, bundle, data, condition, seed + k)
        rollouts[condition] = {
            "prediction": np.asarray(prediction),
            "target": np.asarray(data["target"]),
            "measured": np.asarray(data["measured"]),
            "channel_names": tuple(data["channel_names"]),
            "time_labels": tuple(time_labels),
            "boundary": np.asarray(data["boundary"]).squeeze().astype(bool),
            "train_replicates": tuple(bundle.config.data.micropattern.train_replicates or ()),
        }
    return rollouts


def baseline_rollout(store_root, model_id, data_root, replicate, seed):
    """Baseline (no KO) rollout of a registry model; see ``model_rollouts``."""
    return model_rollouts(store_root, model_id, data_root, replicate, seed)["baseline"]


# --- Cell types ----------------------------------------------------------------


def crop_box(mask):
    """Row and column slices of the bounding box of ``mask``."""
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)


def masked_rgba(rgb, mask, crop_mask=None):
    """``rgb`` [H, W, 3] cropped to the box around ``crop_mask`` (default: the colony
    ``mask``), transparent outside the colony."""
    crop = crop_box(mask if crop_mask is None else crop_mask)
    return np.concatenate([rgb[crop], mask[crop][..., None].astype(float)], axis=-1)


def cell_type_rgba(shares, colours, mask, crop_mask=None):
    """RGBA of per-pixel label shares ``{label: [H, W]}`` blended with ``{label: colour}``
    (labels missing from ``colours`` are left out), cropped as ``masked_rgba``."""
    rgb = sum(
        (np.asarray(shares[label], dtype=float)[..., None] * np.asarray(to_rgb(colours.get(label, UNKNOWN_CELL_TYPE)))
         for label in shares if label in colours),
        np.zeros(np.shape(mask) + (3,)),
    )
    return masked_rgba(np.clip(rgb, 0.0, 1.0), np.asarray(mask, dtype=bool), crop_mask)

from einops import rearrange,repeat
from NCA.NCA_visualiser import (
	plot_to_image,
	plot_weight_matrices,
	plot_weight_kernel_boxplot,
)
import numpy as np
from Common.utils import squarish
from tqdm import tqdm
from jaxtyping import Float,Array,Key,PyTree
import os
import jax
import jax.random as jr
import time
from dotenv import load_dotenv
load_dotenv()
PVC_PATH = os.getenv("PVC_PATH")
LOG_BACKEND = os.environ.get("LOG_BACKEND", "wandb")
# PVC_PATH = "mnt/ceph/ar-dp/"  # Path to the PVC where the data is stored
#if LOG_BACKEND=="wandb":
from Common.utils import get_jax_memory_stats
from Common.trainer.abstract_wandb_log import Train_log
from Common.trainer.experiment_channel_grouping import duplicate_x_channels_9ch
#elif LOG_BACKEND=="tensorboard":
#	from Common.trainer.abstract_tensorboard_log import Train_log


def _is_grouped_9ch_colony_augmenter(data_augmenter):
	return (
		getattr(data_augmenter, "OBS_CHANNELS", None) == 12
		and any(
			cls.__module__ == "NCA.trainer.data_augmenter.colony_9ch"
			and cls.__name__ == "DataAugmenter"
			for cls in type(data_augmenter).__mro__
		)
	)


def _trajectory_snapshot_channels(T, data_augmenter, t, channel_schema=None):
	T_snapshot = T[::t]
	schema = channel_schema or getattr(data_augmenter, "schema", None)
	if schema is not None:
		return T_snapshot[:, np.asarray(schema.target_to_state)]
	if _is_grouped_9ch_colony_augmenter(data_augmenter):
		return duplicate_x_channels_9ch(T_snapshot[:,:9])
	return T_snapshot[:,:data_augmenter.OBS_CHANNELS]


def _target_aligned_diagnostic_channels(outputs, grouped_channels=False, channel_schema=None):
	"""Convert model outputs to the target channel layout used by diagnostics."""
	outputs = np.array(outputs)
	if channel_schema is not None:
		if outputs.shape[2] < channel_schema.n_state_channels:
			raise ValueError("Diagnostic outputs have fewer channels than the channel schema")
		return outputs[:, :, np.asarray(channel_schema.target_to_state)]
	if not grouped_channels:
		return outputs
	if outputs.shape[2] < 9:
		raise ValueError("Grouped micropattern diagnostics require at least 9 model channels")
	return np.concatenate(
		[
			outputs[:, :, 0:4],
			outputs[:, :, 0:3],
			outputs[:, :, 4:8],
			outputs[:, :, 8:9],
		],
		axis=2,
	)


def compute_channel_time_diagnostics(
	predictions,
	targets,
	boundary_masks=None,
	radial_bins=16,
	radial_extent=1.5,
):
	"""Compute total intensity and mean radial intensity per channel/timestep.

	Inputs use [batch, time, channel, x, y]. Radial distance is measured from
	the centroid of each boundary mask and normalized by its maximum in-mask
	radius. Profiles are averaged over batches and over pixels in each annulus.
	The radial profile includes all pixels out to ``radial_extent`` times the
	boundary radius, allowing the diagnostic to show the immediate exterior.
	"""
	predictions = np.asarray(predictions, dtype=np.float32)
	targets = np.asarray(targets, dtype=np.float32)
	if predictions.shape != targets.shape:
		raise ValueError(
			f"Diagnostic prediction/target shapes must match, got {predictions.shape} and {targets.shape}"
		)
	if predictions.ndim != 5:
		raise ValueError("Diagnostics expect [batch, time, channel, x, y] arrays")
	if radial_bins <= 0:
		raise ValueError("radial_bins must be positive")
	if radial_extent <= 1.0:
		raise ValueError("radial_extent must be greater than 1.0")

	batch_count, time_count, channel_count, width, height = predictions.shape
	if boundary_masks is None:
		boundary_masks = np.ones((batch_count, width, height), dtype=bool)
	else:
		boundary_masks = np.asarray(boundary_masks)
		if boundary_masks.ndim == 4 and boundary_masks.shape[1] == 1:
			boundary_masks = boundary_masks[:, 0]
		if boundary_masks.shape != (batch_count, width, height):
			raise ValueError(
				"Boundary masks must have shape [batch, 1, x, y] or [batch, x, y]; "
				f"got {boundary_masks.shape}"
			)
		boundary_masks = boundary_masks.astype(bool)

	prediction_totals = np.zeros((batch_count, time_count, channel_count), dtype=np.float64)
	target_totals = np.zeros_like(prediction_totals)
	prediction_profiles = np.zeros(
		(batch_count, time_count, channel_count, radial_bins), dtype=np.float64
	)
	target_profiles = np.zeros_like(prediction_profiles)
	bin_edges = np.linspace(0.0, radial_extent, radial_bins + 1)

	grid_x, grid_y = np.meshgrid(
		np.arange(width, dtype=np.float32),
		np.arange(height, dtype=np.float32),
		indexing="ij",
	)
	for batch_index in range(batch_count):
		mask = boundary_masks[batch_index]
		if not np.any(mask):
			continue
		centre_x = float(np.mean(grid_x[mask]))
		centre_y = float(np.mean(grid_y[mask]))
		radius = np.sqrt((grid_x - centre_x) ** 2 + (grid_y - centre_y) ** 2)
		max_radius = float(np.max(radius[mask]))
		normalized_radius = radius / max(max_radius, 1e-8)

		prediction_totals[batch_index] = np.sum(
			predictions[batch_index] * mask[None, None], axis=(-1, -2)
		)
		target_totals[batch_index] = np.sum(
			targets[batch_index] * mask[None, None], axis=(-1, -2)
		)
		prediction_pixels = predictions[batch_index].reshape(
			time_count, channel_count, -1
		)
		target_pixels = targets[batch_index].reshape(
			time_count, channel_count, -1
		)

		for bin_index in range(radial_bins):
			if bin_index == radial_bins - 1:
				annulus = (normalized_radius >= bin_edges[bin_index]) & (normalized_radius <= bin_edges[bin_index + 1])
			else:
				annulus = (normalized_radius >= bin_edges[bin_index]) & (normalized_radius < bin_edges[bin_index + 1])
			if not np.any(annulus):
				continue
			annulus_flat = annulus.reshape(-1)
			prediction_profiles[batch_index, :, :, bin_index] = np.mean(
				prediction_pixels[:, :, annulus_flat], axis=-1
			)
			target_profiles[batch_index, :, :, bin_index] = np.mean(
				target_pixels[:, :, annulus_flat], axis=-1
			)

	return {
		"prediction_total_intensity": np.mean(prediction_totals, axis=0),
		"target_total_intensity": np.mean(target_totals, axis=0),
		"prediction_radial_profile": np.mean(prediction_profiles, axis=0),
		"target_radial_profile": np.mean(target_profiles, axis=0),
		"radius": 0.5 * (bin_edges[:-1] + bin_edges[1:]),
	}


def compute_channel_correlation_diagnostics(
	predictions,
	targets,
	boundary_masks=None,
	epsilon=1e-8,
	experiment_group_sizes=None,
):
	"""Compute masked pixelwise Pearson channel correlations per timestep.

	Correlations are calculated independently for each batch from all pixels
	inside its adhesion mask, then averaged across batches. Constant channels
	are assigned zero correlation because Pearson correlation is undefined.
	"""
	predictions = np.asarray(predictions, dtype=np.float32)
	targets = np.asarray(targets, dtype=np.float32)
	if predictions.shape != targets.shape:
		raise ValueError(
			f"Correlation prediction/target shapes must match, got {predictions.shape} and {targets.shape}"
		)
	if predictions.ndim != 5:
		raise ValueError("Correlations expect [batch, time, channel, x, y] arrays")

	batch_count, time_count, channel_count, width, height = predictions.shape
	if boundary_masks is None:
		boundary_masks = np.ones((batch_count, width, height), dtype=bool)
	else:
		boundary_masks = np.asarray(boundary_masks)
		if boundary_masks.ndim == 4 and boundary_masks.shape[1] == 1:
			boundary_masks = boundary_masks[:, 0]
		if boundary_masks.shape != (batch_count, width, height):
			raise ValueError(
				"Boundary masks must have shape [batch, 1, x, y] or [batch, x, y]; "
				f"got {boundary_masks.shape}"
			)
		boundary_masks = boundary_masks.astype(bool)

	def correlation_matrix(values):
		values = values - np.mean(values, axis=1, keepdims=True)
		channel_norms = np.sqrt(np.sum(values**2, axis=1))
		denominator = channel_norms[:, None] * channel_norms[None, :]
		numerator = values @ values.T
		return np.divide(
			numerator,
			denominator,
			out=np.zeros((channel_count, channel_count), dtype=np.float64),
			where=denominator > epsilon,
		)

	prediction_correlations = np.zeros(
		(batch_count, time_count, channel_count, channel_count), dtype=np.float64
	)
	target_correlations = np.zeros_like(prediction_correlations)
	for batch_index in range(batch_count):
		mask_flat = boundary_masks[batch_index].reshape(-1)
		if not np.any(mask_flat):
			continue
		prediction_pixels = predictions[batch_index].reshape(
			time_count, channel_count, -1
		)[:, :, mask_flat]
		target_pixels = targets[batch_index].reshape(
			time_count, channel_count, -1
		)[:, :, mask_flat]
		for time_index in range(time_count):
			prediction_correlations[batch_index, time_index] = correlation_matrix(
				prediction_pixels[time_index]
			)
			target_correlations[batch_index, time_index] = correlation_matrix(
				target_pixels[time_index]
			)

	prediction_mean = np.mean(prediction_correlations, axis=0)
	target_mean = np.mean(target_correlations, axis=0)
	if experiment_group_sizes is not None:
		if sum(experiment_group_sizes) != channel_count:
			raise ValueError("Experiment group sizes must cover all diagnostic channels")
		co_measured = np.zeros((channel_count, channel_count), dtype=bool)
		start = 0
		for size in experiment_group_sizes:
			co_measured[start:start + size, start:start + size] = True
			start += size
		prediction_mean = np.where(co_measured, prediction_mean, np.nan)
		target_mean = np.where(co_measured, target_mean, np.nan)
	return {
		"prediction_channel_correlation": prediction_mean,
		"target_channel_correlation": target_mean,
		"channel_correlation_difference": prediction_mean - target_mean,
	}


def plot_total_intensity_diagnostics(diagnostics, channel_names, timepoint_names=None):
	"""Plot target, prediction, and absolute-error totals as compact heatmaps."""
	import matplotlib.pyplot as plt

	target = diagnostics["target_total_intensity"].T
	prediction = diagnostics["prediction_total_intensity"].T
	error = np.abs(prediction - target)
	channel_count, time_count = target.shape
	if channel_names is None or len(channel_names) != channel_count:
		channel_names = [f"channel_{index + 1}" for index in range(channel_count)]
	if timepoint_names is None or len(timepoint_names) != time_count:
		timepoint_names = [f"t{index + 1}" for index in range(time_count)]

	shared_min = min(float(np.min(target)), float(np.min(prediction)), 0.0)
	shared_max = max(float(np.max(target)), float(np.max(prediction)), 1e-8)
	figure, axes = plt.subplots(1, 3, figsize=(14, max(5.0, 0.35 * channel_count)),dpi=150)
	for axis, values, title, cmap, vmin, vmax in (
		(axes[0], target, "Target total intensity", "viridis", shared_min, shared_max),
		(axes[1], prediction, "Prediction total intensity", "viridis", shared_min, shared_max),
		(axes[2], error, "Absolute total-intensity error", "magma", 0.0, max(float(np.max(error)), 1e-8)),
	):
		image = axis.imshow(
			values,
			aspect="auto",
			interpolation="nearest",
			cmap=cmap,
			vmin=vmin,
			vmax=vmax,
		)
		axis.set_title(title)
		axis.set_xticks(
			np.arange(time_count),
			labels=timepoint_names,
			rotation=45,
			ha="right",
			fontsize=7,
		)
		axis.set_yticks(
			np.arange(channel_count),
			labels=channel_names,
			fontsize=7,
		)
		figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
	figure.tight_layout()
	return plot_to_image(figure)


def plot_radial_intensity_diagnostics(diagnostics, channel_names, timepoint_names=None):
	"""Plot target, prediction, and absolute-error radial profiles as heatmaps."""
	import matplotlib.pyplot as plt

	target = diagnostics["target_radial_profile"]
	prediction = diagnostics["prediction_radial_profile"]
	time_count, channel_count, _ = target.shape
	if timepoint_names is None or len(timepoint_names) != time_count:
		timepoint_names = [f"t{index + 1}" for index in range(time_count)]
	target_rows = rearrange(target, "t c r -> (c t) r")
	prediction_rows = rearrange(prediction, "t c r -> (c t) r")
	error_rows = np.abs(prediction_rows - target_rows)
	row_labels = [
		f"{channel_names[channel_index]} · {timepoint_names[time_index]}"
		for channel_index in range(channel_count)
		for time_index in range(time_count)
	]
	shared_max = max(float(np.max(target_rows)), float(np.max(prediction_rows)), 1e-8)
	figure_height = max(6.0, 0.18 * len(row_labels))
	figure, axes = plt.subplots(1, 3, figsize=(15, figure_height), sharey=True)
	for axis, values, title, vmax in (
		(axes[0], target_rows, "Target radial mean intensity", shared_max),
		(axes[1], prediction_rows, "Prediction radial mean intensity", shared_max),
		(axes[2], error_rows, "Absolute profile error", max(float(np.max(error_rows)), 1e-8)),
	):
		image = axis.imshow(values, aspect="auto", origin="upper", vmin=0.0, vmax=vmax)
		axis.set_title(title)
		axis.set_xlabel("Normalized radius (centre → boundary → exterior)")
		axis.set_xticks(
			np.linspace(0, values.shape[1] - 1, 4),
			labels=["0.0", "0.5", "1.0", "1.5"],
		)
		figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
	axes[0].set_yticks(np.arange(len(row_labels)), labels=row_labels, fontsize=6)
	figure.tight_layout()
	return plot_to_image(figure)


def plot_radial_intensity_line_diagnostics(
	diagnostics, channel_names, timepoint_names=None
):
	"""Plot one sequentially-coloured radial profile line per timestep/channel."""
	import matplotlib.pyplot as plt

	target = np.asarray(diagnostics["target_radial_profile"])
	prediction = np.asarray(diagnostics["prediction_radial_profile"])
	radius = np.asarray(diagnostics.get("radius"))
	time_count, channel_count, radial_count = target.shape
	if timepoint_names is None or len(timepoint_names) != time_count:
		timepoint_names = [f"t{index + 1}" for index in range(time_count)]
	if radius.shape != (radial_count,):
		radius = np.linspace(0.0, 1.5, radial_count)

	figure, axes = plt.subplots(
		channel_count,
		2,
		figsize=(12, max(3.0, 2.5 * channel_count)),
		sharex=True,
		squeeze=False,
	)
	colours = plt.get_cmap("viridis")(np.linspace(0.15, 0.95, max(time_count, 1)))
	for channel_index, channel_name in enumerate(channel_names[:channel_count]):
		for axis, values, label in (
			(axes[channel_index, 0], target[:, channel_index], "Target"),
			(axes[channel_index, 1], prediction[:, channel_index], "Prediction"),
		):
			for time_index, (profile, colour) in enumerate(zip(values, colours)):
				axis.plot(
					radius,
					profile,
					color=colour,
					linewidth=1.5,
					label=timepoint_names[time_index],
				)
			axis.axvline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
			axis.set_title(f"{channel_name} · {label}")
			axis.grid(True, alpha=0.2)
			if channel_index == channel_count - 1:
				axis.set_xlabel("Normalized radius")
			if channel_index == 0 and label == "Target":
				axis.legend(fontsize=7, ncol=min(time_count, 4), title="Timestep")
	figure.suptitle("Radial intensity profiles over time (dashed line = boundary)")
	figure.tight_layout()
	return plot_to_image(figure)


def plot_channel_correlation_diagnostics(
	diagnostics,
	channel_names,
	timepoint_names=None,
	experiment_group_sizes=None,
):
	"""Plot predicted channel correlation and prediction-minus-target per time."""
	import matplotlib.pyplot as plt

	prediction = diagnostics["prediction_channel_correlation"]
	difference = diagnostics["channel_correlation_difference"]
	time_count, channel_count, _ = prediction.shape
	if timepoint_names is None or len(timepoint_names) != time_count:
		timepoint_names = [f"t{index + 1}" for index in range(time_count)]
	if channel_names is None or len(channel_names) != channel_count:
		channel_names = [f"channel_{index + 1}" for index in range(channel_count)]
	group_boundaries = []
	if experiment_group_sizes is not None and sum(experiment_group_sizes) == channel_count:
		group_boundaries = np.cumsum(experiment_group_sizes)[:-1] - 0.5

	difference_limit = max(float(np.nanmax(np.abs(difference))), 0.05)
	figure, axes = plt.subplots(
		time_count,
		2,
		figsize=(13, max(4.0, 3.4 * time_count)),
		squeeze=False,
	)
	for time_index in range(time_count):
		panels = (
			(axes[time_index, 0], prediction[time_index], "NCA correlation", 1.0),
			(
				axes[time_index, 1],
				difference[time_index],
				"NCA − true correlation",
				difference_limit,
			),
		)
		for axis, values, title, limit in panels:
			image = axis.imshow(
				values,
				cmap="coolwarm",
				vmin=-limit,
				vmax=limit,
				interpolation="nearest",
			)
			axis.set_title(f"{timepoint_names[time_index]} · {title}")
			axis.set_xticks(
				np.arange(channel_count),
				labels=channel_names,
				rotation=90,
				fontsize=6,
			)
			axis.set_yticks(
				np.arange(channel_count),
				labels=channel_names,
				fontsize=6,
			)
			for boundary in group_boundaries:
				axis.axhline(boundary, color="black", linewidth=1.25)
				axis.axvline(boundary, color="black", linewidth=1.25)
			figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
	figure.tight_layout()
	return plot_to_image(figure)


def plot_channel_time_grid(values, channel_names=None, timepoint_names=None, title=None):
	"""Render ``[time, channel, x, y]`` images with channel and time labels."""
	import matplotlib.pyplot as plt

	values = np.asarray(values)
	time_count, channel_count, width, height = values.shape
	if channel_names is None or len(channel_names) != channel_count:
		channel_names = [f"channel_{index + 1}" for index in range(channel_count)]
	if timepoint_names is None or len(timepoint_names) != time_count:
		timepoint_names = [f"t{index}" for index in range(time_count)]
	grid = rearrange(values, "t c x y -> (c x) (t y)")
	value_min, value_max = np.nanmin(grid), np.nanmax(grid)
	figure, axis = plt.subplots(
		figsize=(max(7.0, 1.8 * time_count), max(5.0, 0.38 * channel_count)), dpi=150
	)
	axis.imshow(grid, cmap="gray", vmin=value_min, vmax=max(value_max, value_min + 1e-8))
	axis.set_xticks((np.arange(time_count) + 0.5) * height, labels=timepoint_names)
	axis.set_yticks(
		(np.arange(channel_count) + 0.5) * width,
		labels=channel_names,
		rotation=90,
		fontsize=5,
		va="center",
	)
	for boundary in np.arange(1, time_count) * height:
		axis.axvline(boundary - 0.5, color="white", linewidth=0.4, alpha=0.5)
	for boundary in np.arange(1, channel_count) * width:
		axis.axhline(boundary - 0.5, color="white", linewidth=0.4, alpha=0.5)
	if title:
		axis.set_title(title)
	figure.tight_layout()
	return plot_to_image(figure)


def _timepoint_labels(names, count):
	if len(names) == count:
		return names
	if len(names) + 1 == count:
		return ["t0h", *names]
	return [f"t{index}" for index in range(count)]


def _biomarker_name(name):
	"""Return the biomarker suffix from a schema channel name."""
	return str(name).rsplit("/", 1)[-1]


def _singular_value_logging_config(config=None):
	defaults = {
		"enabled": False,
		"plot_spectra": True,
		"epsilon": 1e-8,
	}
	if config is None:
		return defaults
	for key in defaults:
		try:
			if config.get(key) is not None:
				defaults[key] = config.get(key)
		except AttributeError:
			if key in config:
				defaults[key] = config[key]
	defaults["enabled"] = bool(defaults["enabled"])
	defaults["plot_spectra"] = bool(defaults["plot_spectra"])
	defaults["epsilon"] = float(defaults["epsilon"])
	return defaults


def _flatten_weight_tree(weights):
	if isinstance(weights, (list, tuple)):
		for weight in weights:
			yield from _flatten_weight_tree(weight)
	else:
		yield weights


def extract_dense_weight_singular_values(nca, epsilon=1e-8):
	"""Return singular-value diagnostics for squeezed 2D weights."""
	diagnostics = []
	for idx, weight in enumerate(_flatten_weight_tree(nca.get_weights())):
		matrix = np.squeeze(np.array(weight))
		if matrix.ndim != 2:
			continue
		singular_values = np.linalg.svd(
			matrix.astype(np.float32),
			compute_uv=False,
		)
		if singular_values.size == 0:
			continue
		total = np.sum(singular_values)
		if total > epsilon:
			probs = singular_values / total
			effective_rank = float(np.exp(-np.sum(probs * np.log(probs + epsilon))))
		else:
			effective_rank = 0.0
		max_singular = float(np.max(singular_values))
		min_singular = float(np.min(singular_values))
		diagnostics.append({
			"idx": idx,
			"shape": matrix.shape,
			"singular_values": singular_values,
			"summary": {
				"max": max_singular,
				"min": min_singular,
				"mean": float(np.mean(singular_values)),
				"median": float(np.median(singular_values)),
				"condition_number": max_singular / max(min_singular, epsilon),
				"effective_rank": effective_rank,
			},
		})
	return diagnostics


def plot_singular_value_spectrum(singular_values, title):
	import matplotlib.pyplot as plt

	plot_values = np.maximum(np.array(singular_values), 1e-12)
	figure = plt.figure(figsize=(6,4), dpi=150)
	ax = figure.add_subplot(111)
	ax.plot(np.arange(len(plot_values)), plot_values, marker="o", linewidth=1)
	ax.set_yscale("log")
	ax.set_xlabel("Index")
	ax.set_ylabel("Singular value")
	ax.set_title(title)
	ax.grid(True, which="both", alpha=0.25)
	figure.tight_layout()
	return plot_to_image(figure)


class NCA_Train_log(Train_log):
	"""
		Class for logging NCA training behaviour.
	"""

	def __init__(
		self,
		*args,
		singular_value_config=None,
		boundary_mask=None,
		channel_names=None,
		channel_schema=None,
		timepoint_names=None,
		data_augmenter=None,
		radial_bins=16,
		radial_extent=1.5,
		**kwargs,
	):
		data = kwargs.get("data", args[0] if args else None)
		data_values = [] if data is None else list(data)
		uniform_data = bool(data_values) and len(
			{tuple(value.shape) for value in data_values}
		) == 1
		self.diagnostic_targets = (
			np.stack(data_values)[:, 1:] if uniform_data else None
		)
		self.diagnostic_boundary_mask = None if boundary_mask is None else np.array(boundary_mask)
		self.diagnostic_grouped_channels = (
			False if data_augmenter is None else _is_grouped_9ch_colony_augmenter(data_augmenter)
		)
		self.diagnostic_channel_schema = channel_schema or getattr(data_augmenter, "schema", None)
		self.diagnostic_group_sizes = (
			self.diagnostic_channel_schema.group_sizes
			if self.diagnostic_channel_schema is not None
			else (4, 4, 3, 1) if self.diagnostic_grouped_channels else None
		)
		self.radial_bins = int(radial_bins)
		self.radial_extent = float(radial_extent)
		channel_count = (
			self.diagnostic_targets.shape[2]
			if self.diagnostic_targets is not None
			else 0 if not data_values else data_values[0].shape[1]
		)
		if (
			self.diagnostic_channel_schema is not None
			and self.diagnostic_channel_schema.n_measurement_channels == channel_count
		):
			self.channel_names = [
				channel.marker
				for channel in self.diagnostic_channel_schema.measurement_channels
			]
		elif channel_names is not None and len(channel_names) == channel_count:
			self.channel_names = [_biomarker_name(name) for name in channel_names]
		else:
			self.channel_names = [f"channel_{index + 1}" for index in range(channel_count)]
		time_count = (
			0 if not data_values else data_values[0].shape[0] - 1
		)
		if timepoint_names is None or len(timepoint_names) != time_count:
			self.timepoint_names = [f"t{index + 1}" for index in range(time_count)]
		else:
			self.timepoint_names = [str(name) for name in timepoint_names]
		super().__init__(*args, **kwargs)
		self.singular_value_config = _singular_value_logging_config(singular_value_config)

	def log_data_at_init(self, data):
		"""Log every true batch as a labelled channel-by-time grid."""
		images = [
			plot_channel_time_grid(
				batch,
				self.channel_names,
				_timepoint_labels(self.timepoint_names, batch.shape[0]),
				f"True measurements · batch {batch_index + 1}",
			)
			for batch_index, batch in enumerate(data)
		]
		self.log_image("True sequence labelled", np.concatenate(images, axis=0), step=None)

	def log_channel_time_diagnostics(self, log_dict, i):
		"""Log per-channel/timestep totals and radial profiles to W&B."""
		if self.diagnostic_targets is None or "states" not in log_dict:
			return
		try:
			predictions = np.array(log_dict["states"])
			predictions = _target_aligned_diagnostic_channels(
				predictions,
				grouped_channels=self.diagnostic_grouped_channels,
				channel_schema=getattr(self, "diagnostic_channel_schema", None),
			)
			targets = self.diagnostic_targets
			predictions = predictions[:, :targets.shape[1], :targets.shape[2]]
			if predictions.shape != targets.shape:
				raise ValueError(
					f"aligned predictions have shape {predictions.shape}, targets have shape {targets.shape}"
				)
			diagnostics = compute_channel_time_diagnostics(
				predictions,
				targets,
				boundary_masks=self.diagnostic_boundary_mask,
				radial_bins=self.radial_bins,
				radial_extent=getattr(self, "radial_extent", 1.5),
			)
			correlation_diagnostics = compute_channel_correlation_diagnostics(
				predictions,
				targets,
				boundary_masks=self.diagnostic_boundary_mask,
				experiment_group_sizes=getattr(self, "diagnostic_group_sizes", None),
			)
			prediction_totals = diagnostics["prediction_total_intensity"]
			target_totals = diagnostics["target_total_intensity"]
			self.log_scalar(
				"Diagnostics/total_intensity/mean_absolute_error",
				float(np.mean(np.abs(prediction_totals - target_totals))),
				step=i,
			)
			self.log_image(
				"Diagnostics/total_intensity",
				plot_total_intensity_diagnostics(
					diagnostics,
					self.channel_names,
					self.timepoint_names,
				),
				step=i,
			)
			self.log_image(
				"Diagnostics/radial_intensity_profiles",
				plot_radial_intensity_diagnostics(
					diagnostics,
					self.channel_names,
					self.timepoint_names,
				),
				step=i,
			)
			self.log_image(
				"Diagnostics/radial_intensity_lines",
				plot_radial_intensity_line_diagnostics(
					diagnostics,
					self.channel_names,
					self.timepoint_names,
				),
				step=i,
			)
			self.log_image(
				"Diagnostics/channel_correlation",
				plot_channel_correlation_diagnostics(
					correlation_diagnostics,
					self.channel_names,
					self.timepoint_names,
					experiment_group_sizes=getattr(self, "diagnostic_group_sizes", None),
				),
				step=i,
			)
		except Exception as exc:
			print(f"Warning: Failed to log channel/time diagnostics: {exc}", flush=True)

	def log_model_parameters(self,nca,i):  # type: ignore
		"""Log model parameters

		Args:
			nca : nca model class (PyTree)
			i : training step
		"""
		
		for idx, w in enumerate(nca.get_weights()):
			w = np.squeeze(w)
			self.log_histogram(f"Train/weight_{idx}", w, step=i)
			# print("Weight shape ",w.shape)
			if len(w.shape) == 2:
				w = repeat(w,"W H -> W H 3")
				self.log_image(f"Train/weight_image_{idx}", self.normalise_images(w), step=i)
		self.log_singular_value_spectra(nca,i)

	def log_singular_value_spectra(self,nca,i):
		if not self.singular_value_config["enabled"]:
			return
		diagnostics = extract_dense_weight_singular_values(
			nca,
			epsilon=self.singular_value_config["epsilon"],
		)
		for diagnostic in diagnostics:
			idx = diagnostic["idx"]
			tag_prefix = f"Train/SVD/weight_{idx}"
			self.log_histogram(
				f"{tag_prefix}/singular_values",
				diagnostic["singular_values"],
				step=i,
			)
			for name, value in diagnostic["summary"].items():
				self.log_scalar(f"{tag_prefix}/{name}", value, step=i)
			if self.singular_value_config["plot_spectra"]:
				self.log_image(
					f"{tag_prefix}/spectrum",
					plot_singular_value_spectrum(
						diagnostic["singular_values"],
						f"Weight {idx} singular values",
					),
					step=i,
				)
			

	def log_model_outputs(self,x,i):
		"""
			x: Dict {"states": PyTree[Float[Array, "N CHANNELS x y"], "B"]}
			i: training step
		"""
		memory_stats = get_jax_memory_stats()
		for key in memory_stats:
			self.log_scalar(f"Memory/{key}",memory_stats[key],step=i)
		states = x["states"]
		BATCHES = len(states)
		if len({tuple(value.shape) for value in states}) == 1:
			visible = np.stack(states)[:, :, :3]
			self.log_image(
				'Train/visible_batches',
				self.normalise_images(rearrange(visible,"b t c x y -> (b x) (t y) c")),
				step=i)
		else:
			for b in range(BATCHES):
				self.log_image(
					'Train/visible_batch_'+str(b),
					self.normalise_images(rearrange(states[b][:,:3,...],"Batch Channel x y -> Batch x y Channel")),
					step=i)
			
		if states[0].shape[1] > 3:
			b=0
			hidden_channels = states[b][:,3:]
			extra_zeros = (-hidden_channels.shape[1])%3
			hidden_channels = np.pad(hidden_channels,((0,0),(0,extra_zeros),(0,0),(0,0)))
			_cy,_cx = squarish(hidden_channels.shape[1]//3) # type: ignore
			hidden_channels_r = rearrange(hidden_channels,"Batch (cx cy C) x y -> Batch (cx x) (cy y) C",C=3,cy=_cy,cx=_cx)
			hidden_channels_r = (np.tanh(hidden_channels_r)+1.0)/2.0
			self.log_image(
				f'Train/batch_{b}_hidden_channels',
				hidden_channels_r,
				step=i)
	
	def tb_training_loop_log_sequence(self,log_dict,i,model,write_images=True,LOG_EVERY=10):
		detail_losses = []
		for name in log_dict.keys():
			if name != "states":
				if name.startswith("loss_detail/"):
					if i % LOG_EVERY == 0 and name.endswith("/total"):
						detail_losses.append(np.asarray(log_dict[name]).reshape(-1))
				elif name.startswith("pool/"):
					self.log_scalar(f"StatePool/{name.removeprefix('pool/')}",log_dict[name],step=i)
				elif name.startswith("runtime/"):
					self.log_scalar(f"Runtime/{name.removeprefix('runtime/')}",log_dict[name],step=i)
				elif name.startswith("validation/"):
					self.log_scalar(f"Validation/{name.removeprefix('validation/')}",log_dict[name],step=i)
				elif name.startswith("validation_rollout/"):
					self.log_scalar(f"ValidationRollout/{name.removeprefix('validation_rollout/')}",log_dict[name],step=i)
				# elif name == "learning_rate":
					# self.log_scalar("Train/learning_rate", log_dict[name], step=i)
				else:
					self.log_scalar(f"Train/{name}",log_dict[name],step=i)
		if detail_losses:
			self.log_histogram(
				"Train/loss_detail/group_timestep",
				np.concatenate(detail_losses),
				step=i,
			)
		if i%LOG_EVERY==0 and i>0:
			self.log_model_parameters(model,i)
			self.log_channel_time_diagnostics(log_dict,i)
			if write_images:
				self.log_model_outputs(log_dict,i)

	
	def tb_training_end_log(self, # type: ignore
						 	nca,
							# x: PyTree[Float[Array, "N CHANNELS x y"], "B"],  # noqa: F722, F821
							DATA_AUGMENTER,
							t,
								boundary_callback,
								SAVE_TRAJECTORY=False,
								write_images=True,
								key=None):
		"""
			Log trained NCA model trajectory after training

		"""
		if key is None:
			key = jr.PRNGKey(int(time.time()))
		x,y = DATA_AUGMENTER.split_x_y(1)
		x,y = DATA_AUGMENTER.advance_pool(x,y,0,key)
		NUMBER_OF_IMAGES=x[0].shape[0]
		# Log true data for side by side comparison
		schema = self.diagnostic_channel_schema or getattr(DATA_AUGMENTER, "schema", None)
		channel_count = schema.n_measurement_channels if schema else DATA_AUGMENTER.OBS_CHANNELS
		true_images = [
			plot_channel_time_grid(
				batch[:, :channel_count],
				self.channel_names,
				_timepoint_labels(self.timepoint_names, batch.shape[0]),
				f"True measurements · batch {batch_index + 1}",
			)
			for batch_index, batch in enumerate(DATA_AUGMENTER.return_observed_data())
		]
		self.log_image(
			'TrainingRollout/true_data',
			np.concatenate(true_images, axis=0),
			step=None
		)
		BATCHES = len(x)
		CHANNELS = x[0].shape[1]

		print("Running final trained model for "+str(t)+" steps")
		
		SNAPSHOTS = []
		for b in tqdm(range(BATCHES)):
			initial_state = nca.prepare_pool_state(x[b][0])
			T = nca.run(t*NUMBER_OF_IMAGES, initial_state, boundary_callback[b])  # Shape T C x y
			self.log_video(f"TrainingRollout/trajectory_batch_{b + 1}",T[:,:3],step=None)
			T_snapshot = _trajectory_snapshot_channels(
				T, DATA_AUGMENTER, t, self.diagnostic_channel_schema
			)
			SNAPSHOTS.append(plot_channel_time_grid(
				T_snapshot,
				self.channel_names,
				_timepoint_labels(self.timepoint_names, T_snapshot.shape[0]),
				f"NCA predictions · batch {b + 1}",
			))
			
			if SAVE_TRAJECTORY:
				np.save(f"{PVC_PATH}output/{self.wandb_config['name']}_trajectory_{b}.npy",T[::t,:3])  # type: ignore

			if T.shape[1] > 3:
				hidden = T[:, 3:]
				extra_zeros = (-hidden.shape[1])%3
				hidden = np.pad(hidden,((0,0),(0,extra_zeros),(0,0),(0,0)))
				_cy,_cx = squarish(hidden.shape[1]//3)
				hidden = rearrange(hidden,"Time (cx cy C) x y  -> Time C (cx x) (cy y)",C=3,cy=_cy,cx=_cx)
				hidden = (np.tanh(hidden)+1.0)/2.0
				self.log_video(f"TrainingRollout/hidden_trajectory_batch_{b + 1}",hidden,step=None)

		self.log_image(
			'TrainingRollout/trajectory_snapshot',
			np.concatenate(SNAPSHOTS, axis=0),
			step=None
		)

class NCA_knockout_Train_log(NCA_Train_log):

	def __init__(
        self,
        data,
        wandb_config=None,
		knockout_time=None,
		knockout_channel=None,
		singular_value_config=None,
		boundary_mask=None,
		channel_names=None,
		channel_schema=None,
		timepoint_names=None,
		data_augmenter=None,
		radial_bins=16,
    ):
		super().__init__(
			data,
			wandb_config,
			singular_value_config=singular_value_config,
			boundary_mask=boundary_mask,
			channel_names=channel_names,
			channel_schema=channel_schema,
			timepoint_names=timepoint_names,
			data_augmenter=data_augmenter,
			radial_bins=radial_bins,
		)
		assert knockout_time is not None, "knockout_time must be provided for NCA_knockout_Train_log"
		assert knockout_channel is not None, "knockout_channel must be provided for NCA_knockout_Train_log"
		self.knockout_time = knockout_time
		self.knockout_channel = knockout_channel

	def tb_training_end_log(self,
						 	nca,
							# x: PyTree[Float[Array, "N CHANNELS x y"], "B"],  # noqa: F722, F821
							DATA_AUGMENTER,
							t,
								boundary_callback,
								SAVE_TRAJECTORY=False,
								write_images=True,
								key=None):
		"""
		

			Log trained NCA model trajectory after training

		"""
		if key is None:
			key = jr.PRNGKey(int(time.time()))
		x,y = DATA_AUGMENTER.split_x_y(1)
		x,y = DATA_AUGMENTER.advance_pool(x,y,0,key)
		NUMBER_OF_IMAGES=x[0].shape[0]
		# Log true data for side by side comparison
		schema = self.diagnostic_channel_schema or getattr(DATA_AUGMENTER, "schema", None)
		channel_count = schema.n_measurement_channels if schema else DATA_AUGMENTER.OBS_CHANNELS
		true_images = [
			plot_channel_time_grid(
				batch[:, :channel_count],
				self.channel_names,
				_timepoint_labels(self.timepoint_names, batch.shape[0]),
				f"True measurements · batch {batch_index + 1}",
			)
			for batch_index, batch in enumerate(DATA_AUGMENTER.return_observed_data())
		]
		self.log_image(
			'TrainingRollout/true_data',
			np.concatenate(true_images, axis=0),
			step=None
		)
		BATCHES = len(x)
		CHANNELS = x[0].shape[1]

		print("Running final trained model for "+str(t)+" steps")
		
		SNAPSHOTS = []
		for b in tqdm(range(BATCHES)):

			# T =nca.run(t*NUMBER_OF_IMAGES,x[b][0],boundary_callback[b]) # Shape T C x y
			T = []
			xb = x[b][0] # C x y
			
			for step in range(t*NUMBER_OF_IMAGES):
				key = jr.fold_in(key,step)
				if step/t >= self.knockout_time:
					xb = xb.at[self.knockout_channel].set(0.0) # Set nodal channel to 0 at and after knockout time
				xb = nca(xb,boundary_callback[b],key)
				T.append(xb)
			T = np.array(T) # Shape T C x y
			
			self.log_video(f"TrainingRollout/trajectory_comp_batch_{b + 1}",rearrange(T[:,:9],"T (cx cy) X Y -> T cx X (cy Y)",cx=3,cy=3),step=None) # type: ignore
			_T_mono = rearrange(T[:,:9],"T (cx cy) X Y -> T () (cx X) (cy Y)",cx=3,cy=3)
			_T_mono = repeat(_T_mono,"T () x y -> T 3 x y")
			self.log_video(f"TrainingRollout/trajectory_monochrome_batch_{b + 1}",_T_mono,step=None) # type: ignore
			T_snapshot = _trajectory_snapshot_channels(
				T, DATA_AUGMENTER, t, self.diagnostic_channel_schema
			)
			SNAPSHOTS.append(plot_channel_time_grid(
				T_snapshot,
				self.channel_names,
				_timepoint_labels(self.timepoint_names, T_snapshot.shape[0]),
				f"NCA predictions · batch {b + 1}",
			))
			
			if SAVE_TRAJECTORY:
				np.save(f"{PVC_PATH}output/{self.wandb_config['name']}_trajectory_{b}.npy",T[::t,:3]) # type: ignore

		self.log_image(
			'TrainingRollout/trajectory_snapshot',
			np.concatenate(SNAPSHOTS, axis=0),
			step=None
		)



class aNCA_Train_log(NCA_Train_log):
	def log_model_parameters(self,nca,i):
		#Log weights and biasses of model every 10 training epochs
		
		pass


# class uNCA_Train_log(NCA_Train_log):
# 	def log_model_parameters(self, nca, i):
# 		# uNCA exposes additional trainable arrays; log all weights generically.
# 		for idx, w in enumerate(nca.get_weights()):
# 			self.log_histogram(f"Train/weight_{idx}", np.squeeze(w), step=i)




class mNCA_Train_log(NCA_Train_log):
	
	def log_model_parameters(self,nca,i):
		#Log weights and biasses of model every 10 training epochs
		
		for scale,W in enumerate(nca.get_weights()):
			w1,w2,b2 = W
			w1 = np.squeeze(w1)
			w2 = np.squeeze(w1)
			b2 = np.squeeze(b2)		
			self.log_histogram(f'Input layer weights, scale {scale}',w1,step=i)
			self.log_histogram(f'Output layer weights, scale {scale}',w2,step=i)
			self.log_histogram(f'Output layer bias, scale {scale}',b2,step=i)				
			weight_matrix_figs = plot_weight_matrices(nca.subNCAs[scale])
			self.log_image(f"Weight matrices, scale {scale}",np.array(weight_matrix_figs)[:,0],step=i)
					
			kernel_weight_figs = plot_weight_kernel_boxplot(nca.subNCAs[scale])
			self.log_image(f"Input weights per kernel, scale {scale}",np.array(kernel_weight_figs)[:,0],step=i)

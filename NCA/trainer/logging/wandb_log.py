"""Weights & Biases logging for NCA training.

``NCALogger`` logs losses and pool statistics every step, weight histograms,
diagnostic plots and example states every ``log_every`` steps, and a full
rollout of the trained model at the end. The plots themselves are made in
``diagnostics.py``.
"""

import os

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import time
from dotenv import load_dotenv
from einops import rearrange, repeat
from tqdm import tqdm

from Common.trainer.wandb_logger import WandbLogger
from Common.utils import get_jax_memory_stats, squarish
from NCA.trainer.intervention import (
	apply_model_with_blocked_channel,
	intervention_slot,
	rollout_model_sampled,
	rollout_model_with_blocked_channel_sampled,
)
from NCA.trainer.interval_schedule import IntervalSchedule, uniform_schedule
from NCA.trainer.logging.diagnostics import (
	biomarker_name,
	compute_channel_correlation_diagnostics,
	compute_channel_time_diagnostics,
	extract_dense_weight_singular_values,
	plot_channel_correlation_diagnostics,
	plot_channel_time_grid,
	plot_radial_intensity_diagnostics,
	plot_radial_intensity_line_diagnostics,
	plot_singular_value_spectrum,
	plot_total_intensity_diagnostics,
	target_aligned_diagnostic_channels,
	timepoint_labels,
)

load_dotenv()
PVC_PATH = os.getenv("PVC_PATH")


def _trajectory_snapshot_channels(T, data_augmenter, t, channel_schema=None):
	"""Select observed channels at observation frames.

	``t`` is either a frame stride or a sequence of frame indices.
	"""
	T_snapshot = T[::t] if isinstance(t, int) else T[np.asarray(t)]
	schema = channel_schema or getattr(data_augmenter, "schema", None)
	if schema is not None:
		return T_snapshot[:, np.asarray(schema.target_to_state)]
	return T_snapshot[:,:data_augmenter.OBS_CHANNELS]


def _rollout_schedule(t, number_of_images):
	"""Interval schedule for a sequential rollout through ``number_of_images`` transitions."""
	if isinstance(t, IntervalSchedule):
		if t.is_uniform:
			return uniform_schedule(t.scan_length, number_of_images, t.times)
		if t.n_slots != number_of_images:
			raise ValueError(
				f"Interval schedule has {t.n_slots} slots for {number_of_images} logged transitions"
			)
		return t
	return uniform_schedule(t, number_of_images)


def _frame_indices(schedule, frame_count):
	"""Frames of a dense rollout that correspond to observed images."""
	return [step for step in schedule.observation_steps if step < frame_count]


def _knockout_step(schedule, knockout_time):
	"""First rollout step with the NODAL read blocked; past the end if never.

	``knockout_time`` is in hours; uniform schedules keep the 12-hour slot rule.
	"""
	if knockout_time is None or knockout_time < 0:
		return schedule.total_steps + 1
	times = None if schedule.mode == "uniform" else schedule.times
	slot = int(intervention_slot(int(knockout_time), times))
	return schedule.observation_steps[min(slot, schedule.n_slots)]


def _steps_at_image_index(schedule, index):
	"""Rollout step at a (possibly fractional) image index; ``index * t`` when uniform."""
	slot = min(int(index), schedule.n_slots - 1)
	return schedule.observation_steps[slot] + (index - slot) * schedule.steps[slot]


def _trajectory_condition_labels(data_augmenter, batch_count, default_knockout_time=None):
	"""Return human-readable condition labels aligned with rollout batches."""
	intervention_times = getattr(data_augmenter, "intervention_times", None)
	if intervention_times is None:
		if default_knockout_time is None:
			return ("baseline",) * batch_count
		intervention_times = (default_knockout_time,) * batch_count
	if len(intervention_times) != batch_count:
		raise ValueError(
			"Intervention times must match the number of logged trajectory batches"
		)
	return tuple(
		"baseline" if time is None or int(time) < 0 else f"Nodal KO at {int(time)}h"
		for time in intervention_times
	)


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


class NCALogger(WandbLogger):
	"""Log NCA training to W&B.

	``knockout_time`` / ``knockout_channel`` (older colony knockout runs): the
	channel is blocked from that image index onwards in the final rollout.
	Newer knockout runs take per-batch times from the augmenter instead.
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
		knockout_time=None,
		knockout_channel=None,
		**kwargs,
	):
		if (knockout_time is None) != (knockout_channel is None):
			raise ValueError("knockout_time and knockout_channel must be set together")
		self.knockout_time = knockout_time
		self.knockout_channel = knockout_channel
		data = kwargs.get("data", args[0] if args else None)
		data_values = [] if data is None else list(data)
		uniform_data = bool(data_values) and len(
			{tuple(value.shape) for value in data_values}
		) == 1
		self.diagnostic_targets = (
			np.stack(data_values)[:, 1:] if uniform_data else None
		)
		self.diagnostic_boundary_mask = None if boundary_mask is None else np.array(boundary_mask)
		self.diagnostic_channel_schema = channel_schema or getattr(data_augmenter, "schema", None)
		self.diagnostic_group_sizes = (
			self.diagnostic_channel_schema.group_sizes
			if self.diagnostic_channel_schema is not None
			else None
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
			self.channel_names = [biomarker_name(name) for name in channel_names]
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
				timepoint_labels(self.timepoint_names, batch.shape[0]),
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
			predictions = target_aligned_diagnostic_channels(
				predictions,
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
		"""Log weight histograms (and images of 2D weights) at training step ``i``."""
		
		for idx, w in enumerate(nca.get_weights()):
			w = np.squeeze(w)
			self.log_histogram(f"Train/weight_{idx}", w, step=i)
			if len(w.shape) == 2:
				w = repeat(w,"W H -> W H 3")
				self.log_image(f"Train/weight_image_{idx}", self.normalise_images(w), step=i)
		self.log_singular_value_spectra(nca,i)

	def log_model_diagnostics(self,model,log_dict,i):
		"""Weight diagnostics every LOG_EVERY steps (the KAN logger adds its own)."""
		self.log_model_parameters(model,i)

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
		"""Log memory use and example states at training step ``i``.

		Parameters
		----------
		x : dict
			``{"states": list (one per batch) of [N, CHANNELS, x, y] arrays}``
		i : int
			Training step.
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
	
	def log_training_step(self,log_dict,i,model,write_images=True,LOG_EVERY=10):
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
			self.log_model_diagnostics(model,log_dict,i)
			self.log_channel_time_diagnostics(log_dict,i)
			if write_images:
				self.log_model_outputs(log_dict,i)

	def _knockout(self, data_augmenter, schedule, batch):
		"""(blocked channel, first blocked step) for one batch, or None."""
		intervention_times = getattr(data_augmenter, "intervention_times", None)
		if intervention_times is not None:
			return data_augmenter.nodal_channel, _knockout_step(schedule, intervention_times[batch])
		if self.knockout_time is not None:
			# knockout_time is an image index, the same for every batch
			return self.knockout_channel, _steps_at_image_index(schedule, self.knockout_time)
		return None

	def _log_rollout_videos(self, T, b):
		if self.knockout_time is not None:
			# Older colony knockout runs: the 9 state channels as a 3x3 grid
			self.log_video(f"TrainingRollout/trajectory_comp_batch_{b + 1}",rearrange(T[:,:9],"T (cx cy) X Y -> T cx X (cy Y)",cx=3,cy=3),step=None) # type: ignore
			_T_mono = rearrange(T[:,:9],"T (cx cy) X Y -> T () (cx X) (cy Y)",cx=3,cy=3)
			_T_mono = repeat(_T_mono,"T () x y -> T 3 x y")
			self.log_video(f"TrainingRollout/trajectory_monochrome_batch_{b + 1}",_T_mono,step=None) # type: ignore
			return
		self.log_video(f"TrainingRollout/trajectory_batch_{b + 1}",T[:,:3],step=None)
		if T.shape[1] > 3:
			hidden = T[:, 3:]
			extra_zeros = (-hidden.shape[1])%3
			hidden = np.pad(hidden,((0,0),(0,extra_zeros),(0,0),(0,0)))
			_cy,_cx = squarish(hidden.shape[1]//3)
			hidden = rearrange(hidden,"Time (cx cy C) x y  -> Time C (cx x) (cy y)",C=3,cy=_cy,cx=_cx)
			hidden = (np.tanh(hidden)+1.0)/2.0
			self.log_video(f"TrainingRollout/hidden_trajectory_batch_{b + 1}",hidden,step=None)

	def log_training_end(self,
						 nca,
						 DATA_AUGMENTER,
						 t,
						 boundary_callback,
						 SAVE_TRAJECTORY=False,
						 write_images=True,
						 write_videos=True,
						 boundary_masks=None,
						 boundary_mode="soft",
						 key=None):
		"""Roll out the trained NCA from every initial condition and log it."""
		if key is None:
			key = jr.PRNGKey(int(time.time()))
		x,y = DATA_AUGMENTER.split_x_y(1)
		x,y = DATA_AUGMENTER.advance_pool(x,y,0,key)
		schedule = _rollout_schedule(t, x[0].shape[0])
		total_steps = schedule.total_steps
		observation_steps = jnp.asarray(schedule.observation_steps)
		condition_labels = _trajectory_condition_labels(
			DATA_AUGMENTER, len(x), default_knockout_time=self.knockout_time
		)
		# Log true data for side by side comparison
		schema = self.diagnostic_channel_schema or getattr(DATA_AUGMENTER, "schema", None)
		channel_count = schema.n_measurement_channels if schema else DATA_AUGMENTER.OBS_CHANNELS
		true_images = [
			plot_channel_time_grid(
				batch[:, :channel_count],
				self.channel_names,
				timepoint_labels(self.timepoint_names, batch.shape[0]),
				f"True measurements · {condition_labels[batch_index]} · batch {batch_index + 1}",
			)
			for batch_index, batch in enumerate(DATA_AUGMENTER.return_observed_data())
		]
		self.log_image(
			'TrainingRollout/true_data',
			np.concatenate(true_images, axis=0),
			step=None
		)

		print("Running final trained model for "+str(total_steps)+" steps")
		SNAPSHOTS = []
		for b in tqdm(range(len(x))):
			initial_state = nca.prepare_pool_state(x[b][0])
			knockout = self._knockout(DATA_AUGMENTER, schedule, b)
			if not write_videos:
				# Only the observation frames are kept
				boundary_mask = (
					jnp.empty((0,), dtype=initial_state.dtype)
					if boundary_masks is None
					else boundary_masks[b]
				)
				mode = "none" if boundary_masks is None else boundary_mode
				if knockout is None:
					T = rollout_model_sampled(
						nca, initial_state, boundary_mask, mode,
						jr.fold_in(key, b), total_steps, observation_steps,
					)
				else:
					T = rollout_model_with_blocked_channel_sampled(
						nca, initial_state, boundary_mask, mode,
						jr.fold_in(key, b), total_steps,
						knockout[0], int(knockout[1]), observation_steps,
					)
			elif knockout is None:
				T = nca.run(total_steps, initial_state, boundary_callback[b])
			else:
				state = initial_state
				trajectory = [state]
				rollout_key = jr.fold_in(key, b)
				blocked_channel, knockout_step = knockout
				for step in range(total_steps):
					rollout_key = jr.fold_in(rollout_key, step)
					state = apply_model_with_blocked_channel(
						nca,
						state,
						boundary_callback[b],
						rollout_key,
						blocked_channel,
						step >= knockout_step,
					)
					trajectory.append(state)
				T = jnp.stack(trajectory)
			if write_videos:
				self._log_rollout_videos(T, b)
			frames = _frame_indices(schedule, T.shape[0]) if write_videos else 1
			T_snapshot = _trajectory_snapshot_channels(
				T, DATA_AUGMENTER, frames, self.diagnostic_channel_schema
			)
			SNAPSHOTS.append(plot_channel_time_grid(
				T_snapshot,
				self.channel_names,
				timepoint_labels(self.timepoint_names, T_snapshot.shape[0]),
				f"NCA predictions · {condition_labels[b]} · batch {b + 1}",
			))
			if SAVE_TRAJECTORY:
				np.save(f"{PVC_PATH}output/{self.wandb_config['name']}_trajectory_{b}.npy",T[frames if write_videos else slice(None),:3])  # type: ignore

		self.log_image(
			'TrainingRollout/trajectory_snapshot',
			np.concatenate(SNAPSHOTS, axis=0),
			step=None
		)

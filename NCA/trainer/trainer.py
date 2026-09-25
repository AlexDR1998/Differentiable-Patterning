"""Public entry point of the NCA trainer.

``NcaTrainer`` holds what is derived from the loaded data (augmenter, boundary
callbacks, loss masks, channel counts) and the logger. ``train`` then runs:

    preparation.prepare_training -> step.build_train_step -> runner.run_loop
"""

from pathlib import Path
from dataclasses import asdict

import jax.numpy as jnp
import jax.tree_util as jtu
import datetime
from NCA.trainer.logging.tensorboard import (
	NCA_Train_log,
	NCA_knockout_Train_log,
)
from NCA.trainer.logging.kan_tensorboard import (
	kaNCA_Train_log,
	uses_fast_kan_diagnostics,
)
from NCA.trainer.context import TrainerContext
from einops import repeat, rearrange
from Common.model.boundary import model_boundary, hard_boundary, no_boundary


def describe_batch_shapes(value):
	if hasattr(value, "shape"):
		return str(value.shape)
	return str([leaf.shape for leaf in jtu.tree_leaves(value)])


def select_wandb_train_logger_class(model, knockout_time=None):
	if knockout_time is not None:
		return NCA_knockout_Train_log
	if uses_fast_kan_diagnostics(model):
		return kaNCA_Train_log
	return NCA_Train_log

class NcaTrainer:
	"""Config-driven NCA trainer.

	User choices come from the experiment config; the trainer keeps only the
	sections it uses. Values derived from loaded data come in through
	:class:`TrainerContext`.
	"""

	def __init__(self, config, model, data, context: TrainerContext):
		self.run_config = config.run
		self.trainer_config = config.trainer
		self.optimiser_config = config.optimiser
		self.loss_config = config.loss
		self.logging_config = config.logging
		self.knockout_config = config.data.knockout
		self.context = context
		trainer_config = config.trainer
		self.model = model
		data_augmenter = context.data_augmenter
		channel_schema = context.channel_schema or getattr(data_augmenter, "schema", None)
		self.channel_schema = channel_schema
		self.channel_names = context.channel_names
		self.timepoint_names = context.timepoint_names
		self.intervention_times = context.training_intervention_times
		self.nodal_channel = (
			channel_schema.state_channels.index("NODAL")
			if self.intervention_times is not None
			and channel_schema is not None
			and "NODAL" in channel_schema.state_channels
			else None
		)
		boundary_mask = context.boundary_mask
		self.diagnostic_boundary_mask = boundary_mask
		
		# Set up variables 
		self.channels = self.model.N_CHANNELS
		if context.observed_channels is None and channel_schema is not None:
			self.observed_channels = channel_schema.n_state_channels
		elif context.observed_channels is None:
			self.observed_channels = data[0].shape[1]
		else:
			self.observed_channels = context.observed_channels
		# For some loss functions, the NCA observable channels don't necessarily match the data channels. Handle this here.
		if context.data_channels is None and channel_schema is not None:
			self.data_channels = channel_schema.n_measurement_channels
		elif context.data_channels is None:
			self.data_channels = self.observed_channels
		else:
			self.data_channels = context.data_channels
		
		
		self.sharding = trainer_config.sharding
		if self.intervention_times is not None and self.sharding not in (None, 1):
			raise ValueError(
				"NODAL read-block interventions currently require trainer.sharding=null or 1"
			)
		self.grad_loss = trainer_config.grad_loss
		self.loss_time_channel_mask = context.loss_time_channel_mask
		# Set up data and data augmenter class
		self._data_raw = data
		augmenter_kwargs = dict(
			data_true=data,
			hidden_channels=0 if channel_schema is not None else self.channels-self.data_channels,
			nca_model=self.model,
			)
		self.data_augmenter = data_augmenter(**augmenter_kwargs)
		self.data_augmenter.data_init(self.sharding)
		self.data = self.data_augmenter.return_saved_data()
		self.batch_count = len(self.data)
		print("Batches = "+str(self.batch_count))
		
		# Set up partial mask of channels / timesteps
		if self.loss_time_channel_mask is None:
			timepoints = data.shape[1] if hasattr(data, "shape") else data[0].shape[0]
			self.loss_time_channel_mask = jnp.ones((self.batch_count,timepoints-1,self.data_channels),dtype=jnp.float32)

		_model_kernel_length = len(self.model.KERNEL_STR)
		if "GRAD" in self.model.KERNEL_STR:
			_model_kernel_length+=1
		if self.grad_loss:
			self.loss_time_channel_mask = repeat(self.loss_time_channel_mask,"b n c -> b n (gc c) () ()",gc=_model_kernel_length)
			print("Timestep / Channel mask: ")
			print(self.loss_time_channel_mask[:,:,:,0,0])
		else:
			self.loss_time_channel_mask = rearrange(self.loss_time_channel_mask,"b n c -> b n c () ()")
			print("Timestep / Channel mask: ")
			print(self.loss_time_channel_mask[:,:,:,0,0])

		self.loss_time_channel_mask = list(self.loss_time_channel_mask)
		# Set up boundary augmenter class
		# length of BOUNDARY_MASK PyTree should be same as number of batches
		

		self.boundary_callbacks = []
		for b in range(self.batch_count):
			if boundary_mask is not None:
				if trainer_config.boundary_mode == "soft":
					self.boundary_callbacks.append(model_boundary(boundary_mask[b]))
				elif trainer_config.boundary_mode == "hard":
					self.boundary_callbacks.append(hard_boundary(boundary_mask[b]))
				else:
					raise ValueError("trainer.boundary_mode must be 'soft' or 'hard'")
			else:
				self.boundary_callbacks.append(no_boundary())
		
		self._log_root = trainer_config.log_directory
		self._model_root = context.model_directory
		# Keep human-readable names in logging metadata. Checkpoint paths use a
		# bounded, collision-resistant storage ID supplied by the entrypoint.
		self.model_filename = context.storage_id or context.run_name
		
	def setup_logging(self):
		"""Create the logger chosen by logging.backend, and the checkpoint path."""
		logging_backend = self.logging_config.backend
		settings = self.logging_config.singular_values
		singular_value_settings = {
			"enabled": bool(settings.enabled),
			"plot_spectra": bool(settings.plot_spectra),
			"epsilon": float(settings.epsilon),
		}
		wandb_args = {
			"project": self.logging_config.wandb.project,
			"group": self.logging_config.wandb.group,
			"tags": list(self.context.wandb_tags),
			"name": self.context.run_name,
		}
		knockout = {
			"time": self.knockout_config.time,
			"channel": self.knockout_config.channel,
		}
		# Set logging behvaiour based on provided filename
		print(f"Raw data shape(s): {describe_batch_shapes(self._data_raw)}")
		logging_data = self.data_augmenter.return_observed_data()
		if self.model_filename is None:
			self.model_filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
			self.is_logging = False
		else:
			if logging_backend == "none":
				self.is_logging = False
			elif logging_backend=="tensorboard":
				self.is_logging = True
				self.log_directory = str(
					Path(self._log_root) / self.model_filename / "train"
				)
				if uses_fast_kan_diagnostics(self.model):
					self.logger = kaNCA_Train_log(self.log_directory,logging_data)
				else:
					self.logger = NCA_Train_log(
						self.log_directory,
						logging_data,
						singular_value_config=singular_value_settings,
					)
				print("Logging training to: "+self.log_directory)
			elif logging_backend=="wandb":
				self.is_logging = True
				self.log_directory = str(
					Path(self._log_root) / self.model_filename / "train"
				)
				wandb_args["config"] = {
					"model": self.model.get_config(),
					"run": asdict(self.run_config),
					"trainer": asdict(self.trainer_config),
					"optimiser": asdict(self.optimiser_config),
					"loss": asdict(self.loss_config),
				}
				
				if knockout["time"] is not None: # Nodal KO has differet logging behaviour
					self.logger = NCA_knockout_Train_log(
						data=logging_data,
						wandb_config=wandb_args,
						boundary_mask=self.diagnostic_boundary_mask,
						channel_names=self.channel_names,
						channel_schema=self.channel_schema,
						timepoint_names=self.timepoint_names,
						data_augmenter=self.data_augmenter,
						knockout_time=knockout["time"],
						knockout_channel=knockout["channel"],
						singular_value_config=singular_value_settings)
				else:
					logger_class = select_wandb_train_logger_class(self.model)
					self.logger = logger_class(
						data=logging_data,
						wandb_config=wandb_args,
						boundary_mask=self.diagnostic_boundary_mask,
						channel_names=self.channel_names,
						channel_schema=self.channel_schema,
						timepoint_names=self.timepoint_names,
						data_augmenter=self.data_augmenter,
						singular_value_config=singular_value_settings,
					)
				print("Logging training to: "+self.log_directory)
			else:
					raise ValueError(
					"logging.backend must be 'none', 'wandb' or 'tensorboard'"
				)
		self.model_path = str(Path(self._model_root) / self.model_filename)
		print("Saving model to: "+self.model_path)

	def train(
		self,
		*,
		key,
		timesteps=None,
		loss_overrides=None,
		progress_callback=None,
	):
		"""Prepare, compile and execute one configured training run."""
		from NCA.trainer.preparation import prepare_training
		from NCA.trainer.runner import run_loop
		from NCA.trainer.step import build_train_step
		from NCA.trainer.validation import build_validation_evaluator

		setup = prepare_training(
			self,
			key=key,
			timesteps=timesteps,
			loss_overrides=loss_overrides,
		)
		self.setup_logging()
		step = build_train_step(self, setup)
		validation_evaluator = build_validation_evaluator(self, setup)
		return run_loop(
			self,
			setup,
			step,
			progress_callback=progress_callback,
			validation_evaluator=validation_evaluator,
		)


def build_trainer(config, model, data, context: TrainerContext) -> NcaTrainer:
	"""Construct an NCA trainer from an experiment config."""
	return NcaTrainer(config, model, data, context)


__all__ = [
	"NcaTrainer",
	"TrainerContext",
	"build_trainer",
	"describe_batch_shapes",
	"select_wandb_train_logger_class",
]

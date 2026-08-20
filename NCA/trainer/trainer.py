from pathlib import Path
from dataclasses import asdict

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import equinox as eqx
import datetime
from NCA.trainer.logging.tensorboard import (
	NCA_Train_log,
	mNCA_Train_log,
	aNCA_Train_log,
	NCA_knockout_Train_log,
)
from NCA.trainer.logging.kan_tensorboard import (
	kaNCA_Train_log,
	uses_fast_kan_diagnostics,
)
from NCA.model.NCA_KAN_model import kaNCA
from NCA.model.NCA_multi_scale import mNCA
from NCA.model.NCA_multihead_attention import aNCA
from NCA.trainer.training_execution import TrainingExecution
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
	"""Config-driven NCA trainer with an explicit numerical lifecycle.

	Configuration is immutable and comes from one experiment config.  Values
	derived from loaded data live in :class:`TrainerContext` instead of being
	exposed as a second collection of trainer options.
	"""

	def __init__(self, config, model, data, context: TrainerContext):
		self.config = config
		self.context = context
		trainer_config = config.training.trainer
		self.model = model
		data_augmenter = context.data_augmenter
		channel_schema = context.channel_schema or getattr(data_augmenter, "schema", None)
		self.channel_schema = channel_schema
		self.channel_names = context.channel_names
		self.timepoint_names = context.timepoint_names
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
		self.model_filename = context.run_name
		#print(jax.tree_util.tree_structure(self.boundary_callbacks))
		
	def setup_logging(self,logging_backend,wandb_args,knockout,singular_value_settings=None):
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
				if isinstance(self.model ,kaNCA) or uses_fast_kan_diagnostics(self.model):
					self.logger = kaNCA_Train_log(self.log_directory,logging_data)
				elif isinstance(self.model , mNCA):
					self.logger = mNCA_Train_log(self.log_directory,logging_data)
				elif isinstance(self.model , aNCA):
					self.logger = aNCA_Train_log(self.log_directory,logging_data)
				# elif isinstance(self.model, uNCA):
					# self.logger = uNCA_Train_log(self.log_directory, self._data_raw)
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
					"training": asdict(self.config.training),
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

	def _make_batched_nca(self, nca):
		"""Build the established vmap/tree-map NCA application path.

		Accelerator-specific trainers may override this hook without adding
		backend flags or batching branches to the core training loop.
		"""
		apply_with_boundary = jax.vmap(
			nca, in_axes=(0, None, 0), out_axes=0, axis_name="N"
		)
		return lambda x, callback, key_array: jtu.tree_map(
			apply_with_boundary, x, callback, key_array
		)

	def _training_execution(self):
		return TrainingExecution(self)

	def _run_nca_steps(
		self,
		nca,
		vv_nca,
		states,
		reg_logs_internal,
		t,
		key,
		loop_autodiff,
		apply_intermediate_regs,
		training_execution,
	):
		"""Reference one-step scan; SYCL trainers may override the rollout."""
		state_shape = states[0].shape[0]

		def nca_step(carry, j):
			step_key, state, reg_logs = carry
			step_key = jr.fold_in(step_key, j)
			key_array = list(jr.randint(
				step_key,
				shape=(self.batch_count, state_shape, 2),
				minval=0,
				maxval=2_147_483_647,
				dtype=jnp.uint32,
			))
			new_state = vv_nca(
				state, training_execution.boundary_callbacks(), key_array
			)
			reg_logs = apply_intermediate_regs(
				reg_logs,
				state,
				new_state,
				{"model": vv_nca},
				step_key,
			)
			return (step_key, new_state, reg_logs), None

		carry, _ = eqx.internal.scan(
			nca_step,
			(key, states, reg_logs_internal),
			xs=jnp.arange(t),
			kind=loop_autodiff,
		)
		return carry
	
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
		from NCA.trainer.runner import run_training
		from NCA.trainer.step import build_train_step

		setup = prepare_training(
			self,
			key=key,
			timesteps=timesteps,
			loss_overrides=loss_overrides,
		)
		step = build_train_step(self, setup)
		return run_training(
			self,
			setup,
			step,
			progress_callback=progress_callback,
		)


def build_trainer(config, model, data, context: TrainerContext) -> NcaTrainer:
	"""Construct the trainer selected by the typed backend configuration."""
	backend_type = config.training.trainer.backend.type
	is_sycl_model = config.model.family in {"NCA_sycl", "gNCA_sycl"}
	if (backend_type == "sycl") != is_sycl_model:
		raise ValueError(
			"model.family in {'NCA_sycl', 'gNCA_sycl'} and trainer.backend.type='sycl' "
			"must be selected together"
		)
	if backend_type == "sycl":
		from NCA.trainer.backend.sycl.trainer import SyclNcaTrainer

		return SyclNcaTrainer(config, model, data, context)
	if backend_type not in {"none", "nvidia"}:
		raise ValueError(f"Unsupported trainer backend {backend_type!r}")
	return NcaTrainer(config, model, data, context)


__all__ = [
	"NcaTrainer",
	"TrainerContext",
	"build_trainer",
	"describe_batch_shapes",
	"select_wandb_train_logger_class",
]

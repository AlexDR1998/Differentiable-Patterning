import jax
import jax.numpy as jnp
import equinox as eqx
from pathlib import Path
from typing import Any, Union
import abc


class AbstractModel(eqx.Module):
	
	@abc.abstractmethod
	def __call__(self, *args, **kwargs) -> Any:
		raise NotImplementedError

	def partition(self):
		"""
		Behaves like eqx.partition. Overwrite in subclasses to account for hard coded array parameters

		Returns
		-------
		diff : PyTree
			PyTree of same structure as AbstractModel, with all non trainable parameters set to None
		static : PyTree
			PyTree of same structure as AbstractModel with all trainable parameters set to None

		"""
		diff,static = eqx.partition(self,eqx.is_array)
		return diff,static

	def boundary_regulariser_state(self, state):
		"""Return the part of ``state`` governed by the spatial boundary.

		Most models evolve one grid, so the complete state is appropriate.
		Hierarchical models can override this view to exclude auxiliary grids
		from fine-resolution boundary penalties.
		"""
		return state

	def prepare_pool_state(self, state):
		"""Prepare a state before it is used as a new training rollout input.

		Ordinary models preserve the complete recurrent state. Hierarchical
		models can override this to reconstruct auxiliary levels that should
		not themselves be persisted by the training pool.
		"""
		return state
	
	def get_weights(self):
		"""Returns a flat list of the trainable arrays (squeezed), for plotting and logging.

		Returns:
			weights : list of arrays of trainable parameters
		"""
		diff_self,_ = self.partition()
		return [jnp.squeeze(w) for w in jax.tree_util.tree_leaves(diff_self)]
	
	def save(self, path: Union[str, Path], overwrite: bool = False):
		"""
		Save the model with eqx.tree_serialise_leaves. A ".eqx" suffix is added if missing.

		Parameters
		----------
		path : Union[str, Path]
			path to filename.
		overwrite : bool, optional
			Overwrite existing filename. The default is False.

		Raises
		------
		RuntimeError
			file already exists.

		"""
		path = Path(path)
		if path.suffix != ".eqx":
			path = path.with_suffix(".eqx")
		path.parent.mkdir(parents=True, exist_ok=True)
		if path.exists() and not overwrite:
			raise RuntimeError(f'File {path} already exists.')
		eqx.tree_serialise_leaves(path, self)

	
	def load(self, path: Union[str, Path]):
		"""
		Load saved parameters into a model with the same structure as self,
		with eqx.tree_deserialise_leaves. A ".eqx" suffix is added if missing.

		Parameters
		----------
		path : Union[str, Path]
			path to filename.

		Raises
		------
		ValueError
			Not a file.

		Returns
		-------
		AbstractModel
			the loaded model.

		"""
		path = Path(path)
		if path.suffix != ".eqx":
			path = path.with_suffix(".eqx")
		if not path.is_file():
			raise ValueError(f'Not a file: {path}')
		return eqx.tree_deserialise_leaves(path,self)

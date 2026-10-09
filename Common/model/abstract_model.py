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
		"""Split the model into trainable and non-trainable parts, like ``eqx.partition``.

		Subclasses override this to keep fixed array parameters out of ``diff``.

		Returns
		-------
		diff : PyTree
			Same structure as the model, with non-trainable parameters set to None.
		static : PyTree
			Same structure as the model, with trainable parameters set to None.
		"""
		diff,static = eqx.partition(self,eqx.is_array)
		return diff,static

	def boundary_regulariser_state(self, state):
		"""Part of ``state`` that boundary penalties apply to (the whole state by default).

		Hierarchical models override this to leave out their coarser grids.
		"""
		return state

	def prepare_pool_state(self, state):
		"""Prepare a pool state before it starts a new training rollout (unchanged by default).

		Hierarchical models override this to rebuild the coarser levels, which
		the training pool does not store.
		"""
		return state
	
	def get_weights(self):
		"""Flat list of the trainable arrays (squeezed), for plotting and logging."""
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

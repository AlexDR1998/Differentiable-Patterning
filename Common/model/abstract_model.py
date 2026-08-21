import jax
import json
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
	
	def combine(self,diff,static):
		"""
		Wrapper for eqx.combine

		Parameters
		----------
		diff : PyTree
			PyTree of same structure as AbstractModel, with all non trainable parameters set to None
		static : PyTree
			PyTree of same structure as AbstractModel, with all trainable parameters set to None

		"""
		self = eqx.combine(diff,static)
	
	def get_weights(self):
		"""Returns list of arrays of weights, for plotting purposes, or for manually adjusting weights with
		code that doesn't `just work' on PyTrees

		Returns:
			weights : list of arrays of trainable parameters 
		"""
		
		
		diff_self,_ = self.partition()
		ws,tree_def = jax.tree_util.tree_flatten(diff_self)
		#return #list(map(jnp.squeeze,ws))
		return ws,tree_def
	
	
	def set_weights(self,tree_def,weights):

		#raise NotImplementedError
		return jax.tree_util.tree_unflatten(tree_def,weights)
	
	def save(self, path: Union[str, Path], overwrite: bool = False, hyperparams: dict = {}):
		"""
		Wrapper for saving model via eqx.tree_serialise_leaves. Taken from https://github.com/google/jax/issues/2116

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

		Returns
		-------
		None.

		"""

		# suffix = ".eqx"
		# path = Path(path)
		# if path.suffix != suffix:
		# 	path = path.with_suffix(suffix)
		# 	path.parent.mkdir(parents=True, exist_ok=True)
		# if path.exists():
		# 	if overwrite:
		# 		path.unlink()
		# 	else:
		# 		raise RuntimeError(f'File {path} already exists.')
		suffix = ".eqx"
		path = Path(path)
		if path.suffix != suffix:
			path = path.with_suffix(suffix)
		path.parent.mkdir(parents=True, exist_ok=True)
		if path.exists() and not overwrite:
			raise RuntimeError(f'File {path} already exists.')
		eqx.tree_serialise_leaves(path, self)

	
	def load(self, path: Union[str, Path]):
		"""
		Wrapper for loading model via eqx.tree_deserialise_leaves

		Parameters
		----------
		path : Union[str, Path]
			path to filename.

		Raises
		------
		ValueError
			Not a file or incorrect file type.

		Returns
		-------
		AbstractModel
			AbstractModel loaded from pickle.

		"""
		suffix = ".eqx"
		path = Path(path)
		if path.suffix != suffix:
			#raise ValueError(f'Not a {suffix} file: {path}')
			# path = path.with_suffix(suffix)
			path = path + suffix # type: ignore
		# with open(path, "rb") as f:
		if not path.is_file(): # type: ignore
			raise ValueError(f'Not a file: {path}')
		# 	hyperparams = json.loads(f.readline().decode())
		# 	#func = F_rda(key=jr.PRNGKey(0), **hyperparams["pde"])
		# 	#pde = PDE_solver(func,**hyperparams["solver"])
		# 	model = self.__init__(**hyperparams)
		# 	return eqx.tree_deserialise_leaves(f, model)
		return eqx.tree_deserialise_leaves(path,self)
	
	def load_with_hyperparam_dict(self, path: Union[str, Path]):
		"""
		Wrapper for loading model via eqx.tree_deserialise_leaves

		Parameters
		----------
		path : Union[str, Path]
			path to filename.

		Raises
		------
		ValueError
			Not a file or incorrect file type.

		Returns
		-------
		tuple
			model and hyperparams loaded from pickle.

		"""
		
		suffix = ".eqx"
		path = Path(path)
		
		if not path.is_file():
			raise ValueError(f'Not a file: {path}')
		
		if path.suffix != suffix:
			raise ValueError(f'Not a {suffix} file: {path}')
		
		with open(path, "rb") as f:
			hyperparams = json.loads(f.readline().decode())
			print("Hyperparams: ",hyperparams)
			model = self.__init__(**hyperparams)
			return eqx.tree_deserialise_leaves(f, model), hyperparams

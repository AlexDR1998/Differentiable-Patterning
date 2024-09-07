#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:46:20 2023

@author: s1605376
"""
import jax
import equinox as eqx
import jax.numpy as jnp
import time
from PDE.model.reaction_diffusion_advection.advection import V
from PDE.model.reaction_diffusion_advection.reaction import R
from PDE.model.reaction_diffusion_advection.diffusion_nonlinear import D
from PDE.model.reaction_diffusion_advection.identity import I
from jaxtyping import Array, Float, PyTree, Scalar

class F(eqx.Module):
	f_v: eqx.Module
	f_r: eqx.Module
	f_d: eqx.Module
	TERMS: list
	N_CHANNELS: int
	N_LAYERS: int
	ORDER: int
	PADDING: str
	dx: float
	def __init__(self,
			  	 N_CHANNELS,
				 PADDING,
				 dx,
				 TERMS=["reaction","advection","diffusion"],
				 INTERNAL_ACTIVATION="tanh",
				 ADVECTION_OUTER_ACTIVATION="tanh",
				 INIT_SCALE={"reaction":0.5,"advection":0.5,"diffusion":0.5},
				 INIT_TYPE={"reaction":"normal","advection":"normal","diffusion":"normal"},
				 USE_BIAS=True,
				 STABILITY_FACTOR=0.01,
				 ORDER=1,
				 N_LAYERS=1,
				 ZERO_INIT={"reaction":True,"advection":True,"diffusion":False},
				 key=jax.random.PRNGKey(int(time.time()))):
		
		if INTERNAL_ACTIVATION=="tanh":
			INTERNAL_ACTIVATION = jax.nn.tanh
		elif INTERNAL_ACTIVATION=="relu":
			INTERNAL_ACTIVATION = jax.nn.relu
		elif INTERNAL_ACTIVATION=="relu6":
			INTERNAL_ACTIVATION = jax.nn.relu6
		
		if ADVECTION_OUTER_ACTIVATION=="tanh":
			ADVECTION_OUTER_ACTIVATION = jax.nn.tanh
		elif ADVECTION_OUTER_ACTIVATION=="relu":
			ADVECTION_OUTER_ACTIVATION = jax.nn.relu
		elif ADVECTION_OUTER_ACTIVATION=="linear":
			ADVECTION_OUTER_ACTIVATION = lambda x:x

		self.N_CHANNELS = N_CHANNELS
		self.PADDING = PADDING
		self.dx = dx
		self.N_LAYERS = N_LAYERS
		self.ORDER = ORDER
		self.TERMS = TERMS
		key1,key2,key3 = jax.random.split(key,3)
		if "reaction" in self.TERMS:
			self.f_r = R(N_CHANNELS=N_CHANNELS,
						INTERNAL_ACTIVATION=INTERNAL_ACTIVATION,
						OUTER_ACTIVATION=jax.nn.relu6, # SHOULD BE STRICTLY NON NEGATIVE FOR INTERPRETABILITY
						INIT_SCALE=INIT_SCALE["reaction"], # Should be much smaller initial scaling
						INIT_TYPE=INIT_TYPE["reaction"],
						USE_BIAS=USE_BIAS,
						STABILITY_FACTOR=STABILITY_FACTOR,
						ORDER=ORDER,
						N_LAYERS=N_LAYERS,
						ZERO_INIT=ZERO_INIT["reaction"],
						key=key1)
		else:
			self.f_r = I(TYPE="reaction")
		
		if "advection" in self.TERMS:
			self.f_v = V(N_CHANNELS=N_CHANNELS,
						PADDING=PADDING,
						dx=dx,
						INTERNAL_ACTIVATION=INTERNAL_ACTIVATION,
						OUTER_ACTIVATION=ADVECTION_OUTER_ACTIVATION, # Can be any activation function
						INIT_SCALE=INIT_SCALE["advection"],
						INIT_TYPE=INIT_TYPE["advection"],
						USE_BIAS=USE_BIAS,
						ORDER=ORDER,
						N_LAYERS=N_LAYERS,
						ZERO_INIT=ZERO_INIT["advection"],
						DIM = 2,																
						key=key2)
		else:
			self.f_v = I(TYPE="advection")
		
		if "diffusion" in self.TERMS:
			self.f_d = D(N_CHANNELS=N_CHANNELS,
						PADDING=PADDING,
						dx=dx,
						INTERNAL_ACTIVATION=INTERNAL_ACTIVATION,
						OUTER_ACTIVATION=jax.nn.relu6, # MUST BE STRICTLY NON NEGATIVE FOR NUMERICAL STABILITY
						INIT_SCALE=INIT_SCALE["diffusion"],
						INIT_TYPE=INIT_TYPE["diffusion"],
						USE_BIAS=USE_BIAS,
						ORDER=ORDER,
						N_LAYERS=N_LAYERS,
						ZERO_INIT=ZERO_INIT["diffusion"],
						key=key3)
		else:
			self.f_d = I(TYPE="diffusion")



	@eqx.filter_jit
	def __call__(self,
			  	 t: Float[Scalar, ""],
				 X: Float[Array, "{self.N_CHANNELS} x y"],
				 args)-> Float[Array, "{self.N_CHANNELS} x y"]:
		"""

		Parameters
		----------
		t : float32
			timestep
		
		X : float32 [N_CHANNELS,_,_]
			input PDE lattice state.
		
		args : None
			Required for format of diffrax.ODETerm

		Returns
		-------
		X_update : float32 [N_CHANNELS,_,_]
			update to PDE lattice state state.

		"""
		return self.f_d(X) - self.f_v(X) + self.f_r(X)
	
	def partition(self):
		r_diff,r_static = self.f_r.partition()
		v_diff,v_static = self.f_v.partition()
		d_diff,d_static = self.f_d.partition()
		total_diff,total_static = eqx.partition(self,eqx.is_array)
		#print(total_diff)
		#print(total_static)
		where_r = lambda m:m.f_r
		where_v = lambda m:m.f_v
		where_d = lambda m:m.f_d
		
		total_diff = eqx.tree_at(where_r,total_diff,r_diff)
		total_diff = eqx.tree_at(where_v,total_diff,v_diff)
		total_diff = eqx.tree_at(where_d,total_diff,d_diff)
		
		total_static = eqx.tree_at(where_r,total_static,r_static)
		total_static = eqx.tree_at(where_v,total_static,v_static)
		total_static = eqx.tree_at(where_d,total_static,d_static)
		return total_diff,total_static
	
	def combine(self,diff,static):
		self = eqx.combine(diff,static)
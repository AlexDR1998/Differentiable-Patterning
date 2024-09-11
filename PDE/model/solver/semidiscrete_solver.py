import jax
import equinox as eqx
import time
import json
import jax.random as jr
import diffrax
from PDE.model.reaction_diffusion_advection.update import F

from Common.model.abstract_model import AbstractModel

class PDE_solver(AbstractModel):
	func: eqx.Module
	SOLVER: diffrax.AbstractSolver
	stepsize_controller: diffrax.AbstractStepSizeController
	dt0: float
	def __init__(self,F,dt=0.1,SOLVER="heun",ADAPTIVE=False,DTYPE="float32",rtol=1e-3,atol=1e-3):
		
		self.dt0 = dt
		if SOLVER=="heun":
			self.SOLVER = diffrax.Heun()
		elif SOLVER=="euler":
			self.SOLVER = diffrax.Euler()
		elif SOLVER=="tsit5":
			self.SOLVER = diffrax.Tsit5()
		elif SOLVER=="kencarp3":
			self.SOLVER = diffrax.KenCarp3()

		if ADAPTIVE:
			self.stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol)
		else:
			self.stepsize_controller=diffrax.ConstantStepSize()

		if DTYPE=="bfloat16":
			def to_bfloat16(x):
				if eqx.is_inexact_array(x):
					return x.astype(jax.dtypes.bfloat16)
				else:
					return x	
			self.func = jax.tree_util.tree_map(to_bfloat16, F)
		else:
			self.func = F

	def __call__(self, ts, y0):
		solution = diffrax.diffeqsolve(diffrax.ODETerm(self.func),
									   self.SOLVER,
									   t0=ts[0],t1=ts[-1],
									   dt0=self.dt0,
									   y0=y0,
									   max_steps=1000*ts.shape[0],
									   #stepsize_controller=diffrax.ConstantStepSize(),
									   #stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
									   stepsize_controller=self.stepsize_controller,
									   saveat=diffrax.SaveAt(ts=ts))
		return solution.ts,solution.ys
	
	def partition(self):
		func_diff,func_static = self.func.partition()
		where = lambda s:s.func
		total_diff,total_static = eqx.partition(self,eqx.is_array)
		total_diff = eqx.tree_at(where,total_diff,func_diff)
		total_static=eqx.tree_at(where,total_static,func_static)
		return total_diff,total_static
		
def load(path):
	with open(path, "rb") as f:
		hyperparams = json.loads(f.readline().decode())
		print("PDE hyperparameters: ")
		print(json.dumps(hyperparams["pde"],sort_keys=True, indent=4))
		print("Solver hyperparameters: ")
		print(json.dumps(hyperparams["solver"],sort_keys=True, indent=4))
		func = F(key=jr.PRNGKey(0), **hyperparams["pde"])
		pde = PDE_solver(func,**hyperparams["solver"])
		return eqx.tree_deserialise_leaves(f, pde)
	#def set_weights(self,weights):
	#	self.func.set_weights(weights)
def save(filename,hyperparams,model):
	filename = filename+".eqx"
	with open(filename, "wb") as f:
		hyperparam_str = json.dumps(hyperparams)
		f.write((hyperparam_str + "\n").encode())
		eqx.tree_serialise_leaves(f, model)
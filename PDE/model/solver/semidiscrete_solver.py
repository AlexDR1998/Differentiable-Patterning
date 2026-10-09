import jax
import equinox as eqx
import diffrax

SOLVERS = {
	"heun": diffrax.Heun,
	"euler": diffrax.Euler,
	"tsit5": diffrax.Tsit5,
	"kencarp3": diffrax.KenCarp3,
	"dopri5": diffrax.Dopri5,
	"kvaerno3": diffrax.Kvaerno3,
	"dopri8": diffrax.Dopri8,
}


class PDE_solver(eqx.Module):
	"""Method-of-lines solver: integrates dX/dt = F(t, X, args) with diffrax.

	Calling ``PDE_solver(F)(ts, y0)`` returns ``(ts, ys)`` where ``ys`` holds the
	state at every time in ``ts`` (including ``ts[0]``), stacked on axis 0.
	"""
	func: eqx.Module
	SOLVER: diffrax.AbstractSolver
	stepsize_controller: diffrax.AbstractStepSizeController
	dt0: float
	max_steps: int = eqx.field(static=True)

	def __init__(self,F,dt=0.1,SOLVER="heun",ADAPTIVE=False,DTYPE="float32",rtol=1e-3,atol=1e-3,max_steps=100_000):
		if SOLVER not in SOLVERS:
			raise ValueError(f"Unknown solver {SOLVER!r}; choose from {sorted(SOLVERS)}")
		self.SOLVER = SOLVERS[SOLVER]()
		self.dt0 = dt
		self.max_steps = max_steps

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
									   max_steps=self.max_steps,
									   stepsize_controller=self.stepsize_controller,
									   saveat=diffrax.SaveAt(ts=ts))
		return solution.ts,solution.ys

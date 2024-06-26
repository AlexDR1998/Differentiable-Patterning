import optax
import jax
import equinox as eqx


def non_negative_diffusion(schedule):
	#schedule = optax.exponential_decay(1e-3, transition_steps=iters, decay_rate=0.99)
	opt_ra = optax.nadamw(schedule) # Adam with weight decay for reaction and advection
	opt_d = optax.chain(optax.keep_params_nonnegative(),optax.nadam(schedule)) # Non-negative adam on diffusive terms (no weight decay)
	
	def label_diffusive(tree):
		# Returns True for the diffusion terms that should remain non-negative
		
		filter_spec = jax.tree_util.tree_map(lambda _:False,tree)
		filter_spec = eqx.tree_at(lambda t:t.func.f_d.layers[-1].weight,filter_spec,replace=True)
		return filter_spec
	
	def label_not_diffusive(tree):
		# Returns True for the parameters that are NOT the non-negative diffusive terms
		filter_spec = jax.tree_util.tree_map(lambda _:True,tree)
		filter_spec = eqx.tree_at(lambda t:t.func.f_d.layers[-1].weight,filter_spec,replace=False)
		return filter_spec
	
	opt_ra = optax.masked(opt_ra,label_not_diffusive)
	opt_d = optax.masked(opt_d,label_diffusive)
	
	
	return optax.chain(opt_d,opt_ra)


def non_negative_diffusion_chemotaxis(schedule,optimiser=optax.nadamw):
	
	opt_ra = optax.chain(optax.scale_by_param_block_norm(),
					  	 optimiser(schedule)) # Adam with weight decay for reaction and advection
	opt_d = optax.chain(optax.keep_params_nonnegative(),
					    optax.scale_by_param_block_norm(),
						optax.nadam(schedule)) # Non-negative adam on diffusive terms (no weight decay)
	
	def label_diffusive(tree):
		# Returns True for the diffusion terms that should remain non-negative
		
		filter_spec = jax.tree_util.tree_map(lambda _:False,tree)
		filter_spec = eqx.tree_at(lambda t:t.func.signal_diffusion.diffusion_constants.weight,filter_spec,replace=True)
		return filter_spec
	
	def label_not_diffusive(tree):
		# Returns True for the parameters that are NOT the non-negative diffusive terms
		filter_spec = jax.tree_util.tree_map(lambda _:True,tree)
		filter_spec = eqx.tree_at(lambda t:t.func.signal_diffusion.diffusion_constants.weight,filter_spec,replace=False)
		return filter_spec
	
	opt_ra = optax.masked(opt_ra,label_not_diffusive)
	opt_d = optax.masked(opt_d,label_diffusive)
	
	
	return optax.chain(opt_d,opt_ra)
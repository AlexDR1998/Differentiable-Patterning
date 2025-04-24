import optax
import jax 
import jax.numpy as jnp
import equinox as eqx





def unitary_decoder_transform(norm=1.0,eps=1e-8,axis=1):
    """
    A projection function that projects the decoder weights to a unit norm.
    This is useful for ensuring that the decoder weights do not shrink the outputs of the latent space to minimise the loss.
    Args:
        norm: The norm to project to. Default is 1.0.
        eps: A small value to avoid division by zero. Default is 1e-8.
        axis: The axis to project along. Default is 1.
    Returns:
        An optax.GradientTransformation that applies the projection to the decoder weights.
    """

    def init_fn(params):
        return optax.EmptyState()
    def update_fn(updates, state, params=None, **kwargs):
        # Apply the updates to the parameters
        if params is None:
            raise ValueError("params must be provided to unit_norm_transform for projection.")
        
        def project(p,apply_projection):
            if apply_projection:
                pnorm = jnp.linalg.norm(p,axis=axis,keepdims=True) + eps
                p = p/pnorm * norm
            return p
        
        def label_decoder(tree):
            # Returns True for the decoder terms
            filter_spec = jax.tree_util.tree_map(lambda _:False,tree)
            filter_spec = eqx.tree_at(lambda t:t.decoder.weight,filter_spec,replace=True)
            return filter_spec

        
        new_params = optax.apply_updates(params, updates)
        mask = label_decoder(new_params)
        projected_params = jax.tree.map(project, new_params, mask)
        new_updates = jax.tree.map(lambda proj, orig: proj - orig, projected_params, params)

        return new_updates, state
    
    return optax.GradientTransformation(init_fn, update_fn)





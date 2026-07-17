import jax
import jax.numpy as jnp


# Checks which hardware backend is being used
print(jax.default_backend())
print(jax.devices())
key = jax.random.PRNGKey(0)

# Do some linear algebra
A = jax.random.uniform(key,(4,500,500))
key = jax.random.fold_in(key,1)
B = jax.random.uniform(key,(4,500,500))
print(jnp.einsum("aij,bij->ab",A,B))



import jax
import jax.numpy as jnp
import einops

a = jnp.linspace(0,1,100)

AA = einops.repeat(a,"h -> h w", w=100)
AAT= einops.repeat(a,"w -> h w", h=100)
print(AA)
print(AAT)
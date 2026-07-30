import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
print(jax.default_backend())
print(jax.devices())
key = jax.random.PRNGKey(0)

A = jax.random.uniform(key,(4,500,500))
key = jax.random.fold_in(key,1)
B = jax.random.uniform(key,(4,500,500))

def matmul(a,b):
	return jnp.einsum("ij,jk->ik",a,b)

vmat = jax.vmap(matmul)
pmat = jax.pmap(matmul)

shard = jax.sharding.PositionalSharding(mesh_utils.create_device_mesh((4,1,1)))
A_dev,B_dev = jax.device_put((A,B),shard)
jax.debug.visualize_array_sharding(A_dev[0])
jax.debug.visualize_array_sharding(A_dev[:,0])

#shard = jax.sharding.PmapSharding(jax.devices(),A.shape,sharded_dim=0)
#print(shard)
#A_dev,B_dev = jax.device_put((A,B),shard)

C_dev = pmat(A,B)
C = vmat(A,B)
print(C)
print(C_dev)

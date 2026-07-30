import jax
import jax.numpy as jnp
import time

B = 10
S = 16
def matmuladd(a,b,c):
	return a@b + c
key0 = jax.random.PRNGKey(int(time.time()))
key1,key2 = jax.random.split(key0,2)
a = list(jax.random.uniform(key0,shape=(B,S,S)))
b = list(jax.random.uniform(key1,shape=(B,S,S)))
c = list(jax.random.uniform(key2,shape=(B,S,S)))




#print(a)
D = jax.tree_util.tree_map(matmuladd,a,b,c)
D0 = matmuladd(a[0],b[0],c[0])
print(D[0])
print(D0)
print(jnp.sum(D[0]-D0))
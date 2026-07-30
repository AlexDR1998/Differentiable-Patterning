import jax 


a = jax.numpy.zeros((32,100,100))

print(jax.nn.glu(a,axis=0).shape)
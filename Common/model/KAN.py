import jax
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array,Float,Key
import time
from einops import einsum,reduce,repeat
### Equinox module that applies Kolmogorov Arnold network as a 1*1 convolutional network


class nKAN(eqx.Module):
    in_features: int
    out_features: int
    INTRINSIC_HIDDEN_LAYERS: int
    weight: Array
    bias: Array
    activation: callable

    def __init__(self,
                 in_features,
                 out_features,
                 INTRINSIC_HIDDEN_LAYERS,
                 scale,
                 use_bias=True,
                 activation= jax.nn.relu,
                 key=None):
        ### Rather than using B-splines, each learnable edge activation is itself a very small 1->H->H->1 neural network with hidden layer size H as a hyperparameter
        if key is None:
            key = jax.random.PRNGKey(int(time.time()))
        key1,key2,key3,key4 = jax.random.split(key,4)
        self.weight = scale*jax.random.normal(key=key1,shape=(in_features,out_features,INTRINSIC_HIDDEN_LAYERS,2+INTRINSIC_HIDDEN_LAYERS))#/(1.0*in_features)
        
        if use_bias:
            self.bias = [
                 scale*jax.random.normal(key=key2,shape=(in_features,out_features,1)),
                 scale*jax.random.normal(key=key3,shape=(in_features,out_features,1)),
                 scale*jax.random.normal(key=key4,shape=(out_features,))
                ]
        else:
            self.bias = jnp.zeros(3)

        self.INTRINSIC_HIDDEN_LAYERS = INTRINSIC_HIDDEN_LAYERS
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        

    
    def __call__(self,x: Float[Array, "{self.in_features}"],key=None)->Float[Array, "{self.out_features}"]:
        h1 = einsum(x,self.weight[...,0],"I , I O hidden -> I O hidden") + self.bias[0]
        h1 = self.activation(h1)
        h2 = einsum(h1,self.weight[...,1:-1],"I O hidden_in, I O hidden_in hidden_out -> I O hidden_out") + self.bias[1]
        h2 = self.activation(h2)
        return einsum(h2,self.weight[...,-1],"I O hidden, I O hidden -> O") + self.bias[2]# / (1.0*self.INTRINSIC_HIDDEN_LAYERS)

        #return reduce(h3,"I O hidden -> O", "mean")
    


class funcKAN(eqx.Module):
    in_features: int
    out_features: int
    ORDER: int
    weight: Array

    def __init__(self,
                 in_features,
                 out_features,
                 ORDER,
                 scale,
                 key=None):
        """ General superclass for KANs with a function of type f:R x R^ORDER -> R to each edge.
            Subclass into using polynomials or rbfs 

        Args:
            in_features (_type_): _description_
            out_features (_type_): _description_
            ORDER (_type_): _description_
            scale (_type_): _description_
            use_bias (bool, optional): _description_. Defaults to True.
            activation (_type_, optional): _description_. Defaults to jax.nn.relu.
            key (_type_, optional): _description_. Defaults to jax.random.PRNGKey(int(time.time())).
        """
        if key is None:
            key = jax.random.PRNGKey(int(time.time()))
        self.weight = scale*jax.random.normal(key=key,shape=(in_features,out_features,ORDER))#/(1.0*in_features)
        self.ORDER = ORDER
        self.in_features = in_features
        self.out_features = out_features

    def inner_func(self,x: Float[Array, ""],coeffs: Float[Array, "{self.ORDER}"])->Float[Array, ""]:

            return jnp.mean(coeffs)*x
    
    def __call__(self,x:Float[Array,"{self.in_features}"],key=None)->Float[Array, "{self.out_features}"]:
        h1 = repeat(x,"I -> I O",O=self.out_features)
        
        #in_func = eqx.filter_jit(lambda x,coeffs:self.inner_func(x,coeffs))
        vfunc = jax.vmap(self.inner_func,in_axes=(0,0),out_axes=0)
        vvfunc = jax.vmap(vfunc,in_axes=(0,0),out_axes=0)

        h2 = vvfunc(h1,self.weight)
        return reduce(h2,"I O -> O","sum")
    

    def set_weights(self,weight):
        w_where = lambda m:m.weight
        return eqx.tree_at(w_where,self,weight)
        #self.weight = weight

class polyKAN(funcKAN):
    in_features: int
    out_features: int
    ORDER: int
    weight: Array

    def inner_func(self,x: Float[Array, ""],coeffs: Float[Array, "{self.ORDER}"])->Float[Array, ""]:
        output = 0
        for i in range(self.ORDER+1):
            output += coeffs[i]*x**i
        return output



class legKAN(funcKAN):
    in_features: int
    out_features: int
    ORDER: int
    weight: Array
    
    def inner_func(self,x: Float[Array, ""],coeffs: Float[Array, "{self.ORDER}"])->Float[Array, ""]:
        legendre = [
             lambda x: 1,
             lambda x: x,
             lambda x: (3*x**2-1)/2.0,
             lambda x: (5*x**3-3*x)/2.0,
             lambda x: (32*x**4-30*x**2+3)/8.0,
             lambda x: (63*x**5-70*x**3+15*x)/8.0,
             lambda x: (213*x**6-315*x**4+105*x**2-5)/16.0,
             lambda x: (429*x**7-693*x**5+315*x**3-35*x)/16.0
        ]
        output = 0
        for i in range(self.ORDER+1):
            output += coeffs[i]*legendre[i](x)
        return output
    


class gaussKAN(funcKAN):
    in_features: int
    out_features: int
    ORDER: int
    weight: Array
    bounds: float
    width: float
    basis_grid: callable
    def __init__(self,
                 in_features,
                 out_features,
                 ORDER,
                 scale,
                 bounds,
                 width,
                 key=None):
        #super().__init__(in_features,out_features,ORDER,scale,key)
        if key is None:
            key = jax.random.PRNGKey(int(time.time()))
        key1,key2 = jax.random.split(key,2)
        self.weight = scale*jax.random.normal(key=key1,shape=(in_features,out_features,ORDER))
            
        self.ORDER = ORDER
        self.in_features = in_features
        self.out_features = out_features
        self.bounds = bounds
        self.width = width
        def basis_grid():
            def gaussian(x,m):
                return jnp.exp(-0.5*((x-m)*float(self.ORDER)/self.width)**2)*(self.ORDER)/(self.width*jnp.sqrt(2*jnp.pi))
            ms = jnp.linspace(-self.bounds,self.bounds,self.ORDER)
            v_gaussian = jax.vmap(gaussian,in_axes=(None,0),out_axes=0)

            return eqx.filter_jit(lambda x: v_gaussian(x,ms))
        self.basis_grid = basis_grid()
    
    
    
    def inner_func(self,x: Float[Array, ""],coeffs: Float[Array, "{self.ORDER}"])->Float[Array, ""]:
        
        return jnp.dot(self.basis_grid(x),coeffs)
        




    # def __call__(self,x:Float[Array,"{self.in_features}"],key=None)->Float[Array, "{self.out_features}"]:
    #     h1 = repeat(x,"I -> I O",O=self.out_features)
        
    #     in_func = eqx.filter_jit(lambda x,coeffs:self.inner_func(x,coeffs))
    #     vfunc = jax.vmap(in_func,in_axes=(0,0),out_axes=0)
    #     vvfunc = jax.vmap(vfunc,in_axes=(0,0),out_axes=0)

    #     h2 = vvfunc(h1,self.weight[0])
    #     #h3 = self.weight[1]*(jax.nn.silu(h1)+h2)
    #     return reduce(h2,"I O -> O","sum")

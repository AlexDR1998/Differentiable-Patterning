import jax
import equinox as eqx
import jax.numpy as jnp
from einops import rearrange,reduce,einsum
from jaxtyping import Array, Float


class Ops(eqx.Module):
    
    
    PADDING: str
    dx: float
    grad_x: eqx.Module
    grad_y: eqx.Module
    laplacian: eqx.Module
    average: eqx.Module
    #partial_laplacian_inverse: eqx.Module
    def __init__(self,PADDING,dx,KERNEL_SCALE=1,SMOOTHING=1):
        """
        Equinox module for computing finite difference approximation of gradient, divergence, curl or laplacian of 
        scalar or vector fields
        Partition and combine are properly defined such that this isn't changed during training.
            div and curl: f32[2,C, x, y] -> f32[C, x, y]
            grad        : f32[C, x, y] -> f32[2, C, x, y]
            lap         : f32[C, x, y] -> f32[C, x, y]

            Args:
                PADDING: str
                    'ZEROS'
                    'REFLECT'
                    'REPLICATE'
                    'CIRCULAR'
                dx: float
                    step size
            
        """
        key = jax.random.PRNGKey(0) # Dummy key for conv2d init
        self.PADDING = PADDING
        self.dx = dx
        nstds = 2
        self.grad_x = eqx.nn.Conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=2*KERNEL_SCALE+1,
                                    use_bias=False,
                                    key=key,
                                    padding="SAME",
                                    padding_mode=PADDING,
                                    groups=1)
        self.grad_y = eqx.nn.Conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=2*KERNEL_SCALE+1,
                                    use_bias=False,
                                    key=key,
                                    padding="SAME",
                                    padding_mode=PADDING,
                                    groups=1)
        self.laplacian = eqx.nn.Conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=2*KERNEL_SCALE+1,
                                    use_bias=False,
                                    key=key,
                                    padding="SAME",
                                    padding_mode=PADDING,
                                    groups=1)
        self.average = eqx.nn.Conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=2*KERNEL_SCALE+1,
                                    use_bias=False,
                                    key=key,
                                    padding="SAME",
                                    padding_mode=PADDING,
                                    groups=1)
        
        # self.partial_laplacian_inverse = eqx.nn.Conv2d(in_channels=1,
        #                                                 out_channels=1,
        #                                                 kernel_size=2*KERNEL_SCALE+1,
        #                                                 use_bias=False,
        #                                                 key=key,
        #                                                 padding="SAME",
        #                                                 padding_mode=PADDING,
        #                                                 groups=1)

        def gabor(sigma, theta, Lambda, psi, gamma,nstds):
            # General odd gabor filter 
            sigma_x = sigma
            sigma_y = float(sigma) / gamma

            # Bounding box
              # Number of standard deviation sigma
            xmax = max(
                abs(nstds * sigma_x * jnp.cos(theta)), abs(nstds * sigma_y * jnp.sin(theta))
            )
            xmax = jnp.ceil(max(1, xmax))
            ymax = max(
                abs(nstds * sigma_x * jnp.sin(theta)), abs(nstds * sigma_y * jnp.cos(theta))
            )
            ymax = jnp.ceil(max(1, ymax))
            xmin = -xmax
            ymin = -ymax
            (y, x) = jnp.meshgrid(jnp.arange(ymin, ymax + 1), jnp.arange(xmin, xmax + 1))

            # Rotation
            x_theta = x * jnp.cos(theta) + y * jnp.sin(theta)
            y_theta = -x * jnp.sin(theta) + y * jnp.cos(theta)

            gb = jnp.exp(
                -0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)
            ) * jnp.sin(2 * jnp.pi / Lambda * x_theta + psi) 
            return gb / jnp.sum(jnp.abs(gb))


        def gradx(scale,nstds,SMOOTHING):
            # Gabor filter with large wavelength, x direction
            if scale==1:
                if SMOOTHING == 0:
                    return jnp.outer(jnp.array([0,1.0,0]),jnp.array([-1.0,0.0,1.0]))/2.0
                elif SMOOTHING == 1:
                    return jnp.outer(jnp.array([1.0,2.0,1.0]),jnp.array([-1.0,0.0,1.0]))/8.0
            else:
                sigma = scale*0.5

                # Bounding box
                # Number of standard deviation sigma
                xmax = nstds * sigma
                #max(
                #    abs( )
                #)
                xmax = jnp.ceil(max(1, xmax))
                ymax = nstds * sigma
                #max(
                #    abs(nstds * sigma_x * np.sin(theta)), abs( * np.cos(theta))
                #)
                ymax = jnp.ceil(max(1, ymax))
                xmin = -xmax
                ymin = -ymax
                (y, x) = jnp.meshgrid(jnp.arange(ymin, ymax + 1), jnp.arange(xmin, xmax + 1))


                gb = -(x/sigma**2)*jnp.exp(
                    -0.5 * (x**2 / sigma**2 + y**2 / sigma**2)
                )
                return gb / jnp.sum(jnp.abs(gb))

        def gaussian(sigma,nstds):
            # Gaussian
            sigma_x = sigma*0.5
            

            # Bounding box
             # Number of standard deviation sigma
            xmax = nstds * sigma_x
            #max(
            #    abs( )
            #)
            xmax = jnp.ceil(max(1, xmax))
            ymax = nstds * sigma_x
            
            ymax = jnp.ceil(max(1, ymax))
            xmin = -xmax
            ymin = -ymax
            (y, x) = jnp.meshgrid(jnp.arange(ymin, ymax + 1), jnp.arange(xmin, xmax + 1))


            gb = jnp.exp(
                -0.5 * (x**2 / sigma_x**2 + y**2 / sigma_x**2)
            )
            return gb / jnp.sum(gb)

        def laplacian(sigma,nstds,SMOOTHING):
            # Laplacian of gaussian
            if sigma==1:
                if SMOOTHING == 0:
                    lap = jnp.array([[0.0,1.0,0.0],
                                     [1.0,-4.0,1.0],
                                     [0.0,1.0,0.0]])
                elif SMOOTHING == 1:
                    lap = jnp.array([[0.25,0.5,0.25],
                                    [0.5,-3,0.5],
                                    [0.25,0.5,0.25]])
                # lap = jnp.array([[0.0,1.0,0.0],
                #                  [1.0,-4.0,1.0],
                #                  [0.0,1.0,0.0]])
            else:
                sigma_x = sigma*0.5
                
                # Bounding box
                # Number of standard deviation sigma
                xmax = nstds * sigma_x
                #max(
                #    abs( )
                #)
                xmax = jnp.ceil(max(1, xmax))
                ymax = nstds * sigma_x
                
                ymax = jnp.ceil(max(1, ymax))
                xmin = -xmax
                ymin = -ymax
                (y, x) = jnp.meshgrid(jnp.arange(ymin, ymax + 1), jnp.arange(xmin, xmax + 1))


                gb = jnp.exp(
                    -0.5 * (x**2 / sigma_x**2 + y**2 / sigma_x**2)
                )
                lgb = (1-4*(x**2+y**2)/(sigma_x**2))*gb
                
                lap = -lgb
                lap = lap - jnp.mean(lap)
                lap = lap/jnp.sum(jnp.abs(lap))
            
            return lap
        


        def partial_laplacian_inverse(SMOOTHING):
            # Used for jacobi relaxation for inverting poisson equation
            if SMOOTHING == 0:
                return jnp.array([[0.0,1.0,0.0],
                                [1.0,0,1.0],
                                [0.0,1.0,0.0]])
            elif SMOOTHING == 1:
                return jnp.array([[0.25,0.5,0.25],
                        [0.5,0,0.5],
                        [0.25,0.5,0.25]])
    
        _lap = laplacian(KERNEL_SCALE,nstds=nstds,SMOOTHING=SMOOTHING) / (dx*dx)
        _grad_x = gradx(KERNEL_SCALE,nstds=nstds,SMOOTHING=SMOOTHING) / dx
        _grad_y = _grad_x.T
        _av = gaussian(KERNEL_SCALE,nstds=nstds)
        _lap_inv = partial_laplacian_inverse(SMOOTHING=SMOOTHING)
        
        kernel_dx = rearrange(_grad_x,"x y -> () () x y")
        kernel_dy = rearrange(_grad_y,"x y -> () () x y")
        kernel_lap= rearrange(_lap,"x y -> () () x y")
        kernel_av= rearrange(_av,"x y -> () () x y")
        kernel_lap_inv = rearrange(_lap_inv,"x y -> () () x y") 
        w_where = lambda l: l.weight
        self.grad_x = eqx.tree_at(w_where,self.grad_x,kernel_dx)
        self.grad_y = eqx.tree_at(w_where,self.grad_y,kernel_dy)
        self.laplacian = eqx.tree_at(w_where,self.laplacian,kernel_lap)
        self.average = eqx.tree_at(w_where,self.average,kernel_av)
        #self.partial_laplacian_inverse = eqx.tree_at(w_where,self.partial_laplacian_inverse,kernel_lap_inv)


    # @eqx.filter_jit
    # def LapInv(self,X: Float[Array,"C x y"])->Float[Array, "C x y"]:
    #     v_lap_inv = jax.vmap(self.partial_laplacian_inverse,in_axes=0,out_axes=0,axis_name="CHANNELS")
    #     X = rearrange(X,"C x y -> C () x y")
    #     return v_lap_inv(X)[:,0]



    @eqx.filter_jit
    def Grad(self,X: Float[Array,"C x y"])->Float[Array, "dim C x y"]:
        v_grad_x = jax.vmap(self.grad_x,in_axes=0,out_axes=0,axis_name="CHANNELS")
        v_grad_y = jax.vmap(self.grad_y,in_axes=0,out_axes=0,axis_name="CHANNELS")
        X = rearrange(X,"C x y -> C () x y")
        gx = v_grad_x(X)[:,0]
        gy = v_grad_y(X)[:,0]
        #gx = reduce(gx,"C () x y -> C x y", "min")
        #gy = reduce(gy,"C () x y -> C x y", "min")
        return rearrange([gx,gy],"dim k x y -> dim k x y")
    

    @eqx.filter_jit
    def VecGrad(self,X: Float[Array,"dim C x y"])->Float[Array, "dim dim C x y"]:
        """
        Gradient of vector field - returns jacobian tensor. ASSUMES DIM=2
        """
        grad_x = self.Grad(X[0])
        grad_y = self.Grad(X[1])
        return rearrange([grad_x,grad_y],"d1 d2 k x y -> d1 d2 k x y")
    
    @eqx.filter_jit
    def GradNorm(self,X: Float[Array,"C x y"])->Float[Array, "C x y"]:
        v_grad_x = jax.vmap(self.grad_x,in_axes=0,out_axes=0,axis_name="CHANNELS")
        v_grad_y = jax.vmap(self.grad_y,in_axes=0,out_axes=0,axis_name="CHANNELS")
        X = rearrange(X,"C x y -> C () x y")
        gx = v_grad_x(X)[:,0]
        gy = v_grad_y(X)[:,0]
        return jnp.sqrt(gx**2+gy**2)
    
    @eqx.filter_jit
    def Div(self,X: Float[Array,"dim C x y"])->Float[Array, "C x y"]:
        v_grad_x = jax.vmap(self.grad_x,in_axes=0,out_axes=0,axis_name="CHANNELS")
        v_grad_y = jax.vmap(self.grad_y,in_axes=0,out_axes=0,axis_name="CHANNELS")
        Xx = rearrange(X[0],"C x y -> C () x y")
        Xy = rearrange(X[1],"C x y -> C () x y")
        gx = v_grad_x(Xx)[:,0]
        gy = v_grad_y(Xy)[:,0]
        return gx + gy
    @eqx.filter_jit
    def Lap(self,X: Float[Array,"C x y"])->Float[Array, "C x y"]:
        v_lap = jax.vmap(self.laplacian,in_axes=0,out_axes=0,axis_name="CHANNELS")
        X = rearrange(X,"C x y -> C () x y")
        return v_lap(X)[:,0]

    @eqx.filter_jit
    def Curl(self,X: Float[Array,"dim C x y"])->Float[Array, "C x y"]:
        v_grad_x = jax.vmap(self.grad_x,in_axes=0,out_axes=0,axis_name="CHANNELS")
        v_grad_y = jax.vmap(self.grad_y,in_axes=0,out_axes=0,axis_name="CHANNELS")
        Xx = rearrange(X[0],"C x y -> C () x y")
        Xy = rearrange(X[1],"C x y -> C () x y")
        gx = v_grad_x(Xy)[:,0]
        gy = v_grad_y(Xx)[:,0]
        
        return gx - gy
    
    @eqx.filter_jit
    def Average(self,X:Float [Array, "C x y"])->Float[Array,"C x y"]:
        v_av = jax.vmap(self.average,in_axes=0,out_axes=0,axis_name="CHANNELS")
        X = rearrange(X,"C x y -> C () x y")
        return v_av(X)[:,0]
    
    @eqx.filter_jit
    def NonlinearDiffusion(self,f: Float[Array,"C x y"],g: Float[Array, "C x y"])->Float[Array, "C x y"]:
        """ Computes the anisotropic/nonlinear diffusion: Div(f(x) Grad g(x)) in a stable way. 
        If f is constant, reduces to: f Laplacian g
        """
        grad_f = self.Grad(f)
        grad_g = self.Grad(g)
        lap_g = self.Lap(g)
        dot_grad = einsum(grad_f,grad_g,"dim C x y, dim C x y -> C x y")

        
        return dot_grad + f*lap_g


    @eqx.filter_jit
    def VectorLaplacian(self,X: Float[Array,"dim C x y"])->Float[Array, "dim C x y"]:
        """
        Computes the laplacian of a vector field - in cartesian coordinates it reduces to a vectorised Lap
        """
        lX = self.Lap(X[0])
        lY = self.Lap(X[1])
        return rearrange([lX,lY],"dim k x y -> dim k x y")
    

    @eqx.filter_jit
    def VectorMatDiff(self,
               X: Float[Array,"dim C x y"],
               Y: Float[Array,"dim C x y"])->Float[Array, "dim C x y"]:
        """
        Compute the spacial part of a material derivative - (X dot Grad) Y
        Where Y is a vector field 
        
        (X dot Grad) Y = X dot (Grad Y)

        """
        #TODO: check the axes are correct here  
        grad_y = self.VecGrad(Y)
        return einsum(X,grad_y,"S C x y, dim S C x y -> dim C x y")

    @eqx.filter_jit
    def MatDiff(self,X: Float[Array,"dim C x y"],Y: Float[Array,"C x y"])->Float[Array, "C x y"]:
        """
        Compute the spacial part of a material derivative - (X dot Grad) Y
        Where Y is a scalar field 
        
        (X dot Grad) Y = X dot (Grad Y)
        """
        grad_y = self.Grad(Y)
        return einsum(X,grad_y,"dim C x y, dim C x y -> C x y")
        


        

    def partition(self):
        where_x = lambda m: m.grad_x.weight
        where_y = lambda m: m.grad_y.weight
        where_l = lambda m: m.laplacian.weight
        where_av = lambda m: m.average.weight
        #where_lap_inv = lambda m: m.partial_laplacian_inverse.weight
        sobel_x = self.grad_x.weight
        sobel_y = self.grad_y.weight
        l_weight = self.laplacian.weight
        av_weight= self.average.weight
        #lap_inv_weight = self.partial_laplacian_inverse.weight
        diff,static = eqx.partition(self,eqx.is_array)
        diff = eqx.tree_at(where_x,diff,None)
        diff = eqx.tree_at(where_y,diff,None)
        diff = eqx.tree_at(where_l,diff,None)
        diff = eqx.tree_at(where_av,diff,None)
        #diff = eqx.tree_at(where_lap_inv,diff,None)
        static = eqx.tree_at(where_x,static,sobel_x,is_leaf=lambda x: x is None)
        static = eqx.tree_at(where_y,static,sobel_y,is_leaf=lambda x: x is None)
        static = eqx.tree_at(where_l,static,l_weight,is_leaf=lambda x: x is None)
        static = eqx.tree_at(where_av,static,av_weight,is_leaf=lambda x: x is None)
        #static = eqx.tree_at(where_lap_inv,static,lap_inv_weight,is_leaf=lambda x: x is None)
        return diff, static
    
    def combine(self,diff,static):
        self = eqx.combine(diff,static)
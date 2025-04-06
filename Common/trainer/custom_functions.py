import jax
import jax.numpy as np
import equinox as eqx
import jax.random as jr


def check_training_diverged(mean_loss,x,step,loss_thresh=1e16):
    error = 0
    if np.isnan(mean_loss):
        error = 1
        print("|-|-|-|-|-|-  Loss reached NaN at step "+str(step)+" -|-|-|-|-|-|")
    elif any(list(map(lambda x: np.any(np.isnan(x)), x))):
        error = 2
        print("|-|-|-|-|-|-  X reached NaN at step "+str(step)+" -|-|-|-|-|-|")
    elif mean_loss > loss_thresh:
        error = 3
        print( "|-|-|-|-|-|-  Loss exceded "+str(loss_thresh)+" at step "+str(step)+", optimisation probably diverging  -|-|-|-|-|-|")
    return error

			



def perlin(size,cutoff,key):
    # Generates multi-frequency smooth-ish perlin noise
    @jax.jit
    def _perlin(x, y, key):
        # permutation table
        #np.random.seed(seed)
        p = jr.permutation(key,size)
        #p = np.arange(256, dtype=int)
        #np.random.shuffle(p)
        p = np.stack([p, p]).flatten()
        # coordinates of the top-left
        xi, yi = x.astype(int), y.astype(int)
        # internal coordinates
        xf, yf = x - xi, y - yi
        # fade factors
        u, v = fade(xf), fade(yf)
        # noise components
        n00 = gradient(p[p[xi] + yi], xf, yf)
        n01 = gradient(p[p[xi] + yi + 1], xf, yf - 1)
        n11 = gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
        n10 = gradient(p[p[xi + 1] + yi], xf - 1, yf)
        # combine noises
        x1 = lerp(n00, n10, u)
        x2 = lerp(n01, n11, u)  # FIX1: I was using n10 instead of n01
        return lerp(x1, x2, v)  # FIX2: I also had to reverse x1 and x2 here

    def lerp(a, b, x):
        "linear interpolation"
        return a + x * (b - a)

    def fade(t):
        "6t^5 - 15t^4 + 10t^3"
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    def gradient(h, x, y):
        "grad converts h to the right gradient vector and return the dot product with (x,y)"
        vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        g = vectors[h % 4]
        return g[:, :, 0] * x + g[:, :, 1] * y

    # EDIT : generating noise at multiple frequencies and adding them up
    p = np.zeros(size)
    
    for i in range(cutoff):
        key = jr.fold_in(key, i)
        freq = 2**i
        lin = np.linspace(0, freq, size, endpoint=False)
        #liny = np.linspace(0, fr
        # eq, shape[1], endpoint=False)
        x, y = np.meshgrid(lin, lin)  # FIX3: I thought I had to invert x and y here but it was a mistake
        p = _perlin(x, y, key) / freq + p
    #p = p - np.mean(p)
    p = 2*(p - np.min(p)) / (np.max(p) - np.min(p)) -1
    return p



def multi_channel_perlin_noise(size,channels,cutoff,key):
    v_perlin = jax.vmap(perlin,in_axes=(None,None,0),out_axes=0)
    return v_perlin(size,cutoff,jr.split(key,channels))




def multi_batch_perlin_noise(X,cutoff,key):
    batch_size = X.shape[0]
    size = X.shape[-1]
    channels = X.shape[2]
    output = []
    for i in range(batch_size):
        key = jr.fold_in(key, i)
        output.append(multi_channel_perlin_noise(size,channels,cutoff,key))
    return np.array(output)
    
    #v_perlin = jax.vmap(multi_batch_perlin_noise,in_axes=(None,None,None,0),out_axes=0)
    #return v_perlin(size,channels,cutoff,jr.split(key,channels))
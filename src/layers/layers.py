import jax
from flax import linen as nn
import jax.numpy as jnp
import jax.random as jr
    
class ParallelDense(nn.Module):
    num_units: int  # Number of units in each dense layer
    num_par: int  # Number of parallel transformations

    @nn.compact
    def __call__(self, inputs):
        feature_in = inputs.shape[-1]
        # Initialize the weight matrix with shape (block_size, feature_in, num_units)
        kernel = self.param('kernel',
                            jax.nn.initializers.lecun_normal(),
                            (self.num_par, feature_in, self.num_units))

        # Initialize the bias vector with shape (block_size, num_units)
        bias = self.param('bias',
                          jax.nn.initializers.zeros,
                          (self.num_par, self.num_units))

        # Perform the parallel dense transformation
        if inputs.shape[-2] == self.num_par or inputs.shape[-2] == 1:
            return jnp.sum(inputs[...,None] * kernel,-2) + bias
        else:
            raise ValueError(f"Input shape {inputs.shape} does broadcast to {self.block_size}")

class ParallelMLP(nn.Module):
    features: tuple
    block_size: int

    @nn.compact
    def __call__(self, inputs):        
        x = inputs[...,None,:]
        for i, feature in enumerate(self.features):
            x = ParallelDense(feature, self.block_size)(x)
            if i != len(self.features) - 1:
                x = nn.relu(x)
        return x


class ParallelRealNVPNode(nn.Module):
    num_par: int  # Number of parallel transformations
    mask: jnp.ndarray # Mask determines the dimensions that are NOT changed by the transform


    def setup(self):
        # Create a mask matrix for the parallel transformations
        mask_dims = jnp.where(self.mask)
        self.mask_mat = jnp.eye(self.mask.shape[-1])[mask_dims]

    @nn.compact
    def __call__(self, inputs):
        x, log_det_jac = inputs

        if not (x.shape[-2] == self.num_par or x.shape[-2] == 1):
            raise ValueError(f"Input shape {x.shape} does broadcast to (..., {self.num_par}, :)")

        y = x@self.mask_mat.T

        s_kernel = self.param('s_kernel',jax.nn.initializers.lecun_normal(),
                            (self.num_par, y.shape[-1]))
        s_bias = self.param('s_bias',jax.nn.initializers.zeros,
                          (self.num_par,))
        
        t_kernel = self.param('t_kernel',jax.nn.initializers.lecun_normal(),
                            (self.num_par, y.shape[-1]))
        t_bias = self.param('t_bias',jax.nn.initializers.zeros,
                          (self.num_par,))
        
        # Compute the scale and translation parameters
        s = jnp.sum(y * s_kernel,-1) + s_bias
        t = jnp.sum(y* t_kernel,-1) + t_bias
        
        x = x*(~self.mask) + (y*jnp.exp(s[...,None]) + t[...,None])@self.mask_mat

        log_det_jac += 0.0
        return x, log_det_jac

# self = ParallelRealNVPNode(4, jnp.array([False, True, True]))
# x = jnp.ones((5, 1, 3))
# params = self.init(jr.PRNGKey(0), (x, 0.0))
# self.apply(params, (x, 0.0))
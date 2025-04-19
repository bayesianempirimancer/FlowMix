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

class ParallelRealNVP(nn.Module):
    num_units: int
    num_parallel: int  # Number of parallel transformations
    non_linearity: function = nn.relu  # Non-linearity to apply after each dense layer


class FlowBlock(nn.Module):
    # goal of the flow block is to parallelize the computation of the normalizing flows associated with each object

    num_flows: int  # Number of independent normalizing flows

    @nn.compact
    def __call__(self, inputs):
        return None


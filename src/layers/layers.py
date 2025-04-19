from flax import linen as nn
import jax.numpy as jnp
    

class MLP(nn.Module):
    features: tuple

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for i, feature in enumerate(self.features):
            x = nn.Dense(feature)(x)
            if i != len(self.features) - 1:
                x = nn.relu(x)
        return x

from flax import linen as nn
import jax
import jax.numpy as jnp

class ParallelMLPBatch(nn.Module):
    num_units: int
    num_parallel: int  # Number of parallel transformations
    non_linearity: function = nn.relu  # Non-linearity to apply after each dense layer

    @nn.compact
    def __call__(self, inputs):
        feature_in = inputs.shape[-1]
        # Initialize the weight matrix with shape (num_parallel, feature_in, num_units)
        kernel = self.param('kernel',
                            jax.nn.initializers.lecun_normal(),
                            (self.num_parallel, feature_in, self.num_units))

        # Initialize the bias vector with shape (num_parallel, num_units)
        bias = self.param('bias',
                          jax.nn.initializers.zeros,
                          (self.num_parallel, self.num_units))

        # Perform the parallel dense transformation by expanding inputs to include the parallel dimension
        return self.non_linearity(jnp.expand_dims(inputs, axis=-2) @ kernel + bias)

class ParallelRealNVPBatch(nn.Module):


class FlowBlock(nn.Module):
    # goal of the flow block is to parallelize the computation of the normalizing flows associated with each object

    num_flows: int  # Number of independent normalizing flows

    @nn.compact
    def __call__(self, inputs):
        return None


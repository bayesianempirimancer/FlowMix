"""
Concat Squash Layer.

This layer concatenates multiple inputs and applies a squash operation,
which is commonly used in Capsule Networks. The squash operation normalizes
the vector length while preserving direction.

squash(v) = (||v||^2 / (1 + ||v||^2)) * (v / ||v||)

This ensures that:
- Short vectors get shrunk to near zero
- Long vectors get shrunk to unit length
- Direction is preserved
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence, Optional


class ConcatSquash(nn.Module):
    """
    Project each input separately, sum the projections, then apply squash operation.
    
    This avoids concatenation and instead:
    1. Applies a dense layer to each input separately
    2. Sums all projected inputs
    3. Applies squash normalization
    
    Args:
        output_dim: Output dimension after projection (required)
        use_bias: Whether to use bias in projection layers
        epsilon: Small value to prevent division by zero in squash
    """
    output_dim: int
    use_bias: bool = True
    epsilon: float = 1e-7
    
    @nn.compact
    def __call__(self, *inputs):
        """
        Project each input, sum, and squash.
        
        Args:
            *inputs: Variable number of input arrays (can have different shapes)
                    Note: Each input position gets its own projection parameters.
                    If you call with different numbers of inputs, you'll need to
                    reinitialize the layer.
            
        Returns:
            Squashed output of shape (..., output_dim)
        """
        if len(inputs) == 0:
            raise ValueError("At least one input is required")
        
        if self.use_bias:
            summed = nn.parameter(nn.initializers.zeros, (self.output_dim,))
        else:
            summed = jnp.zeros((self.output_dim,))

        for i, x in enumerate(inputs):
            summed = summed + nn.Dense(features=self.output_dim, use_bias=False, name=f"proj_{i}")(x)                

        return self.squash(summed)
    
    def squash(self, v: jnp.ndarray) -> jnp.ndarray:
        """
        Apply squash operation to vector(s).
        
        squash(v) = (||v||^2 / (1 + ||v||^2)) * (v / ||v||)
        
        Args:
            v: Input vector(s) of shape (..., D)
            
        Returns:
            Squashed vector(s) of same shape
        """
        # Compute squared norm along last axis
        v_norm_sq = jnp.sum(v ** 2, axis=-1, keepdims=True)  # (..., 1)
        squash_factor = v_norm_sq / (1.0 + v_norm_sq)  # (..., 1)        
        squash_factor = squash_factor / (jnp.sqrt(v_norm_sq) + self.epsilon)        

        return squash_factor * v




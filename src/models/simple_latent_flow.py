"""
Simple MLP-based flow for latent space.

This is a lightweight flow model specifically designed for transforming
latent vectors z, unlike the more complex CRNs designed for point clouds.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Callable
from src.encoders.embeddings import SinusoidalTimeEmbedding


def get_activation_fn(name: str) -> Callable:
    """Factory function to get activation function by name."""
    if name == 'swish':
        return nn.swish
    elif name == 'relu':
        return nn.relu
    elif name == 'gelu':
        return nn.gelu
    elif name == 'tanh':
        return nn.tanh
    else:
        raise ValueError(f"Unknown activation: {name}")


class SimpleLatentFlow(nn.Module):
    """
    Simple MLP-based flow for latent space.
    
    Takes latent vectors z and time t, outputs velocity field v.
    Much simpler than GlobalAdaLNMLPCRN - designed specifically for latent vectors.
    
    Args:
        hidden_dims: Hidden layer dimensions for MLP
        time_embed_dim: Dimension for time embedding
        activation_fn: Activation function name
    """
    hidden_dims: Sequence[int] = (128, 128, 128, 128)
    time_embed_dim: int = 128
    activation_fn: str = 'swish'
    
    @nn.compact
    def __call__(self, z: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        Compute velocity field for latent flow.
        
        Args:
            z: (B, D) or (B, 1, D) - latent vectors
            t: scalar or (B,) or (B, 1) - time
            
        Returns:
            v: Velocity field with same shape as z
        """
        # Handle different input shapes
        original_shape = z.shape
        if z.ndim == 3:
            B, N, D = z.shape
            z_flat = z.reshape(B, N * D)  # (B, N*D)
            needs_reshape = True
        else:
            B, D = z.shape
            z_flat = z  # (B, D)
            needs_reshape = False
        
        # Normalize time
        if t.ndim == 0:
            t = jnp.broadcast_to(t, (B,))
        elif t.ndim == 2:
            t = t[:, 0]  # (B, 1) -> (B,)
        
        # Time embedding
        time_embed = SinusoidalTimeEmbedding(self.time_embed_dim)(t)  # (B, time_embed_dim)
        
        # Concatenate z and time embedding
        h = jnp.concatenate([z_flat, time_embed], axis=-1)  # (B, D + time_embed_dim)
        
        # MLP layers
        activation = get_activation_fn(self.activation_fn)
        for dim in self.hidden_dims:
            h = nn.Dense(dim)(h)
            h = activation(h)
        
        # Output velocity (same dimension as input z_flat)
        v = nn.Dense(z_flat.shape[-1])(h)  # (B, D) or (B, N*D)
        
        # Reshape if needed
        if needs_reshape:
            v = v.reshape(original_shape)  # (B, N, D)
        
        return v


class AdaLNLatentFlow(nn.Module):
    """
    MLP-based flow for latent space with Adaptive Layer Normalization (AdaLN).
    
    Uses AdaLN to condition on time, similar to GlobalAdaLNMLPCRN but designed
    specifically for latent vectors (no spatial structure).
    
    The AdaLN mechanism allows time-dependent modulation of each layer's activations
    through learned scale and shift parameters derived from the time embedding.
    
    Args:
        hidden_dims: Hidden layer dimensions for MLP
        time_embed_dim: Dimension for time embedding (also used as conditioning dimension)
        activation_fn: Activation function name
    """
    hidden_dims: Sequence[int] = (128, 128, 128, 128)
    time_embed_dim: int = 128
    activation_fn: str = 'swish'
    
    @nn.compact
    def __call__(self, z: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        Compute velocity field for latent flow with time-dependent AdaLN.
        
        Args:
            z: (B, D) or (B, 1, D) - latent vectors
            t: scalar or (B,) or (B, 1) - time
            
        Returns:
            v: Velocity field with same shape as z
        """
        # Handle different input shapes
        original_shape = z.shape
        if z.ndim == 3:
            B, N, D = z.shape
            z_flat = z.reshape(B, N * D)  # (B, N*D)
            needs_reshape = True
        else:
            B, D = z.shape
            z_flat = z  # (B, D)
            needs_reshape = False
        
        # Normalize time
        if t.ndim == 0:
            t = jnp.broadcast_to(t, (B, 1))
        elif t.ndim == 1:
            t = t[:, None] if t.shape[0] == B else jnp.broadcast_to(t, (B, 1))
        
        # Time embedding (used directly as conditioning vector)
        cond = SinusoidalTimeEmbedding(self.time_embed_dim)(t)  # (B, time_embed_dim)
        
        # Initial projection
        activation = get_activation_fn(self.activation_fn)
        h = nn.Dense(self.hidden_dims[0])(z_flat)  # (B, hidden_dims[0])
        h = activation(h)
        
        # MLP layers with AdaLN conditioning
        for dim in self.hidden_dims:
            # Dense layer
            h = nn.Dense(dim)(h)
            h = activation(h)
            
            # AdaLN: regress scale/shift from time conditioning
            scale_shift = nn.Dense(2 * dim)(cond)  # (B, 2*dim)
            scale, shift = jnp.split(scale_shift, 2, axis=-1)  # Each: (B, dim)
            
            # Apply LayerNorm then modulate with scale and shift
            h = nn.LayerNorm()(h)
            h = h * (1 + scale) + shift
        
        # Output velocity (same dimension as input z_flat)
        v = nn.Dense(z_flat.shape[-1], kernel_init=nn.initializers.zeros)(h)  # (B, D) or (B, N*D)
        
        # Reshape if needed
        if needs_reshape:
            v = v.reshape(original_shape)  # (B, N, D)
        
        return v

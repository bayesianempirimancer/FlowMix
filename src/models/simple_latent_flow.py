"""
Simple MLP-based flow for latent space.

This is a lightweight flow model specifically designed for transforming
latent vectors z, unlike the more complex CRNs designed for point clouds.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence
from src.encoders.embeddings import SinusoidalTimeEmbedding


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
    hidden_dims: Sequence[int] = (256, 256, 256)
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
        for dim in self.hidden_dims:
            h = nn.Dense(dim)(h)
            if self.activation_fn == 'swish':
                h = nn.swish(h)
            elif self.activation_fn == 'relu':
                h = nn.relu(h)
            elif self.activation_fn == 'gelu':
                h = nn.gelu(h)
            else:
                raise ValueError(f"Unknown activation: {self.activation_fn}")
        
        # Output velocity (same dimension as input z_flat)
        v = nn.Dense(z_flat.shape[-1])(h)  # (B, D) or (B, N*D)
        
        # Reshape if needed
        if needs_reshape:
            v = v.reshape(original_shape)  # (B, N, D)
        
        return v





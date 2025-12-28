"""
PointMLP - Pure MLP-based encoder with hierarchical sampling.
Simple, fast, and surprisingly competitive.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Sequence


def farthest_point_sampling(x: jnp.ndarray, num_samples: int, key: jax.random.PRNGKey) -> jnp.ndarray:
    """
    Farthest Point Sampling (FPS).
    
    Args:
        x: (B, N, D) - input points
        num_samples: number of points to sample
        key: random key for initialization
        
    Returns:
        indices: (B, num_samples) - indices of sampled points
    """
    B, N, D = x.shape
    
    # Initialize with random first point
    first_idx = jax.random.randint(key, (B,), 0, N)
    selected_indices = jnp.full((B, num_samples), -1, dtype=jnp.int32)
    selected_indices = selected_indices.at[:, 0].set(first_idx)
    
    # Compute initial distances
    batch_indices = jnp.arange(B)
    first_points = x[batch_indices, first_idx]  # (B, D)
    distances = jnp.sum((x - first_points[:, None, :]) ** 2, axis=-1)  # (B, N)
    
    def fps_step(carry, i):
        sel_indices, dists = carry
        
        # Find farthest point
        farthest_idx = jnp.argmax(dists, axis=-1)  # (B,)
        
        # Update selected indices
        sel_indices = sel_indices.at[:, i].set(farthest_idx)
        
        # Update distances
        farthest_points = x[batch_indices, farthest_idx]  # (B, D)
        new_dists = jnp.sum((x - farthest_points[:, None, :]) ** 2, axis=-1)  # (B, N)
        dists = jnp.minimum(dists, new_dists)
        
        return (sel_indices, dists), None
    
    # Run FPS for remaining points
    (selected_indices, _), _ = jax.lax.scan(
        fps_step, 
        (selected_indices, distances), 
        jnp.arange(1, num_samples)
    )
    
    return selected_indices


class PointMLPBlock(nn.Module):
    """
    Basic PointMLP block: MLP with residual connection.
    """
    hidden_dim: int
    out_dim: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, N, D)
        h = nn.Dense(self.hidden_dim)(x)
        h = nn.relu(h)
        h = nn.Dense(self.out_dim)(h)
        
        # Residual connection if dimensions match
        if x.shape[-1] == self.out_dim:
            h = h + x
        
        h = nn.LayerNorm()(h)
        return h


class PointMLP(nn.Module):
    """
    PointMLP: Pure MLP-based point cloud encoder.
    
    Uses hierarchical sampling and simple MLPs without explicit geometric modeling.
    Surprisingly competitive with more complex architectures while being very fast.
    
    Output: (B, M, embed_dim) where M is the number of points after hierarchical sampling
    """
    embed_dim: int = 64
    num_layers: int = 4
    num_samples: Sequence[int] = (512, 256, 128, 64)  # Progressive sampling
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        # x: (B, N, D)
        B, N, D = x.shape
        
        if key is None:
            key = jax.random.PRNGKey(0)
        
        # Initial embedding
        h = nn.Dense(self.embed_dim)(x)  # (B, N, embed_dim)
        h = nn.relu(h)
        
        # Hierarchical processing with sampling
        for i, num_samp in enumerate(self.num_samples):
            if num_samp >= h.shape[1]:
                # Skip if already fewer points
                continue
            
            # Sample points using FPS
            key, subkey = jax.random.split(key)
            sample_indices = farthest_point_sampling(h, num_samp, subkey)  # (B, num_samp)
            
            # Gather sampled points
            batch_indices = jnp.arange(B)[:, None]
            h = h[batch_indices, sample_indices]  # (B, num_samp, embed_dim)
            
            # Apply MLP block
            h = PointMLPBlock(
                hidden_dim=self.embed_dim * 2,
                out_dim=self.embed_dim
            )(h)
        
        # Final MLP layers
        for _ in range(2):
            h = PointMLPBlock(
                hidden_dim=self.embed_dim * 2,
                out_dim=self.embed_dim
            )(h)
        
        return h  # (B, M, embed_dim)


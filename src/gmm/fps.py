import jax
import jax.numpy as jnp
from typing import Optional, Tuple

def batched_fps(
    x: jnp.ndarray,
    num_points: int,
    mask: Optional[jnp.ndarray] = None,
    seed: int = 42
) -> jnp.ndarray:
    """
    Batched Farthest Point Sampling (FPS).
    
    Args:
        x: (B, N, D) Point cloud.
        num_points: Number of points to sample (K).
        mask: (B, N) Boolean or float mask. 1.0 = valid, 0.0 = invalid.
        seed: Random seed for selecting the first point.
        
    Returns:
        indices: (B, K) Indices of selected points.
    """
    B, N, D = x.shape
    
    if mask is None:
        mask = jnp.ones((B, N), dtype=jnp.float32)
    else:
        mask = mask.astype(jnp.float32)
        if mask.ndim == 3:
            mask = mask.squeeze(-1)
            
    key = jax.random.PRNGKey(seed)
    
    # 1. First Point Selection (Random)
    # Probability proportional to mask
    p_mask = mask / (jnp.sum(mask, axis=1, keepdims=True) + 1e-10)
    
    # Randomly select first index
    idx_0 = jax.random.categorical(key, jnp.log(p_mask + 1e-10), axis=1) # (B,)
    
    # Gather first point
    # idx_0: (B,)
    # first_pt: (B, D)
    first_pt = jnp.take_along_axis(x, idx_0[:, None, None], axis=1).squeeze(1)
    
    # Initialize distances
    # dists: (B, N) - min dist to any selected point so far
    # Initially dist to first point
    dists = jnp.sum((x - first_pt[:, None, :])**2, axis=-1)
    
    # Mask invalid points (set dist to -1 so they aren't picked by argmax)
    dists = jnp.where(mask > 0.5, dists, -1.0)
    
    # Initialize indices array
    indices = jnp.zeros((B, num_points), dtype=jnp.int32)
    indices = indices.at[:, 0].set(idx_0)
    
    def fps_step(carry, i):
        curr_dists = carry
        
        # Select point with max distance
        # idx_new: (B,)
        idx_new = jnp.argmax(curr_dists, axis=1)
        
        # Gather new point
        new_pt = jnp.take_along_axis(x, idx_new[:, None, None], axis=1).squeeze(1) # (B, D)
        
        # Update Distances
        # Dist to new point
        new_dists_to_pt = jnp.sum((x - new_pt[:, None, :])**2, axis=-1) # (B, N)
        
        # Min dist update
        next_dists = jnp.minimum(curr_dists, new_dists_to_pt)
        
        # Re-apply mask
        next_dists = jnp.where(mask > 0.5, next_dists, -1.0)
        
        return next_dists, idx_new
        
    # Run loop for K-1 steps
    step_indices = jnp.arange(1, num_points)
    _, remaining_indices = jax.lax.scan(fps_step, dists, step_indices)
    
    # remaining_indices is (K-1, B). Transpose to (B, K-1)
    remaining_indices = remaining_indices.T
    
    # Concatenate
    indices = indices.at[:, 1:].set(remaining_indices)
    
    return indices






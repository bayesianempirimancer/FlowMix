"""
PointNet++ Set Abstraction - outputs sequence of features (B, M, embed_dim).
Hierarchical point cloud processing with sampling and grouping.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Sequence


def farthest_point_sampling(x: jnp.ndarray, num_samples: int) -> jnp.ndarray:
    """
    Farthest Point Sampling (FPS).
    
    Args:
        x: (B, N, D) - input points
        num_samples: number of points to sample
        
    Returns:
        indices: (B, num_samples) - indices of sampled points
    """
    B, N, D = x.shape
    
    def fps_step(carry, _):
        selected_indices, distances = carry
        
        # Find farthest point
        farthest_idx = jnp.argmax(distances, axis=-1)  # (B,)
        
        # Update selected indices
        batch_indices = jnp.arange(B)
        selected_indices = selected_indices.at[batch_indices, jnp.sum(selected_indices >= 0, axis=-1)].set(farthest_idx)
        
        # Update distances
        farthest_points = x[batch_indices, farthest_idx]  # (B, D)
        new_dists = jnp.sum((x - farthest_points[:, None, :]) ** 2, axis=-1)  # (B, N)
        distances = jnp.minimum(distances, new_dists)
        
        return (selected_indices, distances), None
    
    # Initialize: first point is random
    first_idx = jax.random.randint(jax.random.PRNGKey(0), (B,), 0, N)
    selected_indices = jnp.full((B, num_samples), -1, dtype=jnp.int32)
    selected_indices = selected_indices.at[jnp.arange(B), 0].set(first_idx)
    
    # Compute initial distances
    first_points = x[jnp.arange(B), first_idx]  # (B, D)
    distances = jnp.sum((x - first_points[:, None, :]) ** 2, axis=-1)  # (B, N)
    
    # Run FPS
    (selected_indices, _), _ = jax.lax.scan(fps_step, (selected_indices, distances), None, length=num_samples - 1)
    
    return selected_indices


def ball_query(centers: jnp.ndarray, points: jnp.ndarray, radius: float, max_neighbors: int) -> jnp.ndarray:
    """
    Ball query: find all points within radius of each center.
    
    Args:
        centers: (B, M, D) - center points
        points: (B, N, D) - all points
        radius: query radius
        max_neighbors: maximum number of neighbors per center
        
    Returns:
        indices: (B, M, max_neighbors) - indices of neighbors (padded with -1)
    """
    B, M, D = centers.shape
    _, N, _ = points.shape
    
    # Compute distances: (B, M, N)
    centers_expanded = centers[:, :, None, :]  # (B, M, 1, D)
    points_expanded = points[:, None, :, :]   # (B, 1, N, D)
    distances = jnp.sum((centers_expanded - points_expanded) ** 2, axis=-1)  # (B, M, N)
    
    # Find points within radius
    within_radius = distances < radius ** 2  # (B, M, N)
    
    # Get indices of neighbors
    neighbor_indices = jnp.argsort(jnp.where(within_radius, distances, 1e10), axis=-1)[:, :, :max_neighbors]  # (B, M, max_neighbors)
    
    # Mask out invalid neighbors
    neighbor_distances = jnp.take_along_axis(distances, neighbor_indices, axis=-1)  # (B, M, max_neighbors)
    valid_mask = neighbor_distances < radius ** 2
    neighbor_indices = jnp.where(valid_mask, neighbor_indices, -1)
    
    return neighbor_indices


class PointNetPlusPlusSetAbstraction(nn.Module):
    """
    PointNet++ Set Abstraction layer.
    Samples points, groups neighbors, and applies PointNet to each group.
    """
    num_samples: int = 512  # Number of sampled points
    radius: float = 0.2  # Query radius
    max_neighbors: int = 32  # Max neighbors per sampled point
    embed_dim: int = 64
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        # x: (B, N, D)
        B, N, D = x.shape
        
        # Sample points using FPS
        if key is None:
            key = jax.random.PRNGKey(0)
        sample_indices = farthest_point_sampling(x, self.num_samples)  # (B, num_samples)
        
        # Gather sampled points: (B, num_samples, D)
        batch_indices = jnp.arange(B)[:, None]
        sampled_points = x[batch_indices, sample_indices]
        
        # Ball query: find neighbors for each sampled point
        neighbor_indices = ball_query(sampled_points, x, self.radius, self.max_neighbors)  # (B, num_samples, max_neighbors)
        
        # Group neighbors: (B, num_samples, max_neighbors, D)
        neighbor_points = x[batch_indices[:, :, None], neighbor_indices]  # (B, num_samples, max_neighbors, D)
        
        # Normalize relative to center
        neighbor_points = neighbor_points - sampled_points[:, :, None, :]  # (B, num_samples, max_neighbors, D)
        
        # Apply PointNet to each group
        # Flatten: (B * num_samples, max_neighbors, D)
        neighbor_flat = neighbor_points.reshape(B * self.num_samples, self.max_neighbors, D)
        
        # PointNet: MLP + Max Pool
        h = nn.Dense(self.embed_dim)(neighbor_flat)
        h = nn.relu(h)
        h = nn.Dense(self.embed_dim)(h)
        h = nn.relu(h)
        h = jnp.max(h, axis=1)  # (B * num_samples, embed_dim)
        
        # Reshape back: (B, num_samples, embed_dim)
        output = h.reshape(B, self.num_samples, self.embed_dim)
        
        return output


class PointNetPlusPlus(nn.Module):
    """
    PointNet++ encoder with multiple set abstraction layers.
    Outputs sequence of features (B, M, embed_dim) where M decreases with each layer.
    """
    embed_dim: int = 64
    num_samples: Sequence[int] = (512, 128)  # Number of samples at each level
    radius: Sequence[float] = (0.2, 0.4)  # Query radius at each level
    max_neighbors: int = 32
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        # x: (B, N, D)
        h = x
        
        # Apply multiple set abstraction layers
        for num_samp, rad in zip(self.num_samples, self.radius):
            h = PointNetPlusPlusSetAbstraction(
                num_samples=num_samp,
                radius=rad,
                max_neighbors=self.max_neighbors,
                embed_dim=self.embed_dim
            )(h, mask=mask, key=key)
        
        return h  # (B, M, embed_dim) where M = num_samples[-1]


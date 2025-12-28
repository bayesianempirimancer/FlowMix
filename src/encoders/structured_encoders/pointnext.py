"""
PointNeXt - Improved PointNet++ with better normalization and scaling.
State-of-the-art hierarchical point cloud encoder.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Sequence


def farthest_point_sampling(x: jnp.ndarray, num_samples: int, key: jax.random.PRNGKey) -> jnp.ndarray:
    """Farthest Point Sampling."""
    B, N, D = x.shape
    
    first_idx = jax.random.randint(key, (B,), 0, N)
    selected_indices = jnp.full((B, num_samples), -1, dtype=jnp.int32)
    selected_indices = selected_indices.at[:, 0].set(first_idx)
    
    batch_indices = jnp.arange(B)
    first_points = x[batch_indices, first_idx]
    distances = jnp.sum((x - first_points[:, None, :]) ** 2, axis=-1)
    
    def fps_step(carry, i):
        sel_indices, dists = carry
        farthest_idx = jnp.argmax(dists, axis=-1)
        sel_indices = sel_indices.at[:, i].set(farthest_idx)
        farthest_points = x[batch_indices, farthest_idx]
        new_dists = jnp.sum((x - farthest_points[:, None, :]) ** 2, axis=-1)
        dists = jnp.minimum(dists, new_dists)
        return (sel_indices, dists), None
    
    (selected_indices, _), _ = jax.lax.scan(
        fps_step, (selected_indices, distances), jnp.arange(1, num_samples)
    )
    return selected_indices


def ball_query(centers: jnp.ndarray, points: jnp.ndarray, radius: float, max_neighbors: int) -> jnp.ndarray:
    """Ball query for grouping."""
    B, M, D = centers.shape
    _, N, _ = points.shape
    
    centers_expanded = centers[:, :, None, :]
    points_expanded = points[:, None, :, :]
    distances = jnp.sum((centers_expanded - points_expanded) ** 2, axis=-1)
    
    within_radius = distances < radius ** 2
    neighbor_indices = jnp.argsort(jnp.where(within_radius, distances, 1e10), axis=-1)[:, :, :max_neighbors]
    
    neighbor_distances = jnp.take_along_axis(distances, neighbor_indices, axis=-1)
    valid_mask = neighbor_distances < radius ** 2
    neighbor_indices = jnp.where(valid_mask, neighbor_indices, -1)
    
    return neighbor_indices


class InvertedResidualBlock(nn.Module):
    """
    Inverted residual block (MobileNet-style) for PointNeXt.
    Expands channels, applies transformation, then projects back down.
    """
    in_dim: int
    out_dim: int
    expansion: int = 4
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, N, in_dim)
        hidden_dim = self.in_dim * self.expansion
        
        # Expansion
        h = nn.Dense(hidden_dim)(x)
        h = nn.gelu(h)
        h = nn.LayerNorm()(h)
        
        # Projection
        h = nn.Dense(self.out_dim)(h)
        h = nn.LayerNorm()(h)
        
        # Residual connection
        if self.in_dim == self.out_dim:
            h = h + x
        
        return h


class PointNeXtSetAbstraction(nn.Module):
    """
    Improved set abstraction layer with inverted residuals.
    """
    num_samples: int
    radius: float
    max_neighbors: int
    out_dim: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, features: jnp.ndarray, key: jax.random.PRNGKey) -> tuple:
        # x: (B, N, D) - point coordinates
        # features: (B, N, C) - point features
        B, N, D = x.shape
        _, _, C = features.shape
        
        # Sample points
        sample_indices = farthest_point_sampling(x, self.num_samples, key)
        batch_indices = jnp.arange(B)[:, None]
        sampled_points = x[batch_indices, sample_indices]  # (B, M, D)
        
        # Ball query
        neighbor_indices = ball_query(sampled_points, x, self.radius, self.max_neighbors)
        
        # Gather neighbor features
        neighbor_features = features[batch_indices[:, :, None], neighbor_indices]  # (B, M, K, C)
        
        # Relative positions
        neighbor_points = x[batch_indices[:, :, None], neighbor_indices]  # (B, M, K, D)
        relative_pos = neighbor_points - sampled_points[:, :, None, :]  # (B, M, K, D)
        
        # Concatenate features with relative positions
        combined = jnp.concatenate([neighbor_features, relative_pos], axis=-1)  # (B, M, K, C+D)
        
        # Process with inverted residual blocks
        combined_flat = combined.reshape(B * self.num_samples * self.max_neighbors, C + D)
        h = InvertedResidualBlock(in_dim=C + D, out_dim=self.out_dim)(combined_flat)
        h = h.reshape(B, self.num_samples, self.max_neighbors, self.out_dim)
        
        # Max pooling over neighbors
        output_features = jnp.max(h, axis=2)  # (B, M, out_dim)
        
        return sampled_points, output_features


class PointNeXt(nn.Module):
    """
    PointNeXt: Improved PointNet++ with inverted residuals and better normalization.
    
    State-of-the-art hierarchical point cloud encoder with:
    - Inverted residual bottlenecks
    - Better normalization (LayerNorm instead of BatchNorm)
    - Improved local feature aggregation
    
    Output: (B, M, embed_dim) where M is the final number of sampled points
    """
    embed_dim: int = 64
    num_samples: Sequence[int] = (512, 128)
    radius: Sequence[float] = (0.2, 0.4)
    max_neighbors: int = 32
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        # x: (B, N, D)
        B, N, D = x.shape
        
        if key is None:
            key = jax.random.PRNGKey(0)
        
        # Initial feature embedding
        features = nn.Dense(self.embed_dim)(x)  # (B, N, embed_dim)
        features = nn.gelu(features)
        features = nn.LayerNorm()(features)
        
        points = x
        
        # Hierarchical set abstraction
        for i, (num_samp, rad) in enumerate(zip(self.num_samples, self.radius)):
            key, subkey = jax.random.split(key)
            points, features = PointNeXtSetAbstraction(
                num_samples=num_samp,
                radius=rad,
                max_neighbors=self.max_neighbors,
                out_dim=self.embed_dim
            )(points, features, subkey)
        
        return features  # (B, M, embed_dim)


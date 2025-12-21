"""
KPConv (Kernel Point Convolution) - Point cloud specific convolution.
Uses learnable kernel points to define local neighborhoods.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Sequence


def knn_query(query_points: jnp.ndarray, support_points: jnp.ndarray, k: int) -> jnp.ndarray:
    """
    K-nearest neighbor query.
    
    Args:
        query_points: (B, M, D) - query points
        support_points: (B, N, D) - support points
        k: number of neighbors
        
    Returns:
        indices: (B, M, k) - indices of k nearest neighbors
    """
    B, M, D = query_points.shape
    _, N, _ = support_points.shape
    
    # Compute pairwise distances
    query_expanded = query_points[:, :, None, :]  # (B, M, 1, D)
    support_expanded = support_points[:, None, :, :]  # (B, 1, N, D)
    distances = jnp.sum((query_expanded - support_expanded) ** 2, axis=-1)  # (B, M, N)
    
    # Get k nearest neighbors
    indices = jnp.argsort(distances, axis=-1)[:, :, :k]  # (B, M, k)
    
    return indices


class KPConvLayer(nn.Module):
    """
    Kernel Point Convolution layer.
    
    Uses K learnable kernel points to define convolution weights.
    For each input point, finds neighbors and computes weighted sum based on
    distance to kernel points.
    """
    in_dim: int
    out_dim: int
    num_kernel_points: int = 15
    kernel_radius: float = 0.2
    num_neighbors: int = 16
    
    @nn.compact
    def __call__(self, points: jnp.ndarray, features: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            points: (B, N, D) - point coordinates
            features: (B, N, in_dim) - point features
            
        Returns:
            output: (B, N, out_dim) - output features
        """
        B, N, D = points.shape
        
        # Initialize kernel points (learnable)
        # Kernel points are positioned in a local sphere
        kernel_points = self.param(
            'kernel_points',
            nn.initializers.uniform(scale=self.kernel_radius),
            (self.num_kernel_points, D)
        )  # (K, D)
        
        # Convolution weights for each kernel point
        kernel_weights = self.param(
            'kernel_weights',
            nn.initializers.xavier_uniform(),
            (self.num_kernel_points, self.in_dim, self.out_dim)
        )  # (K, in_dim, out_dim)
        
        # Find neighbors for each point
        neighbor_indices = knn_query(points, points, self.num_neighbors)  # (B, N, k)
        
        # Gather neighbor points and features
        batch_indices = jnp.arange(B)[:, None, None]
        neighbor_points = points[batch_indices, neighbor_indices]  # (B, N, k, D)
        neighbor_features = features[batch_indices, neighbor_indices]  # (B, N, k, in_dim)
        
        # Compute relative positions
        relative_pos = neighbor_points - points[:, :, None, :]  # (B, N, k, D)
        
        # Compute distances to each kernel point
        # relative_pos: (B, N, k, D)
        # kernel_points: (K, D)
        # distances: (B, N, k, K)
        distances = jnp.sum(
            (relative_pos[:, :, :, None, :] - kernel_points[None, None, None, :, :]) ** 2,
            axis=-1
        )  # (B, N, k, K)
        
        # Compute kernel weights using Gaussian influence
        sigma = self.kernel_radius / 2.5
        influence = jnp.exp(-distances / (2 * sigma ** 2))  # (B, N, k, K)
        
        # Normalize influence
        influence = influence / (jnp.sum(influence, axis=-1, keepdims=True) + 1e-8)
        
        # Apply convolution
        # neighbor_features: (B, N, k, in_dim)
        # influence: (B, N, k, K)
        # kernel_weights: (K, in_dim, out_dim)
        
        # Reshape for batch matrix multiplication
        neighbor_features_flat = neighbor_features.reshape(B * N * self.num_neighbors, self.in_dim)
        influence_flat = influence.reshape(B * N * self.num_neighbors, self.num_kernel_points)
        
        # Compute weighted features for each kernel point
        # (B*N*k, K) @ (K, in_dim, out_dim) -> (B*N*k, in_dim, out_dim)
        weighted = jnp.einsum('nk,kio->nio', influence_flat, kernel_weights)
        
        # Sum over neighbors
        weighted = weighted.reshape(B, N, self.num_neighbors, self.in_dim, self.out_dim)
        weighted = jnp.sum(weighted, axis=2)  # (B, N, in_dim, out_dim)
        
        # Apply to features
        output = jnp.einsum('bni,bnio->bno', neighbor_features, weighted)
        
        return output


class KPConv(nn.Module):
    """
    KPConv: Kernel Point Convolution network.
    
    Uses learnable kernel points to define convolution operations on point clouds.
    Adapts to the geometric structure of the data.
    
    Output: (B, N, embed_dim)
    """
    embed_dim: int = 64
    num_layers: int = 4
    num_kernel_points: int = 15
    kernel_radius: float = 0.2
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        # x: (B, N, D)
        B, N, D = x.shape
        
        # Initial feature embedding
        features = nn.Dense(self.embed_dim)(x)  # (B, N, embed_dim)
        features = nn.relu(features)
        
        points = x
        
        # Stack KPConv layers
        for i in range(self.num_layers):
            features = KPConvLayer(
                in_dim=self.embed_dim,
                out_dim=self.embed_dim,
                num_kernel_points=self.num_kernel_points,
                kernel_radius=self.kernel_radius
            )(points, features)
            
            features = nn.relu(features)
            features = nn.LayerNorm()(features)
        
        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask[:, :, None]
            features = features * mask_expanded
        
        return features  # (B, N, embed_dim)


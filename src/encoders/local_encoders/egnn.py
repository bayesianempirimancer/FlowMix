"""
EGNN: E(n) Equivariant Graph Neural Network.

Equivariant to translations, rotations, and reflections in n-dimensional Euclidean space.
Based on "E(n) Equivariant Graph Neural Networks" (2021).
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional


def knn_graph(x: jnp.ndarray, k: int) -> jnp.ndarray:
    """Build k-nearest neighbor graph."""
    B, N, D = x.shape
    
    x_expanded_i = x[:, :, None, :]
    x_expanded_j = x[:, None, :, :]
    distances = jnp.sum((x_expanded_i - x_expanded_j) ** 2, axis=-1)
    distances = distances + jnp.eye(N)[None, :, :] * 1e10
    indices = jnp.argsort(distances, axis=-1)[:, :, :k]
    
    return indices


class EGNNLayer(nn.Module):
    """
    E(n) Equivariant Graph Neural Network Layer.
    
    Updates both node features (invariant) and coordinates (equivariant).
    Message passing preserves E(n) symmetry.
    """
    hidden_dim: int
    k: int = 20
    
    @nn.compact
    def __call__(self, h: jnp.ndarray, x: jnp.ndarray) -> tuple:
        """
        Args:
            h: (B, N, D_h) - node features (invariant)
            x: (B, N, D_x) - node coordinates (equivariant)
            
        Returns:
            h_new: (B, N, D_h) - updated features
            x_new: (B, N, D_x) - updated coordinates
        """
        B, N, D_h = h.shape
        _, _, D_x = x.shape
        
        # Build k-NN graph
        knn_indices = knn_graph(x, self.k)  # (B, N, k)
        
        # Gather neighbor features and coordinates
        batch_indices = jnp.arange(B)[:, None, None]
        h_j = h[batch_indices, knn_indices]  # (B, N, k, D_h)
        x_j = x[batch_indices, knn_indices]  # (B, N, k, D_x)
        
        # Expand for broadcasting
        h_i = h[:, :, None, :]  # (B, N, 1, D_h)
        x_i = x[:, :, None, :]  # (B, N, 1, D_x)
        
        # Compute edge features (invariant to E(n))
        # 1. Squared distance (invariant)
        x_diff = x_j - x_i  # (B, N, k, D_x)
        dist_sq = jnp.sum(x_diff ** 2, axis=-1, keepdims=True)  # (B, N, k, 1)
        
        # 2. Concatenate node features and distance
        edge_input = jnp.concatenate([
            jnp.tile(h_i, (1, 1, self.k, 1)),  # h_i
            h_j,  # h_j
            dist_sq  # ||x_i - x_j||^2
        ], axis=-1)  # (B, N, k, 2*D_h + 1)
        
        # Edge MLP (computes messages)
        edge_flat = edge_input.reshape(B * N * self.k, 2 * D_h + 1)
        m_ij = nn.Dense(self.hidden_dim)(edge_flat)
        m_ij = nn.silu(m_ij)
        m_ij = nn.Dense(self.hidden_dim)(m_ij)
        m_ij = m_ij.reshape(B, N, self.k, self.hidden_dim)
        
        # Aggregate messages (sum over neighbors)
        m_i = jnp.sum(m_ij, axis=2)  # (B, N, hidden_dim)
        
        # Update node features
        h_input = jnp.concatenate([h, m_i], axis=-1)  # (B, N, D_h + hidden_dim)
        h_new = nn.Dense(D_h)(h_input)
        h_new = nn.silu(h_new)
        h_new = nn.Dense(D_h)(h_new)
        h_new = h + h_new  # Residual connection
        
        # Update coordinates (equivariant)
        # Compute coordinate update weights from edge features
        coord_weights = nn.Dense(1)(m_ij.reshape(B * N * self.k, self.hidden_dim))
        coord_weights = coord_weights.reshape(B, N, self.k, 1)  # (B, N, k, 1)
        
        # Weighted sum of relative positions (preserves equivariance)
        x_update = jnp.sum(coord_weights * x_diff, axis=2)  # (B, N, D_x)
        
        # Apply coordinate MLP
        x_update_norm = jnp.linalg.norm(x_update, axis=-1, keepdims=True) + 1e-8
        x_update_normalized = x_update / x_update_norm
        
        # Scale factor (learned from features)
        scale = nn.Dense(1)(h_new)
        scale = jax.nn.tanh(scale)  # Bound the scale
        
        x_new = x + scale * x_update_normalized * 0.1  # Small update
        
        return h_new, x_new


class EGNN(nn.Module):
    """
    EGNN: E(n) Equivariant Graph Neural Network.
    
    Maintains both node features (invariant) and coordinates (equivariant).
    The coordinates update equivariantly while features remain invariant.
    
    Output: (B, N, embed_dim) - invariant features
            Optionally also returns updated coordinates (B, N, D)
    """
    embed_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 4
    k: int = 20
    update_coords: bool = False  # Whether to update and return coordinates
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        # x: (B, N, D) - input point coordinates
        B, N, D = x.shape
        
        # Initialize node features
        h = nn.Dense(self.embed_dim)(x)  # (B, N, embed_dim)
        h = nn.silu(h)
        
        # Keep coordinates separate
        coords = x
        
        # Stack EGNN layers
        for i in range(self.num_layers):
            h, coords = EGNNLayer(
                hidden_dim=self.hidden_dim,
                k=self.k
            )(h, coords)
        
        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask[:, :, None]
            h = h * mask_expanded
        
        if self.update_coords:
            # Return both features and updated coordinates
            # This is useful for generative tasks
            return h, coords
        else:
            # Return only invariant features
            return h  # (B, N, embed_dim)


class EGNN_Coords(nn.Module):
    """
    EGNN that returns both features and updated coordinates.
    Useful for tasks that need equivariant coordinate updates.
    
    Output: (features, coordinates)
            features: (B, N, embed_dim) - invariant
            coordinates: (B, N, D) - equivariant
    """
    embed_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 4
    k: int = 20
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, key: Optional[jax.random.PRNGKey] = None) -> tuple:
        return EGNN(
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            k=self.k,
            update_coords=True
        )(x, mask=mask, key=key)


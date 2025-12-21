"""
VN-DGCNN: Vector Neuron DGCNN - SO(3)-equivariant point cloud encoder.

Replaces scalar features with 3D vector features to achieve rotation equivariance.
Based on "Vector Neurons: A General Framework for SO(3)-Equivariant Networks" (2021).
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


class VNLinear(nn.Module):
    """
    Vector Neuron Linear layer.
    Maps (B, N, 3, in_dim) -> (B, N, 3, out_dim)
    Each feature is a 3D vector, and the layer preserves SO(3) equivariance.
    """
    in_dim: int
    out_dim: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, N, 3, in_dim)
        # Apply linear transformation to the feature dimension
        # This is equivariant because we don't mix the 3D vector components
        
        # Reshape to (B*N*3, in_dim)
        B, N, _, D_in = x.shape
        x_flat = x.reshape(B * N * 3, D_in)
        
        # Apply dense layer
        out_flat = nn.Dense(self.out_dim, use_bias=False)(x_flat)
        
        # Reshape back to (B, N, 3, out_dim)
        out = out_flat.reshape(B, N, 3, self.out_dim)
        
        return out


class VNReLU(nn.Module):
    """
    Vector Neuron ReLU.
    Applies ReLU based on vector norm, preserving direction.
    """
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, N, 3, D)
        # Compute norm for each vector: (B, N, D)
        norm = jnp.linalg.norm(x, axis=2, keepdims=True)  # (B, N, 1, D)
        
        # Compute direction (unit vector)
        direction = x / (norm + 1e-8)  # (B, N, 3, D)
        
        # Apply ReLU to norm
        norm_relu = jax.nn.relu(norm)  # (B, N, 1, D)
        
        # Reconstruct: norm * direction
        out = norm_relu * direction  # (B, N, 3, D)
        
        return out


class VNLayerNorm(nn.Module):
    """
    Vector Neuron Layer Normalization.
    Normalizes vector norms across features.
    """
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, N, 3, D)
        # Compute norm for each vector: (B, N, D)
        norm = jnp.linalg.norm(x, axis=2, keepdims=True)  # (B, N, 1, D)
        
        # Normalize across feature dimension
        mean_norm = jnp.mean(norm, axis=-1, keepdims=True)  # (B, N, 1, 1)
        std_norm = jnp.std(norm, axis=-1, keepdims=True) + 1e-8  # (B, N, 1, 1)
        
        # Normalize
        norm_normalized = (norm - mean_norm) / std_norm
        
        # Compute direction
        direction = x / (norm + 1e-8)
        
        # Reconstruct
        out = norm_normalized * direction
        
        return out


class VNEdgeConv(nn.Module):
    """
    Vector Neuron Edge Convolution.
    Equivariant version of EdgeConv from DGCNN.
    """
    out_dim: int
    k: int = 20
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, N, 3, D)
        B, N, _, D = x.shape
        
        # Build k-NN graph based on spatial positions (first 3 components of first feature)
        # For simplicity, use the mean position across features
        positions = jnp.mean(x, axis=-1)  # (B, N, 3)
        knn_indices = knn_graph(positions, self.k)  # (B, N, k)
        
        # Gather neighbor features
        batch_indices = jnp.arange(B)[:, None, None]
        neighbor_features = x[batch_indices, knn_indices]  # (B, N, k, 3, D)
        
        # Expand x to match neighbor shape
        x_expanded = x[:, :, None, :, :]  # (B, N, 1, 3, D)
        
        # Edge features: concatenate [x_i, x_j - x_i]
        edge_features = jnp.concatenate([
            jnp.tile(x_expanded, (1, 1, self.k, 1, 1)),  # (B, N, k, 3, D)
            neighbor_features - x_expanded  # (B, N, k, 3, D)
        ], axis=-1)  # (B, N, k, 3, 2*D)
        
        # Apply VN-MLP to edge features
        # Reshape to (B*N*k, 3, 2*D)
        edge_flat = edge_features.reshape(B * N * self.k, 3, 2 * D)
        
        # VN-Linear layers
        h = VNLinear(in_dim=2 * D, out_dim=self.out_dim)(edge_flat)
        h = VNReLU()(h)
        h = VNLinear(in_dim=self.out_dim, out_dim=self.out_dim)(h)
        
        # Reshape back
        h = h.reshape(B, N, self.k, 3, self.out_dim)
        
        # Max pooling over neighbors (based on norm)
        norms = jnp.linalg.norm(h, axis=3)  # (B, N, k, out_dim)
        max_indices = jnp.argmax(norms, axis=2)  # (B, N, out_dim)
        
        # Gather max features
        batch_idx = jnp.arange(B)[:, None, None]
        point_idx = jnp.arange(N)[None, :, None]
        feat_idx = jnp.arange(self.out_dim)[None, None, :]
        
        output = h[batch_idx, point_idx, max_indices, :, feat_idx]  # (B, N, 3, out_dim)
        
        return output


class VN_DGCNN(nn.Module):
    """
    VN-DGCNN: Vector Neuron Dynamic Graph CNN.
    
    SO(3)-equivariant point cloud encoder using vector neurons.
    Features are 3D vectors that rotate with the input.
    
    Output: (B, N, 3, embed_dim) - each feature is a 3D vector
    """
    embed_dim: int = 64
    k: int = 20
    num_layers: int = 4
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        # x: (B, N, D) - input point coordinates
        B, N, D = x.shape
        
        # Convert to vector features: (B, N, 3, 1)
        # Each point becomes a 3D vector feature
        if D == 2:
            # 2D points: pad with zeros
            x_padded = jnp.concatenate([x, jnp.zeros((B, N, 1))], axis=-1)
            h = x_padded[:, :, :, None]  # (B, N, 3, 1)
        elif D == 3:
            h = x[:, :, :, None]  # (B, N, 3, 1)
        else:
            raise ValueError(f"Expected 2D or 3D points, got {D}D")
        
        # Initial projection to embed_dim
        h = VNLinear(in_dim=1, out_dim=self.embed_dim)(h)  # (B, N, 3, embed_dim)
        h = VNReLU()(h)
        
        # Stack VN-EdgeConv layers
        for i in range(self.num_layers):
            h_new = VNEdgeConv(out_dim=self.embed_dim, k=self.k)(h)
            h = h + h_new  # Residual connection
            h = VNLayerNorm()(h)
        
        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask[:, :, None, None]  # (B, N, 1, 1)
            h = h * mask_expanded
        
        return h  # (B, N, 3, embed_dim)


class VN_DGCNN_Invariant(nn.Module):
    """
    VN-DGCNN with invariant output (for global pooling).
    Converts equivariant features to invariant features by taking norms.
    
    Output: (B, N, embed_dim) - scalar features (rotation invariant)
    """
    embed_dim: int = 64
    k: int = 20
    num_layers: int = 4
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        # Get equivariant features
        h = VN_DGCNN(
            embed_dim=self.embed_dim,
            k=self.k,
            num_layers=self.num_layers
        )(x, mask=mask, key=key)  # (B, N, 3, embed_dim)
        
        # Convert to invariant features by taking norms
        h_invariant = jnp.linalg.norm(h, axis=2)  # (B, N, embed_dim)
        
        return h_invariant


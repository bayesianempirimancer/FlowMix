"""
Modern Dynamic Graph CNN (DGCNN) - outputs sequence of features (B, N, embed_dim).

Modern features:
- Pre-normalization
- Improved edge convolution with attention
- Residual connections
- Dropout and stochastic depth
- Better aggregation strategies
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Literal


def knn_graph(x: jnp.ndarray, k: int, exclude_self: bool = True) -> jnp.ndarray:
    """
    Build k-nearest neighbor graph.
    
    Args:
        x: (B, N, D) - input points
        k: number of neighbors
        exclude_self: whether to exclude self from neighbors
        
    Returns:
        indices: (B, N, k) - indices of k nearest neighbors for each point
    """
    B, N, D = x.shape
    
    # Compute pairwise distances: (B, N, N)
    x_expanded_i = x[:, :, None, :]  # (B, N, 1, D)
    x_expanded_j = x[:, None, :, :]   # (B, 1, N, D)
    distances = jnp.sum((x_expanded_i - x_expanded_j) ** 2, axis=-1)  # (B, N, N)
    
    if exclude_self:
        # Add large value to diagonal to exclude self
        distances = distances + jnp.eye(N)[None, :, :] * 1e10
    
    # Get top-k indices: (B, N, k)
    indices = jnp.argsort(distances, axis=-1)[:, :, :k]
    
    return indices


class ModernEdgeConv(nn.Module):
    """
    Modern Edge Convolution layer with attention and better aggregation.
    """
    out_dim: int
    k: int = 20
    aggregation: Literal['max', 'mean', 'attention'] = 'max'
    use_residual: bool = True
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Args:
            x: (B, N, D)
            deterministic: whether to use dropout
        
        Returns:
            output: (B, N, out_dim)
        """
        B, N, D = x.shape
        
        # Build k-NN graph
        knn_indices = knn_graph(x, self.k)  # (B, N, k)
        
        # Gather neighbor features: (B, N, k, D)
        batch_indices = jnp.arange(B)[:, None, None]  # (B, 1, 1)
        neighbor_features = x[batch_indices, knn_indices]  # (B, N, k, D)
        
        # Expand x to match neighbor_features: (B, N, 1, D)
        x_expanded = x[:, :, None, :]
        
        # Edge features: concatenate [x_i, x_j - x_i, ||x_j - x_i||²]
        diff = neighbor_features - x_expanded  # (B, N, k, D)
        dist_sq = jnp.sum(diff ** 2, axis=-1, keepdims=True)  # (B, N, k, 1)
        
        edge_features = jnp.concatenate([
            jnp.tile(x_expanded, (1, 1, self.k, 1)),  # x_i
            diff,  # x_j - x_i
            dist_sq  # distance
        ], axis=-1)  # (B, N, k, 2*D + 1)
        
        # Apply MLP to edge features (Pre-LN)
        edge_feat_dim = 2 * D + 1
        edge_features_flat = edge_features.reshape(B * N * self.k, edge_feat_dim)
        
        # Two-layer MLP with Pre-LN
        h = nn.LayerNorm()(edge_features_flat)
        h = nn.Dense(self.out_dim * 2)(h)
        h = nn.swish(h)
        if self.dropout_rate > 0.0 and not deterministic:
            h = nn.Dropout(self.dropout_rate)(h, deterministic=deterministic)
        h = nn.Dense(self.out_dim)(h)
        
        edge_mlp = h.reshape(B, N, self.k, self.out_dim)
        
        # Aggregation
        if self.aggregation == 'max':
            output = jnp.max(edge_mlp, axis=2)  # (B, N, out_dim)
        elif self.aggregation == 'mean':
            output = jnp.mean(edge_mlp, axis=2)  # (B, N, out_dim)
        elif self.aggregation == 'attention':
            # Attention-based aggregation
            # Compute attention scores from edge features
            attn_logits = nn.Dense(1)(edge_mlp)  # (B, N, k, 1)
            attn_weights = nn.softmax(attn_logits, axis=2)  # (B, N, k, 1)
            output = jnp.sum(edge_mlp * attn_weights, axis=2)  # (B, N, out_dim)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        return output


class DGCNN(nn.Module):
    """
    Modern Dynamic Graph CNN encoder.
    
    Modern features:
    - Pre-normalization
    - Improved edge convolution with attention
    - Residual connections
    - Dropout and stochastic depth
    - Better aggregation strategies
    
    Outputs sequence of features (B, N, embed_dim).
    """
    embed_dim: int = 64
    k: int = 20  # number of neighbors
    num_layers: int = 4
    aggregation: Literal['max', 'mean', 'attention'] = 'max'
    dropout_rate: float = 0.0
    drop_path_rate: float = 0.0
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None,
                 key: Optional[jax.random.PRNGKey] = None,
                 deterministic: bool = True) -> jnp.ndarray:
        """
        Args:
            x: (B, N, D) - input point cloud
            mask: (B, N) - validity mask
            key: random key (unused, for compatibility)
            deterministic: whether to use dropout
        
        Returns:
            (B, N, embed_dim) - sequence of features
        """
        B, N, D = x.shape
        
        # Initial projection with Pre-LN
        h = nn.LayerNorm()(x)
        h = nn.Dense(self.embed_dim)(h)  # (B, N, embed_dim)
        
        # Stochastic depth: linearly increasing drop rate
        dpr = [self.drop_path_rate * i / max(self.num_layers - 1, 1)
               for i in range(self.num_layers)]
        
        # Stack EdgeConv layers with residual connections
        for i in range(self.num_layers):
            h_new = ModernEdgeConv(
                out_dim=self.embed_dim,
                k=self.k,
                aggregation=self.aggregation,
                dropout_rate=self.dropout_rate,
                name=f'edge_conv_{i}'
            )(h, deterministic=deterministic)
            
            # Stochastic depth
            if dpr[i] > 0.0 and not deterministic:
                keep_prob = 1.0 - dpr[i]
                shape = (B,) + (1,) * (h_new.ndim - 1)
                random_tensor = jax.random.bernoulli(
                    self.make_rng('dropout'), keep_prob, shape
                )
                h_new = h_new * random_tensor / keep_prob
            
            # Residual connection
            h = h + h_new
            h = nn.LayerNorm()(h)
        
        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask[:, :, None]  # (B, N, 1)
            h = h * mask_expanded
        
        return h  # (B, N, embed_dim)

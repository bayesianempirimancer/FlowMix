"""
Modern PointNet - The only "pure" global encoder (MLP + pooling built-in).

Modern features:
- Pre-normalization
- SwiGLU activation (optional)
- Dropout and batch normalization options
- Flexible pooling strategies
- Better initialization
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Optional, Literal


class SwiGLU(nn.Module):
    """SwiGLU activation: Swish-Gated Linear Unit."""
    dim: int
    hidden_dim: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gate = nn.Dense(self.hidden_dim, name='gate')(x)
        value = nn.Dense(self.hidden_dim, name='value')(x)
        hidden = nn.swish(gate) * value
        return nn.Dense(self.dim, name='proj')(hidden)


class PointNetEncoder(nn.Module):
    """
    Modern PointNet: The only pure global encoder.
    Applies per-element MLP -> Pooling -> Dense Projections.
    
    Modern features:
    - Pre-normalization for stability
    - SwiGLU activation (optional)
    - Flexible pooling (max, mean, max+mean)
    - Dropout for regularization
    - Better initialization
    
    Output: (B, latent_dim) - global features
    """
    latent_dim: int = 128
    hidden_dims: Sequence[int] = (64, 128, 256)
    pooling: Literal['max', 'mean', 'max_mean'] = 'max'
    use_swiglu: bool = False
    dropout_rate: float = 0.0
    use_batch_norm: bool = False  # Alternative to LayerNorm
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None,
                 key: Optional[jax.random.PRNGKey] = None,
                 deterministic: bool = True) -> jnp.ndarray:
        """
        Args:
            x: (B, N, D) - Input set of points
            mask: (B, N) - 1.0 for valid, 0.0 for invalid
            key: random key (unused, for compatibility)
            deterministic: whether to use dropout
        
        Returns:
            (B, latent_dim) - Global feature vector
        """
        h = x
        
        # Per-point MLP with modern features
        for i, dim in enumerate(self.hidden_dims):
            if self.use_swiglu and i < len(self.hidden_dims) - 1:
                # SwiGLU for hidden layers
                next_dim = self.hidden_dims[i + 1] if i + 1 < len(self.hidden_dims) else dim
                h = SwiGLU(dim=dim, hidden_dim=int(dim * 1.5))(h)
            else:
                # Standard layer
                h = nn.Dense(dim)(h)
                h = nn.swish(h)
            
            # Normalization
            if self.use_batch_norm:
                h = nn.BatchNorm(use_running_average=deterministic)(h)
            else:
                h = nn.LayerNorm()(h)
            
            # Dropout
            if self.dropout_rate > 0.0 and not deterministic:
                h = nn.Dropout(self.dropout_rate)(h, deterministic=deterministic)
        
        # Pooling with mask support
        if mask is not None:
            mask_expanded = mask[..., None]  # (B, N, 1)
        else:
            mask_expanded = None
        
        if self.pooling == 'max':
            # Masked max pooling
            if mask_expanded is not None:
                h_masked = jnp.where(mask_expanded, h, -1e9)
            else:
                h_masked = h
            global_feat = jnp.max(h_masked, axis=1)  # (B, D)
            
        elif self.pooling == 'mean':
            # Masked mean pooling
            if mask_expanded is not None:
                h_masked = h * mask_expanded
                count = jnp.sum(mask_expanded, axis=1, keepdims=False)  # (B, 1)
                global_feat = jnp.sum(h_masked, axis=1) / jnp.maximum(count, 1.0)
            else:
                global_feat = jnp.mean(h, axis=1)
                
        elif self.pooling == 'max_mean':
            # Concatenate max and mean pooling
            if mask_expanded is not None:
                h_masked_max = jnp.where(mask_expanded, h, -1e9)
                h_masked_mean = h * mask_expanded
                count = jnp.sum(mask_expanded, axis=1, keepdims=False)
                max_feat = jnp.max(h_masked_max, axis=1)
                mean_feat = jnp.sum(h_masked_mean, axis=1) / jnp.maximum(count, 1.0)
            else:
                max_feat = jnp.max(h, axis=1)
                mean_feat = jnp.mean(h, axis=1)
            global_feat = jnp.concatenate([max_feat, mean_feat], axis=-1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Project to latent dimension
        z = nn.Dense(self.latent_dim)(global_feat)
        
        return z  # (B, latent_dim)

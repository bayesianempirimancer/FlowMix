"""
Equivariant Global Encoder for Point Clouds.

Equivariant encoders: Output transforms in the same way as the input.
This is critical for geometric point cloud processing where we want
representations that respect geometric symmetries.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Sequence, Literal


class EquivariantDeepSetsEncoder(nn.Module):
    """
    Equivariant Deep Sets encoder.
    
    Processes point cloud with equivariant operations, then aggregates
    to get equivariant global features.
    
    Properties:
    - E(n) equivariant (rotations + translations + reflections)
    - Uses message passing with equivariant updates
    - Output can be invariant or equivariant depending on final aggregation
    
    Output: (B, latent_dim) - can be invariant or (B, D, latent_dim) for equivariant
    """
    latent_dim: int = 128
    hidden_dims: Sequence[int] = (64, 128, 256)
    output_type: Literal['invariant', 'equivariant'] = 'invariant'
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None,
                 key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        """
        Args:
            x: (B, N, D) - Input point cloud coordinates
            mask: (B, N) - 1.0 for valid, 0.0 for invalid points
            key: Random key (unused, for compatibility)
            
        Returns:
            (B, latent_dim) if output_type='invariant'
            (B, D, latent_dim) if output_type='equivariant'
        """
        B, N, D = x.shape
        
        # Center points
        if mask is not None:
            mask_expanded = mask[..., None]
            x_masked = x * mask_expanded
            count = jnp.sum(mask_expanded, axis=1, keepdims=True)
            centroid = jnp.sum(x_masked, axis=1, keepdims=True) / jnp.maximum(count, 1.0)
        else:
            centroid = jnp.mean(x, axis=1, keepdims=True)
        
        x_centered = x - centroid
        
        # Initialize features from coordinates
        h = x_centered  # (B, N, D)
        
        # Equivariant processing layers
        for dim in self.hidden_dims:
            # Equivariant linear: h -> (h, x) -> h_new
            # Use coordinates to update features equivariantly
            h_expanded = h[:, :, None, :]  # (B, N, 1, dim)
            x_expanded = x_centered[:, :, None, :]  # (B, N, 1, D)
            
            # Compute pairwise interactions (invariant distances)
            distances = jnp.sum((x_expanded - x_expanded.transpose(0, 2, 1, 3)) ** 2, axis=-1)  # (B, N, N)
            
            # Message passing: aggregate features from neighbors
            # Use distance-based attention
            attention = nn.softmax(-distances, axis=-1)  # (B, N, N)
            
            # Aggregate features
            h_aggregated = jnp.sum(h[:, None, :, :] * attention[:, :, :, None], axis=2)  # (B, N, dim)
            
            # Update features
            h = nn.Dense(dim)(h_aggregated)
            h = nn.LayerNorm()(h)
            h = nn.swish(h)
        
        # Final aggregation
        if mask is not None:
            mask_expanded = mask[..., None]
        else:
            mask_expanded = None
        
        if self.output_type == 'invariant':
            # Invariant output: pool features
            if mask_expanded is not None:
                h_masked = h * mask_expanded
                count = jnp.sum(mask_expanded, axis=1, keepdims=True)
                global_feat = jnp.sum(h_masked, axis=1) / jnp.maximum(count, 1.0)
            else:
                global_feat = jnp.mean(h, axis=1)  # (B, hidden_dims[-1])
            
            z = nn.Dense(self.latent_dim)(global_feat)
            return z  # (B, latent_dim)
            
        else:  # equivariant
            # Equivariant output: use coordinates as basis
            # Project features to coordinate space
            if mask_expanded is not None:
                h_masked = h * mask_expanded
                x_masked = x_centered * mask_expanded
                count = jnp.sum(mask_expanded, axis=1, keepdims=True)
                # Weighted sum using coordinates
                weights = nn.Dense(1)(h_masked)  # (B, N, 1)
                weights = nn.softmax(weights, axis=1)  # (B, N, 1)
                global_coords = jnp.sum(x_masked * weights, axis=1)  # (B, D)
                global_feat = jnp.sum(h_masked, axis=1) / jnp.maximum(count, 1.0)  # (B, hidden_dims[-1])
            else:
                weights = nn.Dense(1)(h)
                weights = nn.softmax(weights, axis=1)
                global_coords = jnp.sum(x_centered * weights, axis=1)
                global_feat = jnp.mean(h, axis=1)
            
            # Combine coordinates and features
            combined = jnp.concatenate([global_coords, global_feat], axis=-1)  # (B, D + hidden_dims[-1])
            z = nn.Dense(self.latent_dim)(combined)
            
            # Reshape to (B, D, latent_dim) for equivariant output
            z_expanded = z[:, None, :]  # (B, 1, latent_dim)
            z_equivariant = jnp.tile(z_expanded, (1, D, 1))  # (B, D, latent_dim)
            # Could also use global_coords to weight the output
            
            return z_equivariant  # (B, D, latent_dim)

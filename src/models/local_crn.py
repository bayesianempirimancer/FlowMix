"""
Local Conditional ResNets (CRN) for Flow Models.

Local CRNs expect context with shape (B, K, Dc) where K is the number of abstract
latent representations (slots, latent queries, mixture components, etc.).

**Key Property: N = K (One-to-One Correspondence)**

Local CRNs sample exactly K points (one per latent) with direct 1-to-1 conditioning:
- Point 0 ← Latent 0
- Point 1 ← Latent 1
- ...
- Point K-1 ← Latent K-1

This is the most efficient pattern when you want to sample one point per latent,
such as:
- One point per slot (Slot Attention)
- One point per mixture component (GMM)
- One point per latent query (Perceiver)

A Local CRN is any nn.Module that takes arguments (x, c, t, mask) where:
- x: input tensor (K points being flowed) - shape (B, K, D)
- c: structured context (K latent representations) - shape (B, K, Dc)
- t: time - scalar or (B,) or (B, 1)
- mask: (B, K) - optional mask for valid latents (1 = valid, 0 = masked)
and returns output with the same shape as x: (B, K, D)

**Constraint:** x.shape[1] must equal c.shape[1] (N = K)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Sequence, Optional, Callable
from src.encoders.embeddings import SinusoidalTimeEmbedding, PositionalEmbedding2D
from src.layers.self_attention import SelfAttention


def get_activation_fn(name: str) -> Callable:
    """Factory function to get activation function by name."""
    name_lower = name.lower()
    
    if name_lower in ('swish', 'silu'):
        return nn.swish
    elif name_lower == 'relu':
        return nn.relu
    elif name_lower == 'gelu':
        return nn.gelu
    elif name_lower == 'tanh':
        return jnp.tanh
    elif name_lower == 'sigmoid':
        return nn.sigmoid
    elif name_lower == 'elu':
        return nn.elu
    elif name_lower == 'leaky_relu':
        return nn.leaky_relu
    elif name_lower in ('none', 'identity'):
        return lambda x: x
    else:
        raise ValueError(f"Unknown activation function: {name}")


def get_time_embedding(embed_type: str, dim: int) -> nn.Module:
    """Factory function to get time embedding module by type."""
    if embed_type.lower() == 'sinusoidal':
        return SinusoidalTimeEmbedding(dim=dim)
    else:
        raise ValueError(f"Unknown time embedding type: {embed_type}")


class LocalAdaLNMLPCRN(nn.Module):
    """
    Local CRN using MLP with per-latent AdaLN.
    
    **Pattern:** N = K (One-to-One)
    
    Expects structured context: c has shape (B, K, Dc) and x has shape (B, K, D).
    Each point gets conditioned by its corresponding latent (1-to-1 mapping).
    
    Use case: Sampling one point per slot/latent/component.
    
    Args:
        hidden_dims: Hidden layer dimensions
        time_embed_dim: Dimension for time embedding
        time_embed_type: Type of time embedding ('sinusoidal')
        cond_dim: Dimension of conditioning vector
        position_embed_type: Type of positional embedding for x
        activation_fn: Name of activation function
    """
    hidden_dims: Sequence[int] = (64, 64, 64, 64, 64, 64)
    time_embed_dim: int = 32
    time_embed_type: str = 'sinusoidal'
    cond_dim: int = 256
    position_embed_type: str = 'linear'
    activation_fn: str = 'swish'
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, c: jnp.ndarray, t: Any,
                 mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Args:
            x: (B, K, D) - K input points (one per latent)
            c: (B, K, Dc) - K structured context (one per point)
            t: scalar or (B,) or (B, 1) - time
            mask: (B, K) - optional mask for valid latents
            
        Returns:
            Output with same shape as x: (B, K, D)
        """
        # Convert to JAX arrays
        x = jnp.asarray(x)
        c = jnp.asarray(c)
        t = jnp.asarray(t)
        
        # Validate context shape
        assert c.ndim == 3, f"Local CRN expects c with shape (B, K, Dc), got {c.shape}"
        assert x.ndim == 3, f"Local CRN expects x with shape (B, K, D), got {x.shape}"
        assert x.shape[1] == c.shape[1], f"N must equal K for Local CRN: x has N={x.shape[1]}, c has K={c.shape[1]}"
        
        B = x.shape[0]
        K = x.shape[1]
        
        # Normalize time
        if t.ndim == 0:
            t = jnp.broadcast_to(t, (B, 1))
        elif t.ndim == 1:
            t = t[:, None] if t.shape[0] == B else jnp.broadcast_to(t, (B, 1))
        
        # Broadcast c if needed
        if c.shape[0] == 1 and B > 1:
            c = jnp.broadcast_to(c, (B, c.shape[1], c.shape[2]))
        
        activation = get_activation_fn(self.activation_fn)
        
        # Build per-latent conditioning from structured context and time
        time_embed = get_time_embedding(self.time_embed_type, self.time_embed_dim)
        t_feat = time_embed(t)  # (B, time_embed_dim)
        
        # Broadcast time to all K latents
        t_feat = t_feat[:, None, :]  # (B, 1, time_embed_dim)
        t_feat = jnp.broadcast_to(t_feat, (B, K, self.time_embed_dim))  # (B, K, time_embed_dim)
        
        # Concatenate time and per-latent context (1-to-1 correspondence)
        cond = jnp.concatenate([t_feat, c], axis=-1)  # (B, K, time_embed_dim + Dc)
        cond = nn.swish(nn.Dense(self.cond_dim)(cond))
        cond = nn.swish(nn.Dense(self.cond_dim)(cond))  # (B, K, cond_dim)
        
        # Apply mask to conditioning if provided
        if mask is not None:
            mask_expanded = mask[:, :, None]  # (B, K, 1)
            cond = cond * mask_expanded
        
        # Positional embedding for x
        if self.position_embed_type == 'linear':
            h = nn.Dense(self.hidden_dims[0])(x)
        elif self.position_embed_type == 'sinusoidal':
            pos_embed = PositionalEmbedding2D(dim=self.hidden_dims[0])
            h = pos_embed(x)
        elif self.position_embed_type == 'fourier':
            pos_embed = PositionalEmbedding2D(dim=self.hidden_dims[0], fourier=True)
            h = pos_embed(x)
        else:
            raise ValueError(f"Unknown position_embed_type: {self.position_embed_type}")
        
        h = activation(h)
        
        # MLP with per-latent AdaLN conditioning (1-to-1)
        for dim in self.hidden_dims:
            h = nn.Dense(dim)(h)
            h = activation(h)
            
            # AdaLN: regress scale/shift from per-latent conditioning
            # Each of K points gets its own scale/shift from its corresponding latent
            scale_shift = nn.Dense(2 * dim)(cond)  # (B, K, 2*dim)
            scale, shift = jnp.split(scale_shift, 2, axis=-1)  # Each: (B, K, dim)
            
            h = nn.LayerNorm()(h)
            h = h * (1 + scale) + shift
        
        # Output projection
        dx = nn.Dense(x.shape[-1], kernel_init=nn.initializers.zeros)(h)
        
        # Apply mask to output if provided
        if mask is not None:
            mask_expanded = mask[:, :, None]
            dx = dx * mask_expanded
        
        return dx


class LocalDiTCRN(nn.Module):
    """
    Local CRN using Transformer blocks with per-latent AdaLN.
    
    **Pattern:** N = K (One-to-One)
    
    Expects structured context: c has shape (B, K, Dc) and x has shape (B, K, D).
    Each point gets conditioned by its corresponding latent, plus self-attention among the K points.
    
    Args:
        embed_dim: Embedding dimension for points
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        mlp_ratio: Ratio for MLP hidden dimension
        time_embed_dim: Dimension for time embedding
        time_embed_type: Type of time embedding ('sinusoidal')
        cond_dim: Dimension of conditioning vector
        position_embed_type: Type of positional embedding
    """
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 4
    mlp_ratio: float = 2.0
    time_embed_dim: int = 32
    time_embed_type: str = 'sinusoidal'
    cond_dim: int = 256
    position_embed_type: str = 'linear'
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, c: jnp.ndarray, t: Any,
                 mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Args:
            x: (B, K, D) - K input points (one per latent)
            c: (B, K, Dc) - K structured context (one per point)
            t: scalar or (B,) or (B, 1) - time
            mask: (B, K) - optional mask for valid latents
            
        Returns:
            Output with same shape as x: (B, K, D)
        """
        # Convert to JAX arrays
        x = jnp.asarray(x)
        c = jnp.asarray(c)
        t = jnp.asarray(t)
        
        # Validate shapes
        assert c.ndim == 3, f"Local CRN expects c with shape (B, K, Dc), got {c.shape}"
        assert x.ndim == 3, f"Local CRN expects x with shape (B, K, D), got {x.shape}"
        assert x.shape[1] == c.shape[1], f"N must equal K for Local CRN: x has N={x.shape[1]}, c has K={c.shape[1]}"
        
        B = x.shape[0]
        K = x.shape[1]
        
        # Normalize time
        if t.ndim == 0:
            t = jnp.broadcast_to(t, (B, 1))
        elif t.ndim == 1:
            t = t[:, None] if t.shape[0] == B else jnp.broadcast_to(t, (B, 1))
        
        # Broadcast c if needed
        if c.shape[0] == 1 and B > 1:
            c = jnp.broadcast_to(c, (B, c.shape[1], c.shape[2]))
        
        # Build per-latent conditioning
        time_embed = get_time_embedding(self.time_embed_type, self.time_embed_dim)
        t_feat = time_embed(t)  # (B, time_embed_dim)
        
        # Broadcast time to all K latents
        t_feat = t_feat[:, None, :]
        t_feat = jnp.broadcast_to(t_feat, (B, K, self.time_embed_dim))
        
        # Concatenate time and per-latent context (1-to-1)
        cond = jnp.concatenate([t_feat, c], axis=-1)  # (B, K, time_embed_dim + Dc)
        cond = nn.swish(nn.Dense(self.cond_dim)(cond))
        cond = nn.swish(nn.Dense(self.cond_dim)(cond))  # (B, K, cond_dim)
        
        # Apply mask to conditioning if provided
        if mask is not None:
            mask_expanded = mask[:, :, None]
            cond = cond * mask_expanded
        
        # Positional embedding + project to embed_dim
        if self.position_embed_type == 'linear':
            h = nn.Dense(self.embed_dim)(x)
        elif self.position_embed_type == 'sinusoidal':
            pos_embed = PositionalEmbedding2D(dim=self.embed_dim)
            h = pos_embed(x)
        elif self.position_embed_type == 'fourier':
            pos_embed = PositionalEmbedding2D(dim=self.embed_dim, fourier=True)
            h = pos_embed(x)
        else:
            raise ValueError(f"Unknown position_embed_type: {self.position_embed_type}")
        
        # Prepare attention mask
        attn_mask = None
        if mask is not None:
            # Convert to attention mask format: (B, 1, 1, K)
            attn_mask = mask[:, None, None, :] > 0.5
        
        # Transformer blocks with per-latent AdaLN
        for _ in range(self.num_layers):
            # Self-attention among K points
            h_norm = nn.LayerNorm()(h)
            attn_out = SelfAttention(
                num_heads=self.num_heads,
                head_dim=self.embed_dim // self.num_heads
            )(h_norm, mask=attn_mask)
            
            # Per-latent AdaLN for attention (1-to-1)
            scale_shift = nn.Dense(2 * self.embed_dim)(cond)
            scale, shift = jnp.split(scale_shift, 2, axis=-1)
            attn_out = attn_out * (1 + scale) + shift
            
            h = h + attn_out
            
            # MLP
            h_norm = nn.LayerNorm()(h)
            mlp_dim = int(self.embed_dim * self.mlp_ratio)
            mlp = nn.Dense(mlp_dim)(h_norm)
            mlp = nn.swish(mlp)
            mlp = nn.Dense(self.embed_dim)(mlp)
            
            # Per-latent AdaLN for MLP (1-to-1)
            scale_shift = nn.Dense(2 * self.embed_dim)(cond)
            scale, shift = jnp.split(scale_shift, 2, axis=-1)
            mlp = mlp * (1 + scale) + shift
            
            h = h + mlp
        
        # Final layer norm and output projection
        h = nn.LayerNorm()(h)
        dx = nn.Dense(x.shape[-1], kernel_init=nn.initializers.zeros)(h)
        
        # Apply mask to output if provided
        if mask is not None:
            mask_expanded = mask[:, :, None]
            dx = dx * mask_expanded
        
        return dx

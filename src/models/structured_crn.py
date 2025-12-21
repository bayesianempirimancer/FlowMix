"""
Structured Conditional ResNets (CRN) for Flow Models.

Structured CRNs expect context with shape (B, K, Dc) - a set of K structured latent representations.

The K dimension represents abstract/structured features such as:
- Slots (from Slot Attention) representing different objects
- Latent queries (from Perceiver) representing learned features
- Mixture components (from GMM) representing modes
- Abstract "super points" or collocation points

**Key Property: K << N (Many Points, Few Latents)**

Two architectural patterns:
1. **Pool-Based**: Pool K latents → global conditioning (efficient)
2. **Attention-Based**: Each of N points cross-attends to K latents (expressive)

Note: For the case where N = K (one point per latent), see local_crn.py

A Structured CRN is any nn.Module that takes arguments (x, c, t, mask) where:
- x: input tensor (points being flowed) - shape (B, N, D) where N >> K
- c: structured context (set of K latent representations) - shape (B, K, Dc)
- t: time - scalar or (B,) or (B, 1)
- mask: (B, K) - optional mask for valid latents (1 = valid, 0 = masked)
and returns output with the same shape as x.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Sequence, Optional, Callable
from src.encoders.embeddings import SinusoidalTimeEmbedding, PositionalEmbedding2D
from src.layers.self_attention import SelfAttention
from src.layers.cross_attention import CrossAttention


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


# ============================================================================
# Pool-Based Structured CRNs (K << N)
# Pool K latents → global conditioning → AdaLN
# ============================================================================

class StructuredAdaLNMLPCRN(nn.Module):
    """
    Structured CRN using MLP with Adaptive Layer Normalization (AdaLN).
    
    **Pattern:** Pool-Based (K << N)
    
    Expects structured context: c has shape (B, K, Dc) where K is the number of
    abstract latent representations (slots, latent queries, mixture components, etc.).
    
    Pools K latents into a single global conditioning vector, then applies AdaLN.
    
    Args:
        hidden_dims: Hidden layer dimensions
        time_embed_dim: Dimension for time embedding
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
            x: (B, N, D) or (B, D) - input points
            c: (B, K, Dc) - structured context (K latent representations)
            t: scalar or (B,) or (B, 1) - time
            mask: (B, K) - optional mask for valid latents
            
        Returns:
            Output with same shape as x
        """
        # Convert to JAX arrays
        x = jnp.asarray(x)
        c = jnp.asarray(c)
        t = jnp.asarray(t)
        
        # Validate context shape
        assert c.ndim == 3, f"Structured CRN expects c with shape (B, K, Dc), got {c.shape}"
        
        B = x.shape[0]
        
        # Normalize time
        if t.ndim == 0:
            t = jnp.broadcast_to(t, (B, 1))
        elif t.ndim == 1:
            t = t[:, None] if t.shape[0] == B else jnp.broadcast_to(t, (B, 1))
        
        # Broadcast c if needed
        if c.shape[0] == 1 and B > 1:
            c = jnp.broadcast_to(c, (B, c.shape[1], c.shape[2]))
        
        activation = get_activation_fn(self.activation_fn)
        
        # Build conditioning from structured context and time
        time_embed = get_time_embedding(self.time_embed_type, self.time_embed_dim)
        t_feat = time_embed(t)  # (B, time_embed_dim)
        
        # Broadcast time to all K latent representations
        K = c.shape[1]
        t_feat = t_feat[:, None, :]  # (B, 1, time_embed_dim)
        t_feat = jnp.broadcast_to(t_feat, (B, K, self.time_embed_dim))  # (B, K, time_embed_dim)
        
        # Concatenate time and structured context
        cond = jnp.concatenate([t_feat, c], axis=-1)  # (B, K, time_embed_dim + Dc)
        cond = nn.swish(nn.Dense(self.cond_dim)(cond))
        cond = nn.swish(nn.Dense(self.cond_dim)(cond))  # (B, K, cond_dim)
        
        # Apply mask to conditioning if provided
        if mask is not None:
            mask_expanded = mask[:, :, None]  # (B, K, 1)
            cond = cond * mask_expanded
        
        # Pool K structured latents into single global conditioning vector
        # Use max pooling to aggregate information from all K latents
        cond = jnp.max(cond, axis=1)  # (B, cond_dim)
        
        # Handle 2D input (add spatial dimension)
        squeeze_output = False
        if x.ndim == 2:
            x = x[:, None, :]  # (B, D) -> (B, 1, D)
            squeeze_output = True
        
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
        
        # MLP with AdaLN conditioning from pooled structured context
        for dim in self.hidden_dims:
            h = nn.Dense(dim)(h)
            h = activation(h)
            
            # AdaLN: regress scale/shift from global conditioning (pooled from K latents)
            scale_shift = nn.Dense(2 * dim)(cond)  # (B, 2*dim)
            scale, shift = jnp.split(scale_shift, 2, axis=-1)  # Each: (B, dim)
            
            # Broadcast to match h: (B, N, dim)
            scale = scale[:, None, :]
            shift = shift[:, None, :]
            
            h = nn.LayerNorm()(h)
            h = h * (1 + scale) + shift
        
        # Output projection
        dx = nn.Dense(x.shape[-1], kernel_init=nn.initializers.zeros)(h)
        
        # Apply mask to output if provided
        if mask is not None and not squeeze_output:
            # For pool-based, mask was already applied to conditioning
            pass
        
        # Remove spatial dimension if input was 2D
        if squeeze_output:
            dx = dx[:, 0, :]
        
        return dx


# Similar implementations for StructuredDiTCRN would go here...
# (Keeping the file manageable, showing the pattern)


# ============================================================================
# Attention-Based Structured CRNs (K << N)
# Each of N points cross-attends to K latents
# ============================================================================

class StructuredCrossAttentionCRN(nn.Module):
    """
    Structured CRN using Cross-Attention to structured latent representations.
    
    **Pattern:** Attention-Based (K << N)
    
    Expects structured context: c has shape (B, K, Dc) where K is the number of
    abstract latent representations.
    Each input point cross-attends to the K structured context features.
    
    Args:
        latent_dim: Dimension for processing
        num_heads: Number of attention heads
        num_layers: Number of cross-attention layers
        time_embed_dim: Dimension for time embedding
        cond_dim: Dimension of conditioning vector
        position_embed_type: Type of positional embedding
    """
    latent_dim: int = 64
    num_heads: int = 4
    num_layers: int = 3
    time_embed_dim: int = 64
    time_embed_type: str = 'sinusoidal'
    cond_dim: int = 256
    position_embed_type: str = 'linear'
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, c: jnp.ndarray, t: Any,
                 mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Args:
            x: (B, N, D) or (B, D) - input points
            c: (B, K, Dc) - structured context (K latent representations)
            t: scalar or (B,) or (B, 1) - time
            mask: (B, K) - optional mask for context points
            
        Returns:
            Output with same shape as x
        """
        # Convert to JAX arrays
        x = jnp.asarray(x)
        c = jnp.asarray(c)
        t = jnp.asarray(t)
        
        # Validate context shape
        assert c.ndim == 3, f"Structured CRN expects c with shape (B, K, Dc), got {c.shape}"
        
        B = x.shape[0]
        
        # Normalize time
        if t.ndim == 0:
            t = jnp.broadcast_to(t, (B, 1))
        elif t.ndim == 1:
            t = t[:, None] if t.shape[0] == B else jnp.broadcast_to(t, (B, 1))
        
        # Broadcast c if needed
        if c.shape[0] == 1 and B > 1:
            c = jnp.broadcast_to(c, (B, c.shape[1], c.shape[2]))
        
        # Build time embedding
        time_embed = get_time_embedding(self.time_embed_type, self.time_embed_dim)
        t_feat = time_embed(t)  # (B, time_embed_dim)
        
        # Broadcast time to all context points
        K = c.shape[1]
        t_feat_expanded = t_feat[:, None, :]
        t_feat_expanded = jnp.broadcast_to(t_feat_expanded, (B, K, self.time_embed_dim))
        
        # Concatenate time with structured context
        context_with_time = jnp.concatenate([t_feat_expanded, c], axis=-1)  # (B, K, time_embed_dim + Dc)
        context_features = nn.Dense(self.latent_dim)(context_with_time)  # (B, K, latent_dim)
        
        # Apply mask to context if provided
        if mask is not None:
            mask_expanded = mask[:, :, None]
            context_features = context_features * mask_expanded
        
        # Handle 2D input
        squeeze_output = False
        if x.ndim == 2:
            x = x[:, None, :]
            squeeze_output = True
        
        # Positional embedding + project to latent_dim
        if self.position_embed_type == 'linear':
            h = nn.Dense(self.latent_dim)(x)
        elif self.position_embed_type == 'sinusoidal':
            pos_embed = PositionalEmbedding2D(dim=self.latent_dim)
            h = pos_embed(x)
        elif self.position_embed_type == 'fourier':
            pos_embed = PositionalEmbedding2D(dim=self.latent_dim, fourier=True)
            h = pos_embed(x)
        else:
            raise ValueError(f"Unknown position_embed_type: {self.position_embed_type}")
        
        # Cross-attention layers: h attends to context_features
        for _ in range(self.num_layers):
            # Cross-attention
            h_norm = nn.LayerNorm()(h)
            attn_out = CrossAttention(
                num_heads=self.num_heads,
                head_dim=self.latent_dim // self.num_heads
            )(h_norm, context_features, mask=mask)
            h = h + attn_out
            
            # MLP
            h_norm = nn.LayerNorm()(h)
            mlp = nn.Dense(self.latent_dim * 2)(h_norm)
            mlp = nn.swish(mlp)
            mlp = nn.Dense(self.latent_dim)(mlp)
            h = h + mlp
        
        # Final layer norm and output projection
        h = nn.LayerNorm()(h)
        dx = nn.Dense(x.shape[-1], kernel_init=nn.initializers.zeros)(h)
        
        if squeeze_output:
            dx = dx[:, 0, :]
        
        return dx


# Note: For Direct CRNs (N = K, one-to-one correspondence), see local_crn.py

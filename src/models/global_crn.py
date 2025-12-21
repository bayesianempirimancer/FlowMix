"""
Global Conditional ResNets (CRN) for Flow Models.

Global CRNs expect context with shape (B, Dc) - a single global vector per batch item.

A Global CRN is any nn.Module that takes arguments (x, c, t) where:
- x: input tensor (covariates/points being flowed) - shape (B, N, D) or (B, D)
- c: global context tensor (latent encoding) - shape (B, Dc)
- t: time - scalar or (B,) or (B, 1)
and returns output with the same shape as x.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Sequence, Optional, Callable
from src.encoders.embeddings import SinusoidalTimeEmbedding, PositionalEmbedding2D
from src.layers.self_attention import SelfAttention
from src.layers.cross_attention import CrossAttention
from src.layers.concat_squash import ConcatSquash


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


class GlobalAdaLNMLPCRN(nn.Module):
    """
    Global CRN using MLP with Adaptive Layer Normalization (AdaLN).
    
    Expects global context: c has shape (B, Dc).
    Uses AdaLN to condition on context and time.
    
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
    def __call__(self, x: jnp.ndarray, c: jnp.ndarray, t: Any) -> jnp.ndarray:
        """
        Args:
            x: (B, N, D) or (B, D) - input points
            c: (B, Dc) - global context (latent encoding)
            t: scalar or (B,) or (B, 1) - time
            
        Returns:
            Output with same shape as x
        """
        # Convert to JAX arrays
        x = jnp.asarray(x)
        c = jnp.asarray(c)
        t = jnp.asarray(t)
        
        # Validate context shape
        assert c.ndim == 2, f"Global CRN expects c with shape (B, Dc), got {c.shape}"
        
        B = x.shape[0]
        
        # Normalize time
        if t.ndim == 0:
            t = jnp.broadcast_to(t, (B, 1))
        elif t.ndim == 1:
            t = t[:, None] if t.shape[0] == B else jnp.broadcast_to(t, (B, 1))
        
        # Broadcast c if needed
        if c.shape[0] == 1 and B > 1:
            c = jnp.broadcast_to(c, (B, c.shape[1]))
        
        activation = get_activation_fn(self.activation_fn)
        
        # Build conditioning vector from global context and time
        time_embed = get_time_embedding(self.time_embed_type, self.time_embed_dim)
        t_feat = time_embed(t)  # (B, time_embed_dim)
        
        # Concatenate time and global context
#        cond = jnp.concatenate([t_feat, c], axis=-1)  # (B, time_embed_dim + Dc)
        cond = nn.Dense(self.cond_dim)(c) 
        cond = cond + nn.Dense(self.cond_dim, use_bias=False)(t_feat)
        cond = nn.swish(cond)
        cond = nn.swish(nn.Dense(self.cond_dim)(cond))  # (B, cond_dim)
        
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
        
        # MLP with AdaLN conditioning
        for dim in self.hidden_dims:
            h = nn.Dense(dim)(h)
            h = activation(h)
            
            # AdaLN: regress scale/shift from global conditioning
            scale_shift = nn.Dense(2 * dim)(cond)  # (B, 2*dim)
            scale, shift = jnp.split(scale_shift, 2, axis=-1)  # Each: (B, dim)
            
            # Broadcast to match h: (B, N, dim)
            scale = scale[:, None, :]
            shift = shift[:, None, :]
            
            h = nn.LayerNorm()(h)
            h = h * (1 + scale) + shift
        
        # Output projection
        dx = nn.Dense(x.shape[-1], kernel_init=nn.initializers.zeros)(h)
        
        # Remove spatial dimension if input was 2D
        if squeeze_output:
            dx = dx[:, 0, :]
        
        return dx


class GlobalDiTCRN(nn.Module):
    """
    Global CRN using Transformer blocks with AdaLN (DiT-style).
    
    Expects global context: c has shape (B, Dc).
    Uses self-attention over input points with AdaLN conditioning.
    
    Args:
        embed_dim: Embedding dimension for points
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        mlp_ratio: Ratio for MLP hidden dimension
        time_embed_dim: Dimension for time embedding
        cond_dim: Dimension of conditioning vector
        position_embed_type: Type of positional embedding
        use_concat_squash: Whether to use ConcatSquash for conditioning
    """
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 4
    mlp_ratio: float = 2.0
    time_embed_dim: int = 32
    time_embed_type: str = 'sinusoidal'
    cond_dim: int = 256
    position_embed_type: str = 'linear'
    use_concat_squash: bool = True
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, c: jnp.ndarray, t: Any) -> jnp.ndarray:
        """
        Args:
            x: (B, N, D) or (B, D) - input points
            c: (B, Dc) - global context (latent encoding)
            t: scalar or (B,) or (B, 1) - time
            
        Returns:
            Output with same shape as x
        """
        # Convert to JAX arrays
        x = jnp.asarray(x)
        c = jnp.asarray(c)
        t = jnp.asarray(t)
        
        # Validate context shape
        assert c.ndim == 2, f"Global CRN expects c with shape (B, Dc), got {c.shape}"
        
        B = x.shape[0]
        
        # Normalize time
        if t.ndim == 0:
            t = jnp.broadcast_to(t, (B, 1))
        elif t.ndim == 1:
            t = t[:, None] if t.shape[0] == B else jnp.broadcast_to(t, (B, 1))
        
        # Broadcast c if needed
        if c.shape[0] == 1 and B > 1:
            c = jnp.broadcast_to(c, (B, c.shape[1]))
        
        # Build conditioning vector from global context and time
        time_embed = get_time_embedding(self.time_embed_type, self.time_embed_dim)
        t_feat = time_embed(t)  # (B, time_embed_dim)
        
        # Concatenate time and global context
        cond = jnp.concatenate([t_feat, c], axis=-1)  # (B, time_embed_dim + Dc)
        cond = nn.swish(nn.Dense(self.cond_dim)(cond))
        cond = nn.swish(nn.Dense(self.cond_dim)(cond))  # (B, cond_dim)
        
        # Handle 2D input
        squeeze_output = False
        if x.ndim == 2:
            x = x[:, None, :]
            squeeze_output = True
        
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
        
        # Transformer blocks with AdaLN
        for _ in range(self.num_layers):
            # Self-attention
            h_norm = nn.LayerNorm()(h)
            attn_out = SelfAttention(
                num_heads=self.num_heads,
                head_dim=self.embed_dim // self.num_heads
            )(h_norm)
            
            # AdaLN for attention
            scale_shift = nn.Dense(2 * self.embed_dim)(cond)
            scale, shift = jnp.split(scale_shift, 2, axis=-1)
            scale = scale[:, None, :]
            shift = shift[:, None, :]
            attn_out = attn_out * (1 + scale) + shift
            
            h = h + attn_out
            
            # MLP
            h_norm = nn.LayerNorm()(h)
            mlp_dim = int(self.embed_dim * self.mlp_ratio)
            mlp = nn.Dense(mlp_dim)(h_norm)
            mlp = nn.swish(mlp)
            mlp = nn.Dense(self.embed_dim)(mlp)
            
            # AdaLN for MLP
            scale_shift = nn.Dense(2 * self.embed_dim)(cond)
            scale, shift = jnp.split(scale_shift, 2, axis=-1)
            scale = scale[:, None, :]
            shift = shift[:, None, :]
            mlp = mlp * (1 + scale) + shift
            
            h = h + mlp
        
        # Final layer norm and output projection
        h = nn.LayerNorm()(h)
        dx = nn.Dense(x.shape[-1], kernel_init=nn.initializers.zeros)(h)
        
        if squeeze_output:
            dx = dx[:, 0, :]
        
        return dx


class GlobalCrossAttentionCRN(nn.Module):
    """
    Global CRN using Cross-Attention to latent points with AdaLN.
    
    Expects global context: c has shape (B, Dc).
    Projects global context into M latent points, then each input point cross-attends to these latents.
    
    Args:
        num_latents: Number of latent points to project context into
        latent_dim: Dimension of latent points
        num_heads: Number of attention heads
        num_layers: Number of cross-attention layers
        time_embed_dim: Dimension for time embedding
        cond_dim: Dimension of conditioning vector
        position_embed_type: Type of positional embedding
    """
    num_latents: int = 32
    latent_dim: int = 64
    num_heads: int = 4
    num_layers: int = 3
    time_embed_dim: int = 64
    time_embed_type: str = 'sinusoidal'
    cond_dim: int = 256
    position_embed_type: str = 'linear'
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, c: jnp.ndarray, t: Any) -> jnp.ndarray:
        """
        Args:
            x: (B, N, D) or (B, D) - input points
            c: (B, Dc) - global context (latent encoding)
            t: scalar or (B,) or (B, 1) - time
            
        Returns:
            Output with same shape as x
        """
        # Convert to JAX arrays
        x = jnp.asarray(x)
        c = jnp.asarray(c)
        t = jnp.asarray(t)
        
        # Validate context shape
        assert c.ndim == 2, f"Global CRN expects c with shape (B, Dc), got {c.shape}"
        
        B = x.shape[0]
        
        # Normalize time
        if t.ndim == 0:
            t = jnp.broadcast_to(t, (B, 1))
        elif t.ndim == 1:
            t = t[:, None] if t.shape[0] == B else jnp.broadcast_to(t, (B, 1))
        
        # Broadcast c if needed
        if c.shape[0] == 1 and B > 1:
            c = jnp.broadcast_to(c, (B, c.shape[1]))
        
        # Build conditioning vector from global context and time
        time_embed = get_time_embedding(self.time_embed_type, self.time_embed_dim)
        t_feat = time_embed(t)  # (B, time_embed_dim)
        
        # Concatenate time and global context
        cond = jnp.concatenate([t_feat, c], axis=-1)  # (B, time_embed_dim + Dc)
        cond = nn.swish(nn.Dense(self.cond_dim)(cond))
        cond = nn.swish(nn.Dense(self.cond_dim)(cond))  # (B, cond_dim)
        
        # Project global context into M latent points
        latents = nn.Dense(self.num_latents * self.latent_dim)(cond)  # (B, M*D)
        latents = latents.reshape(B, self.num_latents, self.latent_dim)  # (B, M, D)
        
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
        
        # Cross-attention layers
        for _ in range(self.num_layers):
            # Cross-attention: h (queries) attends to latents (keys/values)
            h_norm = nn.LayerNorm()(h)
            attn_out = CrossAttention(
                num_heads=self.num_heads,
                head_dim=self.latent_dim // self.num_heads
            )(h_norm, latents)
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


class GlobalSimpleConcatCRN(nn.Module):
    """
    Simple Global CRN that concatenates global context and time with input.
    
    Expects global context: c has shape (B, Dc).
    
    Args:
        hidden_dims: Hidden layer dimensions
        time_embed_dim: Dimension for time embedding
        position_embed_type: Type of positional embedding
        activation_fn: Name of activation function
    """
    hidden_dims: Sequence[int] = (64, 64, 64, 64)
    time_embed_dim: int = 32
    time_embed_type: str = 'sinusoidal'
    position_embed_type: str = 'linear'
    activation_fn: str = 'swish'
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, c: jnp.ndarray, t: Any) -> jnp.ndarray:
        """
        Args:
            x: (B, N, D) or (B, D) - input points
            c: (B, Dc) - global context (latent encoding)
            t: scalar or (B,) or (B, 1) - time
            
        Returns:
            Output with same shape as x
        """
        # Convert to JAX arrays
        x = jnp.asarray(x)
        c = jnp.asarray(c)
        t = jnp.asarray(t)
        
        # Validate context shape
        assert c.ndim == 2, f"Global CRN expects c with shape (B, Dc), got {c.shape}"
        
        B = x.shape[0]
        
        # Normalize time
        if t.ndim == 0:
            t = jnp.broadcast_to(t, (B, 1))
        elif t.ndim == 1:
            t = t[:, None] if t.shape[0] == B else jnp.broadcast_to(t, (B, 1))
        
        # Broadcast c if needed
        if c.shape[0] == 1 and B > 1:
            c = jnp.broadcast_to(c, (B, c.shape[1]))
        
        activation = get_activation_fn(self.activation_fn)
        
        # Time embedding
        time_embed = get_time_embedding(self.time_embed_type, self.time_embed_dim)
        t_feat = time_embed(t)  # (B, time_embed_dim)
        
        # Handle 2D input
        squeeze_output = False
        if x.ndim == 2:
            x = x[:, None, :]
            squeeze_output = True
        
        # Positional embedding
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
        
        # Concatenate time and global context, then broadcast to all points
        cond = jnp.concatenate([t_feat, c], axis=-1)  # (B, time_embed_dim + Dc)
        cond = cond[:, None, :]  # (B, 1, time_embed_dim + Dc)
        cond = jnp.broadcast_to(cond, (B, h.shape[1], cond.shape[-1]))  # (B, N, time_embed_dim + Dc)
        
        # Concatenate with h
        h = jnp.concatenate([h, cond], axis=-1)  # (B, N, hidden_dims[0] + time_embed_dim + Dc)
        
        # MLP
        for dim in self.hidden_dims:
            h = nn.Dense(dim)(h)
            h = activation(h)
            h = nn.LayerNorm()(h)
        
        # Output projection
        dx = nn.Dense(x.shape[-1], kernel_init=nn.initializers.zeros)(h)
        
        if squeeze_output:
            dx = dx[:, 0, :]
        
        return dx


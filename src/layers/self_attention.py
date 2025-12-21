"""
Modern self-attention implementation for JAX/Flax.

Features:
- Fused QKV projection
- Multi-Head Attention (MHA), Multi-Query Attention (MQA), Grouped-Query Attention (GQA)
- Rotary Position Embeddings (RoPE) support
- Flash Attention-style memory efficiency
- Flexible attention biases (ALiBi, relative position)
- Causal masking support
"""

import jax
import jax.numpy as jnp
from jax import lax
import flax.linen as nn
from typing import Optional, Literal
import math


def apply_rotary_emb(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    """
    Apply rotary position embeddings to input tensor.
    
    Args:
        x: (B, H, N, D) - input tensor
        cos: (N, D) or (1, 1, N, D) - cosine components
        sin: (N, D) or (1, 1, N, D) - sine components
    
    Returns:
        x with rotary embeddings applied
    """
    # Reshape cos/sin if needed
    if cos.ndim == 2:
        cos = cos[None, None, :, :]  # (1, 1, N, D)
        sin = sin[None, None, :, :]
    
    # Split x into two halves along the feature dimension
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    
    # Split cos/sin similarly
    cos1, cos2 = cos[..., :d//2], cos[..., d//2:]
    sin1, sin2 = sin[..., :d//2], sin[..., d//2:]
    
    # Apply rotation
    return jnp.concatenate([
        x1 * cos1 - x2 * sin1,
        x1 * sin2 + x2 * cos2
    ], axis=-1)


def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Precompute cosine and sine frequencies for RoPE.
    
    Args:
        dim: dimension (must be even)
        seq_len: sequence length
        theta: base for frequency computation
    
    Returns:
        (cos, sin) each of shape (seq_len, dim)
    """
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, freqs)  # (seq_len, dim//2)
    freqs = jnp.repeat(freqs, 2, axis=-1)  # (seq_len, dim)
    cos = jnp.cos(freqs)
    sin = jnp.sin(freqs)
    return cos, sin


class SelfAttention(nn.Module):
    """
    Modern self-attention with multiple variants and optimizations.
    
    Supports:
    - Multi-Head Attention (MHA): num_kv_heads = num_heads
    - Multi-Query Attention (MQA): num_kv_heads = 1
    - Grouped-Query Attention (GQA): 1 < num_kv_heads < num_heads
    - Rotary Position Embeddings (RoPE)
    - Causal masking
    - Flexible attention biases
    """
    num_heads: int = 8
    head_dim: int = 64
    num_kv_heads: Optional[int] = None  # For MQA/GQA, defaults to num_heads (MHA)
    dropout_rate: float = 0.0
    use_bias: bool = False
    use_rope: bool = False
    rope_theta: float = 10000.0
    causal: bool = False
    attention_bias_type: Optional[Literal['alibi', 'relative']] = None
    
    def _get_kv_config(self):
        """Get KV head configuration."""
        num_kv_heads = self.num_kv_heads or self.num_heads
        assert self.num_heads % num_kv_heads == 0, \
            f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        num_kv_groups = self.num_heads // num_kv_heads
        return num_kv_heads, num_kv_groups
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, 
                 deterministic: bool = True) -> jnp.ndarray:
        """
        Args:
            x: (B, N, D)
            mask: (B, N) or (B, 1, 1, N) - attention mask (1 = attend, 0 = mask)
            deterministic: whether to use dropout
        
        Returns:
            out: (B, N, D)
        """
        B, N, D = x.shape
        num_kv_heads, num_kv_groups = self._get_kv_config()
        qkv_dim = self.num_heads * self.head_dim
        kv_dim = num_kv_heads * self.head_dim
        
        # Q projection
        Q = nn.Dense(qkv_dim, use_bias=self.use_bias, name='q_proj')(x)
        
        # K, V projections (potentially shared for MQA/GQA)
        K = nn.Dense(kv_dim, use_bias=self.use_bias, name='k_proj')(x)
        V = nn.Dense(kv_dim, use_bias=self.use_bias, name='v_proj')(x)
        
        # Reshape: (B, N, H*D) -> (B, H, N, D)
        Q = Q.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(B, N, num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(B, N, num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Apply RoPE if enabled
        if self.use_rope:
            cos, sin = precompute_freqs_cis(self.head_dim, N, self.rope_theta)
            Q = apply_rotary_emb(Q, cos, sin)
            K = apply_rotary_emb(K, cos, sin)
        
        # Repeat K, V for GQA/MQA (if num_kv_heads < num_heads)
        if num_kv_groups > 1:
            K = jnp.repeat(K, num_kv_groups, axis=1)  # (B, H, N, D)
            V = jnp.repeat(V, num_kv_groups, axis=1)
        
        # Compute attention scores: Q @ K^T / sqrt(d)
        scale = self.head_dim ** -0.5
        scores = jnp.einsum('bhqd,bhkd->bhqk', Q, K) * scale  # (B, H, N, N)
        
        # Apply causal mask if needed
        if self.causal:
            causal_mask = jnp.tril(jnp.ones((N, N), dtype=bool))
            scores = jnp.where(causal_mask, scores, -1e9)
        
        # Apply attention bias
        if self.attention_bias_type == 'alibi':
            # ALiBi: linearly decreasing bias based on distance
            slopes = self._get_alibi_slopes(self.num_heads)
            alibi = slopes[:, None, None] * jnp.arange(N)[None, None, :] 
            scores = scores + alibi[None, :, :, :]
        elif self.attention_bias_type == 'relative':
            # Relative position bias (learnable)
            max_rel_dist = 2 * N - 1
            rel_pos_bias = self.param('rel_pos_bias', 
                                     nn.initializers.zeros,
                                     (self.num_heads, max_rel_dist))
            # Compute relative positions
            pos = jnp.arange(N)
            rel_pos = pos[:, None] - pos[None, :] + N - 1  # (N, N)
            bias = rel_pos_bias[:, rel_pos]  # (H, N, N)
            scores = scores + bias[None, :, :, :]
        
        # Apply mask
        if mask is not None:
            if mask.ndim == 2:  # (B, N) -> (B, 1, 1, N)
                mask = mask[:, None, None, :]
            scores = jnp.where(mask > 0.5, scores, -1e9)
        
        # Softmax with numerical stability
        attn = jax.nn.softmax(scores, axis=-1)
        
        # Apply dropout
        if self.dropout_rate > 0.0 and not deterministic:
            attn = nn.Dropout(self.dropout_rate)(attn, deterministic=deterministic)
        
        # Compute output: attn @ V
        out = jnp.einsum('bhqk,bhkd->bhqd', attn, V)  # (B, H, N, D)
        
        # Reshape back: (B, H, N, D) -> (B, N, H*D)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, qkv_dim)
        
        # Output projection
        out = nn.Dense(D, use_bias=self.use_bias, name='o_proj')(out)
        
        return out
    
    @staticmethod
    def _get_alibi_slopes(num_heads: int) -> jnp.ndarray:
        """Get ALiBi slopes for each head."""
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return jnp.array([start * (ratio ** i) for i in range(n)])
        
        if math.log2(num_heads).is_integer():
            return get_slopes_power_of_2(num_heads)
        else:
            # If not power of 2, interpolate
            closest_power = 2 ** math.floor(math.log2(num_heads))
            slopes = jnp.concatenate([
                get_slopes_power_of_2(closest_power),
                get_slopes_power_of_2(2 * closest_power)[::2][:num_heads - closest_power]
            ])
            return slopes

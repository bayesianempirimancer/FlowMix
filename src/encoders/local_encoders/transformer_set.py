"""
Modern Transformer Set Encoder - outputs sequence of features (B, N+1, embed_dim).
Includes CLS token for global context.

Modern features:
- Pre-normalization (Pre-LN) for better training stability
- SwiGLU activation for better expressiveness
- Configurable attention (MQA/GQA/RoPE support)
- Stochastic depth for regularization
- Layer scaling for deep networks
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Literal
from src.layers.self_attention import SelfAttention


class SwiGLU(nn.Module):
    """
    SwiGLU activation: Swish-Gated Linear Unit.
    Used in modern transformers (PaLM, LLaMA, etc.)
    
    Output = Swish(W1 @ x) ⊙ (W2 @ x)
    """
    dim: int
    hidden_dim: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gate = nn.Dense(self.hidden_dim, name='gate')(x)
        value = nn.Dense(self.hidden_dim, name='value')(x)
        hidden = nn.swish(gate) * value
        return nn.Dense(self.dim, name='proj')(hidden)


class TransformerBlock(nn.Module):
    """
    Modern transformer block with Pre-LN, SwiGLU, and optional features.
    """
    embed_dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    num_kv_heads: Optional[int] = None
    use_rope: bool = False
    dropout_rate: float = 0.0
    drop_path_rate: float = 0.0
    use_swiglu: bool = True
    layer_scale_init: Optional[float] = None
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None,
                 deterministic: bool = True) -> jnp.ndarray:
        """
        Args:
            x: (B, N, D)
            mask: attention mask
            deterministic: whether to use dropout
        
        Returns:
            x: (B, N, D)
        """
        head_dim = self.embed_dim // self.num_heads
        
        # Pre-LN + Self-Attention
        attn_out = SelfAttention(
            num_heads=self.num_heads,
            head_dim=head_dim,
            num_kv_heads=self.num_kv_heads,
            use_rope=self.use_rope,
            dropout_rate=self.dropout_rate,
        )(nn.LayerNorm()(x), mask=mask, deterministic=deterministic)
        
        # Layer scale for attention
        if self.layer_scale_init is not None:
            scale = self.param('attn_scale', 
                             lambda rng, shape: jnp.full(shape, self.layer_scale_init),
                             (self.embed_dim,))
            attn_out = attn_out * scale
        
        # Stochastic depth for attention
        if self.drop_path_rate > 0.0 and not deterministic:
            keep_prob = 1.0 - self.drop_path_rate
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = jax.random.bernoulli(
                self.make_rng('dropout'), keep_prob, shape
            )
            attn_out = attn_out * random_tensor / keep_prob
        
        x = x + attn_out
        
        # Pre-LN + MLP
        mlp_dim = int(self.embed_dim * self.mlp_ratio)
        if self.use_swiglu:
            # SwiGLU: more expressive, used in modern LLMs
            mlp_out = SwiGLU(dim=self.embed_dim, hidden_dim=mlp_dim)(nn.LayerNorm()(x))
        else:
            # Standard MLP
            h = nn.LayerNorm()(x)
            h = nn.Dense(mlp_dim)(h)
            h = nn.swish(h)
            if self.dropout_rate > 0.0 and not deterministic:
                h = nn.Dropout(self.dropout_rate)(h, deterministic=deterministic)
            mlp_out = nn.Dense(self.embed_dim)(h)
        
        # Layer scale for MLP
        if self.layer_scale_init is not None:
            scale = self.param('mlp_scale',
                             lambda rng, shape: jnp.full(shape, self.layer_scale_init),
                             (self.embed_dim,))
            mlp_out = mlp_out * scale
        
        # Stochastic depth for MLP
        if self.drop_path_rate > 0.0 and not deterministic:
            keep_prob = 1.0 - self.drop_path_rate
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = jax.random.bernoulli(
                self.make_rng('dropout'), keep_prob, shape
            )
            mlp_out = mlp_out * random_tensor / keep_prob
        
        x = x + mlp_out
        
        return x


class TransformerSetEncoder(nn.Module):
    """
    Modern Transformer that outputs a sequence of features (Set-to-Set).
    Outputs (B, N+1, embed_dim) where the first token is a CLS token.
    
    Modern features:
    - Pre-normalization for better training stability
    - SwiGLU activation (optional)
    - Multi-Query/Grouped-Query Attention support
    - Rotary Position Embeddings (optional)
    - Stochastic depth regularization
    - Layer scaling for deep networks
    """
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 3
    mlp_ratio: float = 4.0
    num_kv_heads: Optional[int] = None  # For MQA/GQA
    use_rope: bool = False
    dropout_rate: float = 0.0
    drop_path_rate: float = 0.0
    use_swiglu: bool = True
    layer_scale_init: Optional[float] = None  # e.g., 1e-4 for deep networks
    
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
            (B, N+1, embed_dim) - sequence with CLS token
        """
        B, N, D = x.shape
        
        # Input projection
        h = nn.Dense(self.embed_dim)(x)
        
        # CLS token
        cls_token = self.param('cls_token', nn.initializers.normal(0.02), 
                              (1, 1, self.embed_dim))
        cls_token = jnp.tile(cls_token, (B, 1, 1))
        
        # Prepend CLS token
        h = jnp.concatenate([cls_token, h], axis=1)  # (B, N+1, embed_dim)
        
        # Handle mask
        if mask is not None:
            # mask is (B, N). CLS token is always valid.
            cls_mask = jnp.ones((B, 1), dtype=mask.dtype)
            full_mask = jnp.concatenate([cls_mask, mask], axis=1)  # (B, N+1)
            attn_mask = full_mask[:, None, None, :] > 0.5
        else:
            attn_mask = None
            
        # Stochastic depth: linearly increasing drop rate
        dpr = [self.drop_path_rate * i / max(self.num_layers - 1, 1) 
               for i in range(self.num_layers)]
        
        # Transformer blocks
        for i in range(self.num_layers):
            h = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                num_kv_heads=self.num_kv_heads,
                use_rope=self.use_rope,
                dropout_rate=self.dropout_rate,
                drop_path_rate=dpr[i],
                use_swiglu=self.use_swiglu,
                layer_scale_init=self.layer_scale_init,
                name=f'block_{i}'
            )(h, mask=attn_mask, deterministic=deterministic)
        
        # Final layer norm (Pre-LN style)
        h = nn.LayerNorm()(h)
        
        return h

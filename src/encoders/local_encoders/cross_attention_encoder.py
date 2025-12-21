"""
Modern Cross-Attention Encoder (Perceiver-style) - outputs sequence of latents (B, M, latent_dim).

Modern features:
- Pre-normalization for better training stability
- SwiGLU activation for better expressiveness
- Configurable attention (MQA/GQA support)
- Self-attention between latents (Perceiver AR)
- Stochastic depth for regularization
- Layer scaling for deep networks
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional
from src.layers.cross_attention import CrossAttention
from src.layers.self_attention import SelfAttention


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


class PerceiverBlock(nn.Module):
    """
    Perceiver block: Cross-attention from latents to inputs + Self-attention among latents.
    """
    latent_dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    num_kv_heads: Optional[int] = None
    use_self_attn: bool = True  # Perceiver AR style
    dropout_rate: float = 0.0
    drop_path_rate: float = 0.0
    use_swiglu: bool = True
    layer_scale_init: Optional[float] = None
    
    @nn.compact
    def __call__(self, latents: jnp.ndarray, x: jnp.ndarray,
                 mask: Optional[jnp.ndarray] = None,
                 deterministic: bool = True) -> jnp.ndarray:
        """
        Args:
            latents: (B, M, D) - latent queries
            x: (B, N, D) - input keys/values
            mask: (B, N) - mask for inputs
            deterministic: whether to use dropout
        
        Returns:
            latents: (B, M, D)
        """
        head_dim = self.latent_dim // self.num_heads
        
        # Pre-LN + Cross-Attention: latents attend to inputs
        cross_attn_out = CrossAttention(
            num_heads=self.num_heads,
            head_dim=head_dim,
            num_kv_heads=self.num_kv_heads,
            dropout_rate=self.dropout_rate,
        )(nn.LayerNorm()(latents), x, mask=mask, deterministic=deterministic)
        
        # Layer scale
        if self.layer_scale_init is not None:
            scale = self.param('cross_attn_scale',
                             lambda rng, shape: jnp.full(shape, self.layer_scale_init),
                             (self.latent_dim,))
            cross_attn_out = cross_attn_out * scale
        
        # Stochastic depth
        if self.drop_path_rate > 0.0 and not deterministic:
            keep_prob = 1.0 - self.drop_path_rate
            shape = (latents.shape[0],) + (1,) * (latents.ndim - 1)
            random_tensor = jax.random.bernoulli(
                self.make_rng('dropout'), keep_prob, shape
            )
            cross_attn_out = cross_attn_out * random_tensor / keep_prob
        
        latents = latents + cross_attn_out
        
        # Pre-LN + Self-Attention among latents (Perceiver AR)
        if self.use_self_attn:
            self_attn_out = SelfAttention(
                num_heads=self.num_heads,
                head_dim=head_dim,
                num_kv_heads=self.num_kv_heads,
                dropout_rate=self.dropout_rate,
            )(nn.LayerNorm()(latents), deterministic=deterministic)
            
            # Layer scale
            if self.layer_scale_init is not None:
                scale = self.param('self_attn_scale',
                                 lambda rng, shape: jnp.full(shape, self.layer_scale_init),
                                 (self.latent_dim,))
                self_attn_out = self_attn_out * scale
            
            # Stochastic depth
            if self.drop_path_rate > 0.0 and not deterministic:
                keep_prob = 1.0 - self.drop_path_rate
                shape = (latents.shape[0],) + (1,) * (latents.ndim - 1)
                random_tensor = jax.random.bernoulli(
                    self.make_rng('dropout'), keep_prob, shape
                )
                self_attn_out = self_attn_out * random_tensor / keep_prob
            
            latents = latents + self_attn_out
        
        # Pre-LN + MLP
        mlp_dim = int(self.latent_dim * self.mlp_ratio)
        if self.use_swiglu:
            mlp_out = SwiGLU(dim=self.latent_dim, hidden_dim=mlp_dim)(nn.LayerNorm()(latents))
        else:
            h = nn.LayerNorm()(latents)
            h = nn.Dense(mlp_dim)(h)
            h = nn.swish(h)
            if self.dropout_rate > 0.0 and not deterministic:
                h = nn.Dropout(self.dropout_rate)(h, deterministic=deterministic)
            mlp_out = nn.Dense(self.latent_dim)(h)
        
        # Layer scale
        if self.layer_scale_init is not None:
            scale = self.param('mlp_scale',
                             lambda rng, shape: jnp.full(shape, self.layer_scale_init),
                             (self.latent_dim,))
            mlp_out = mlp_out * scale
        
        # Stochastic depth
        if self.drop_path_rate > 0.0 and not deterministic:
            keep_prob = 1.0 - self.drop_path_rate
            shape = (latents.shape[0],) + (1,) * (latents.ndim - 1)
            random_tensor = jax.random.bernoulli(
                self.make_rng('dropout'), keep_prob, shape
            )
            mlp_out = mlp_out * random_tensor / keep_prob
        
        latents = latents + mlp_out
        
        return latents


class CrossAttentionEncoder(nn.Module):
    """
    Modern Perceiver-style encoder: learns M latent points that cross-attend to N input points.
    Outputs (B, M, latent_dim) where M << N.
    
    Complexity: O(M × N) instead of O(N²) for self-attention.
    
    Modern features:
    - Pre-normalization for better training stability
    - SwiGLU activation (optional)
    - Self-attention among latents (Perceiver AR style)
    - Multi-Query/Grouped-Query Attention support
    - Stochastic depth regularization
    - Layer scaling for deep networks
    """
    num_latents: int = 32
    latent_dim: int = 64
    num_heads: int = 4
    num_layers: int = 3
    mlp_ratio: float = 4.0
    num_kv_heads: Optional[int] = None  # For MQA/GQA
    use_self_attn: bool = True  # Perceiver AR: self-attention among latents
    dropout_rate: float = 0.0
    drop_path_rate: float = 0.0
    use_swiglu: bool = True
    layer_scale_init: Optional[float] = None
    
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
            (B, M, latent_dim) - compressed latent representation
        """
        B, N, D = x.shape
        
        # Learnable latent queries
        latents = self.param('latents', nn.initializers.normal(0.02),
                            (1, self.num_latents, self.latent_dim))
        latents = jnp.broadcast_to(latents, (B, self.num_latents, self.latent_dim))
        
        # Embed input points to latent_dim
        x_embed = nn.Dense(self.latent_dim)(x)  # (B, N, latent_dim)
        
        # Stochastic depth: linearly increasing drop rate
        dpr = [self.drop_path_rate * i / max(self.num_layers - 1, 1)
               for i in range(self.num_layers)]
        
        # Perceiver blocks
        for i in range(self.num_layers):
            latents = PerceiverBlock(
                latent_dim=self.latent_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                num_kv_heads=self.num_kv_heads,
                use_self_attn=self.use_self_attn,
                dropout_rate=self.dropout_rate,
                drop_path_rate=dpr[i],
                use_swiglu=self.use_swiglu,
                layer_scale_init=self.layer_scale_init,
                name=f'block_{i}'
            )(latents, x_embed, mask=mask, deterministic=deterministic)
        
        # Final layer norm (Pre-LN style)
        latents = nn.LayerNorm()(latents)
        
        return latents

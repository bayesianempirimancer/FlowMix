"""
Pooling strategies for converting local encoders (B, N, D) to global encoders (B, D).
These wrappers can be applied to ANY local encoder.

Usage:
    local_encoder = DGCNN(embed_dim=64)
    global_encoder = MaxPoolingEncoder(local_encoder=local_encoder, latent_dim=128)
    z = global_encoder(x, mask=mask, key=key)  # (B, latent_dim)
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional


class MeanPoolingEncoder(nn.Module):
    """
    Global encoder using mean pooling over local features.
    Wraps any local encoder and applies mean pooling.
    """
    local_encoder: nn.Module
    latent_dim: int = 128
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        # Apply local encoder: (B, K, embed_dim) where K may != N
        # The local encoder handles masking internally
        local_features = self.local_encoder(x, mask=mask, key=key)
        
        # Mean pooling over K dimension (no mask needed here)
        global_feat = jnp.mean(local_features, axis=1)  # (B, embed_dim)
        
        # Project to latent dimension
        z = nn.Dense(self.latent_dim)(global_feat)
        
        return z  # (B, latent_dim)


class MaxPoolingEncoder(nn.Module):
    """
    Global encoder using max pooling over local features.
    Wraps any local encoder and applies max pooling.
    """
    local_encoder: nn.Module
    latent_dim: int = 128
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        # Apply local encoder: (B, K, embed_dim) where K may != N
        # The local encoder handles masking internally
        local_features = self.local_encoder(x, mask=mask, key=key)
        
        # Max pooling over K dimension (no mask needed here)
        global_feat = jnp.max(local_features, axis=1)  # (B, embed_dim)
        
        # Project to latent dimension
        z = nn.Dense(self.latent_dim)(global_feat)
        
        return z  # (B, latent_dim)


class AttentionPoolingEncoder(nn.Module):
    """
    Global encoder using attention-based pooling over local features.
    Learns to attend to important local features.
    """
    local_encoder: nn.Module
    latent_dim: int = 128
    attention_dim: int = 64
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        # Apply local encoder: (B, K, embed_dim) where K may != N
        # The local encoder handles masking internally
        local_features = self.local_encoder(x, mask=mask, key=key)
        B, K, embed_dim = local_features.shape
        
        # Learnable query for attention
        query = self.param('query', nn.initializers.xavier_uniform(), (1, 1, self.attention_dim))
        query = jnp.tile(query, (B, 1, 1))  # (B, 1, attention_dim)
        
        # Project local features to attention space
        keys = nn.Dense(self.attention_dim)(local_features)  # (B, K, attention_dim)
        values = local_features  # (B, K, embed_dim)
        
        # Compute attention scores
        scores = jnp.sum(query * keys, axis=-1)  # (B, K)
        
        # Softmax (no mask needed - local encoder already handled it)
        attn_weights = nn.softmax(scores, axis=-1)  # (B, K)
        
        # Weighted sum
        global_feat = jnp.sum(values * attn_weights[:, :, None], axis=1)  # (B, embed_dim)
        
        # Project to latent dimension
        z = nn.Dense(self.latent_dim)(global_feat)
        
        return z  # (B, latent_dim)


class Set2SetPoolingEncoder(nn.Module):
    """
    Global encoder using Set2Set pooling (LSTM with attention).
    Wraps any local encoder and applies Set2Set pooling.
    """
    local_encoder: nn.Module
    latent_dim: int = 128
    num_steps: int = 3
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        # Apply local encoder: (B, K, embed_dim) where K may != N
        # The local encoder handles masking internally
        local_features = self.local_encoder(x, mask=mask, key=key)
        B, K, embed_dim = local_features.shape
        
        # LSTM cell
        lstm = nn.LSTMCell(features=embed_dim)
        
        # Initialize hidden and cell states
        h = jnp.zeros((B, embed_dim))
        c = jnp.zeros((B, embed_dim))
        
        # Process for num_steps
        for step in range(self.num_steps):
            # Attention over local features
            q = nn.Dense(embed_dim)(h)  # (B, embed_dim)
            
            # Compute attention scores: (B, K)
            scores = jnp.sum(local_features * q[:, None, :], axis=-1)
            
            # Softmax (no mask needed - local encoder already handled it)
            attn_weights = nn.softmax(scores, axis=-1)  # (B, K)
            
            # Weighted sum
            context = jnp.sum(local_features * attn_weights[:, :, None], axis=1)  # (B, embed_dim)
            
            # LSTM step
            lstm_input = jnp.concatenate([h, context], axis=-1)  # (B, 2*embed_dim)
            h, c = lstm((h, c), lstm_input)
        
        # Use final hidden state as global feature
        global_feat = h  # (B, embed_dim)
        
        # Project to latent dimension
        z = nn.Dense(self.latent_dim)(global_feat)
        
        return z  # (B, latent_dim)


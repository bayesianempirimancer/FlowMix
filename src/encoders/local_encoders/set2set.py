"""
Set2Set - outputs sequence of features (B, T, embed_dim).
Uses LSTM with attention to process sets, outputting a sequence of hidden states.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional


class Set2Set(nn.Module):
    """
    Set2Set encoder using LSTM with attention mechanism.
    Outputs sequence of hidden states (B, T, embed_dim) where T is the number of processing steps.
    """
    embed_dim: int = 64
    num_steps: int = 3  # Number of processing steps
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        # x: (B, N, D)
        B, N, D = x.shape
        
        # Project input to embed_dim
        x_embed = nn.Dense(self.embed_dim)(x)  # (B, N, embed_dim)
        
        # LSTM cell
        lstm = nn.LSTMCell(features=self.embed_dim)
        
        # Initialize hidden and cell states
        h = jnp.zeros((B, self.embed_dim))
        c = jnp.zeros((B, self.embed_dim))
        
        outputs = []
        
        # Process for num_steps
        for step in range(self.num_steps):
            # Attention over input set
            # Query from hidden state
            q = nn.Dense(self.embed_dim)(h)  # (B, embed_dim)
            
            # Compute attention scores: (B, N)
            scores = jnp.sum(x_embed * q[:, None, :], axis=-1)  # (B, N)
            
            # Apply mask if provided
            if mask is not None:
                scores = jnp.where(mask > 0.5, scores, -1e9)
            
            # Softmax
            attn_weights = nn.softmax(scores, axis=-1)  # (B, N)
            
            # Weighted sum: (B, embed_dim)
            context = jnp.sum(x_embed * attn_weights[:, :, None], axis=1)
            
            # Concatenate context with hidden state
            lstm_input = jnp.concatenate([h, context], axis=-1)  # (B, 2*embed_dim)
            
            # LSTM step
            h, c = lstm((h, c), lstm_input)
            
            outputs.append(h)
        
        # Stack outputs: (B, num_steps, embed_dim)
        output = jnp.stack(outputs, axis=1)
        
        return output


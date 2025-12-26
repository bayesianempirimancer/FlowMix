"""
Global GMFlow CRN.

This module implements a Global CRN that predicts parameters for a Gaussian Mixture Model (GMM)
instead of a single velocity vector. This allows the flow matching model to capture multi-modal
velocity distributions.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Sequence, Optional, Callable, Dict

from src.models.global_crn import get_activation_fn, get_time_embedding, GlobalAdaLNMLPCRN

class GlobalGMFlowCRN(nn.Module):
    """
    Global CRN for GMFlow that predicts Gaussian Mixture parameters.
    
    Instead of predicting a single vector v, it predicts:
    - Logits (mixing coefficients) for K components
    - Means for K components
    - Log-variances (diagonal) for K components
    
    Args:
        num_components: Number of Gaussian components (K)
        hidden_dims: Hidden layer dimensions
        time_embed_dim: Dimension for time embedding
        cond_dim: Dimension of conditioning vector
        dropout_rate: Dropout rate (not used in current MLP implementation but kept for API consistency)
        activation_fn: Name of activation function
    """
    num_components: int = 4
    hidden_dims: Sequence[int] = (64, 64, 64, 64, 64, 64)
    time_embed_dim: int = 32
    time_embed_type: str = 'sinusoidal'
    cond_dim: int = 256
    position_embed_type: str = 'linear'
    activation_fn: str = 'swish'
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, c: jnp.ndarray, t: Any) -> Dict[str, jnp.ndarray]:
        """
        Args:
            x: (B, N, D) or (B, D) - input points
            c: (B, Dc) - global context (latent encoding)
            t: scalar or (B,) or (B, 1) - time
            
        Returns:
            Dictionary containing:
            - 'logits': (B, N, K) or (B, K) - Mixture logits
            - 'means': (B, N, K, D) or (B, K, D) - Component means
            - 'logvars': (B, N, K, D) or (B, K, D) - Component log-variances
        """
        # Reuse GlobalAdaLNMLPCRN architecture for feature extraction
        # We'll use it as a backbone, but we need to intercept the output projection
        
        # 1. Feature Extraction (same as GlobalAdaLNMLPCRN until the end)
        B = x.shape[0]
        # Handle 2D input (add spatial dimension for processing, remove later)
        x_in = x
        if x.ndim == 2:
            x_in = x[:, None, :] # (B, 1, D)
        
        spatial_dim = x.shape[-1]
        
        # We can reuse the backbone logic by composing or copying. 
        # For flexibility and to ensure access to internal features, let's reimplement the backbone here
        # (It's lightweight enough).
        
        # --- Backbone Start ---
        x_arr = jnp.asarray(x_in)
        c_arr = jnp.asarray(c)
        t_arr = jnp.asarray(t)
        
        # Validate context
        assert c_arr.ndim == 2, f"GlobalGMFlowCRN expects c with shape (B, Dc), got {c_arr.shape}"
        
        # Normalize time
        if t_arr.ndim == 0:
            t_arr = jnp.broadcast_to(t_arr, (B, 1))
        elif t_arr.ndim == 1:
            t_arr = t_arr[:, None] if t_arr.shape[0] == B else jnp.broadcast_to(t_arr, (B, 1))
            
         # Broadcast c if needed
        if c_arr.shape[0] == 1 and B > 1:
            c_arr = jnp.broadcast_to(c_arr, (B, c_arr.shape[1]))
            
        activation = get_activation_fn(self.activation_fn)
        
        # Time embedding
        time_embed = get_time_embedding(self.time_embed_type, self.time_embed_dim)
        t_feat = time_embed(t_arr) # (B, time_embed_dim)
        
        # Conditioning vector
        # cond = jnp.concatenate([t_feat, c_arr], axis=-1)
        cond = nn.Dense(self.cond_dim)(c_arr) 
        cond = cond + nn.Dense(self.cond_dim, use_bias=False)(t_feat)
        cond = activation(cond)
        cond = activation(nn.Dense(self.cond_dim)(cond)) # (B, cond_dim)
        
        # Input embedding
        if self.position_embed_type == 'linear':
            h = nn.Dense(self.hidden_dims[0])(x_arr)
        else:
            # Fallback for complex embeddings if needed, essentially copy from GlobalAdaLNMLPCRN
            # For brevity assuming linear/sinusoidal logic exists in imported module or basic Dense is fine
            # Reusing the exact logic from GlobalAdaLNMLPCRN for consistency
            from src.encoders.embeddings import PositionalEmbedding2D
            if self.position_embed_type == 'sinusoidal':
                pos_embed = PositionalEmbedding2D(dim=self.hidden_dims[0])
                h = pos_embed(x_arr)
            elif self.position_embed_type == 'fourier':
                 pos_embed = PositionalEmbedding2D(dim=self.hidden_dims[0], fourier=True)
                 h = pos_embed(x_arr)
            else:
                 h = nn.Dense(self.hidden_dims[0])(x_arr) # Fallback/Linear expectation

        h = activation(h)
        
        # MLP Blocks with AdaLN
        for dim in self.hidden_dims:
            h = nn.Dense(dim)(h)
            h = activation(h)
            
            # AdaLN
            scale_shift = nn.Dense(2 * dim)(cond)
            scale, shift = jnp.split(scale_shift, 2, axis=-1)
            
            scale = scale[:, None, :]
            shift = shift[:, None, :]
            
            h = nn.LayerNorm()(h)
            h = h * (1 + scale) + shift
            
        # --- Backbone End ---
        
        # 2. Heads for GMM parameters
        # We need to predict:
        # - Pi (logits): K values (mixing weights)
        # - Mu: K * D values (means)
        # - Sigma (logvar): K * D values (variances)
        
        # Shared projection or separate heads? Typically separate heads from the shared feature `h`.
        
        # Logits head: (B, N, K)
        logits = nn.Dense(self.num_components)(h)
        
        # Means head: (B, N, K * D) -> reshape to (B, N, K, D)
        means_flat = nn.Dense(self.num_components * spatial_dim)(h)
        means = means_flat.reshape(B, -1, self.num_components, spatial_dim)
        
        # Logvars: shared across K and dimension D. Only depends on cond.
        # Initialize small/negative for stability if possible, but Dense default is 0 bias.
        logvars = nn.Dense(1)(cond) # (B, 1)
        logvars = logvars[:, None, None, :] # (B, 1, 1, 1)
        
        # Collapse spatial dimension if input was 2D (N=1)
        if x.ndim == 2:
            logits = logits[:, 0, :]       # (B, K)
            means = means[:, 0, :, :]      # (B, K, D)
            logvars = logvars[:, 0, :, :]  # (B, 1, 1) -> (B, 1, 1) broadcasts to (B, K, D)
            
        return {
            'logits': logits,
            'means': means,
            'logvars': logvars
        }

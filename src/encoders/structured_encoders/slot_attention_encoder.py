"""
Slot Attention Encoder - outputs sequence of slots (B, K, slot_dim).
"""
import jax
import jax.numpy as jnp
from jax import lax
import flax.linen as nn
from typing import Optional


class SlotAttentionEncoder(nn.Module):
    """
    Slot Attention encoder that outputs a sequence of slots.
    
    Args:
        num_slots: Number of slots K
        slot_dim: Dimension of each slot
        iters: Number of attention iterations
        hidden_dim: Hidden dimension for MLP
        epsilon: Small constant for numerical stability
        init_type: Slot initialization ('fps', 'kmeans', 'learned')
    
    Returns:
        (B, K, slot_dim) - Sequence of slot representations
    """
    num_slots: int = 8
    slot_dim: int = 64
    iters: int = 3
    hidden_dim: int = 128
    epsilon: float = 1e-8
    init_type: str = 'fps' # 'fps', 'kmeans', 'learned'
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, 
                 key: Optional[jax.random.PRNGKey] = None):
        # x: (B, N, D)
        B, N, D = x.shape
        
        # 1. Initialize Slots
        if self.init_type in ['fps', 'kmeans']:
            # Data-dependent initialization
            if key is None:
                key = jax.random.PRNGKey(11)
                
            def get_init_centroids(points, k, rng, strategy):
                # points: (N, D)
                # k: int
                
                # 1. FPS Initialization
                # Random first point
                idx_0 = jax.random.randint(rng, (), 0, points.shape[0])
                m_0 = points[idx_0]
                
                means = jnp.zeros((k, points.shape[-1]))
                means = means.at[0].set(m_0)
                
                dists = jnp.sum((points - m_0)**2, axis=-1)
                
                def fps_step(carry, i):
                    curr_means, curr_dists = carry
                    idx_new = jnp.argmax(curr_dists)
                    new_pt = points[idx_new]
                    # Dynamic update
                    curr_means = jax.lax.dynamic_update_slice(curr_means, new_pt[None, :], (i, 0))
                    new_dists = jnp.sum((points - new_pt)**2, axis=-1)
                    curr_dists = jnp.minimum(curr_dists, new_dists)
                    return (curr_means, curr_dists), None
                    
                # Run FPS
                init_val = (means, dists)
                (means, _), _ = jax.lax.scan(fps_step, init_val, jnp.arange(1, k))
                
                if strategy == 'kmeans':
                    # 2. K-Means Refinement (Simple Lloyd's)
                    def step(carry, _):
                        m = carry
                        # Assign
                        d = jnp.sum((points[:, None, :] - m[None, :, :])**2, axis=-1)
                        assign = jnp.argmin(d, axis=1)
                        one_hot = jax.nn.one_hot(assign, k)
                        # Update
                        cnts = jnp.sum(one_hot, axis=0)
                        sums = jnp.matmul(one_hot.T, points)
                        new_m = sums / (cnts[:, None] + 1e-10)
                        # Handle empty clusters (stay same)
                        new_m = jnp.where(cnts[:, None] > 0, new_m, m)
                        return new_m, None
                        
                    means, _ = jax.lax.scan(step, means, None, length=5) # 5 iters
                
                return means

            # vmap over batch
            keys = jax.random.split(key, B)
            centroids = jax.vmap(get_init_centroids, in_axes=(0, None, 0, None))(
                x, self.num_slots, keys, self.init_type
            ) # (B, K, D)
            
            # Project centroids to slot_dim
            # This aligns the spatial centroid with the slot latent space
            slots = nn.Dense(self.slot_dim)(centroids) # (B, K, slot_dim)
            
        elif self.init_type == 'learned':
            # Mu and Sigma for slot initialization
            mu = self.param('slot_mu', nn.initializers.xavier_uniform(), (1, 1, self.slot_dim))
            log_sigma = self.param('slot_log_sigma', nn.initializers.xavier_uniform(), (1, 1, self.slot_dim))
            
            mu = jnp.tile(mu, (B, self.num_slots, 1))
            sigma = jnp.exp(log_sigma)
            sigma = jnp.tile(sigma, (B, self.num_slots, 1))
            
            # Random initialization
            if key is None:
                key = jax.random.PRNGKey(0)
                
            slots = mu + sigma * jax.random.normal(key, mu.shape)
        else:
            raise ValueError(f"Unknown init_type: {self.init_type}")
        
        # 2. Project Inputs to Keys/Values
        # inputs: x (B, N, D)
        # We use LayerNorm first
        x_norm = nn.LayerNorm()(x)
        
        # Fused KV projection
        kv_proj = nn.Dense(self.slot_dim * 2, use_bias=False)(x_norm)  # (B, N, 2*slot_dim)
        k, v = jnp.split(kv_proj, 2, axis=-1)  # Each: (B, N, slot_dim)
        
        # GRU for updates
        gru = nn.GRUCell(features=self.slot_dim)
        
        # Masking preparation
        if mask is not None:
            # mask: (B, N)
            # We want to ignore attention to masked inputs.
            # Attention logits shape: (B, K, N)
            # Mask shape for addition: (B, 1, N)
            attn_mask_bias = jnp.where(mask[:, None, :], 0.0, -1e9)
        else:
            attn_mask_bias = 0.0
            
        # 3. Iterative Updates
        for _ in range(self.iters):
            slots_prev = slots
            slots_ln = nn.LayerNorm()(slots)
            
            # Query
            q = nn.Dense(self.slot_dim, use_bias=False)(slots_ln) # (B, K, slot_dim)
            
            # Attention: Softmax(Q @ K.T / sqrt(d))
            # q: (B, K, D), k: (B, N, D) -> (B, K, N)
            # Use dot_general for better optimization
            # Contract D dim (axis 2), batch B dim (axis 0)
            dots = lax.dot_general(
                q, k,
                (((2,), (2,)), ((0,), (0,))),  # Contract D, batch B
                precision=lax.Precision.DEFAULT
            ) * (self.slot_dim ** -0.5)
            
            # Apply mask
            dots = dots + attn_mask_bias
            
            attn = jax.nn.softmax(dots, axis=-1) # (B, K, N)
            
            # Weighted sum of V using dot_general
            # attn: (B, K, N), v: (B, N, D) -> (B, K, D)
            # Contract N dim, batch B dim
            updates = lax.dot_general(
                attn + self.epsilon, v,
                (((2,), (1,)), ((0,), (0,))),  # Contract N, batch B
                precision=lax.Precision.DEFAULT
            )
            
            # Normalize by sum of weights (handling mask implicitly via softmax)
            attn_sum = jnp.sum(attn, axis=-1, keepdims=True)
            updates = updates / (attn_sum + self.epsilon)
            
            # GRU Update
            # Flatten batch and slots
            slots_flat = slots.reshape(-1, self.slot_dim)
            updates_flat = updates.reshape(-1, self.slot_dim)
            
            new_slots_flat, _ = gru(updates_flat, slots_flat)
            slots = new_slots_flat.reshape(B, self.num_slots, self.slot_dim)
            
            # MLP Residual
            mlp = nn.Dense(self.hidden_dim)(slots)
            mlp = nn.relu(mlp)
            mlp = nn.Dense(self.slot_dim)(mlp)
            slots = slots + mlp
        
        return slots  # (B, K, slot_dim)


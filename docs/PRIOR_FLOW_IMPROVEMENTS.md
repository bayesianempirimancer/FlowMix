# Prior Flow and Latent Distribution Improvements

## Issues Identified

1. **Unconditional samples look terrible** - Prior flow not learning well
2. **Variance collapse** - Some dimensions of z have very small variance
3. **Marginal distribution not unit normal** - Total variance of z not trending towards 1

## Root Causes

### 1. VAE KL Loss Limitation
- VAE KL loss: `KL(q(z|x) || N(0,I))` encourages each **conditional** distribution to be unit normal
- But it doesn't directly ensure the **marginal** `p(z) = E_x[q(z|x)]` is unit normal
- The marginal can have different statistics even if each q(z|x) is close to N(0,I)

### 2. Architecture Mismatch
- Using `GlobalAdaLNMLPCRN` (designed for point clouds) with dummy context for latent vectors
- This architecture expects spatial structure and meaningful context
- For simple latent vectors, a simpler architecture might work better

### 3. Weak Regularization
- VAE KL weight is very small (0.00001)
- May not be strong enough to prevent variance collapse

## Solutions

### Solution 1: Add Marginal KL Loss
Compute empirical statistics of z in the batch and penalize deviation from N(0,I):

```python
# Compute empirical mean and variance of z in batch
z_mean_empirical = jnp.mean(z, axis=0)  # (D,)
z_var_empirical = jnp.var(z, axis=0)    # (D,)

# KL divergence between empirical N(μ_emp, σ²_emp) and N(0, I)
marginal_kl = 0.5 * (
    jnp.sum(z_var_empirical) +  # sum of variances
    jnp.sum(z_mean_empirical ** 2) -  # sum of squared means
    latent_dim -  # subtract D
    jnp.sum(jnp.log(z_var_empirical + 1e-8))  # log determinant
)
```

### Solution 2: Use Simpler Prior Flow Architecture
Instead of GlobalAdaLNMLPCRN, use a simple MLP-based flow:

```python
class SimpleLatentFlow(nn.Module):
    """Simple MLP-based flow for latent space."""
    hidden_dims: Sequence[int] = (256, 256)
    time_embed_dim: int = 128
    
    @nn.compact
    def __call__(self, z: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        # z: (B, D) or (B, 1, D)
        # t: scalar or (B,)
        
        # Time embedding
        time_embed = SinusoidalTimeEmbedding(self.time_embed_dim)(t)
        
        # Concatenate z and time embedding
        if z.ndim == 3:
            z_flat = z.reshape(z.shape[0], -1)  # (B, D)
        else:
            z_flat = z
        
        h = jnp.concatenate([z_flat, time_embed], axis=-1)
        
        # MLP
        for dim in self.hidden_dims:
            h = nn.Dense(dim)(h)
            h = nn.swish(h)
        
        # Output velocity
        v = nn.Dense(z_flat.shape[-1])(h)
        
        if z.ndim == 3:
            v = v[:, None, :]  # (B, 1, D)
        
        return v
```

### Solution 3: Increase VAE KL Weight
- Current: 0.00001 (very weak)
- Try: 0.001 or 0.01 to prevent variance collapse

### Solution 4: Add Variance Regularization
Directly penalize variance that's too small or too large:

```python
# Penalize variance far from 1
variance_penalty = jnp.mean((z_var_empirical - 1.0) ** 2)
```

### Solution 5: Use Batch Statistics in Prior Flow
Instead of dummy context, use batch statistics as context:

```python
# Compute batch statistics
z_mean_batch = jnp.mean(z_stopped, axis=0, keepdims=True)  # (1, D)
z_std_batch = jnp.std(z_stopped, axis=0, keepdims=True)   # (1, D)

# Use as context (concatenate or use as AdaIN parameters)
context = jnp.concatenate([z_mean_batch, z_std_batch], axis=-1)  # (1, 2*D)
```

## Recommended Approach

1. **Add marginal KL loss** - Most direct way to ensure p(z) is unit normal
2. **Simplify prior flow architecture** - Use MLP instead of GlobalAdaLNMLPCRN
3. **Increase VAE KL weight** - Help prevent variance collapse
4. **Combine all three** - Most robust solution

## Implementation Priority

1. First: Add marginal KL loss (quickest to implement, addresses core issue)
2. Second: Simplify prior flow architecture (better suited for task)
3. Third: Increase VAE KL weight (if still needed after above)





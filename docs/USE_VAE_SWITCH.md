# use_vae Switch - Conditional VAE Behavior

**Date**: December 16, 2025

## Overview

The `use_vae` parameter in `MnistFlow2D` controls whether the model uses variational inference with a VAE framework or directly encodes to a latent representation.

## What the Switch Does

The `use_vae` flag controls three things:

1. **Encoder Output Dimension**
   - `use_vae=True`: Encoder outputs `2*latent_dim`
   - `use_vae=False`: Encoder outputs `latent_dim`

2. **Mu/Logvar Split**
   - `use_vae=True`: Split encoder output into `(mu, logvar)` and sample `z ~ N(mu, exp(logvar))`
   - `use_vae=False`: Use encoder output directly as `z`, no split

3. **KL Loss Computation**
   - `use_vae=True`: Compute KL divergence `KL(q(z|x) || N(0,I))`
   - `use_vae=False`: No KL loss (KL = 0)

## Usage

### With VAE (Default)

```python
from src.models.mnist_flow_2d import MnistFlow2D

# VAE mode (default)
model = MnistFlow2D(
    latent_dim=64,
    encoder_type='pointnet',
    use_vae=True  # Default
)

# Encoder outputs 2*64 = 128 dimensions
# Split into mu (64) and logvar (64)
# Sample z ~ N(mu, exp(logvar))
# Compute KL loss

params = model.init(key, x, key)
loss, metrics = model.apply(params, x, key)
# metrics['kl_loss'] will be non-zero
```

### Without VAE

```python
# Direct encoding mode
model = MnistFlow2D(
    latent_dim=64,
    encoder_type='pointnet',
    use_vae=False
)

# Encoder outputs 64 dimensions directly
# z = encoder(x)
# No mu/logvar split
# No KL loss

params = model.init(key, x, key)
loss, metrics = model.apply(params, x, key)
# metrics['kl_loss'] will be 0.0
```

## Implementation Details

### 1. Encoder Setup

```python
def setup(self):
    # Determine encoder output dimension based on use_vae
    encoder_output_dim = 2 * self.latent_dim if self.use_vae else self.latent_dim
    
    if self.encoder_type == 'pointnet':
        self.encoder = PointNetEncoder(latent_dim=encoder_output_dim, **enc_kwargs)
    # ... similar for other encoder types
```

### 2. Encoding

```python
def encode(self, x, key, mask=None):
    """Encode input to latent code."""
    z_encoded = self.encoder(x, mask=mask, key=key)
    
    if self.use_vae:
        # Split into mu and logvar
        z_mu, z_logvar = jnp.split(z_encoded, 2, axis=-1)
        z = self.reparameterize(z_mu, z_logvar, key)
        return z, z_mu, z_logvar
    else:
        # Use encoder output directly
        return z_encoded, None, None
```

### 3. Loss Computation

```python
def compute_loss(self, x, key, mask=None):
    # ... encode and compute reconstruction loss ...
    
    # VAE KL loss - only if use_vae=True
    if self.use_vae:
        vae_kl_per_latent = 0.5 * (
            jnp.exp(logvar) + jnp.square(mu) - 1.0 - logvar
        )
        vae_kl = jnp.mean(jnp.sum(vae_kl_per_latent, axis=-1))
    else:
        vae_kl = 0.0
    
    # ... combine losses ...
```

## Examples

### Example 1: Global Encoder with VAE

```python
model = MnistFlow2D(
    latent_dim=64,
    encoder_type='pointnet',
    encoder_output_type='global',
    use_vae=True
)

# Encoder: (B, N, 2) -> (B, 128)
# Split: (B, 128) -> mu=(B, 64), logvar=(B, 64)
# Sample: z ~ N(mu, exp(logvar))
# KL: KL(q(z|x) || N(0, I))
```

### Example 2: Global Encoder without VAE

```python
model = MnistFlow2D(
    latent_dim=64,
    encoder_type='pointnet',
    encoder_output_type='global',
    use_vae=False
)

# Encoder: (B, N, 2) -> (B, 64)
# Direct: z = encoder(x)
# No KL loss
```

### Example 3: Structured Encoder with VAE

```python
model = MnistFlow2D(
    latent_dim=64,
    encoder_type='slot_attention',
    encoder_output_type='local',  # Structured
    use_vae=True
)

# Encoder: (B, N, 2) -> (B, K, 128)
# Split: (B, K, 128) -> mu=(B, K, 64), logvar=(B, K, 64)
# Sample: z ~ N(mu, exp(logvar)) for each slot
# KL: Sum over K slots and latent dims
```

### Example 4: Structured Encoder without VAE

```python
model = MnistFlow2D(
    latent_dim=64,
    encoder_type='slot_attention',
    encoder_output_type='local',  # Structured
    use_vae=False
)

# Encoder: (B, N, 2) -> (B, K, 64)
# Direct: z = encoder(x) (slots directly)
# No KL loss
```

## When to Use Each Mode

### Use VAE Mode (`use_vae=True`) When:

✅ You want a probabilistic latent representation  
✅ You want to regularize the latent space  
✅ You need to sample from the prior `p(z) = N(0, I)`  
✅ You want a smooth, continuous latent space  
✅ You're doing generative modeling with sampling

**Pros:**
- Regularized latent space
- Can sample from prior
- Better generalization (in theory)
- Principled probabilistic framework

**Cons:**
- Higher dimensional encoder output (2x)
- Additional KL loss term to balance
- More complex optimization

### Use Direct Mode (`use_vae=False`) When:

✅ You want deterministic encoding  
✅ You don't need to sample from the prior  
✅ You want simpler optimization  
✅ You want maximum encoding capacity  
✅ You're doing conditional generation only

**Pros:**
- Simpler model
- Full encoder capacity for representation
- Easier optimization (no KL balancing)
- Faster inference (no sampling)

**Cons:**
- No latent space regularization
- Can't sample from prior
- Latent space may be irregular
- Purely deterministic

## Testing

```bash
cd /home/jebeck/GitHub/OC-Flow-Mix
python3 << 'EOF'
import jax
import jax.numpy as jnp
from src.models.mnist_flow_2d import MnistFlow2D

x = jnp.ones((2, 50, 2))
key = jax.random.PRNGKey(0)

# Test VAE mode
model_vae = MnistFlow2D(latent_dim=64, encoder_type='pointnet', use_vae=True)
params_vae = model_vae.init(key, x, key)
z, mu, logvar = model_vae.apply(params_vae, x, key, method=model_vae.encode)
print(f"VAE mode: z={z.shape}, mu={mu.shape}, logvar={logvar.shape}")

# Test direct mode
model_direct = MnistFlow2D(latent_dim=64, encoder_type='pointnet', use_vae=False)
params_direct = model_direct.init(key, x, key)
z, mu, logvar = model_direct.apply(params_direct, x, key, method=model_direct.encode)
print(f"Direct mode: z={z.shape}, mu={mu}, logvar={logvar}")
EOF
```

## Comparison

| Feature | `use_vae=True` | `use_vae=False` |
|---------|----------------|-----------------|
| Encoder output dim | `2*latent_dim` | `latent_dim` |
| Mu/logvar split | ✅ Yes | ❌ No |
| Sampling | ✅ z ~ N(μ, σ²) | ❌ z = encoder(x) |
| KL loss | ✅ Computed | ❌ Zero |
| Latent regularization | ✅ Yes | ❌ No |
| Prior sampling | ✅ Can sample | ❌ Cannot sample |
| Optimization | More complex | Simpler |
| Capacity | Half (split) | Full |

## Migration from Old Code

If you have code that assumed VAE was always on:

**Before:**
```python
model = MnistFlow2D(latent_dim=64, encoder_type='pointnet')
# Always did VAE
```

**After:**
```python
# Explicit VAE mode (same behavior as before)
model = MnistFlow2D(latent_dim=64, encoder_type='pointnet', use_vae=True)

# Or new direct mode
model = MnistFlow2D(latent_dim=64, encoder_type='pointnet', use_vae=False)
```

**No breaking changes** - `use_vae=True` is the default, so old code continues to work!

## Implementation Notes

1. **Encoder dimension is automatically adjusted** based on `use_vae`
2. **All encoder types support both modes** (global, structured, local)
3. **KL loss is automatically included/excluded** from total loss
4. **Sampling still works in both modes** (conditional on input)
5. **Prior sampling only works with VAE mode** (needs regularized latent space)

## Summary

The `use_vae` switch provides a clean way to toggle between:
- **VAE mode**: Probabilistic, regularized, can sample from prior
- **Direct mode**: Deterministic, full capacity, simpler optimization

Both modes work with all encoder types and CRN architectures. The switch controls encoder output dimension, mu/logvar splitting, and KL loss computation in a unified way.

✅ **Default is `use_vae=True`** for backward compatibility  
✅ **No breaking changes** to existing code  
✅ **Clean, unified implementation** across all encoder types






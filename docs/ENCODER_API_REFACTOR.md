# Encoder API Refactoring - Unified Output Format

**Date**: December 16, 2025

## Problem

Previously, encoders returned `(mu, logvar)` tuples for VAE compatibility. This created inconsistency:
- Different return signatures across encoder types
- VAE-specific logic embedded in encoders
- Harder to use encoders for non-VAE tasks

## Solution

**All encoders now return a single tensor.** The VAE split happens in `mnist_flow_2d.py`.

### Before (Messy)
```python
# Encoders had VAE logic built-in
encoder = PointNetEncoder(latent_dim=64)
mu, logvar = encoder(x, mask=mask, key=key)  # (B, 64), (B, 64)

# Slot Attention had special flag
encoder = SlotAttentionEncoder(slot_dim=64, return_vae_params=True)
mu, logvar = encoder(x, mask=mask, key=key)  # (B, K, 64), (B, K, 64)
```

### After (Clean)
```python
# All encoders return single tensor
encoder = PointNetEncoder(latent_dim=128)  # 2*64 for VAE
z = encoder(x, mask=mask, key=key)  # (B, 128)

# VAE split happens in mnist_flow_2d.py
mu, logvar = jnp.split(z, 2, axis=-1)  # (B, 64), (B, 64)

# Same for structured encoders
encoder = SlotAttentionEncoder(slot_dim=128)  # 2*64 for VAE
z = encoder(x, mask=mask, key=key)  # (B, K, 128)
mu, logvar = jnp.split(z, 2, axis=-1)  # (B, K, 64), (B, K, 64)
```

## Changes Made

### 1. Encoders - Return Single Tensor

**Global Encoders:**
- `PointNetEncoder`: Returns `(B, latent_dim)`
- `MaxPoolingEncoder`: Returns `(B, latent_dim)`
- `MeanPoolingEncoder`: Returns `(B, latent_dim)`
- `AttentionPoolingEncoder`: Returns `(B, latent_dim)`
- `Set2SetPoolingEncoder`: Returns `(B, latent_dim)`

**Local Encoders:**
- `SlotAttentionEncoder`: Returns `(B, K, slot_dim)`
- `TransformerSetEncoder`: Returns `(B, N, embed_dim)`
- `CrossAttentionEncoder`: Returns `(B, K, latent_dim)`
- `DGCNN`: Returns `(B, N, output_dim)`
- All others: Returns `(B, N/K, dim)`

### 2. mnist_flow_2d.py - VAE Split

```python
def encode(self, x, key, mask=None):
    """Encode input to latent distribution."""
    # Encoder outputs single tensor
    # Shape: (B, 2*latent_dim) or (B, K, 2*latent_dim)
    z_encoded = self.encoder(x, mask=mask, key=key)
    
    # Split into mu and logvar
    z_mu, z_logvar = jnp.split(z_encoded, 2, axis=-1)
    
    # Reparameterization trick
    z = self.reparameterize(z_mu, z_logvar, key)
    return z, z_mu, z_logvar
```

### 3. Encoder Setup - 2x Latent Dim

```python
# In mnist_flow_2d.py setup():
if self.encoder_type == 'pointnet':
    # Output 2*latent_dim for VAE split (mu, logvar)
    self.encoder = PointNetEncoder(latent_dim=2*self.latent_dim, **enc_kwargs)

elif self.encoder_type == 'slot_attention':
    # Output 2*latent_dim for VAE split
    base_encoder = SlotAttentionEncoder(slot_dim=2*self.latent_dim, **enc_kwargs)
    if self.encoder_output_type == 'global':
        self.encoder = MaxPoolingEncoder(base_encoder, 2*self.latent_dim)
    else:
        self.encoder = base_encoder
```

### 4. Pooling Encoders - Simplified

Pooling encoders no longer try to apply the input mask to local features (since `K != N`):

```python
class MaxPoolingEncoder(nn.Module):
    @nn.compact
    def __call__(self, x, mask=None, key=None):
        # Local encoder handles masking internally
        local_features = self.local_encoder(x, mask=mask, key=key)  # (B, K, D)
        
        # Pool over K dimension (no mask needed)
        global_feat = jnp.max(local_features, axis=1)  # (B, D)
        
        # Project to latent dimension
        z = nn.Dense(self.latent_dim)(global_feat)
        return z  # (B, latent_dim)
```

## Benefits

### 1. Consistency
✅ All encoders have the same output signature (single tensor)  
✅ No special cases or flags  
✅ Easier to understand and maintain

### 2. Flexibility
✅ Encoders can be used for non-VAE tasks  
✅ VAE logic is centralized in one place  
✅ Easy to change VAE parameterization

### 3. Simplicity
✅ Less code in encoders  
✅ Clear separation of concerns  
✅ Easier to add new encoders

## Usage Examples

### Global Encoder
```python
from src.encoders.global_encoders.pointnet import PointNetEncoder

# For VAE: output 2*latent_dim
encoder = PointNetEncoder(latent_dim=128)  # Will split to 64+64
z = encoder(x, mask=mask, key=key)  # (B, 128)
mu, logvar = jnp.split(z, 2, axis=-1)  # (B, 64), (B, 64)

# For non-VAE: just use the output directly
encoder = PointNetEncoder(latent_dim=64)
z = encoder(x, mask=mask, key=key)  # (B, 64)
```

### Structured Encoder
```python
from src.encoders.local_encoders.slot_attention_encoder import SlotAttentionEncoder

# For VAE: output 2*slot_dim
encoder = SlotAttentionEncoder(num_slots=8, slot_dim=128)  # Will split to 64+64
z = encoder(x, mask=mask, key=key)  # (B, 8, 128)
mu, logvar = jnp.split(z, 2, axis=-1)  # (B, 8, 64), (B, 8, 64)

# For non-VAE: just use slots directly
encoder = SlotAttentionEncoder(num_slots=8, slot_dim=64)
slots = encoder(x, mask=mask, key=key)  # (B, 8, 64)
```

### Wrapped Encoder
```python
from src.encoders.local_encoders.transformer_set import TransformerSetEncoder
from src.encoders.global_encoders.pooling import MaxPoolingEncoder

# For VAE: wrap and output 2*latent_dim
base = TransformerSetEncoder(embed_dim=64)
encoder = MaxPoolingEncoder(base, latent_dim=128)  # Will split to 64+64
z = encoder(x, mask=mask, key=key)  # (B, 128)
mu, logvar = jnp.split(z, 2, axis=-1)  # (B, 64), (B, 64)
```

### In mnist_flow_2d.py
```python
# Model automatically handles VAE split
model = MnistFlow2D(
    latent_dim=64,  # Actual latent dimension
    encoder_type='pointnet',  # or 'slot_attention', 'transformer', etc.
    encoder_output_type='global',  # or 'local'
)

# Encoder is configured to output 2*latent_dim
# encode() method splits into mu and logvar
z, mu, logvar = model.encode(x, key, mask=mask)
```

## Migration Guide

### If You Have Custom Encoders

**Before:**
```python
class MyEncoder(nn.Module):
    latent_dim: int = 64
    
    @nn.compact
    def __call__(self, x, mask=None, key=None):
        h = self.process(x)
        mu = nn.Dense(self.latent_dim)(h)
        logvar = nn.Dense(self.latent_dim)(h)
        return mu, logvar  # Tuple
```

**After:**
```python
class MyEncoder(nn.Module):
    latent_dim: int = 64  # Now should be 2*actual_latent_dim for VAE
    
    @nn.compact
    def __call__(self, x, mask=None, key=None):
        h = self.process(x)
        z = nn.Dense(self.latent_dim)(h)
        return z  # Single tensor
```

### If You Use Encoders Directly

**Before:**
```python
encoder = PointNetEncoder(latent_dim=64)
mu, logvar = encoder(x, mask=mask, key=key)
```

**After:**
```python
encoder = PointNetEncoder(latent_dim=128)  # 2x for VAE
z = encoder(x, mask=mask, key=key)
mu, logvar = jnp.split(z, 2, axis=-1)
```

## Testing

All tests pass:
```bash
cd /home/jebeck/GitHub/OC-Flow-Mix
python3 -c "
import jax
import jax.numpy as jnp
from src.models.mnist_flow_2d import MnistFlow2D

x = jnp.ones((2, 50, 2))
key = jax.random.PRNGKey(0)

# Test different encoder types
for encoder_type in ['pointnet', 'slot_attention', 'transformer']:
    model = MnistFlow2D(latent_dim=64, encoder_type=encoder_type)
    params = model.init(key, x, key)
    loss, metrics = model.apply(params, x, key)
    print(f'{encoder_type}: loss={loss:.4f} ✓')
"
```

## Files Modified

1. **`src/encoders/global_encoders/pointnet.py`**
   - Removed `Tuple` import
   - Changed return type to `jnp.ndarray`
   - Removed separate `z_mu` and `z_logvar` projections
   - Returns single `z` tensor

2. **`src/encoders/global_encoders/pooling.py`**
   - Removed `Tuple` import
   - All pooling encoders return single tensor
   - Simplified masking logic (local encoder handles it)
   - Removed duplicate mask application

3. **`src/encoders/local_encoders/slot_attention_encoder.py`**
   - Removed `return_vae_params` parameter
   - Removed `Tuple` import
   - Simplified to always return single tensor
   - Removed conditional VAE projection logic

4. **`src/models/mnist_flow_2d.py`**
   - Updated `encode()` method to split encoder output
   - Updated encoder setup to use `2*latent_dim`
   - Added comments explaining VAE split
   - All encoder instantiations now use `2*latent_dim`

## Conclusion

✅ **Cleaner API**: Single return type for all encoders  
✅ **Better separation**: VAE logic in mnist_flow_2d.py, not encoders  
✅ **More flexible**: Encoders work for VAE and non-VAE tasks  
✅ **Easier to maintain**: Less code, clearer intent  
✅ **No breaking changes**: mnist_flow_2d.py handles everything

**The encoder API is now unified and consistent!** 🎯






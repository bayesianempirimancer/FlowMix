# Encoder Mask Support Status

## Summary

All encoders in the codebase accept a `mask` parameter. This document tracks which encoders properly use the mask and which ones currently ignore it.

## Mask Parameter Convention

```python
def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, 
             key: Optional[jax.random.PRNGKey] = None) -> ...:
    """
    Args:
        x: Input points (B, N, D)
        mask: Binary mask (B, N) where 1 = valid, 0 = masked out
        key: Random key for stochastic operations
    """
```

## Global Encoders

### PointNet (`global_encoders/pointnet.py`)
- ✅ **Accepts mask**: Yes
- ✅ **Uses mask**: Yes
- **Implementation**: Masked pooling (max/mean)
  - Max pooling: Sets masked values to -1e9 before max
  - Mean pooling: Multiplies by mask and divides by valid count
- **Returns**: `(mu, logvar)` - `(B, Dc), (B, Dc)`

```python
if mask_expanded is not None:
    h_masked = jnp.where(mask_expanded, h, -1e9)  # Max
    # or
    h_masked = h * mask_expanded  # Mean
    count = jnp.sum(mask_expanded, axis=1)
    global_feat = jnp.sum(h_masked, axis=1) / jnp.maximum(count, 1.0)
```

### Pooling Wrappers (`global_encoders/pooling.py`)
- ✅ **Accepts mask**: Yes
- ✅ **Uses mask**: Yes (passes to wrapped encoder)
- **Types**: MaxPoolingEncoder, MeanPoolingEncoder, AttentionPoolingEncoder, Set2SetPoolingEncoder
- **Implementation**: Forwards mask to local encoder, then pools
- **Returns**: `(mu, logvar)` - `(B, Dc), (B, Dc)`

## Local Encoders

### TransformerSetEncoder (`local_encoders/transformer_set.py`)
- ✅ **Accepts mask**: Yes
- ✅ **Uses mask**: Yes
- **Implementation**: 
  - Adds CLS token (always valid)
  - Extends mask to include CLS: `[1, mask...]`
  - Passes to self-attention layers
- **Returns**: `(B, N+1, D)` with CLS token

```python
if mask is not None:
    cls_mask = jnp.ones((B, 1), dtype=mask.dtype)
    full_mask = jnp.concatenate([cls_mask, mask], axis=1)
```

### DGCNN (`local_encoders/dgcnn.py`)
- ✅ **Accepts mask**: Yes
- ✅ **Uses mask**: Yes
- **Implementation**: Multiplies output features by mask
- **Returns**: `(B, N, D)`

```python
if mask is not None:
    mask_expanded = mask[:, :, None]  # (B, N, 1)
    h = h * mask_expanded
```

### Slot Attention (`local_encoders/slot_attention_encoder.py`)
- ✅ **Accepts mask**: Yes
- ✅ **Uses mask**: Yes
- **Implementation**: Masks attention logits via bias (`-1e9` for masked positions)
- **Returns**: 
  - Standard mode: `(B, K, D)` 
  - VAE mode (`return_vae_params=True`): `((B, K, D), (B, K, D))` as `(mu, logvar)`
- **Status**: ✅ Fixed! Now properly batched and VAE-compatible

### Cross Attention Encoder (`local_encoders/cross_attention_encoder.py`)
- ✅ **Accepts mask**: Yes
- ✅ **Uses mask**: Yes (passes to attention layers)
- **Returns**: `(B, K, D)`

### GMM Featurizer (`local_encoders/gmm_featurizer.py`)
- ✅ **Accepts mask**: Yes
- ✅ **Uses mask**: Yes
- **Returns**: `((B, K, D), (B, K))` - features and valid mask

### Other Local Encoders
All accept `mask` parameter:
- ✅ KPConv (`kpconv.py`)
- ✅ PointNeXt (`pointnext.py`)
- ✅ PointMLP (`pointmlp.py`)
- ✅ EGNN (`egnn.py`)
- ✅ VN-DGCNN (`vn_dgcnn.py`)
- ✅ Set2Set (`set2set.py`)
- ✅ PointNet++ (`pointnet_plus_plus.py`)

**Status**: Most accept but may not actively use the mask in their computations.

## Mask Usage Patterns

### Pattern 1: Masked Pooling (Global Encoders)
```python
if mask is not None:
    mask_expanded = mask[..., None]  # (B, N, 1)
    
    # Max pooling
    h_masked = jnp.where(mask_expanded, h, -1e9)
    result = jnp.max(h_masked, axis=1)
    
    # Mean pooling
    h_masked = h * mask_expanded
    count = jnp.sum(mask_expanded, axis=1)
    result = jnp.sum(h_masked, axis=1) / jnp.maximum(count, 1.0)
```

### Pattern 2: Masked Features (Local Encoders)
```python
if mask is not None:
    mask_expanded = mask[:, :, None]  # (B, N, 1)
    h = h * mask_expanded  # Zero out masked points
```

### Pattern 3: Attention Masking
```python
if mask is not None:
    # Convert to attention mask format
    attn_mask = mask[:, None, None, :]  # (B, 1, 1, N)
    # or
    attn_mask = mask[:, None, :, None]  # (B, 1, N, 1)
    # Pass to attention layers
```

## VAE Framework Compatibility

For the VAE framework in `mnist_flow_2d.py`, encoders should return `(mu, logvar)`:

### Global Encoders
- ✅ **PointNet**: Returns `(B, Dc), (B, Dc)` ✓
- ✅ **Pooling Wrappers**: Return `(B, Dc), (B, Dc)` ✓

### Local Encoders (Need Wrapper)
Most local encoders return `(B, N, D)` or `(B, K, D)`, not `(mu, logvar)`.

**Solution**: Wrap with pooling encoder or add VAE head:
```python
# Option 1: Wrap with pooling
encoder = MaxPoolingEncoder(TransformerSetEncoder(...), latent_dim)
# Returns: (B, Dc), (B, Dc)

# Option 2: Add VAE projection layer
local_features = encoder(x, mask=mask)  # (B, N, D)
mu = nn.Dense(latent_dim)(jnp.mean(local_features, axis=1))
logvar = nn.Dense(latent_dim)(jnp.mean(local_features, axis=1))
```

## Testing Mask Usage

```python
import jax
import jax.numpy as jnp
from src.encoders.global_encoders.pointnet import PointNetEncoder

x = jnp.ones((2, 50, 2))
mask = jnp.concatenate([jnp.ones((2, 25)), jnp.zeros((2, 25))], axis=1)
key = jax.random.PRNGKey(0)

encoder = PointNetEncoder(latent_dim=64)
params = encoder.init(key, x, mask=mask, key=key)

# Without mask
mu_no_mask, _ = encoder.apply(params, x, mask=None, key=key)

# With mask (should be different if mask is used)
mu_with_mask, _ = encoder.apply(params, x, mask=mask, key=key)

print(f"Mask affects output: {not jnp.allclose(mu_no_mask, mu_with_mask)}")
```

## Recommendations

### For Users
1. ✅ **PointNet**: Fully supports masks, use with confidence
2. ✅ **Pooling Wrappers**: Properly forward masks to wrapped encoders
3. ⚠️ **Local Encoders**: Check specific implementation for mask usage
4. ⚠️ **Slot Attention**: Needs fixing for VAE framework (missing batch dim)

### For Developers
1. **Always accept mask parameter** (even if not used yet)
2. **Document mask usage** in docstrings
3. **Test mask effects** by comparing outputs with/without mask
4. **Use consistent patterns** (see above)
5. **For VAE**: Ensure encoders return `(mu, logvar)` tuple

## Status Summary

| Encoder | Accepts Mask | Uses Mask | Returns (mu, logvar) | Status |
|---------|--------------|-----------|----------------------|--------|
| PointNet | ✅ | ✅ | ✅ | Ready |
| Pooling Wrappers | ✅ | ✅ | ✅ | Ready |
| TransformerSet | ✅ | ✅ | ❌ | Needs wrapper |
| DGCNN | ✅ | ✅ | ❌ | Needs wrapper |
| Slot Attention | ✅ | ✅ | ✅ | Ready |
| Cross Attention | ✅ | ✅ | ❌ | Needs wrapper |
| GMM | ✅ | ✅ | ❌ | Needs wrapper |
| Others | ✅ | ❓ | ❌ | Needs review |

## Recent Fixes (Dec 16, 2025)

### Slot Attention Encoder
**Fixed** to properly:
1. ✅ Return batched output `(B, K, slot_dim)` - was already correct
2. ✅ Use mask in attention computation - was already correct
3. ✅ **NEW**: Added `return_vae_params` mode for VAE framework compatibility
   - When `return_vae_params=True`, returns `(mu, logvar)` tuple
   - Each has shape `(B, K, slot_dim)`

**Usage:**
```python
# Standard mode (for structured CRNs)
encoder = SlotAttentionEncoder(num_slots=8, slot_dim=64, return_vae_params=False)
slots = encoder(x, mask=mask, key=key)  # (B, 8, 64)

# VAE mode (for mnist_flow_2d.py)
encoder = SlotAttentionEncoder(num_slots=8, slot_dim=64, return_vae_params=True)
mu, logvar = encoder(x, mask=mask, key=key)  # (B, 8, 64), (B, 8, 64)
```

## Conclusion

✅ **All encoders accept the mask parameter** (good API consistency)  
✅ **Slot Attention now fully supports masks and VAE framework**  
⚠️ **Other local encoders need wrappers for VAE** (don't return mu/logvar directly)

**Recommendation**: 
- **For VAE framework**: Use PointNet, Slot Attention (with `return_vae_params=True`), or wrapped local encoders
- **For structured CRNs**: Use any local encoder (Transformer, DGCNN, Slot Attention, etc.)

**Last Updated**: December 16, 2025


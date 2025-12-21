# Per-Point Context Handling in CRNs

## What "Using First Point" Means

When a CRN receives context `c` with shape `(B, N, Dc)`, it means:
- **B**: batch size
- **N**: number of points (spatial dimension)
- **Dc**: context dimension

This represents **per-point context** - each of the N points has its own context vector.

However, CRNs use **AdaLN (Adaptive Layer Normalization)** conditioning, which requires:
- A single conditioning vector per batch item to generate AdaLN parameters (scale/shift)
- These parameters are then broadcast to all N points

## The Code

```python
# Handle per-point context: if c has spatial dimension, use first point for conditioning
if c.ndim == 3:
    # c is (B, N, Dc) - use first point for global conditioning
    c_for_cond = c[:, 0, :]  # (B, Dc) - takes first point's context
else:
    c_for_cond = c  # (B, Dc) - already global context
```

## What This Means Precisely

**Input:**
- `c` has shape `(B, N, Dc)` - e.g., `(2, 500, 128)` means 2 batches, 500 points, 128-dim context per point

**Operation:**
- `c[:, 0, :]` selects the context vector at index 0 for each batch
- Result: `(B, Dc)` - e.g., `(2, 128)` - one context vector per batch

**Effect:**
- All N points in each batch use the **same** AdaLN conditioning parameters
- These parameters are derived from the **first point's context** only
- The other N-1 points' context vectors are **ignored** for conditioning

## Why This Design?

1. **AdaLN Architecture**: AdaLN generates scale/shift parameters from a single conditioning vector
2. **Broadcasting**: These parameters are broadcast to all N points: `(B, cond_dim)` → `(B, 1, cond_dim)` → applied to `(B, N, hidden_dim)`
3. **Efficiency**: Using one context vector is simpler and faster than per-point conditioning

## Example

```python
# Batch 1: 500 points, each with 128-dim context
c = jnp.array([
    [c_0_0, c_0_1, ..., c_0_499],  # Batch 0: 500 context vectors
    [c_1_0, c_1_1, ..., c_1_499],  # Batch 1: 500 context vectors
])  # Shape: (2, 500, 128)

# After c[:, 0, :]
c_for_cond = jnp.array([
    c_0_0,  # Batch 0: uses first point's context
    c_1_0,  # Batch 1: uses first point's context
])  # Shape: (2, 128)

# This is used to generate AdaLN parameters
cond = MLP(concat([c_for_cond, t_feat]))  # (2, 256)

# AdaLN parameters are broadcast to all 500 points
gamma, beta = split(Dense(cond))  # (2, hidden_dim)
gamma = gamma[:, None, :]  # (2, 1, hidden_dim) - broadcast to (2, 500, hidden_dim)
beta = beta[:, None, :]   # (2, 1, hidden_dim) - broadcast to (2, 500, hidden_dim)

# Applied to all points
h = LayerNorm(h)  # (2, 500, hidden_dim)
h = h * (1 + gamma) + beta  # All 500 points use same gamma/beta
```

## Implications

1. **Information Loss**: Context from points 1..N-1 is discarded for conditioning
2. **Global Conditioning**: All points in a batch share the same AdaLN parameters
3. **Design Choice**: This assumes the first point's context is representative, or that global conditioning is sufficient

## Alternative Approaches (Not Currently Used)

1. **Average Context**: `c_for_cond = jnp.mean(c, axis=1)` - average all N context vectors
2. **Per-Point Conditioning**: Generate different AdaLN params for each point (more expensive)
3. **Attention Pooling**: Use attention to pool N context vectors into one

## Current Usage

In the flow model:
- Encoder outputs `z` with shape `(B, Dz)` - **global context** (one per batch)
- This is passed as `c` to CRN
- So `c` is typically `(B, Dc)`, not `(B, N, Dc)`
- The per-point handling is for **compatibility** - if someone passes `(B, N, Dc)`, it gracefully handles it by using the first point

## Computational Efficiency Concern

**Question**: If we're going to use `c[:, 0, :]` when `c` has shape `(B, N, Dc)`, why compute per-point context at all?

**Answer**: 
1. **We don't compute per-point context** - encoders output global context `(B, Dz)`
2. **The per-point handling is defensive** - it allows the CRN to accept `(B, N, Dc)` without error
3. **If per-point context were computed**, using only the first point would waste computation

**Recommendation**: 
- If you need per-point context, consider **averaging** instead: `c_for_cond = jnp.mean(c, axis=1)` 
- Or use **attention pooling** to aggregate all N context vectors into one
- Otherwise, ensure encoders output global context `(B, Dz)` to avoid wasted computation

**Current Status**: 
- ✅ No wasted computation - encoders output `(B, Dz)` 
- ✅ Per-point handling exists but is not used
- ⚠️ If future code computes `(B, N, Dc)`, it will waste N-1 points worth of computation

## Slot Attention Case

**Slot Attention Encoder**:
1. Computes `slots` with shape `(B, K, slot_dim)` where K is the number of slots (e.g., 8)
2. **Currently**: Pools slots using max pooling: `global_feat = jnp.max(slots, axis=1)` → `(B, slot_dim)`
3. Projects to `(B, latent_dim)` for `z_mu` and `z_logvar`
4. This pooled `z` is passed as context `c` to the CRN

**Current Status**: ✅ **No waste** - slots are pooled before being used as context

**Potential Issue**: 
- If someone passed slots `(B, K, slot_dim)` directly as context `c` to a CRN
- And the CRN uses `c[:, 0, :]` to extract `(B, slot_dim)`
- Then we'd only use the **first slot** and ignore the other K-1 slots
- This would waste computation and lose the multi-object representation that Slot Attention captures

**Why This Matters**:
- Slots are **not per-point context** - they're a learned set representation
- Each slot typically captures a different object/part of the scene
- Using only the first slot loses this multi-object structure

**Solution**:
- **Current approach (correct)**: Pool slots before passing to CRN (max, mean, or attention pooling)
- **If passing slots directly**: Use pooling in the CRN: `c_for_cond = jnp.mean(c, axis=1)` or `jnp.max(c, axis=1)`
- **Better approach**: Use `CrossAttentionCRN` which projects context into M latents - could project slots into latents directly

**Example**:
```python
# Slot Attention outputs
slots = slot_attention(x)  # (B, K=8, slot_dim=64)

# Current: Pool first
global_feat = jnp.max(slots, axis=1)  # (B, 64)
z = Dense(latent_dim)(global_feat)  # (B, 128)
# Pass z as context - no waste

# If passing slots directly (NOT RECOMMENDED):
# CRN would use c[:, 0, :] → only first slot used
# Wastes slots 1..K-1

# Better: Pool in CRN
if c.ndim == 3 and c.shape[1] == num_slots:  # Slots case
    c_for_cond = jnp.mean(c, axis=1)  # Average all slots
else:
    c_for_cond = c[:, 0, :]  # Per-point context case
```


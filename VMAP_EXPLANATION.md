# Understanding `vmap` in Hutchinson's Trace Estimator

## The Problem

We need to compute the trace of the Jacobian for **every point** in **every batch**:
- Input: `x_t` has shape `(B, N, D)` where:
  - `B` = batch size
  - `N` = number of points per sample
  - `D` = spatial dimension (e.g., 2 for 2D points)
- Output: `traces` has shape `(B, N)` - one trace value per point

## The Base Function

```python
def hutchinson_trace(x_point, z_ctx, t_val, mask_val, key):
    """
    Compute trace for a SINGLE point.
    
    Args:
        x_point: (D,) - single point coordinates
        z_ctx: (Dc,) or (K, Dc) - context for this point
        t_val: () - scalar time value
        mask_val: () - scalar mask value
        key: PRNGKey - random key for sampling
    
    Returns:
        scalar - trace estimate for this point
    """
    # ... Hutchinson's estimator logic ...
    return trace_estimate  # scalar
```

This function operates on **one point at a time**. We need to apply it to all `B * N` points.

## The Vectorization Strategy

We use **nested `vmap`** to vectorize over two dimensions:
1. **Inner `vmap`**: Vectorize over `N` points (within a batch)
2. **Outer `vmap`**: Vectorize over `B` batches

### Step 1: Prepare Inputs

Before vmapping, we need to ensure all inputs have compatible shapes:

```python
# Original shapes:
x_t:   (B, N, D)      # Already has B and N dimensions ✓
z:     (B, Dc)        # Only has B dimension, missing N
t:     (B, 1)         # Only has B dimension, missing N
mask:  (B, N)         # Already has B and N dimensions ✓
keys:  ???            # Need to generate

# After preparation:
x_t:        (B, N, D)      # Unchanged
z_array:    (B, N, Dc)     # Tiled to add N dimension
t_array:    (B, N)         # Tiled to add N dimension
mask_array: (B, N)         # Unchanged or created
keys:       (B, N, 2)      # Generated for each point
```

#### Why Tile `z`?

```python
# If z is global: (B, Dc)
z_array = jnp.tile(z[:, None, :], (1, N, 1))  # -> (B, N, Dc)
# This broadcasts the same context to all N points in each batch

# If z is structured: (B, K, Dc)
z_array = jnp.tile(z[:, None, :, :], (1, N, 1, 1))  # -> (B, N, K, Dc)
# This broadcasts the K latents to all N points in each batch
```

Each point gets the **same context**, but we need to tile it so that when we vmap over `N`, each point has its own copy to work with.

#### Why Tile `t`?

```python
# If t is (B, 1)
t_array = jnp.tile(t, (1, N))  # -> (B, N)
# Each point gets the same time value, but we need it in (B, N) shape for vmap
```

### Step 2: Inner `vmap` (Over Points)

```python
# Apply hutchinson_trace to all N points in a single batch
inner_vmap = jax.vmap(hutchinson_trace, in_axes=(0, 0, 0, 0, 0))
```

**What `in_axes=(0, 0, 0, 0, 0)` means:**

For each of the 5 arguments to `hutchinson_trace`, map over axis 0:

| Argument | Shape Before | `in_axes` | Meaning | Shape Seen by Function |
|----------|-------------|-----------|---------|----------------------|
| `x_point` | `(N, D)` | `0` | Map over N points | `(D,)` |
| `z_ctx` | `(N, Dc)` or `(N, K, Dc)` | `0` | Map over N contexts | `(Dc,)` or `(K, Dc)` |
| `t_val` | `(N,)` | `0` | Map over N time values | `()` scalar |
| `mask_val` | `(N,)` | `0` | Map over N mask values | `()` scalar |
| `key` | `(N, 2)` | `0` | Map over N keys | `(2,)` PRNGKey |

**Result:** The inner vmap takes inputs with leading dimension `N` and produces output with shape `(N,)` - one trace per point.

### Step 3: Outer `vmap` (Over Batches)

```python
# Apply inner_vmap to all B batches
outer_vmap = jax.vmap(inner_vmap, in_axes=(0, 0, 0, 0, 0))
```

**What `in_axes=(0, 0, 0, 0, 0)` means:**

For each of the 5 arguments, map over axis 0 (the batch dimension):

| Argument | Shape Before | `in_axes` | Meaning | Shape Seen by Inner Vmap |
|----------|-------------|-----------|---------|------------------------|
| `x_t` | `(B, N, D)` | `0` | Map over B batches | `(N, D)` |
| `z_array` | `(B, N, Dc)` or `(B, N, K, Dc)` | `0` | Map over B batches | `(N, Dc)` or `(N, K, Dc)` |
| `t_array` | `(B, N)` | `0` | Map over B batches | `(N,)` |
| `mask_array` | `(B, N)` | `0` | Map over B batches | `(N,)` |
| `keys` | `(B, N, 2)` | `0` | Map over B batches | `(N, 2)` |

**Result:** The outer vmap takes inputs with leading dimension `B` and produces output with shape `(B, N)` - one trace per point per batch.

## Complete Flow

Let's trace through a concrete example with `B=2`, `N=3`, `D=2`:

### Initial Shapes

```python
x_t:   (2, 3, 2)   # 2 batches, 3 points each, 2D coordinates
z:     (2, 64)     # 2 batches, 64-dim context (global encoder)
t:     (2, 1)      # 2 batches, 1 time value each
mask:  (2, 3)      # 2 batches, 3 mask values each
```

### After Preparation

```python
x_t:        (2, 3, 2)    # Unchanged
z_array:    (2, 3, 64)   # Tiled: each of 3 points gets the same 64-dim context
t_array:    (2, 3)       # Tiled: each of 3 points gets the same time value
mask_array: (2, 3)       # Unchanged
keys:       (2, 3, 2)    # Generated: unique key for each of 6 points
```

### Outer Vmap (Batch 0)

The outer vmap processes the first batch (index 0):

```python
# Inputs to inner_vmap (batch 0):
x_t[0]:        (3, 2)    # 3 points, 2D each
z_array[0]:    (3, 64)   # 3 contexts, 64-dim each
t_array[0]:    (3,)      # 3 time values
mask_array[0]: (3,)      # 3 mask values
keys[0]:       (3, 2)    # 3 random keys
```

### Inner Vmap (Point 0 of Batch 0)

The inner vmap processes the first point (index 0) of batch 0:

```python
# Inputs to hutchinson_trace (batch 0, point 0):
x_point:  (2,)      # x_t[0, 0, :]     - single 2D point
z_ctx:    (64,)     # z_array[0, 0, :]  - single 64-dim context
t_val:    ()        # t_array[0, 0]     - single scalar time
mask_val: ()        # mask_array[0, 0]  - single scalar mask
key:      (2,)      # keys[0, 0, :]     - single PRNGKey

# Output:
trace:    ()        # Single scalar trace estimate
```

### Inner Vmap (All Points in Batch 0)

The inner vmap applies `hutchinson_trace` to all 3 points:

```python
# Point 0: hutchinson_trace(x_t[0,0], z_array[0,0], t_array[0,0], mask_array[0,0], keys[0,0]) -> scalar
# Point 1: hutchinson_trace(x_t[0,1], z_array[0,1], t_array[0,1], mask_array[0,1], keys[0,1]) -> scalar
# Point 2: hutchinson_trace(x_t[0,2], z_array[0,2], t_array[0,2], mask_array[0,2], keys[0,2]) -> scalar

# Output: (3,) - one trace per point
```

### Outer Vmap (All Batches)

The outer vmap applies the inner vmap to both batches:

```python
# Batch 0: inner_vmap(...) -> (3,)
# Batch 1: inner_vmap(...) -> (3,)

# Output: (2, 3) - one trace per point per batch
```

## Why `in_axes=(0, 0, 0, 0, 0)`?

The key insight is that **all inputs have been prepared to have matching leading dimensions**:

```python
# Before vmapping:
x_t:        (B, N, ...)
z_array:    (B, N, ...)
t_array:    (B, N)
mask_array: (B, N)
keys:       (B, N, ...)
```

All have `(B, N)` as their first two dimensions! This allows us to use simple `in_axes=(0, 0, 0, 0, 0)` for both vmaps.

### Alternative: Different `in_axes`

If we **hadn't tiled** `z` and `t`, we would need different `in_axes`:

```python
# Without tiling:
x_t:   (B, N, D)
z:     (B, Dc)      # Missing N dimension!
t:     (B, 1)       # Missing N dimension!
mask:  (B, N)
keys:  (B, N, 2)

# Inner vmap would need:
inner_vmap = jax.vmap(
    hutchinson_trace,
    in_axes=(0, None, None, 0, 0)
    #        ^  ^^^^  ^^^^  ^  ^
    #        |   |     |    |  |
    #        |   |     |    |  +-- Map over N keys
    #        |   |     |    +----- Map over N masks
    #        |   |     +---------- Don't map over t (broadcast to all N)
    #        |   +---------------- Don't map over z (broadcast to all N)
    #        +-------------------- Map over N points
)
```

But this is more complex! By tiling first, we make the vmap structure simpler and more uniform.

## Visual Representation

```
Input Tensors (after preparation):
┌─────────────────────────────────────┐
│ x_t:        (B, N, D)               │
│ z_array:    (B, N, Dc)              │
│ t_array:    (B, N)                  │
│ mask_array: (B, N)                  │
│ keys:       (B, N, 2)               │
└─────────────────────────────────────┘
                 │
                 ▼
         Outer vmap (axis 0)
         Map over B batches
                 │
         ┌───────┴───────┐
         │               │
      Batch 0         Batch 1
    (N, D) etc.     (N, D) etc.
         │               │
         ▼               ▼
    Inner vmap      Inner vmap
    (axis 0)        (axis 0)
    Map over N      Map over N
         │               │
    ┌────┼────┐     ┌────┼────┐
    │    │    │     │    │    │
   P0   P1   P2    P0   P1   P2
   ()   ()   ()    ()   ()   ()
    │    │    │     │    │    │
    └────┼────┘     └────┼────┘
         │               │
      (N,)            (N,)
         │               │
         └───────┬───────┘
                 ▼
         Output: (B, N)
```

## Key Takeaways

1. **Tiling is for convenience**: We tile `z` and `t` to give them `(B, N)` leading dimensions, making vmap uniform.

2. **`in_axes=(0, 0, 0, 0, 0)` means**: "Map over axis 0 of all 5 arguments"

3. **Nested vmap structure**:
   - **Outer**: Map over `B` batches (axis 0)
   - **Inner**: Map over `N` points (axis 0)

4. **Each function call sees scalars**: The base `hutchinson_trace` function receives:
   - `x_point`: `(D,)` - a single point
   - `z_ctx`: `(Dc,)` or `(K, Dc)` - context for that point
   - `t_val`: `()` - scalar time
   - `mask_val`: `()` - scalar mask
   - `key`: `(2,)` - random key

5. **Efficiency**: JAX compiles this nested vmap into efficient parallel code that processes all `B * N` points simultaneously!

## Alternative Approaches

### Approach 1: Single vmap over flattened dimension

```python
# Flatten to (B*N, D)
x_flat = x_t.reshape(-1, D)
z_flat = jnp.repeat(z, N, axis=0)  # (B*N, Dc)
# ... etc ...

# Single vmap
traces_flat = jax.vmap(hutchinson_trace, in_axes=(0, 0, 0, 0, 0))(
    x_flat, z_flat, t_flat, mask_flat, keys_flat
)  # (B*N,)

# Reshape back
traces = traces_flat.reshape(B, N)
```

**Pros**: Single vmap, simpler structure
**Cons**: More reshaping, less clear what dimensions represent

### Approach 2: Explicit loops (slow!)

```python
traces = []
for b in range(B):
    batch_traces = []
    for n in range(N):
        trace = hutchinson_trace(
            x_t[b, n], z_array[b, n], t_array[b, n], 
            mask_array[b, n], keys[b, n]
        )
        batch_traces.append(trace)
    traces.append(batch_traces)
traces = jnp.array(traces)  # (B, N)
```

**Pros**: Very explicit, easy to understand
**Cons**: Slow! No parallelization, not JIT-friendly

### Approach 3: Our nested vmap (best!)

```python
trace_vmap = jax.vmap(
    jax.vmap(hutchinson_trace, in_axes=(0, 0, 0, 0, 0)),
    in_axes=(0, 0, 0, 0, 0)
)
all_traces = trace_vmap(x_t, z_array, t_array, mask_array, keys)
```

**Pros**: Clear structure (batch → points), efficient, JIT-friendly
**Cons**: Requires tiling inputs first

## Summary

The `in_axes=(0, 0, 0, 0, 0)` specification tells JAX:

> "For each of the 5 input arguments, vectorize over axis 0 (the leading dimension)"

By carefully preparing our inputs to all have `(B, N, ...)` shapes, we can use this simple, uniform vmap specification to efficiently compute traces for all points in all batches in parallel!






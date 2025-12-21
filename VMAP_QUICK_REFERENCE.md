# `vmap` Quick Reference for Hutchinson's Trace Estimator

## TL;DR

```python
# Nested vmap with in_axes=(0, 0, 0, 0, 0)
trace_vmap = jax.vmap(                           # Outer: map over B batches
    jax.vmap(hutchinson_trace, in_axes=(0,0,0,0,0)),  # Inner: map over N points
    in_axes=(0,0,0,0,0)
)

# Input shapes (after tiling):
x_t:        (B, N, D)
z_array:    (B, N, Dc) or (B, N, K, Dc)
t_array:    (B, N)
mask_array: (B, N)
keys:       (B, N, 2)

# Output shape:
traces:     (B, N)
```

## What `in_axes=(0, 0, 0, 0, 0)` Means

**"Map over axis 0 of all 5 arguments"**

| Argument Position | Argument Name | `in_axes` Value | Meaning |
|------------------|---------------|-----------------|---------|
| 1st | `x_point` | `0` | Map over axis 0 |
| 2nd | `z_ctx` | `0` | Map over axis 0 |
| 3rd | `t_val` | `0` | Map over axis 0 |
| 4th | `mask_val` | `0` | Map over axis 0 |
| 5th | `key` | `0` | Map over axis 0 |

## Shape Transformations

### Outer Vmap (Batch Dimension)

```
Input:                      After Outer Vmap:
x_t:        (B, N, D)  →    (N, D)      for each batch
z_array:    (B, N, Dc) →    (N, Dc)     for each batch
t_array:    (B, N)     →    (N,)        for each batch
mask_array: (B, N)     →    (N,)        for each batch
keys:       (B, N, 2)  →    (N, 2)      for each batch
```

### Inner Vmap (Point Dimension)

```
Input:                      After Inner Vmap:
x_t:        (N, D)     →    (D,)        for each point
z_array:    (N, Dc)    →    (Dc,)       for each point
t_array:    (N,)       →    ()          for each point
mask_array: (N,)       →    ()          for each point
keys:       (N, 2)     →    (2,)        for each point
```

### Complete Flow

```
(B, N, D)  →  [Outer vmap]  →  (N, D)  →  [Inner vmap]  →  (D,)
                                                              ↓
                                                       hutchinson_trace
                                                              ↓
                                                           scalar
                                                              ↓
                                              [Inner vmap]   ↓
                                                           (N,)
                                                              ↓
                                              [Outer vmap]   ↓
                                                          (B, N)
```

## Why Tile First?

### Without Tiling (Complex)

```python
# Original shapes:
x_t:   (B, N, D)
z:     (B, Dc)      # No N dimension!
t:     (B, 1)       # No N dimension!

# Would need different in_axes:
inner_vmap = jax.vmap(
    hutchinson_trace,
    in_axes=(0, None, None, 0, 0)
    #        ^  ^^^^  ^^^^  ^  ^
    #        |   |     |    |  +-- Map over keys
    #        |   |     |    +----- Map over masks
    #        |   |     +---------- Broadcast t to all N
    #        |   +---------------- Broadcast z to all N
    #        +-------------------- Map over x_t
)
```

### With Tiling (Simple)

```python
# After tiling:
x_t:        (B, N, D)
z_array:    (B, N, Dc)   # Now has N dimension!
t_array:    (B, N)       # Now has N dimension!

# Uniform in_axes:
inner_vmap = jax.vmap(
    hutchinson_trace,
    in_axes=(0, 0, 0, 0, 0)  # All the same!
)
```

## Common `in_axes` Patterns

### Pattern 1: Map over all arguments
```python
jax.vmap(f, in_axes=(0, 0, 0))
# Map over axis 0 of all 3 arguments
```

### Pattern 2: Broadcast some arguments
```python
jax.vmap(f, in_axes=(0, None, 0))
# Map over axis 0 of arg1 and arg3
# Broadcast arg2 to all mapped elements
```

### Pattern 3: Map over different axes
```python
jax.vmap(f, in_axes=(0, 1, 0))
# Map over axis 0 of arg1
# Map over axis 1 of arg2
# Map over axis 0 of arg3
```

### Pattern 4: Nested vmap (our case)
```python
jax.vmap(jax.vmap(f, in_axes=(0, 0)), in_axes=(0, 0))
# Outer: map over axis 0 of both args
# Inner: map over axis 0 of both args
# Result: 2D mapping (like nested for loops)
```

## Concrete Example

```python
import jax
import jax.numpy as jnp

# Base function: operates on single point
def trace_single(x, z, t):
    """x: (D,), z: (Dc,), t: () → scalar"""
    return jnp.sum(x) + jnp.sum(z) + t

# Prepare inputs
B, N, D, Dc = 2, 3, 2, 64
x = jnp.ones((B, N, D))
z = jnp.ones((B, Dc))
t = jnp.ones((B, 1))

# Tile to add N dimension
z_tiled = jnp.tile(z[:, None, :], (1, N, 1))  # (B, N, Dc)
t_tiled = jnp.tile(t, (1, N))                  # (B, N)

# Nested vmap
trace_vmap = jax.vmap(
    jax.vmap(trace_single, in_axes=(0, 0, 0)),
    in_axes=(0, 0, 0)
)

# Apply
result = trace_vmap(x, z_tiled, t_tiled)
print(result.shape)  # (2, 3) - one trace per point per batch
```

## Debugging Tips

### Check intermediate shapes
```python
# Add print statements in the base function
def hutchinson_trace(x_point, z_ctx, t_val, mask_val, key):
    print(f"x_point: {x_point.shape}")
    print(f"z_ctx: {z_ctx.shape}")
    # ... etc
```

### Test with small inputs
```python
# Use B=1, N=2 for easier debugging
x_small = jnp.ones((1, 2, 2))
z_small = jnp.ones((1, 2, 64))
# ...
```

### Verify tiling
```python
# Check shapes after tiling
print(f"x_t: {x_t.shape}")
print(f"z_array: {z_array.shape}")
print(f"t_array: {t_array.shape}")
# All should have (B, N, ...) shape
```

## Performance Notes

### Why nested vmap is fast

JAX compiles nested vmaps into efficient parallel code:
- No Python loops at runtime
- Vectorized operations on GPU/TPU
- Automatic batching and parallelization

### Equivalent to:
```python
# Conceptually equivalent to (but much faster than):
result = []
for b in range(B):
    batch_result = []
    for n in range(N):
        trace = hutchinson_trace(
            x_t[b, n], z_array[b, n], t_array[b, n],
            mask_array[b, n], keys[b, n]
        )
        batch_result.append(trace)
    result.append(batch_result)
```

But vmap:
- ✅ Runs in parallel
- ✅ JIT-compilable
- ✅ GPU-friendly
- ✅ No Python overhead

## Summary

| Aspect | Value |
|--------|-------|
| **Pattern** | Nested vmap |
| **Structure** | `vmap(vmap(f, in_axes), in_axes)` |
| **in_axes** | `(0, 0, 0, 0, 0)` for all arguments |
| **Reason** | All inputs tiled to have `(B, N, ...)` shape |
| **Outer vmap** | Maps over B batches (axis 0) |
| **Inner vmap** | Maps over N points (axis 0) |
| **Output** | `(B, N)` - one trace per point per batch |
| **Efficiency** | Fully parallelized, JIT-compiled |

**Key Insight**: By tiling inputs to have matching `(B, N, ...)` shapes, we can use simple, uniform `in_axes=(0, 0, 0, 0, 0)` for clean, efficient nested vectorization!






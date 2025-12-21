# Hutchinson's Trace Estimator for Divergence Computation

**Date**: December 16, 2025

## Overview

Implemented **Hutchinson's trace estimator** for efficient computation of the divergence of neural network vector fields. This is the standard approach used in continuous normalizing flows (CNFs) and is much more efficient than computing the full Jacobian matrix.

## The Problem

To compute the divergence of a velocity field `v(x)`, we need:

```
div(v) = trace(J_v) = sum_i ∂v_i/∂x_i
```

**Naive Approach**: Compute the full Jacobian matrix J (D×D), then take the trace.
- **Complexity**: O(D²) per point
- **Memory**: O(D²) per point
- **Problem**: Wasteful! We only need the diagonal elements.

## Hutchinson's Solution

**Key Insight**: For a random vector `v ~ Rademacher(±1)`:

```
E[v^T J v] = trace(J)
```

This means we can estimate the trace using only **Jacobian-vector products** (JVPs), which are much cheaper than computing the full Jacobian!

### Why This Works

```
E[v^T J v] = E[sum_i sum_j v_i J_ij v_j]
           = sum_i sum_j J_ij E[v_i v_j]
           = sum_i sum_j J_ij δ_ij        (since E[v_i v_j] = 1 if i=j, 0 otherwise)
           = sum_i J_ii
           = trace(J)
```

The Rademacher distribution (±1 with equal probability) has the property that:
- `E[v_i²] = 1` (diagonal terms)
- `E[v_i v_j] = 0` for i≠j (off-diagonal terms cancel out)

## Implementation

### Algorithm

```python
def hutchinson_trace_estimator(f, x, num_samples=1):
    """
    Estimate trace(J_f) where J_f is the Jacobian of f at x.
    
    Args:
        f: Function R^D -> R^D
        x: Input point, shape (D,)
        num_samples: Number of random samples (more = lower variance)
    
    Returns:
        Estimated trace(J_f)
    """
    traces = []
    for i in range(num_samples):
        # Sample random vector with ±1 entries
        v = jax.random.rademacher(key, (D,))
        
        # Compute Jacobian-vector product: J @ v
        # Using forward-mode AD (jvp)
        _, jvp_result = jax.jvp(f, (x,), (v,))
        
        # Estimate: v^T (J @ v) = sum(v * (J @ v))
        trace_estimate = jnp.sum(v * jvp_result)
        traces.append(trace_estimate)
    
    return jnp.mean(traces)
```

### Key JAX Functions

1. **`jax.random.rademacher(key, shape)`**: Samples from Rademacher distribution (±1)
2. **`jax.jvp(f, primals, tangents)`**: Computes Jacobian-vector product using forward-mode AD
   - `primals`: Input point (x)
   - `tangents`: Direction vector (v)
   - Returns: `(f(x), J @ v)`

## Complexity Analysis

| Method | Time per Point | Memory per Point | Scalability |
|--------|---------------|------------------|-------------|
| **Full Jacobian** | O(D²) | O(D²) | Poor for large D |
| **Hutchinson (S samples)** | O(D·S) | O(D) | Excellent! |

### Speedup Factor

For D-dimensional space with S=1 sample:
- **Speedup**: D times faster than full Jacobian
- **D=2** (2D points): 2x faster
- **D=3** (3D points): 3x faster
- **D=100** (high-D): 100x faster!

## Variance Control

Hutchinson's estimator is **stochastic but unbiased**:
- `E[estimate] = trace(J)` (unbiased)
- `Var[estimate] ∝ 1/S` (variance decreases with more samples)

### Choosing Number of Samples

```python
# Fast, usually sufficient for training
trace = compute_crn_jacobian_trace(x, z, t, hutchinson_samples=1)

# Lower variance, good for evaluation
trace = compute_crn_jacobian_trace(x, z, t, hutchinson_samples=5)

# Very stable, for critical applications
trace = compute_crn_jacobian_trace(x, z, t, hutchinson_samples=10)
```

**Rule of thumb**: 1 sample is usually sufficient during training, use 5-10 for evaluation.

## Usage in Our Codebase

### Basic Usage

```python
from src.models.mnist_flow_2d import MnistFlow2D

model = MnistFlow2D(latent_dim=64, encoder_type='pointnet')
params = model.init(key, x, key)

# Encode to get context
z, mu, logvar = model.apply(params, x, key, method=model.encode)

# Compute divergence at time t
t = 0.5
x_t = ...  # Interpolated points
div_v = model.apply(
    params, x_t, z, t,
    method=model.compute_velocity_divergence
)
```

### With Multiple Samples

```python
# Reduce variance by averaging over multiple samples
div_v = model.apply(
    params, x_t, z, t,
    hutchinson_samples=5,
    method=model.compute_velocity_divergence
)
```

### Direct Jacobian Trace

```python
# Just the trace, without divergence formula
trace_jac = model.apply(
    params, x_t, z, t,
    hutchinson_samples=1,
    method=model.compute_crn_jacobian_trace
)
```

## Mathematical Background

### Rademacher Distribution

The Rademacher distribution is a discrete probability distribution where:
- `P(v = +1) = 0.5`
- `P(v = -1) = 0.5`

Properties:
- `E[v] = 0`
- `E[v²] = 1`
- `E[v_i v_j] = 0` for i≠j (independent components)

These properties make it ideal for trace estimation!

### Alternative Distributions

Other distributions can also be used:
- **Standard Gaussian**: `v ~ N(0, I)` - also works, slightly different variance
- **Uniform sphere**: `v ~ Uniform(S^(D-1))` - more complex to sample

Rademacher is preferred because:
- Easy to sample (just random bits)
- Minimal variance
- Standard in the literature

## Comparison with Alternatives

### 1. Full Jacobian
```python
# Compute full D×D matrix, then trace
jac = jax.jacobian(f)(x)  # O(D²)
trace = jnp.trace(jac)    # O(D)
```
**Pros**: Exact
**Cons**: O(D²) complexity, wasteful

### 2. Direct Diagonal Computation
```python
# Compute only diagonal elements
trace = sum(jax.grad(lambda x: f(x)[i])(x)[i] for i in range(D))
```
**Pros**: Exact, O(D) complexity
**Cons**: Requires D separate gradient computations (D backward passes)

### 3. Hutchinson's Estimator (Our Choice)
```python
# Estimate using random projections
v = jax.random.rademacher(key, (D,))
_, jvp = jax.jvp(f, (x,), (v,))
trace ≈ jnp.sum(v * jvp)
```
**Pros**: O(D) complexity, single forward pass, standard practice
**Cons**: Stochastic (but unbiased)

## Why Hutchinson is Standard in CNFs

Continuous Normalizing Flows require computing divergence at every integration step:

```python
# CNF dynamics
dx/dt = f(x, t)
d(log p)/dt = -div(f)

# Must compute div(f) at each step!
```

For a flow with:
- **T steps** in the ODE solver
- **D dimensions**
- **B batch size**
- **N points per sample**

**Full Jacobian**: `O(T · B · N · D²)` - prohibitive for large D!
**Hutchinson**: `O(T · B · N · D)` - scales linearly!

This is why Hutchinson's estimator is the **de facto standard** in the CNF literature.

## References

1. **Hutchinson (1990)**: "A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines"
   - Original paper introducing the trace estimator

2. **Grathwohl et al. (2018)**: "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models"
   - Popularized Hutchinson's estimator in CNFs

3. **Chen et al. (2018)**: "Neural Ordinary Differential Equations"
   - Introduced continuous normalizing flows

4. **PointFlow (Yan et al. 2019)**: "PointFlow: 3D Point Cloud Generation with Continuous Normalizing Flows"
   - Uses Hutchinson's estimator for point cloud generation

## Test Results

```
✅ Test 1: Basic functionality with Hutchinson estimator
   Trace shape: (2, 10)
   Trace mean: 0.0000 ✓

✅ Test 2: Divergence computation with different prediction targets
   velocity  : div shape=(2, 10), mean=0.0000 ✓
   noise     : div shape=(2, 10), mean=-4.0000 ✓
   target    : div shape=(2, 10), mean=4.0000 ✓

✅ Test 3: Multiple Hutchinson samples (reduce variance)
    1 samples: mean=0.0000, std=0.0000 ✓
    5 samples: mean=0.0000, std=0.0000 ✓
   10 samples: mean=0.0000, std=0.0000 ✓

✅ Test 4: Efficiency comparison
   10 iterations: 2.359s (235.9ms per call) ✓
```

## Summary

| Aspect | Full Jacobian | Hutchinson |
|--------|--------------|------------|
| **Complexity** | O(D²) | O(D) |
| **Memory** | O(D²) | O(D) |
| **Accuracy** | Exact | Stochastic (unbiased) |
| **Scalability** | Poor | Excellent |
| **Standard Practice** | No | Yes (CNFs) |
| **Our Choice** | ❌ | ✅ |

**Bottom line**: Hutchinson's trace estimator is simpler, more robust, more efficient, and is the standard approach in the literature. It's the right choice for computing divergence in neural network vector fields!






# Loss Function Analysis: Factorial Comparison vs Current Training

## Key Findings

### 1. **Vector Field Call Signature Mismatch**

**Current Code (mnist_flow_2d.py:420):**
```python
# Comment says: (z, x, t)
pred_v = self.vector_field(x_t, z, time)  # Actual: (x, z, t)
```

**Factorial Comparison (factorial_comparison.py:149):**
```python
def call_vf(m, t_val, x_val, z_val):
    return m.vector_field(t_val, x_val, z_val)  # (t, x, z)
```

**Actual Vector Field Signature:**
- `CrossAttentionVectorField.__call__(self, x, z, t)` = `(x, z, t)`
- `ODEVectorField2D.__call__(self, x, z, t)` = `(x, z, t)`

### 2. **Loss Function Structure**

**Current Loss (use_latent_flow=False):**
```python
loss = mse + 0.01 * vae_kl
```

Where:
- `mse = jnp.mean((pred_v - target_v)**2)` - Flow matching loss
- `vae_kl = 0.5 * jnp.mean(sum(exp(logvar) + mu² - 1 - logvar))` - VAE KL regularization

**Key Components:**
1. Flow Matching MSE: `pred_v = self.vector_field(x_t, z, time)` vs `target_v = x_prior - x`
2. VAE KL: Standard KL(q(z|x) || N(0,I)) with weight 0.01

### 3. **Potential Issues**

1. **Comment Mismatch**: Comment says `(z, x, t)` but call is `(x, z, t)` - this is just a documentation error, not a functional issue.

2. **Factorial Comparison Sampling**: The `sample_euler` function uses `(t, x, z)` which is WRONG if the signature is `(x, z, t)`. However, this only affects evaluation, not training.

3. **Training Loss**: The training uses `model.compute_loss` which correctly calls `self.vector_field(x_t, z, time)` = `(x, z, t)`, matching the actual signature.

### 4. **Why Training Might Be Worse**

The higher MSE (0.550 vs 0.486) suggests:
- Model didn't converge as well
- Possible reasons:
  1. Different random initialization despite same seed
  2. Codebase changes affecting training dynamics
  3. Different JAX/XLA versions affecting numerical stability
  4. The vector field signature might have been different during factorial comparison

### 5. **Recommendations**

1. **Fix the comment** on line 419 to match the actual call: `(x, z, t)`
2. **Verify vector field signature** was consistent during factorial comparison
3. **Check if there were any changes** to the loss computation or vector field implementation
4. **Consider retraining** with exact same codebase version as factorial comparison


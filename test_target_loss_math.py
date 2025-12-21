"""
Test script to verify the mathematical correctness of TARGET prediction loss computation.
"""

import jax
import jax.numpy as jnp
import numpy as np

# Simulate the loss computation
B, N, D = 2, 10, 2

# Generate test data
key = jax.random.PRNGKey(42)
k1, k2 = jax.random.split(key)
x_0 = jax.random.normal(k1, (B, N, D))  # Data
x_1 = jax.random.normal(k2, (B, N, D))  # Noise
t = jax.random.uniform(k2, (B,))  # Time

# Interpolate
t_exp = t[:, None, None]
x_t = (1 - t_exp) * x_0 + t_exp * x_1

# True velocity
v_true = x_1 - x_0

print("=" * 80)
print("Testing TARGET prediction loss computation")
print("=" * 80)
print(f"x_0 shape: {x_0.shape}")
print(f"x_1 shape: {x_1.shape}")
print(f"x_t shape: {x_t.shape}")
print(f"t shape: {t.shape}")
print(f"t values: {t}")
print()

# Test 1: Perfect prediction (crn_output = x_0 exactly)
print("Test 1: Perfect TARGET prediction")
crn_output_perfect = x_0.copy()
v_pred_perfect = (x_t - crn_output_perfect) / (t[:, None, None] + 1e-8)
v_error_perfect = v_pred_perfect - v_true
loss_perfect = jnp.mean(v_error_perfect ** 2)
print(f"  Predicted velocity error (should be ~0): {jnp.mean(jnp.abs(v_error_perfect)):.6f}")
print(f"  Loss (should be ~0): {loss_perfect:.6f}")
print()

# Test 2: Small error in x_0 prediction
print("Test 2: Small error in TARGET prediction")
error = 0.1
crn_output_error = x_0 + error
v_pred_error = (x_t - crn_output_error) / (t[:, None, None] + 1e-8)
v_error_error = v_pred_error - v_true
loss_error = jnp.mean(v_error_error ** 2)
print(f"  Error in x_0 prediction: {error}")
print(f"  Predicted velocity error: {jnp.mean(jnp.abs(v_error_error)):.6f}")
print(f"  Loss: {loss_error:.6f}")
print(f"  Expected velocity error: {error / jnp.mean(t):.6f} (error / mean(t))")
print()

# Test 3: Check when t is very small
print("Test 3: Behavior when t is very small")
t_small = jnp.array([0.001, 0.01])
t_small_exp = t_small[:, None, None]
x_t_small = (1 - t_small_exp) * x_0[:2] + t_small_exp * x_1[:2]
v_pred_small = (x_t_small - x_0[:2]) / (t_small_exp + 1e-8)
v_true_small = x_1[:2] - x_0[:2]
v_error_small = v_pred_small - v_true_small
print(f"  t values: {t_small}")
print(f"  Velocity error magnitude: {jnp.mean(jnp.abs(v_error_small), axis=(1, 2))}")
print(f"  Note: When t is small, small errors in x_0 get amplified by 1/t")
print()

# Test 4: Compare with direct velocity prediction
print("Test 4: Comparing TARGET vs VELOCITY prediction")
# When predicting velocity directly
v_pred_direct = v_true + 0.1  # Small error
loss_velocity = jnp.mean((v_pred_direct - v_true) ** 2)

# When predicting x_0 with same "effective" error
# If v_error = 0.1, and t = 0.5, then x_0_error = v_error * t = 0.05
t_avg = 0.5
x_0_error_equiv = 0.1 * t_avg
crn_output_equiv = x_0 + x_0_error_equiv
v_pred_equiv = (x_t - crn_output_equiv) / (t[:, None, None] + 1e-8)
loss_target = jnp.mean((v_pred_equiv - v_true) ** 2)

print(f"  Velocity prediction loss: {loss_velocity:.6f}")
print(f"  Target prediction loss (equivalent error): {loss_target:.6f}")
print(f"  Ratio: {loss_target / loss_velocity:.6f}")
print()

# Test 5: Check the actual loss computation path
print("Test 5: Actual loss computation (TARGET prediction, TARGET loss)")
sq_err_target = jnp.mean((crn_output_perfect - x_0) ** 2, axis=-1)
print(f"  Base squared error shape: {sq_err_target.shape}")
print(f"  Base squared error (should be ~0): {jnp.mean(sq_err_target):.6f}")
print()

print("=" * 80)
print("Analysis:")
print("=" * 80)
print("When CRN predicts TARGET (x_0):")
print("  - Loss: (crn_output - x_0)^2")
print("  - This is correct mathematically")
print()
print("Potential issues:")
print("  1. When t is large, x_t ≈ x_1 (noise), so predicting x_0 is very hard")
print("  2. When t is small, x_t ≈ x_0, so the task is easier")
print("  3. The uniform loss weighting doesn't account for this difficulty imbalance")
print("  4. This might cause the model to focus on easy cases (small t) and fail on hard cases (large t)")
print()
print("In contrast, velocity prediction:")
print("  - The velocity v = x_1 - x_0 is the same regardless of t")
print("  - The task difficulty is more uniform across t values")
print("  - This might explain why velocity prediction works better")





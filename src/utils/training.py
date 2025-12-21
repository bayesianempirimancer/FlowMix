"""
Training utilities for flow models.

Includes data augmentation, metrics, and evaluation functions.
"""

import jax
import jax.numpy as jnp
import numpy as np


def augment_batch(x: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
    """
    Apply random augmentation to a batch of point clouds.
    
    Args:
        x: (B, N, 2) point clouds
        key: random key
        
    Returns:
        Augmented point clouds (B, N, 2)
    """
    B, N, D = x.shape
    k1, k2, k3 = jax.random.split(key, 3)
    
    # Random rotation (0 to 2π)
    angles = jax.random.uniform(k1, (B,), minval=0, maxval=2 * jnp.pi)
    cos_a = jnp.cos(angles)[:, None, None]
    sin_a = jnp.sin(angles)[:, None, None]
    
    x_rot = x[..., 0:1] * cos_a - x[..., 1:2] * sin_a
    y_rot = x[..., 0:1] * sin_a + x[..., 1:2] * cos_a
    x_aug = jnp.concatenate([x_rot, y_rot], axis=-1)
    
    # Random scale (0.7 to 1.3)
    scales = jax.random.uniform(k2, (B, 1, 1), minval=0.7, maxval=1.3)
    x_aug = x_aug * scales
    
    # Random translation (-0.2 to 0.2)
    translations = jax.random.uniform(k3, (B, 1, D), minval=-0.2, maxval=0.2)
    x_aug = x_aug + translations
    
    return x_aug


def chamfer_distance(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Chamfer distance between two batches of point clouds.
    
    Args:
        x: (B, N, D) first point clouds
        y: (B, M, D) second point clouds
        
    Returns:
        (B,) Chamfer distances
    """
    # Pairwise squared distances
    x_sq = jnp.sum(x**2, axis=-1, keepdims=True)  # (B, N, 1)
    y_sq = jnp.sum(y**2, axis=-1, keepdims=True)  # (B, M, 1)
    dist_sq = x_sq + jnp.swapaxes(y_sq, -2, -1) - 2 * jnp.matmul(x, jnp.swapaxes(y, -2, -1))
    
    # Min distances
    min_dist_x = jnp.min(dist_sq, axis=2)  # (B, N)
    min_dist_y = jnp.min(dist_sq, axis=1)  # (B, M)
    
    return jnp.mean(min_dist_x, axis=1) + jnp.mean(min_dist_y, axis=1)


def sample_with_euler(model, variables, z: jnp.ndarray, num_points: int, 
                      key: jax.random.PRNGKey, num_steps: int = 20) -> jnp.ndarray:
    """
    Sample point clouds using Euler integration (much faster than diffeqsolve).
    
    This avoids the expensive XLA compilation of diffeqsolve by using
    a simple fixed-step Euler integrator with jax.lax.fori_loop.
    
    Args:
        model: MnistFlow2D model
        variables: model parameters
        z: latent codes (B, Dz)
        num_points: number of points to generate
        key: random key
        num_steps: number of Euler steps
        
    Returns:
        Generated point clouds (B, num_points, 2)
    """
    B = z.shape[0]
    
    # Start from prior (noise at t=1)
    x_init = jax.random.normal(key, (B, num_points, 2))
    
    # Integrate from t=1 to t=0 using Euler with jax.lax.fori_loop
    dt = -1.0 / num_steps
    
    # Define helper to call vector field
    def call_vf(m, t_val, x_val, z_val):
        return m.vector_field(t_val, x_val, z_val)
    
    def euler_step(step, x_t):
        t = 1.0 + step * dt
        v = model.apply(variables, t, x_t, z, method=call_vf)
        return x_t + dt * v
    
    x_final = jax.lax.fori_loop(0, num_steps, euler_step, x_init)
    
    return x_final


def evaluate_chamfer(model, variables, X_test: jnp.ndarray, key: jax.random.PRNGKey, 
                     batch_size: int = 64, num_points: int = 500, 
                     max_samples: int = 200) -> float:
    """
    Evaluate Chamfer distance on test set using fast Euler sampling.
    
    Uses simple Euler integration instead of diffeqsolve for speed.
    Limited to max_samples to keep evaluation fast during training.
    """
    num_samples = min(len(X_test), max_samples)
    total_chamfer = 0.0
    count = 0
    
    for i in range(0, num_samples, batch_size):
        batch_x = X_test[i:i+batch_size]
        actual_batch = len(batch_x)
        
        key, k_enc, k_sample = jax.random.split(key, 3)
        
        # Encode
        z_batch, _, _ = model.apply(variables, batch_x, k_enc, method=model.forward_inference)
        
        # Sample using fast Euler integration
        x_gen = sample_with_euler(model, variables, z_batch, num_points, k_sample)
        
        # Compute Chamfer
        cd_batch = chamfer_distance(batch_x, x_gen)
        total_chamfer += float(jnp.sum(cd_batch))
        count += actual_batch
    
    return total_chamfer / count


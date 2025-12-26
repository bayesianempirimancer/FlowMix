"""
Plotting utilities for sample generation visualization.

Includes conditional and unconditional sample plotting.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


def plot_unconditional_samples(model, params, key, output_dir, num_samples=36, num_points=500):
    """Plot unconditional samples by sampling z and generating point clouds.
    
    If use_prior_flow=True: samples z from learned prior distribution.
    If use_prior_flow=False: samples z from unit normal distribution.
    
    Args:
        model: Trained flow model
        params: Model parameters
        key: JAX random key
        output_dir: Directory to save the plot
        num_samples: Number of samples to generate
        num_points: Number of points per sample
    """
    print(f"    Generating {num_samples} unconditional samples...")
    key, k_sample = jax.random.split(key)
    
    # Generate all samples in a single batch call (more efficient)
    # Use batch_size parameter to generate num_samples at once
    x_gen = model.apply(params, num_points, k_sample, z=None, num_steps=20, batch_size=num_samples, method=model.sample)
    
    # x_gen will have shape (num_samples, num_points, 2) if num_samples > 1
    # or (num_points, 2) if num_samples == 1
    if x_gen.ndim == 2:
        x_gen = x_gen[None, :, :]  # Add batch dimension if single sample
    
    # Plot on 6x6 grid
    n_cols = 6
    n_rows = 6
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(num_samples):
        ax = axes[i]
        ax.scatter(x_gen[i, :, 0], x_gen[i, :, 1], s=1, alpha=0.6)
        ax.set_title(f"Sample {i+1}", fontsize=8)
        ax.set_aspect('equal')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axis('off')  # Remove axes for cleaner look
    
    plt.tight_layout()
    output_path = output_dir / "unconditional_samples.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved unconditional samples to {output_path}")


def plot_samples(model, params, X_test, key, output_dir, num_samples=16, 
                 use_grid_mask=False, grid_size=3, mask_prob=0.2, create_grid_mask_fn=None):
    """Plot generated samples (conditional generation from test data).
    
    Args:
        model: Trained flow model
        params: Model parameters
        X_test: Test point clouds
        key: JAX random key
        output_dir: Directory to save the plot
        num_samples: Number of samples to plot
        use_grid_mask: If True, apply grid mask to ground truth images to show what encoder saw
        grid_size: Size of the grid for masking
        mask_prob: Probability of masking each grid cell
        create_grid_mask_fn: Function to create grid mask (if None and use_grid_mask=True, will error)
    """
    num_plot = min(len(X_test), num_samples)
    
    # Get a batch
    batch_x = X_test[:num_plot]
    key, k_enc, k_sample = jax.random.split(key, 3)
    
    # Generate encoder mask if enabled (same mask used during training)
    enc_mask = None
    if use_grid_mask:
        if create_grid_mask_fn is None:
            raise ValueError("create_grid_mask_fn must be provided when use_grid_mask=True")
        key, k_mask = jax.random.split(key)
        enc_mask = create_grid_mask_fn(batch_x, k_mask, grid_size=grid_size, mask_prob=mask_prob)
    
    # Encode (with enc_mask if enabled - encoder handles mask=None)
    z_batch, _, _ = model.apply(params, batch_x, k_enc, enc_mask, method=model.encode)
    
    # Sample in batch using vmap (much faster!)
    keys_sample = jax.random.split(k_sample, num_plot)
    
    # Sample batch directly (sample method now handles batch shapes)
    # z_batch already has batch dimension, so we can pass it directly
    x_gen = model.apply(params, batch_x.shape[1], k_sample, z=z_batch, num_steps=20, batch_size=None, method=model.sample)
    # x_gen will have shape (num_plot, N, 2)
    
    # Plot
    n_cols = 4
    n_rows = (num_plot + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(12, 3 * n_rows))
    axes = axes.flatten()
    
    for i in range(num_plot):
        # Ground truth (show masked version if grid masking is enabled)
        ax = axes[i * 2]
        if use_grid_mask and enc_mask is not None:
            # Show what encoder saw: visible points in black, masked points in red
            visible_mask = enc_mask[i]
            masked_mask = ~enc_mask[i]
            
            if jnp.any(visible_mask):
                ax.scatter(batch_x[i, visible_mask, 0], batch_x[i, visible_mask, 1], 
                          s=1, alpha=0.5, c='black', label='Visible')
            if jnp.any(masked_mask):
                ax.scatter(batch_x[i, masked_mask, 0], batch_x[i, masked_mask, 1], 
                          s=1, alpha=0.3, c='red', label='Masked')
            ax.set_title(f"GT (masked) {i}")
        else:
            ax.scatter(batch_x[i, :, 0], batch_x[i, :, 1], s=1, alpha=0.5)
            ax.set_title(f"GT {i}")
        ax.set_aspect('equal')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        
        # Generated
        ax = axes[i * 2 + 1]
        ax.scatter(x_gen[i, :, 0], x_gen[i, :, 1], s=1, alpha=0.5, color='orange')
        ax.set_title(f"Gen {i}")
        ax.set_aspect('equal')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
    
    # Hide unused axes
    for i in range(num_plot * 2, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / "samples.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✓ Saved samples to {output_path}")


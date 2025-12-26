"""
Plotting utilities for training flow models.

Includes loss trajectory plots and sample visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from pathlib import Path


def plot_loss_trajectories(loss_history, chamfer_history, output_dir):
    """
    Plot training loss and Chamfer distance trajectories.
    
    Args:
        loss_history: List of (epoch, mse) tuples
        chamfer_history: List of (epoch, chamfer) tuples
        output_dir: Path to save plot
    """
    epochs = [x[0] for x in loss_history]
    mse_values = [x[1] for x in loss_history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # MSE loss
    ax1.plot(epochs, mse_values, 'b-', linewidth=2, label='MSE Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training Loss (MSE)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Chamfer distance
    if chamfer_history:
        chamfer_epochs = [x[0] for x in chamfer_history]
        chamfer_values = [x[1] for x in chamfer_history]
        ax2.plot(chamfer_epochs, chamfer_values, 'r-', linewidth=2, marker='o', label='Chamfer Distance')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Chamfer Distance')
        ax2.set_title('Chamfer Distance (Lower is Better)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.tight_layout()
    output_path = Path(output_dir) / "loss_trajectories.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved loss trajectories to {output_path}")


def plot_samples(model, variables, X_test, output_dir, num_samples=16):
    """
    Generate and plot conditional and unconditional samples.
    
    Args:
        model: Trained flow model
        variables: Model parameters
        X_test: Test point clouds
        output_dir: Directory to save plots
        num_samples: Number of samples to generate
    """
    import jax
    
    key = jax.random.PRNGKey(42)
    
    # Select test samples for conditional generation
    np.random.seed(42)
    test_indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
    X_cond = X_test[test_indices]
    
    # Generate conditional samples (from q(z|x))
    print("Generating conditional samples...")
    conditional_samples = []
    for i in range(len(X_cond)):
        key, k_encode, k_sample = jax.random.split(key, 3)
        # Encode to get z from q(z|x)
        z, z_mu, z_logvar = model.apply(
            variables,
            X_cond[i:i+1],
            k_encode,
            method=model.forward_inference
        )
        # Sample point cloud conditioned on z
        sample = model.apply(
            variables,
            500,  # num_points
            k_sample,  # key
            z,  # z
            20,  # num_steps
            None,  # batch_size (ignored since z is provided)
            method=model.sample
        )
        conditional_samples.append(np.array(sample))
    
    # Generate unconditional samples (from p(z) = N(0,I))
    print("Generating unconditional samples...")
    unconditional_samples = []
    for i in range(num_samples):
        key, k_sample = jax.random.split(key)
        # Sample z from N(0,I) and generate
        sample = model.apply(
            variables,
            500,  # num_points
            k_sample,  # key
            None,  # z (will sample from prior)
            20,  # num_steps
            1,  # batch_size (single sample)
            method=model.sample
        )
        unconditional_samples.append(np.array(sample))
    
    output_path = Path(output_dir)
    
    # Plot conditional samples
    n_cols = 4
    n_rows = (len(conditional_samples) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_rows == 1:
        axes = axes[None, :] if axes.ndim == 1 else axes
    elif n_cols == 1:
        axes = axes[:, None] if axes.ndim == 1 else axes
    
    for i in range(len(conditional_samples)):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # Plot ground truth
        ax.scatter(X_cond[i, :, 0], X_cond[i, :, 1], s=2, c='blue', alpha=0.3, label='GT')
        # Plot generated
        ax.scatter(conditional_samples[i][:, 0], conditional_samples[i][:, 1], 
                  s=2, c='red', alpha=0.5, label='Gen')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        if i == 0:
            ax.legend(fontsize=8, loc='upper right')
    
    # Hide extra subplots
    for i in range(len(conditional_samples), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.axis('off')
    
    plt.suptitle('Conditional Samples (from q(z|x))', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / "conditional_samples.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved conditional samples to {output_path / 'conditional_samples.png'}")
    
    # Plot unconditional samples
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_rows == 1:
        axes = axes[None, :] if axes.ndim == 1 else axes
    elif n_cols == 1:
        axes = axes[:, None] if axes.ndim == 1 else axes
    
    for i in range(len(unconditional_samples)):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        ax.scatter(unconditional_samples[i][:, 0], unconditional_samples[i][:, 1], 
                  s=2, c='black', alpha=0.6)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
    
    # Hide extra subplots
    for i in range(len(unconditional_samples), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.axis('off')
    
    plt.suptitle('Unconditional Samples (from p(z) = N(0,I))', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / "unconditional_samples.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved unconditional samples to {output_path / 'unconditional_samples.png'}")


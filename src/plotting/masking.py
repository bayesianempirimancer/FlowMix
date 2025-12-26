"""
Plotting utilities for masked training visualization.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable


def visualize_masked_digits(X, create_grid_mask_fn: Callable, output_dir: Path, 
                           num_samples: int = 16, grid_size: int = 6, mask_prob: float = 0.2,
                           seed: int = 42):
    """
    Visualize masked digits used during training.
    
    Args:
        X: Point cloud data, shape (N, num_points, 2)
        create_grid_mask_fn: Function to create grid mask
        output_dir: Directory to save the visualization
        num_samples: Number of samples to visualize
        grid_size: Size of the grid (e.g., 6 for 6x6 grid)
        mask_prob: Probability of masking each grid cell
        seed: Random seed
    """
    # Get a batch of samples
    x_batch = X[:num_samples]  # (num_samples, N, 2)
    
    # Generate grid mask
    key = jax.random.PRNGKey(seed)
    key, k_mask = jax.random.split(key)
    mask = create_grid_mask_fn(x_batch, k_mask, grid_size=grid_size, mask_prob=mask_prob)
    
    # Create masked version (set masked points to NaN for visualization)
    x_masked = x_batch.copy()
    x_masked = np.array(x_masked)  # Convert to numpy for easier manipulation
    mask_np = np.array(mask)
    x_masked[~mask_np] = np.nan  # Set masked points to NaN
    
    # Create visualization
    n_cols = 4
    n_rows = (num_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i in range(num_samples):
        # Original (left column)
        ax_orig = axes[i * 2]
        x_orig = x_batch[i]  # (N, 2)
        ax_orig.scatter(x_orig[:, 0], x_orig[:, 1], s=1, c='black', alpha=0.6)
        ax_orig.set_title(f"Original {i+1}", fontsize=10)
        ax_orig.set_aspect('equal')
        ax_orig.set_xlim(-1.5, 1.5)
        ax_orig.set_ylim(-1.5, 1.5)
        ax_orig.axis('off')
        
        # Masked (right column)
        ax_masked = axes[i * 2 + 1]
        x_m = x_masked[i]  # (N, 2)
        # Plot visible points
        visible_mask = mask_np[i]
        if np.any(visible_mask):
            ax_masked.scatter(x_m[visible_mask, 0], x_m[visible_mask, 1], 
                            s=1, c='black', alpha=0.6, label='Visible')
        # Plot masked points (if any are not NaN)
        masked_mask = ~mask_np[i]
        if np.any(masked_mask):
            # Show masked regions as red points or gray
            masked_points = x_batch[i][masked_mask]
            ax_masked.scatter(masked_points[:, 0], masked_points[:, 1], 
                            s=1, c='red', alpha=0.3, label='Masked')
        
        # Draw grid lines to show grid structure
        x_min = np.min(x_batch[i], axis=0)
        x_max = np.max(x_batch[i], axis=0)
        x_range = x_max[0] - x_min[0]
        y_range = x_max[1] - x_min[1]
        
        # Draw grid
        for gx in range(grid_size + 1):
            x_val = x_min[0] + (x_range / grid_size) * gx
            ax_masked.axvline(x_val, color='gray', linestyle='--', alpha=0.2, linewidth=0.5)
        for gy in range(grid_size + 1):
            y_val = x_min[1] + (y_range / grid_size) * gy
            ax_masked.axhline(y_val, color='gray', linestyle='--', alpha=0.2, linewidth=0.5)
        
        # Count masked points
        num_masked = np.sum(masked_mask)
        num_visible = np.sum(visible_mask)
        ax_masked.set_title(f"Masked {i+1} ({num_visible}/{num_visible+num_masked} visible)", fontsize=10)
        ax_masked.set_aspect('equal')
        ax_masked.set_xlim(-1.5, 1.5)
        ax_masked.set_ylim(-1.5, 1.5)
        ax_masked.axis('off')
    
    # Hide unused subplots
    for i in range(num_samples * 2, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Grid Masking Visualization (grid_size={grid_size}, mask_prob={mask_prob})', 
                 fontsize=14, y=0.995)
    plt.tight_layout()
    
    output_path = output_dir / "masked_training_examples.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved masked training examples to {output_path}")


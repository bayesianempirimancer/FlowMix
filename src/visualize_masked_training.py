"""
Visualize masked digits used during training with grid masking.
"""

from pathlib import Path

from src.train_all_digits import create_grid_mask, load_all_digits_data
from src.plotting import visualize_masked_digits


def main():
    # Configuration
    output_dir = Path("artifacts/all_digits_pointnet_adaln_velocity_no_vae_prior_flow_joint_squash_gridmask")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_samples = 16
    grid_size = 3  # Updated to 3x3 grid
    mask_prob = 0.2
    
    print("=" * 80)
    print("Visualizing Masked Training Examples")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Number of samples: {num_samples}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Mask probability: {mask_prob}")
    print()
    
    # Load data
    print("Loading data...")
    X = load_all_digits_data(dataset_path="data/mnist_2d_single.npz", max_samples=None, seed=42)
    
    # Use plotting function
    visualize_masked_digits(X, create_grid_mask, output_dir, 
                           num_samples=num_samples, grid_size=grid_size, mask_prob=mask_prob)
    
    print("\n" + "=" * 80)
    print("Visualization Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

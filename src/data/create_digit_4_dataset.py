"""
Create a filtered dataset containing only digit 4 from the full MNIST 2D dataset.

This script loads MNIST labels to properly identify digit 4 samples, then filters
the point cloud dataset accordingly.

This is a one-time script to create mnist_2d_just_4s.npz from mnist_2d_full_dataset.npz.
Note: Requires tensorflow_datasets for this one-time operation.
"""

import numpy as np
import tensorflow_datasets as tfds
from pathlib import Path
import tqdm

def create_digit_4_dataset():
    """Load full dataset and filter for digit 4 using actual MNIST labels."""
    
    # Paths
    input_path = "data/mnist_2d_full_dataset.npz"
    output_path = "data/mnist_2d_just_4s.npz"
    
    print("=" * 80)
    print("Creating Digit 4 Dataset")
    print("=" * 80)
    
    # Load existing point cloud dataset
    print(f"\nLoading {input_path}...")
    data = np.load(input_path)
    points_all = data['points']  # (60000, 500, 2)
    print(f"Loaded {len(points_all)} point clouds")
    
    # Load MNIST labels to identify digit 4
    print("\nLoading MNIST labels from tensorflow_datasets...")
    print("(This is a one-time operation to get the correct labels)")
    ds = tfds.load('mnist', split='train', shuffle_files=False)
    
    labels = []
    for ex in tqdm.tqdm(tfds.as_numpy(ds), total=60000, desc="Loading labels"):
        labels.append(ex['label'])
    
    labels = np.array(labels)
    print(f"Loaded {len(labels)} labels")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Filter for digit 4
    digit_4_indices = np.where(labels == 4)[0]
    print(f"\nFound {len(digit_4_indices)} digit 4 samples")
    print(f"Indices range: {digit_4_indices.min()} to {digit_4_indices.max()}")
    
    # Extract digit 4 point clouds
    points_4 = points_all[digit_4_indices]
    print(f"\nExtracted point clouds shape: {points_4.shape}")
    
    # Save filtered dataset
    print(f"\nSaving to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, points=points_4)
    
    print(f"✓ Saved {len(points_4)} digit 4 point clouds to {output_path}")
    
    # Verify
    print("\nVerifying saved dataset...")
    data_check = np.load(output_path)
    print(f"  Points shape: {data_check['points'].shape}")
    print(f"  Points range: [{data_check['points'].min():.2f}, {data_check['points'].max():.2f}]")
    
    print("\n" + "=" * 80)
    print("✓ Dataset creation complete!")
    print("=" * 80)


if __name__ == "__main__":
    create_digit_4_dataset()


"""
Analyze encoder latent representations for MNIST digits.

This script:
1. Loads a trained model
2. Encodes all MNIST samples to get latent codes z
3. Computes statistics (mean, variance) overall and per digit
4. Creates a t-SNE visualization colored by digit identity
"""

import jax
import jax.numpy as jnp
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import tensorflow_datasets as tfds

from src.models.mnist_flow_2d import MnistFlow2D, PredictionTarget
from src.plotting import create_tsne_plot


def load_model_and_checkpoint(checkpoint_path):
    """Load model and parameters from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    params = checkpoint['params']
    
    # Reconstruct model with same config as training script
    # Update these to match the training configuration
    model = MnistFlow2D(
        latent_dim=32,  # Reduced from 64
        encoder_type='pointnet',
        encoder_output_type='global',
        crn_type='adaln_mlp',
        crn_kwargs={
            'hidden_dims': (32, 32, 32, 32, 32, 32),  # Reduced from (64, 64, 64, 64, 64, 64)
            'cond_dim': 64,  # Reduced from 128
        },
        prediction_target=PredictionTarget.VELOCITY,
        loss_targets=(PredictionTarget.VELOCITY,),  # Match training config
        use_vae=False,  # Match training config (no VAE, using LayerNorm)
        vae_kl_weight=0.0,  # Match training config
        marginal_kl_weight=0.0,  # Match training config
        use_prior_flow=True,  # Match training config
    )
    
    print(f"Model loaded: latent_dim=32, encoder_type=pointnet, use_vae=False")
    
    return model, params


def load_mnist_data_with_labels(dataset_path="data/mnist_2d_full_dataset.npz", train_split=0.9, seed=42, use_test_only=False):
    """Load MNIST point clouds and labels.
    
    Args:
        dataset_path: Path to dataset file
        train_split: Train/test split ratio
        seed: Random seed for splitting
        use_test_only: If True, only return test set (for t-SNE on validation data)
    """
    print(f"Loading MNIST data from {dataset_path}...")
    
    # Load point clouds
    data = np.load(dataset_path)
    points = data['points']  # (60000, 500, 2)
    
    # Load labels from tensorflow_datasets
    print("Loading MNIST labels...")
    ds = tfds.load('mnist', split='train', shuffle_files=False)
    labels = []
    for ex in tqdm(tfds.as_numpy(ds), total=60000, desc="Loading labels"):
        labels.append(ex['label'])
    labels = np.array(labels)
    
    # Split into train/test (matching training script)
    n = len(points)
    n_train = int(n * train_split)
    
    rng = np.random.RandomState(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    if use_test_only:
        points = points[test_indices]
        labels = labels[test_indices]
        print(f"Using test set only: {len(points)} point clouds")
    else:
        print(f"Loaded {len(points)} point clouds with labels")
    
    print(f"Label distribution: {np.bincount(labels)}")
    
    return jnp.array(points), labels


def encode_all_samples(model, params, X, batch_size=128):
    """Encode all samples to get latent codes z."""
    print(f"Encoding {len(X)} samples...")
    
    all_z = []
    num_batches = (len(X) + batch_size - 1) // batch_size
    
    key = jax.random.PRNGKey(42)
    
    for i in tqdm(range(num_batches), desc="Encoding"):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(X))
        batch_x = X[start_idx:end_idx]
        
        # Encode batch
        key, k_enc = jax.random.split(key)
        z_batch, _, _ = model.apply(params, batch_x, k_enc, method=model.encode)
        
        all_z.append(np.array(z_batch))
    
    z_all = np.concatenate(all_z, axis=0)
    print(f"Encoded latent codes shape: {z_all.shape}")
    
    return z_all


def compute_statistics(z_all, labels):
    """Compute overall and per-digit statistics."""
    print("\n" + "=" * 80)
    print("Statistical Analysis")
    print("=" * 80)
    
    # Overall statistics
    z_mean_overall = np.mean(z_all, axis=0)
    z_var_overall = np.var(z_all, axis=0)
    z_std_overall = np.std(z_all, axis=0)
    
    print(f"\nOverall Statistics (across all digits):")
    print(f"  Mean: {z_mean_overall}")
    print(f"  Mean (per dimension): min={z_mean_overall.min():.4f}, "
          f"max={z_mean_overall.max():.4f}, mean={z_mean_overall.mean():.4f}")
    print(f"  Variance: {z_var_overall}")
    print(f"  Variance (per dimension): min={z_var_overall.min():.4f}, "
          f"max={z_var_overall.max():.4f}, mean={z_var_overall.mean():.4f}")
    print(f"  Std Dev (per dimension): min={z_std_overall.min():.4f}, "
          f"max={z_std_overall.max():.4f}, mean={z_std_overall.mean():.4f}")
    
    # Per-digit statistics
    print(f"\nPer-Digit Statistics:")
    digit_stats = {}
    
    for digit in range(10):
        digit_mask = labels == digit
        z_digit = z_all[digit_mask]
        
        if len(z_digit) > 0:
            z_mean_digit = np.mean(z_digit, axis=0)
            z_var_digit = np.var(z_digit, axis=0)
            z_std_digit = np.std(z_digit, axis=0)
            
            digit_stats[digit] = {
                'mean': z_mean_digit,
                'var': z_var_digit,
                'std': z_std_digit,
                'count': len(z_digit)
            }
            
            print(f"\n  Digit {digit} (n={len(z_digit)}):")
            print(f"    Mean (per dim): min={z_mean_digit.min():.4f}, "
                  f"max={z_mean_digit.max():.4f}, mean={z_mean_digit.mean():.4f}")
            print(f"    Variance (per dim): min={z_var_digit.min():.4f}, "
                  f"max={z_var_digit.max():.4f}, mean={z_var_digit.mean():.4f}")
            print(f"    Std Dev (per dim): min={z_std_digit.min():.4f}, "
                  f"max={z_std_digit.max():.4f}, mean={z_std_digit.mean():.4f}")
    
    return z_mean_overall, z_var_overall, digit_stats




def main():
    """Main analysis function."""
    import sys
    import os
    sys.path.append(os.path.abspath('.'))
    
    # Configuration
    checkpoint_path = Path("artifacts/all_digits_pointnet_adaln_velocity_no_vae_prior_flow_joint_squash_momentum_default/final_model.pkl")
    
    # Check if checkpoint exists
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train a model first or specify a different checkpoint path.")
        return
    
    # Use the checkpoint's directory as the output directory
    output_dir = checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load model
    model, params = load_model_and_checkpoint(checkpoint_path)
    
    # Load all data once (more efficient than loading twice)
    # Use same split as training script: train_split=0.9, seed=42
    X_all, labels_all = load_mnist_data_with_labels(dataset_path="data/mnist_2d_single.npz", train_split=0.9, seed=42, use_test_only=False)
    
    # Split into train/test for t-SNE (use test set only for visualization)
    # Match the split used in training script
    train_split = 0.9
    seed = 42
    n = len(X_all)
    n_train = int(n * train_split)
    
    # Use same shuffling as training script for consistent split
    rng = np.random.RandomState(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    test_indices = indices[n_train:]
    
    X_test = X_all[test_indices]
    labels_test = labels_all[test_indices]
    
    # Encode all samples for overall statistics
    print("\nEncoding all samples for overall statistics...")
    z_all = encode_all_samples(model, params, X_all, batch_size=128)
    
    # Encode test set for t-SNE
    print("\nEncoding test set for t-SNE visualization...")
    z_test = encode_all_samples(model, params, X_test, batch_size=128)
    
    # Compute statistics on all data
    z_mean_overall, z_var_overall, digit_stats = compute_statistics(z_all, labels_all)
    
    # Save statistics
    stats_path = output_dir / "latent_statistics.txt"
    with open(stats_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Encoder Latent Code Statistics\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Overall Statistics (across all digits):\n")
        f.write(f"  Mean (per dimension): min={z_mean_overall.min():.4f}, "
                f"max={z_mean_overall.max():.4f}, mean={z_mean_overall.mean():.4f}\n")
        f.write(f"  Variance (per dimension): min={z_var_overall.min():.4f}, "
                f"max={z_var_overall.max():.4f}, mean={z_var_overall.mean():.4f}\n\n")
        
        f.write("Per-Digit Statistics:\n")
        for digit in range(10):
            if digit in digit_stats:
                stats = digit_stats[digit]
                f.write(f"\n  Digit {digit} (n={stats['count']}):\n")
                f.write(f"    Mean (per dim): min={stats['mean'].min():.4f}, "
                        f"max={stats['mean'].max():.4f}, mean={stats['mean'].mean():.4f}\n")
                f.write(f"    Variance (per dim): min={stats['var'].min():.4f}, "
                        f"max={stats['var'].max():.4f}, mean={stats['var'].mean():.4f}\n")
                f.write(f"    Std Dev (per dim): min={stats['std'].min():.4f}, "
                        f"max={stats['std'].max():.4f}, mean={stats['std'].mean():.4f}\n")
    
    print(f"\nSaved statistics to {stats_path}")
    
    # Create t-SNE plot (using test set only)
    tsne_path = output_dir / "tsne_latent_codes.png"
    create_tsne_plot(z_test, labels_test, tsne_path)
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()


"""
Factorial comparison script for testing different model configurations.

Tests all combinations of:
1. VAE loss (on/off)
2. Marginal KL loss (on/off)  
3. Z normalization (on/off)

Total: 2^3 = 8 combinations
Each model trains for 200 epochs, then t-SNE analysis is run.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from pathlib import Path
from tqdm import tqdm
import pickle
import os
import sys
from datetime import datetime
import itertools

from src.models.mnist_flow_2d import MnistFlow2D, PredictionTarget
from src.utils.training import chamfer_distance
from src.train_all_digits import load_all_digits_data, create_data_loaders
from src.plotting import plot_samples, plot_unconditional_samples, plot_loss_trajectory
from src.analyze_encoder_latents import load_model_and_checkpoint, load_mnist_data_with_labels, encode_all_samples
from src.plotting import create_tsne_plot


def train_single_config(config, base_output_dir, epochs=200):
    """Train a single model configuration.
    
    Args:
        config: Dictionary with keys: use_vae, vae_kl_weight, marginal_kl_weight, normalize_z
        base_output_dir: Base directory for all factorial results
        epochs: Number of training epochs
        
    Returns:
        Path to the final model checkpoint
    """
    # Create output directory for this configuration
    config_name = f"vae_{config['use_vae']}_vae_kl_{config['vae_kl_weight']}_marginal_kl_{config['marginal_kl_weight']}_normalize_z_{config['normalize_z']}"
    output_dir = base_output_dir / config_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"Training Configuration: {config_name}")
    print("=" * 80)
    print(f"  use_vae: {config['use_vae']}")
    print(f"  vae_kl_weight: {config['vae_kl_weight']}")
    print(f"  marginal_kl_weight: {config['marginal_kl_weight']}")
    print(f"  normalize_z: {config['normalize_z']}")
    print(f"  Output directory: {output_dir}")
    print()
    
    # Training configuration
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    MAX_SAMPLES = None
    SEED = 123  # New random seed for this run
    CHECKPOINT_INTERVAL = 50
    BETA1, BETA2 = 0.9, 0.999  # Default momentum
    
    # Load data
    print("Loading full MNIST data...")
    X = load_all_digits_data(dataset_path="data/mnist_2d_single.npz", max_samples=MAX_SAMPLES, seed=SEED)
    X_train, X_test, get_batches = create_data_loaders(X, batch_size=BATCH_SIZE, seed=SEED)
    print()
    
    # Initialize model
    print("Initializing model...")
    model = MnistFlow2D(
        latent_dim=32,
        encoder_type='pointnet',
        encoder_output_type='global',
        crn_type='adaln_mlp',
        crn_kwargs={
            'hidden_dims': (32, 32, 32, 32, 32, 32),
            'cond_dim': 64,
        },
        prediction_target=PredictionTarget.VELOCITY,
        loss_targets=(PredictionTarget.VELOCITY,),
        use_vae=config['use_vae'],
        vae_kl_weight=config['vae_kl_weight'],
        marginal_kl_weight=config['marginal_kl_weight'],
        use_prior_flow=True,
        prior_flow_kwargs={
            'hidden_dims': (128, 128, 128, 128, 128),
            'time_embed_dim': 128,
        },
        optimal_reweighting=False,
        normalize_z=config['normalize_z'],
    )
    
    # Initialize parameters
    key = jax.random.PRNGKey(SEED)
    dummy_x = jnp.ones((2, 500, 2))
    key, k_init = jax.random.split(key)
    params = model.init(k_init, dummy_x, k_init)
    
    # Initialize optimizer
    optimizer = optax.adam(LEARNING_RATE, b1=BETA1, b2=BETA2)
    opt_state = optimizer.init(params)
    
    # Training loop
    print(f"Training for {epochs} epochs...")
    loss_history = []
    chamfer_history = []
    
    @jax.jit
    def train_step(params, opt_state, x_batch, key):
        def loss_fn(p):
            loss, metrics = model.apply(p, x_batch, key, method=model.compute_loss)
            return loss, metrics
        
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, metrics
    
    num_batches = (len(X_train) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for epoch in range(epochs):
        key, k_epoch = jax.random.split(key)
        # Simple sequential batching (same as get_batches from create_data_loaders)
        n_train = len(X_train)
        batches = [X_train[i:i+BATCH_SIZE] for i in range(0, n_train, BATCH_SIZE)]
        
        epoch_losses = []
        epoch_metrics = {'flow_loss': [], 'prior_flow_loss': [], 'vae_kl': [], 'marginal_kl': []}
        
        for batch_idx, batch_x in enumerate(tqdm(batches, desc=f"Epoch {epoch+1}/{epochs}", 
                                                  total=num_batches, ncols=100, mininterval=0.5)):
            key, k_batch = jax.random.split(k_epoch)
            params, opt_state, loss, metrics = train_step(params, opt_state, batch_x, k_batch)
            
            epoch_losses.append(float(loss))
            for k, v in metrics.items():
                if k in epoch_metrics:
                    epoch_metrics[k].append(float(v))
        
        avg_loss = np.mean(epoch_losses)
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        loss_history.append({'epoch': epoch+1, 'loss': avg_loss, **avg_metrics})
        
        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
        if avg_metrics:
            print(f"  Flow: {avg_metrics.get('flow_loss', 0):.4f}, "
                  f"Prior: {avg_metrics.get('prior_flow_loss', 0):.4f}, "
                  f"VAE KL: {avg_metrics.get('vae_kl', 0):.4f}, "
                  f"Marginal KL: {avg_metrics.get('marginal_kl', 0):.4f}")
        
        # Evaluation and checkpointing
        if (epoch + 1) % 10 == 0 or epoch == 0:
            key, k_eval = jax.random.split(key)
            chamfer = evaluate(model, params, X_test, k_eval, batch_size=BATCH_SIZE, num_samples=100)
            chamfer_history.append({'epoch': epoch+1, 'chamfer': chamfer})
            print(f"  Chamfer distance: {chamfer:.6f}")
        
        if CHECKPOINT_INTERVAL and (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pkl"
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({'params': params, 'opt_state': opt_state, 'epoch': epoch+1}, f)
            print(f"  Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = output_dir / "final_model.pkl"
    with open(final_path, 'wb') as f:
        pickle.dump({'params': params, 'opt_state': opt_state, 'epoch': epochs}, f)
    print(f"\n✓ Saved final model to {final_path}")
    
    # Generate plots
    print("\nGenerating plots...")
    key, k_plot = jax.random.split(key)
    plot_loss_trajectory(loss_history, chamfer_history, output_dir)
    
    key, k_samples = jax.random.split(k_plot)
    plot_samples(model, params, X_test, k_samples, output_dir, num_samples=16)
    
    key, k_uncond = jax.random.split(k_samples)
    plot_unconditional_samples(model, params, k_uncond, output_dir, num_samples=36)
    
    print(f"✓ All plots saved to {output_dir}")
    
    return final_path


def evaluate(model, params, X_test, key, batch_size=32, num_samples=100):
    """Evaluate model using Chamfer distance."""
    total_chamfer = 0.0
    count = 0
    
    # Simple sequential batching (same as get_batches from create_data_loaders)
    n = len(X_test)
    for i in range(0, n, batch_size):
        batch_x = X_test[i:i+batch_size]
        actual_batch = len(batch_x)
        key, k_enc, k_sample = jax.random.split(key, 3)
        
        # Encode
        z_batch, _, _ = model.apply(params, batch_x, k_enc, method=model.encode)
        
        # Sample
        keys_sample = jax.random.split(k_sample, actual_batch)
        x_gen = sample_batch(model, params, z_batch, keys_sample, X_test.shape[1])
        
        # Compute Chamfer distance
        cd_batch = chamfer_distance(batch_x, x_gen)
        total_chamfer += float(jnp.sum(cd_batch))
        count += actual_batch
    
    return total_chamfer / count if count > 0 else 0.0


def sample_batch(model, params, z_batch, keys_sample, num_points):
    """Sample a batch of point clouds efficiently."""
    key_batch = keys_sample[0] if len(keys_sample) > 0 else jax.random.PRNGKey(0)
    x_gen = model.apply(params, num_points, key_batch, z=z_batch, num_steps=20, batch_size=None, method=model.sample)
    return x_gen


def run_tsne_analysis(checkpoint_path, config):
    """Run t-SNE analysis on a trained model."""
    print(f"\n{'='*80}")
    print(f"Running t-SNE analysis for: {checkpoint_path.parent.name}")
    print(f"{'='*80}")
    
    # Load model
    model, params = load_model_and_checkpoint(checkpoint_path)
    
    # Load data
    X_all, labels_all = load_mnist_data_with_labels(dataset_path="data/mnist_2d_single.npz", 
                                                     train_split=0.9, seed=42, use_test_only=False)
    
    # Split into test set (same as training)
    train_split = 0.9
    seed = 42
    n = len(X_all)
    n_train = int(n * train_split)
    rng = np.random.RandomState(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    test_indices = indices[n_train:]
    X_test = X_all[test_indices]
    labels_test = labels_all[test_indices]
    
    # Encode test set
    z_test = encode_all_samples(model, params, X_test, batch_size=128)
    
    # Create t-SNE plot
    output_path = checkpoint_path.parent / 'tsne_latent_codes.png'
    create_tsne_plot(z_test, labels_test, output_path)
    
    print(f"✓ t-SNE analysis complete: {output_path}")


def main():
    """Run factorial comparison across all configurations."""
    # Create base output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(f"artifacts/factorial_comparison_{timestamp}")
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("FACTORIAL COMPARISON")
    print("=" * 80)
    print(f"Base output directory: {base_output_dir}")
    print()
    
    # Define all configurations (2^3 = 8 combinations)
    # Factor 1: VAE loss (on/off) - when on, weight = 0.01, but only works if use_vae=True
    # Factor 2: Marginal KL loss (on/off) - when on, weight = 0.01
    # Factor 3: Z normalization (on/off)
    
    # For VAE loss to be "on", we need both use_vae=True AND vae_kl_weight=0.01
    # So we'll treat "VAE loss on" as use_vae=True with vae_kl_weight=0.01
    # and "VAE loss off" as use_vae=False (vae_kl_weight=0.0) OR use_vae=True with vae_kl_weight=0.0
    
    vae_loss_options = [False, True]  # False = no VAE loss, True = VAE loss with weight 0.01
    marginal_kl_options = [False, True]  # False = no marginal KL, True = marginal KL with weight 0.01
    normalize_z_options = [False, True]
    
    # Generate all combinations (2^3 = 8)
    configs = []
    for vae_loss_on, marginal_kl_on, normalize_z in itertools.product(
        vae_loss_options, marginal_kl_options, normalize_z_options
    ):
        # Set use_vae and vae_kl_weight based on vae_loss_on
        if vae_loss_on:
            use_vae = True
            vae_kl_weight = 0.01
        else:
            use_vae = False
            vae_kl_weight = 0.0
        
        # Set marginal_kl_weight based on marginal_kl_on
        if marginal_kl_on:
            marginal_kl_weight = 0.01
        else:
            marginal_kl_weight = 0.0
        
        configs.append({
            'use_vae': use_vae,
            'vae_kl_weight': vae_kl_weight,
            'marginal_kl_weight': marginal_kl_weight,
            'normalize_z': normalize_z
        })
    
    print(f"Total configurations: {len(configs)}")
    print("\nConfiguration list:")
    for i, config in enumerate(configs, 1):
        print(f"  {i}. VAE={config['use_vae']}, VAE_KL={config['vae_kl_weight']}, "
              f"Marginal_KL={config['marginal_kl_weight']}, Normalize_Z={config['normalize_z']}")
    print()
    
    # Train each configuration
    checkpoint_paths = []
    for i, config in enumerate(configs, 1):
        print(f"\n{'#'*80}")
        print(f"Configuration {i}/{len(configs)}")
        print(f"{'#'*80}\n")
        
        try:
            checkpoint_path = train_single_config(config, base_output_dir, epochs=200)
            checkpoint_paths.append((checkpoint_path, config))
            print(f"\n✓ Configuration {i} training complete!")
        except Exception as e:
            print(f"\n✗ Configuration {i} failed with error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Run t-SNE analysis for each completed model
    print(f"\n{'='*80}")
    print("Running t-SNE Analysis")
    print(f"{'='*80}\n")
    
    for checkpoint_path, config in checkpoint_paths:
        try:
            run_tsne_analysis(checkpoint_path, config)
        except Exception as e:
            print(f"\n✗ t-SNE analysis failed for {checkpoint_path.parent.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print("FACTORIAL COMPARISON COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {base_output_dir}")
    print(f"Total configurations trained: {len(checkpoint_paths)}/{len(configs)}")


if __name__ == "__main__":
    main()


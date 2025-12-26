"""
Analyze factorial comparison results.

This script:
1. Loads all final checkpoints from factorial comparison
2. Generates sample and evaluation figures for each model
3. Performs model comparison (losses, Chamfer distance, etc.)
4. Creates t-SNE visualizations for all models
5. Generates comparison summary plots
"""

import jax
import jax.numpy as jnp
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Tuple

from src.models.mnist_flow_2d import MnistFlow2D, PredictionTarget
from src.utils.training import chamfer_distance
from src.train_all_digits import load_all_digits_data, create_data_loaders
from src.plotting import plot_samples, plot_unconditional_samples, create_tsne_plot
from src.analyze_encoder_latents import load_model_and_checkpoint, load_mnist_data_with_labels, encode_all_samples


def load_factorial_results(base_dir: Path) -> List[Dict]:
    """Load all final checkpoints from factorial comparison."""
    results = []
    
    config_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('vae_')])
    
    for config_dir in config_dirs:
        checkpoint_path = config_dir / "final_model.pkl"
        if not checkpoint_path.exists():
            print(f"Warning: {checkpoint_path} not found, skipping...")
            continue
        
        # Parse configuration from directory name
        # Format: vae_False_vae_kl_0.0_marginal_kl_0.0_normalize_z_False
        name = config_dir.name
        # Extract values using string parsing
        use_vae = 'vae_True' in name
        normalize_z = 'normalize_z_True' in name
        
        # Extract weights
        import re
        vae_kl_match = re.search(r'vae_kl_([\d.]+)', name)
        marginal_kl_match = re.search(r'marginal_kl_([\d.]+)', name)
        
        vae_kl_weight = float(vae_kl_match.group(1)) if vae_kl_match else 0.0
        marginal_kl_weight = float(marginal_kl_match.group(1)) if marginal_kl_match else 0.0
        
        config = {
            'directory': config_dir,
            'checkpoint_path': checkpoint_path,
            'use_vae': use_vae,
            'vae_kl_weight': vae_kl_weight,
            'marginal_kl_weight': marginal_kl_weight,
            'normalize_z': normalize_z,
        }
        
        results.append(config)
    
    return results


def evaluate_model(model, params, X_test, key, batch_size=128, num_samples=1000):
    """Evaluate model and return metrics."""
    total_chamfer = 0.0
    count = 0
    
    n = len(X_test)
    num_eval = min(n, num_samples)
    
    for i in range(0, num_eval, batch_size):
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
    
    avg_chamfer = total_chamfer / count if count > 0 else 0.0
    return avg_chamfer


def sample_batch(model, params, z_batch, keys_sample, num_points):
    """Sample a batch of point clouds efficiently."""
    key_batch = keys_sample[0] if len(keys_sample) > 0 else jax.random.PRNGKey(0)
    x_gen = model.apply(params, num_points, key_batch, z=z_batch, num_steps=20, batch_size=None, method=model.sample)
    return x_gen


def generate_figures_for_model(config: Dict, model, params, X_test, key, output_dir: Path):
    """Generate sample and evaluation figures for a single model."""
    print(f"  Generating figures for {output_dir.name}...")
    
    # Generate conditional samples
    key, k_samples = jax.random.split(key)
    plot_samples(model, params, X_test, k_samples, output_dir, num_samples=16)
    
    # Generate unconditional samples
    key, k_uncond = jax.random.split(k_samples)
    plot_unconditional_samples(model, params, k_uncond, output_dir, num_samples=36)
    
    print(f"  ✓ Figures generated")


def create_comparison_summary(results: List[Dict], base_dir: Path):
    """Create comparison summary plots and tables."""
    print("\nCreating comparison summary...")
    
    # Prepare data for comparison
    comparison_data = []
    
    for result in results:
        # Load loss history if available
        loss_file = result['directory'] / "loss_trajectory.png"
        # We can't easily parse the loss history from the PNG, so we'll focus on final metrics
        
        comparison_data.append({
            'Config': result['directory'].name,
            'Directory': str(result['directory']),  # Full path for merging
            'VAE': result['use_vae'],
            'VAE_KL_Weight': result['vae_kl_weight'],
            'Marginal_KL_Weight': result['marginal_kl_weight'],
            'Normalize_Z': result['normalize_z'],
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Save comparison table
    summary_path = base_dir / "comparison_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"  ✓ Saved comparison summary to {summary_path}")
    
    # If we have evaluation results, add performance comparison
    eval_path = base_dir / "evaluation_results.csv"
    if eval_path.exists():
        eval_df = pd.read_csv(eval_path)
        if 'chamfer_distance' in eval_df.columns and len(eval_df) > 0:
            # Merge with config info - match on directory paths
            # Normalize paths for comparison
            eval_df['directory_normalized'] = eval_df['directory'].apply(lambda x: str(Path(x)))
            df['directory_normalized'] = df['Directory'].apply(lambda x: str(Path(x)))
            
            merged_df = df.merge(eval_df[['directory_normalized', 'chamfer_distance']], 
                                on='directory_normalized', how='left')
            
            # Filter out rows with NaN chamfer_distance
            merged_df = merged_df.dropna(subset=['chamfer_distance'])
            
            if len(merged_df) > 0:
                # Create performance comparison plot
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                
                # Plot 1: Chamfer distance by configuration
                ax = axes[0]
                sorted_df = merged_df.sort_values('chamfer_distance', ascending=True)
                y_pos = np.arange(len(sorted_df))
                chamfer_values = sorted_df['chamfer_distance'].values
                
                # Create horizontal bar plot
                bars = ax.barh(y_pos, chamfer_values, alpha=0.7)
                
                # Create labels - use shorter config names
                config_labels = []
                for i, (_, row) in enumerate(sorted_df.iterrows()):
                    config_name = Path(row['Config']).name
                    # Simplify label
                    parts = config_name.split('_')
                    if len(parts) >= 7:
                        label = f"{i+1}. VAE={parts[1][0]}, MKL={parts[5]}, Z={parts[7][0]}"
                    else:
                        label = f"Config {i+1}"
                    config_labels.append(label)
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(config_labels, fontsize=8)
                ax.set_xlabel('Chamfer Distance', fontsize=10)
                ax.set_title('Model Performance (Chamfer Distance)', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
                
                # Add value labels on bars
                for i, val in enumerate(chamfer_values):
                    ax.text(val, i, f" {val:.6f}", va='center', fontsize=7)
                
                # Set x-axis limits to show all data
                ax.set_xlim(left=0, right=chamfer_values.max() * 1.1)
                
                # Plot 2: Chamfer distance by factor
                ax = axes[1]
                # Group by factors and compute means
                factor_means = {}
                factor_stds = {}
                
                for factor in ['VAE', 'Marginal_KL_Weight', 'Normalize_Z']:
                    grouped = merged_df.groupby(factor)['chamfer_distance']
                    factor_means[factor] = grouped.mean().to_dict()
                    factor_stds[factor] = grouped.std().fillna(0.0).to_dict()
                
                # Create grouped bar plot
                factors = list(factor_means.keys())
                x = np.arange(len(factors))
                width = 0.35
                
                # Get unique values across all factors for legend
                all_values = set()
                for factor in factors:
                    all_values.update(factor_means[factor].keys())
                
                # Create bars for each factor
                for i, factor in enumerate(factors):
                    means_dict = factor_means[factor]
                    stds_dict = factor_stds[factor]
                    values = sorted(means_dict.keys())
                    
                    if len(values) >= 1:
                        val0 = values[0]
                        mean0 = means_dict[val0]
                        std0 = stds_dict.get(val0, 0.0)
                        label0 = f"{val0}" if i == 0 else ""  # Only label first factor to avoid duplicates
                        ax.bar(x[i] - width/2, mean0, width, label=label0, alpha=0.7, 
                              yerr=std0, capsize=3, color='steelblue')
                    
                    if len(values) >= 2:
                        val1 = values[1]
                        mean1 = means_dict[val1]
                        std1 = stds_dict.get(val1, 0.0)
                        label1 = f"{val1}" if i == 0 else ""
                        ax.bar(x[i] + width/2, mean1, width, label=label1, alpha=0.7,
                              yerr=std1, capsize=3, color='coral')
                
                ax.set_xlabel('Factor', fontsize=10)
                ax.set_ylabel('Mean Chamfer Distance', fontsize=10)
                ax.set_title('Performance by Factor', fontsize=12, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(factors, rotation=45, ha='right')
                ax.legend(title='Factor Value', fontsize=8, loc='upper right')
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_ylim(bottom=0)
                
                plt.tight_layout()
                perf_path = base_dir / "performance_comparison.png"
                plt.savefig(perf_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved performance comparison to {perf_path}")
            else:
                print(f"  Warning: No valid chamfer distance data found for performance comparison")
        else:
            print(f"  Warning: evaluation_results.csv missing chamfer_distance column or is empty")
    
    # Create visualization of configurations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Configuration heatmap
    ax = axes[0, 0]
    config_matrix = df[['VAE', 'VAE_KL_Weight', 'Marginal_KL_Weight', 'Normalize_Z']].values
    im = ax.imshow(config_matrix.astype(float), aspect='auto', cmap='viridis')
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"Config {i+1}" for i in range(len(df))])
    ax.set_xticks(range(4))
    ax.set_xticklabels(['VAE', 'VAE_KL', 'Marginal_KL', 'Normalize_Z'])
    ax.set_title('Configuration Matrix')
    plt.colorbar(im, ax=ax)
    
    # Plot 2: Factor combinations
    ax = axes[0, 1]
    factor_counts = df[['VAE', 'Marginal_KL_Weight', 'Normalize_Z']].apply(
        lambda x: f"VAE={x.iloc[0]}, MKL={x.iloc[1]}, NormZ={x.iloc[2]}", axis=1
    ).value_counts()
    ax.barh(range(len(factor_counts)), factor_counts.values)
    ax.set_yticks(range(len(factor_counts)))
    ax.set_yticklabels(factor_counts.index, fontsize=8)
    ax.set_xlabel('Count')
    ax.set_title('Factor Combinations')
    
    # Plot 3: VAE vs Non-VAE
    ax = axes[1, 0]
    vae_counts = df['VAE'].value_counts()
    ax.pie(vae_counts.values, labels=['No VAE', 'VAE'], autopct='%1.1f%%')
    ax.set_title('VAE Usage Distribution')
    
    # Plot 4: Normalization distribution
    ax = axes[1, 1]
    norm_counts = df['Normalize_Z'].value_counts()
    ax.pie(norm_counts.values, labels=['No Norm', 'Normalize'], autopct='%1.1f%%')
    ax.set_title('Z Normalization Distribution')
    
    plt.tight_layout()
    summary_plot_path = base_dir / "comparison_summary.png"
    plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved comparison plot to {summary_plot_path}")


def create_combined_tsne_plot(results: List[Dict], base_dir: Path, X_test, labels_test):
    """Create a combined t-SNE plot showing all models side by side."""
    print("\nCreating combined t-SNE visualization...")
    
    n_models = len(results)
    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    if n_rows == 1:
        axes = axes[None, :] if axes.ndim == 1 else axes
    axes = axes.flatten()
    
    for idx, result in enumerate(results):
        ax = axes[idx]
        
        # Load model with correct configuration
        try:
            checkpoint_path = result['checkpoint_path']
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            params = checkpoint['params']
            
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
                use_vae=result['use_vae'],
                vae_kl_weight=result['vae_kl_weight'],
                marginal_kl_weight=result['marginal_kl_weight'],
                use_prior_flow=True,
                prior_flow_kwargs={
                    'hidden_dims': (128, 128, 128, 128, 128),
                    'time_embed_dim': 128,
                },
                optimal_reweighting=False,
                normalize_z=result['normalize_z'],
            )
            
            # Encode test set
            z_test = encode_all_samples(model, params, X_test, batch_size=128)
            
            # Create t-SNE (we'll compute it here)
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
            z_2d = tsne.fit_transform(z_test)
            
            # Plot
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            for digit in range(10):
                digit_mask = labels_test == digit
                if np.any(digit_mask):
                    ax.scatter(z_2d[digit_mask, 0], z_2d[digit_mask, 1],
                              c=[colors[digit]], label=f'{digit}', alpha=0.6, s=10)
            
            config_name = result['directory'].name
            ax.set_title(f"Config {idx+1}\n{config_name}", fontsize=8)
            ax.set_xlabel('t-SNE 1', fontsize=8)
            ax.set_ylabel('t-SNE 2', fontsize=8)
            ax.tick_params(labelsize=6)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{str(e)}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Config {idx+1}\n{result['directory'].name}", fontsize=8)
    
    # Hide unused axes
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    combined_tsne_path = base_dir / "combined_tsne_comparison.png"
    plt.savefig(combined_tsne_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved combined t-SNE plot to {combined_tsne_path}")


def create_unconditional_samples_comparison(results: List[Dict], base_dir: Path):
    """Create a comparison figure showing unconditional samples from all models.
    
    Layout: 8 rows (one per model) × 10 columns (10 samples per model)
    Each subplot shows a single point cloud from one z sample.
    """
    print("\nCreating unconditional samples comparison...")
    
    # Use same random seed for all models for fair comparison
    key = jax.random.PRNGKey(42)
    num_samples_per_model = 10  # Show 10 samples per model
    num_points = 500
    
    n_models = len(results)
    n_cols = 10  # 10 samples per row
    n_rows = n_models  # One row per model
    
    # Create figure: 8 rows × 10 columns
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 2.5 * n_rows))
    if n_rows == 1:
        axes = axes[None, :] if axes.ndim == 1 else axes
    
    for model_idx, result in enumerate(results):
        print(f"  Generating samples for model {model_idx+1}/{n_models}: {result['directory'].name}")
        
        try:
            # Load model
            checkpoint_path = result['checkpoint_path']
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            params = checkpoint['params']
            
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
                use_vae=result['use_vae'],
                vae_kl_weight=result['vae_kl_weight'],
                marginal_kl_weight=result['marginal_kl_weight'],
                use_prior_flow=True,
                prior_flow_kwargs={
                    'hidden_dims': (128, 128, 128, 128, 128),
                    'time_embed_dim': 128,
                },
                optimal_reweighting=False,
                normalize_z=result['normalize_z'],
            )
            
            # Generate unconditional samples (one at a time to ensure each z is independent)
            for sample_idx in range(num_samples_per_model):
                key, k_sample = jax.random.split(key)
                # Generate single sample
                x_gen = model.apply(params, num_points, k_sample, z=None, num_steps=20, 
                                   batch_size=1, method=model.sample)
                
                # x_gen has shape (1, num_points, 2) or (num_points, 2)
                if x_gen.ndim == 2:
                    x_gen = x_gen[None, :, :]
                x_gen = x_gen[0]  # Remove batch dimension: (num_points, 2)
                
                ax = axes[model_idx, sample_idx]
                
                # Plot point cloud
                ax.scatter(x_gen[:, 0], x_gen[:, 1], s=1, alpha=0.6, c='black')
                ax.set_aspect('equal')
                ax.set_xlim(-1.5, 1.5)
                ax.set_ylim(-1.5, 1.5)
                ax.axis('off')
                
                # Add model label on first sample of each row
                if sample_idx == 0:
                    # Create a simplified config name for display
                    config_name = result['directory'].name
                    # Extract key info
                    vae_str = "VAE" if result['use_vae'] else "NoVAE"
                    mkl_str = f"MKL{result['marginal_kl_weight']}"
                    nz_str = "NZ" if result['normalize_z'] else "NoNZ"
                    short_name = f"{vae_str}_{mkl_str}_{nz_str}"
                    ax.text(-1.4, 1.3, f"M{model_idx+1}: {short_name}", 
                           fontsize=7, fontweight='bold', va='top', ha='left',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
        except Exception as e:
            # Show error in first column
            ax = axes[model_idx, 0]
            ax.text(0.5, 0.5, f"Error:\n{str(e)}", ha='center', va='center', 
                   transform=ax.transAxes, fontsize=8)
            # Hide other columns for this model
            for col in range(1, n_cols):
                axes[model_idx, col].axis('off')
            import traceback
            traceback.print_exc()
    
    plt.suptitle('Unconditional Samples Comparison: 10 Samples per Model (8 Models)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    comparison_path = base_dir / "unconditional_samples_comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved unconditional samples comparison to {comparison_path}")


def main():
    """Main analysis function."""
    import sys
    
    # Find the most recent factorial comparison directory
    factorial_dirs = sorted(Path("artifacts").glob("factorial_comparison_*"))
    if not factorial_dirs:
        print("Error: No factorial comparison directories found!")
        print("Please run factorial_comparison.py first.")
        return
    
    base_dir = factorial_dirs[-1]
    print("=" * 80)
    print("FACTORIAL COMPARISON ANALYSIS")
    print("=" * 80)
    print(f"Base directory: {base_dir}")
    print()
    
    # Load all results
    print("Loading factorial comparison results...")
    results = load_factorial_results(base_dir)
    print(f"Found {len(results)} completed configurations")
    
    if len(results) == 0:
        print("Error: No completed configurations found!")
        return
    
    # Load test data
    print("\nLoading test data...")
    X_all, labels_all = load_mnist_data_with_labels(dataset_path="data/mnist_2d_single.npz", 
                                                     train_split=0.9, seed=42, use_test_only=False)
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
    print(f"Test set: {len(X_test)} samples")
    
    # Evaluate each model and generate figures
    print("\n" + "=" * 80)
    print("EVALUATING MODELS AND GENERATING FIGURES")
    print("=" * 80)
    
    evaluation_results = []
    key = jax.random.PRNGKey(42)
    
    for idx, result in enumerate(results, 1):
        print(f"\n[{idx}/{len(results)}] Processing {result['directory'].name}...")
        
        try:
            # Load model with correct configuration matching factorial comparison
            checkpoint_path = result['checkpoint_path']
            print(f"  Loading model from {checkpoint_path}...")
            
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            params = checkpoint['params']
            
            # Reconstruct model with same config as factorial comparison training
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
                use_vae=result['use_vae'],
                vae_kl_weight=result['vae_kl_weight'],
                marginal_kl_weight=result['marginal_kl_weight'],
                use_prior_flow=True,
                prior_flow_kwargs={
                    'hidden_dims': (128, 128, 128, 128, 128),
                    'time_embed_dim': 128,
                },
                optimal_reweighting=False,
                normalize_z=result['normalize_z'],
            )
            
            # Evaluate
            key, k_eval = jax.random.split(key)
            chamfer = evaluate_model(model, params, X_test, k_eval, batch_size=128, num_samples=1000)
            print(f"  Chamfer distance: {chamfer:.6f}")
            
            # Generate figures
            key, k_fig = jax.random.split(k_eval)
            generate_figures_for_model(result, model, params, X_test, k_fig, result['directory'])
            
            # Run t-SNE if not already done
            tsne_path = result['directory'] / 'tsne_latent_codes.png'
            if not tsne_path.exists():
                print(f"  Running t-SNE analysis...")
                z_test = encode_all_samples(model, params, X_test, batch_size=128)
                create_tsne_plot(z_test, labels_test, tsne_path)
            
            evaluation_results.append({
                **result,
                'chamfer_distance': chamfer
            })
            
        except Exception as e:
            print(f"  ✗ Error processing model: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save evaluation results first (needed for comparison summary)
    eval_df = pd.DataFrame(evaluation_results)
    eval_path = base_dir / "evaluation_results.csv"
    eval_df.to_csv(eval_path, index=False)
    print(f"\n✓ Saved evaluation results to {eval_path}")
    
    # Create comparison summary (now includes performance comparison)
    print("\n" + "=" * 80)
    print("CREATING COMPARISON SUMMARY")
    print("=" * 80)
    create_comparison_summary(evaluation_results, base_dir)
    
    # Create combined t-SNE plot
    print("\n" + "=" * 80)
    print("CREATING COMBINED T-SNE VISUALIZATION")
    print("=" * 80)
    create_combined_tsne_plot(evaluation_results, base_dir, X_test, labels_test)
    
    # Create unconditional samples comparison
    print("\n" + "=" * 80)
    print("CREATING UNCONDITIONAL SAMPLES COMPARISON")
    print("=" * 80)
    create_unconditional_samples_comparison(evaluation_results, base_dir)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {base_dir}")
    print(f"Total models analyzed: {len(evaluation_results)}/{len(results)}")


if __name__ == "__main__":
    main()


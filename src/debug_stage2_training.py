"""
Debug script to verify Stage 2 training logic:
1. Check that gradients are properly masked
2. Verify that only prior flow parameters receive gradients
3. Check that encoder/CRN parameters are not being updated
4. Verify loss computation logic
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import pickle

from src.models.mnist_flow_2d import MnistFlow2D, PredictionTarget
import numpy as np
import os

def check_gradients(model, params, x_batch, key, update_prior_only=True):
    """Check which parameters receive gradients."""
    
    def loss_fn(p):
        # Always compute prior loss for this check
        loss, metrics = model.apply(p, x_batch, key, None, True, False, method=model.compute_loss)
        return loss, metrics
    
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
    # Apply gradient masking if update_prior_only
    if update_prior_only:
        def zero_non_prior_grads(path, value):
            path_str = '/'.join(str(p) for p in path)
            if 'prior_vector_field' in path_str:
                return value  # Keep gradient for prior flow
            else:
                return jnp.zeros_like(value)
        
        grads = jax.tree_util.tree_map_with_path(zero_non_prior_grads, grads)
    
    # Check which parameters have non-zero gradients
    print("=" * 80)
    print("GRADIENT ANALYSIS")
    print("=" * 80)
    
    def check_grad(path, value):
        path_str = '/'.join(str(p) for p in path)
        grad_norm = jnp.linalg.norm(value)
        if grad_norm > 1e-8:
            print(f"  ✓ {path_str}: grad_norm = {grad_norm:.6f}")
            return True
        else:
            print(f"  ✗ {path_str}: grad_norm = {grad_norm:.6e} (zeroed)")
            return False
    
    has_grads = []
    jax.tree_util.tree_map_with_path(
        lambda path, value: has_grads.append(check_grad(path, value)),
        grads
    )
    
    num_nonzero = sum(has_grads)
    print(f"\nTotal parameters with non-zero gradients: {num_nonzero}/{len(has_grads)}")
    
    # Check specific parameter groups
    print("\n" + "=" * 80)
    print("PARAMETER GROUP ANALYSIS")
    print("=" * 80)
    
    encoder_grads = []
    crn_grads = []
    prior_grads = []
    
    def categorize_grad(path, value):
        path_str = '/'.join(str(p) for p in path)
        grad_norm = jnp.linalg.norm(value)
        
        if 'encoder' in path_str:
            encoder_grads.append(grad_norm)
        elif 'crn' in path_str and 'prior' not in path_str:
            crn_grads.append(grad_norm)
        elif 'prior_vector_field' in path_str:
            prior_grads.append(grad_norm)
    
    jax.tree_util.tree_map_with_path(categorize_grad, grads)
    
    print(f"Encoder gradients: {len(encoder_grads)} params")
    if encoder_grads:
        print(f"  Max norm: {max(encoder_grads):.6e}, Mean norm: {np.mean(encoder_grads):.6e}")
    
    print(f"CRN gradients: {len(crn_grads)} params")
    if crn_grads:
        print(f"  Max norm: {max(crn_grads):.6e}, Mean norm: {np.mean(crn_grads):.6e}")
    
    print(f"Prior flow gradients: {len(prior_grads)} params")
    if prior_grads:
        print(f"  Max norm: {max(prior_grads):.6e}, Mean norm: {np.mean(prior_grads):.6e}")
    
    return grads, metrics

def check_loss_components(model, params, x_batch, key):
    """Check individual loss components."""
    print("\n" + "=" * 80)
    print("LOSS COMPONENT ANALYSIS")
    print("=" * 80)
    
    # Compute loss with prior
    loss_with_prior, metrics_with = model.apply(
        params, x_batch, key, None, True, False, method=model.compute_loss
    )
    
    # Compute loss without prior
    loss_without_prior, metrics_without = model.apply(
        params, x_batch, key, None, False, False, method=model.compute_loss
    )
    
    print("With prior loss computation:")
    print(f"  Total loss: {loss_with_prior:.6f}")
    for k, v in metrics_with.items():
        print(f"  {k}: {v:.6f}")
    
    print("\nWithout prior loss computation:")
    print(f"  Total loss: {loss_without_prior:.6f}")
    for k, v in metrics_without.items():
        print(f"  {k}: {v:.6f}")
    
    print(f"\nDifference in total loss: {loss_with_prior - loss_without_prior:.6f}")
    print(f"Prior flow loss component: {metrics_with.get('prior_flow_loss', 0.0):.6f}")

def main():
    # Load model from checkpoint
    checkpoint_path = Path("artifacts/all_digits_pointnet_adaln_velocity_no_vae_prior_flow_joint_squash/final_model.pkl")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    params = checkpoint['params']
    
    # Initialize model (must match training config)
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
        use_vae=False,
        vae_kl_weight=0.0,
        marginal_kl_weight=0.0,
        use_prior_flow=True,
    )
    
    # Load a small batch of data
    dataset_path = "data/mnist_2d_full_dataset.npz"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    data = np.load(dataset_path)
    points = data['points']  # (60000, 500, 2)
    x_batch = points[:4]  # Small batch for debugging
    
    key = jax.random.PRNGKey(42)
    
    # Check loss components
    check_loss_components(model, params, x_batch, key)
    
    # Check gradients with update_prior_only=True (Stage 2 mode)
    print("\n" + "=" * 80)
    print("STAGE 2 MODE (update_prior_only=True)")
    print("=" * 80)
    grads_stage2, metrics_stage2 = check_gradients(model, params, x_batch, key, update_prior_only=True)
    
    # Check gradients with update_prior_only=False (normal mode)
    print("\n" + "=" * 80)
    print("NORMAL MODE (update_prior_only=False)")
    print("=" * 80)
    grads_normal, metrics_normal = check_gradients(model, params, x_batch, key, update_prior_only=False)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("If Stage 2 training is working correctly:")
    print("  - Only prior_vector_field parameters should have non-zero gradients")
    print("  - Encoder and CRN gradients should be zero")
    print("  - Prior flow loss should be computed and included in total loss")

if __name__ == "__main__":
    main()


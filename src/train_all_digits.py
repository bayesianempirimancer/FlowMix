"""
Training script for all MNIST digits using PointNet encoder and GlobalAdaLNMLPCRN.

This trains on the full MNIST 2D point cloud dataset (all 10 digits).
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

from src.models.mnist_flow_2d import MnistFlow2D, PredictionTarget
from src.utils.training import chamfer_distance
from src.plotting import plot_samples, plot_unconditional_samples, plot_loss_trajectory


def create_grid_mask(x: jnp.ndarray, key: jax.random.PRNGKey, 
                     grid_size: int = 6, mask_prob: float = 0.2) -> jnp.ndarray:
    """
    Create a grid mask that masks out points inside randomly selected grid boxes.
    
    Args:
        x: Point cloud, shape (B, N, 2) - assumes 2D coordinates
        key: Random key
        grid_size: Size of the grid (e.g., 6 for 6x6 grid)
        mask_prob: Probability of masking each grid cell
        
    Returns:
        mask: Boolean mask, shape (B, N), True for visible points, False for masked points
    """
    B, N, D = x.shape
    assert D == 2, "Grid mask assumes 2D coordinates"
    
    # Get bounds of the point cloud (per batch to handle different scales)
    x_min = jnp.min(x, axis=1, keepdims=True)  # (B, 1, 2)
    x_max = jnp.max(x, axis=1, keepdims=True)  # (B, 1, 2)
    
    # Determine which grid cells to mask (different pattern for each batch element)
    key, k_mask = jax.random.split(key)
    # Sample mask for each grid cell per batch element: (B, grid_size, grid_size)
    # True means the cell should be masked
    # Each batch element gets its own independent grid pattern
    grid_mask = jax.random.bernoulli(k_mask, mask_prob, (B, grid_size, grid_size))
    
    # For each point, determine which grid cell it belongs to
    # Normalize points to [0, 1] range per batch
    x_norm = (x - x_min) / (x_max - x_min + 1e-8)  # (B, N, 2)
    
    # Compute grid cell indices for each point
    grid_x = jnp.floor(x_norm[:, :, 0] * grid_size).astype(jnp.int32)  # (B, N)
    grid_y = jnp.floor(x_norm[:, :, 1] * grid_size).astype(jnp.int32)  # (B, N)
    
    # Clamp to valid grid indices
    grid_x = jnp.clip(grid_x, 0, grid_size - 1)
    grid_y = jnp.clip(grid_y, 0, grid_size - 1)
    
    # Look up mask value for each point's grid cell using batched indexing
    # grid_mask has shape (B, grid_size, grid_size)
    # grid_y and grid_x have shape (B, N)
    # We need to index grid_mask[b, grid_y[b, i], grid_x[b, i]] for each b and i
    
    # Create batch indices for each point
    batch_idx = jnp.broadcast_to(jnp.arange(B)[:, None], (B, N))  # (B, N)
    
    # Use advanced indexing: grid_mask[batch_idx, grid_y, grid_x]
    point_mask = grid_mask[batch_idx, grid_y, grid_x]  # (B, N) - True if cell is masked
    
    # Invert: True means visible (not masked), False means masked
    point_mask = ~point_mask  # True = visible, False = masked
    
    return point_mask


def load_all_digits_data(dataset_path="src/data/mnist_2d_full_dataset.npz", max_samples=None, seed=42):
    """
    Load all MNIST digits point cloud dataset.
    
    Optimized for fast loading - directly loads .npz files without extra overhead.
    
    Args:
        dataset_path: Path to the full MNIST dataset file (.npz)
        max_samples: Maximum number of samples to use (None = use all)
        seed: Random seed for shuffling
        
    Returns:
        X: (num_samples, 500, 2) point clouds
    """
    print(f"Loading full MNIST dataset from {dataset_path}...")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}.\n"
            f"Please ensure the dataset exists."
        )
    
    # Direct loading for speed (optimized path for .npz files)
    data = np.load(dataset_path)
    points = data['points']  # (60000, 500, 2)
    
    print(f"Loaded {len(points)} point clouds (all digits)")
    print(f"Point cloud shape: {points.shape}")
    print(f"Point range: [{points.min():.2f}, {points.max():.2f}]")
    
    # Optionally limit number of samples
    if max_samples is not None and len(points) > max_samples:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(points), max_samples, replace=False)
        points = points[indices]
        print(f"Using {len(points)} samples (randomly selected)")
    
    # Convert to JAX array
    X = jnp.array(points)
    
    return X


def create_data_loaders(X, batch_size=32, train_split=0.9, seed=42):
    """Split data into train/test and create batches."""
    n = len(X)
    n_train = int(n * train_split)
    
    # Shuffle
    rng = np.random.RandomState(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    def get_batches(data, batch_size, shuffle=False):
        """Generator for batches - optimized for speed with no shuffling."""
        n = len(data)
        # Simple sequential batching - no shuffling for speed
        for i in range(0, n, batch_size):
            batch = data[i:i+batch_size]
            yield batch
    
    return X_train, X_test, get_batches


def train_step(model, params, opt_state, x_batch, key, optimizer, update_prior_only=False, freeze_prior=False, use_grid_mask=False, grid_size=4, mask_prob=0.2):
    """Single training step.
    
    Args:
        update_prior_only: If True, only update prior flow parameters, freeze all others.
        freeze_prior: If True, freeze prior flow parameters, update all others.
        use_grid_mask: If True, use grid masking for encoder (encoder sees masked points, flow sees full points).
        grid_size: Size of the grid for masking (default 4).
        mask_prob: Probability of masking each grid cell (default 0.2).
    """
    # Generate grid mask for encoder if enabled
    encoder_mask = None
    if use_grid_mask:
        key, k_mask = jax.random.split(key)
        encoder_mask = create_grid_mask(x_batch, k_mask, grid_size=grid_size, mask_prob=mask_prob)
    
    def loss_fn(p):
        # Skip prior flow loss computation if prior is frozen (no gradients needed)
        compute_prior_loss = not freeze_prior
        # In prior_only_mode, exclude flow_loss from total loss since encoder/CRN are frozen
        prior_only_mode = update_prior_only
        # Pass encoder_mask separately: encoder uses masked points, flow uses full points
        # mask=None for flow (uses full x), encoder_mask for encoder
        loss, metrics = model.apply(p, x_batch, key, None, encoder_mask, compute_prior_loss, prior_only_mode, method=model.compute_loss)
        return loss, metrics
    
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
    # If update_prior_only, zero out gradients for all parameters except prior_vector_field
    if update_prior_only:
        def zero_non_prior_grads(path, value):
            # Check if this is a prior flow parameter
            # Prior flow parameters are under 'prior_vector_field'
            # In Flax, the path is a tuple of keys like ('params', 'prior_vector_field', ...)
            # Use tuple comparison instead of string conversion for speed
            if len(path) >= 2 and path[1] == 'prior_vector_field':
                return value  # Keep gradient for prior flow
            else:
                # Zero out gradient for everything else
                return jnp.zeros_like(value)
        
        grads = jax.tree_util.tree_map_with_path(zero_non_prior_grads, grads)
    
    # If freeze_prior, zero out gradients for prior flow parameters
    elif freeze_prior:
        def zero_prior_grads(path, value):
            # Check if this is a prior flow parameter
            # Use tuple comparison instead of string conversion for speed
            if len(path) >= 2 and path[1] == 'prior_vector_field':
                # Zero out gradient for prior flow
                return jnp.zeros_like(value)
            else:
                # Keep gradient for everything else
                return value
        
        grads = jax.tree_util.tree_map_with_path(zero_prior_grads, grads)
    
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss, metrics


def sample_batch(model, params, z_batch, keys_sample, num_points):
    """Sample a batch of point clouds efficiently using vmap."""
    def sample_single(z_single, key_single):
        """Sample for a single latent code."""
        return model.apply(params, num_points, key_single, z=z_single[None], method=model.sample)
    
    # Vmap over batch dimension
    x_gen = jax.vmap(sample_single, in_axes=(0, 0))(z_batch, keys_sample)
    return x_gen


def evaluate(model, params, X_test, key, batch_size=32, num_samples=100):
    """Evaluate model using Chamfer distance."""
    num_eval = min(len(X_test), num_samples)
    total_chamfer = 0.0
    count = 0
    
    for i in range(0, num_eval, batch_size):
        batch_x = X_test[i:i+batch_size]
        actual_batch = len(batch_x)
        
        key, k_enc, k_sample = jax.random.split(key, 3)
        
        # Encode (already JIT-compiled via model.apply)
        z_batch, _, _ = model.apply(params, batch_x, k_enc, method=model.encode)
        
        # Sample in batch using vmap (efficient, no serial loop)
        keys_sample = jax.random.split(k_sample, actual_batch)
        x_gen = sample_batch(model, params, z_batch, keys_sample, X_test.shape[1])  # (B, N, 2)
        
        # Compute Chamfer distance
        cd_batch = chamfer_distance(batch_x, x_gen)
        total_chamfer += float(jnp.sum(cd_batch))
        count += actual_batch
    
    return total_chamfer / count if count > 0 else 0.0


def main():
    # Configuration
    # Two-stage training protocol:
    # Stage 1: Train model (encoder + main CRN) but NOT prior flow for 100 epochs
    # Stage 2: Freeze everything except prior flow, train only prior flow for 100 epochs
    ENABLE_TWO_STAGE_TRAINING = False  # If True, automatically run Stage 1 then Stage 2
    STAGE_1_EPOCHS = 100  # Train model without prior flow
    STAGE_2_EPOCHS = 100  # Train only prior flow (100 more epochs)
    CHECKPOINT_INTERVAL = 50  # Save checkpoint every N epochs (50 = every 50 epochs)
    BATCH_SIZE = 256  # Doubled from 128
    LEARNING_RATE = 1e-3  # Default learning rate (0.001)
    MAX_SAMPLES = None  # None = use all available samples (60,000)
    SEED = 42
    USE_GRID_MASK = False  # Enable grid masking: encoder sees masked points, flow sees full points
    GRID_SIZE = 3  # 3x3 grid
    GRID_MASK_PROB = 1.0 / 3.0  # Probability of masking each grid cell (1/3)
    RESUME_FROM_CHECKPOINT = None  # Path to checkpoint file to resume from, or None to start fresh
    # If ENABLE_TWO_STAGE_TRAINING=True, these are ignored and stages run automatically
    TRAINING_STAGE = 0  # 0 = Joint training (train everything), 1 = Stage 1 (train model, freeze prior), 2 = Stage 2 (train only prior)
    JOINT_TRAINING_EPOCHS = 100  # Number of epochs for joint training (when TRAINING_STAGE=0)
    
    # Momentum configuration: "doubled" (beta1=0.95, beta2=0.9995) or "default" (beta1=0.9, beta2=0.999)
    MOMENTUM_CONFIG = "default"  # Options: "doubled" or "default"
    
    # Set momentum parameters based on config
    if MOMENTUM_CONFIG == "doubled":
        BETA1, BETA2 = 0.95, 0.9995
        momentum_suffix = "momentum_doubled"
    else:
        BETA1, BETA2 = 0.9, 0.999
        momentum_suffix = "momentum_default"
    
    # Output directory for joint training without VAE
    if USE_GRID_MASK:
        output_dir = Path(f"artifacts/all_digits_pointnet_adaln_velocity_no_vae_prior_flow_joint_gridmask_{momentum_suffix}")
    else:
        output_dir = Path(f"artifacts/all_digits_pointnet_adaln_velocity_no_vae_prior_flow_joint_{momentum_suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Training All MNIST Digits Model")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Momentum config: {MOMENTUM_CONFIG} (beta1={BETA1}, beta2={BETA2})")
    print(f"Training Stage: {TRAINING_STAGE}")
    if TRAINING_STAGE == 0:
        print("  → Joint Training: Training encoder + CRN + prior flow simultaneously")
    elif TRAINING_STAGE == 1:
        print("  → Stage 1: Training main model (encoder + CRN), prior flow FROZEN")
    elif TRAINING_STAGE == 2:
        print("  → Stage 2: Training ONLY prior flow, main model FROZEN")
    if USE_GRID_MASK:
        print(f"  → Grid Masking: ENABLED (grid_size={GRID_SIZE}, mask_prob={GRID_MASK_PROB})")
        print("    Encoder sees masked points, flow sees full points")
    if RESUME_FROM_CHECKPOINT is not None:
        print(f"Resume checkpoint: {RESUME_FROM_CHECKPOINT}")
    print()
    
    # Load data
    print("Loading full MNIST data...")
    X = load_all_digits_data(dataset_path="data/mnist_2d_single.npz", max_samples=MAX_SAMPLES, seed=SEED)
    X_train, X_test, get_batches = create_data_loaders(X, batch_size=BATCH_SIZE, seed=SEED)
    print()
    
    # Model config with VAE enabled
    print("Initializing model...")
    model = MnistFlow2D(
        latent_dim=32,  # Reduced from 64 by factor of 2
        encoder_type='pointnet',
        encoder_output_type='global',
        crn_type='adaln_mlp',
        crn_kwargs={
            'hidden_dims': (32, 32, 32, 32, 32, 32),  # Reduced from (64, 64, 64, 64, 64, 64) by factor of 2
            'cond_dim': 64,  # Reduced from 128 by another factor of 2 (originally 256)
        },
        prediction_target=PredictionTarget.VELOCITY,  # Predict velocity
        loss_targets=(PredictionTarget.VELOCITY,),  # Train on velocity loss only
        use_vae=False,  # VAE mode disabled - using LayerNorm instead
        vae_kl_weight=0.0,  # VAE KL loss weight (set to 0)
        marginal_kl_weight=0.0,  # Marginal KL loss weight (set to 0)
        use_prior_flow=True,  # Enable prior flow
        prior_flow_kwargs={
            'hidden_dims': (128, 128, 128, 128, 128),  # 5 layers of size 128
            'time_embed_dim': 128,  # Time embedding dimension for AdaLN
        },
        optimal_reweighting=False,  # No optimal reweighting
    )
    
    # Determine if we should run two-stage training automatically
    if ENABLE_TWO_STAGE_TRAINING and RESUME_FROM_CHECKPOINT is None:
        # Run both stages automatically
        stages_to_run = [1, 2]
        print("=" * 80)
        print("AUTOMATED TWO-STAGE TRAINING ENABLED")
        print("=" * 80)
        print(f"Will run Stage 1 ({STAGE_1_EPOCHS} epochs), then Stage 2 ({STAGE_2_EPOCHS} epochs)")
        print()
    elif TRAINING_STAGE == 0:
        # Joint training: train everything simultaneously
        stages_to_run = [0]
        print("=" * 80)
        print("JOINT TRAINING MODE")
        print("=" * 80)
        print(f"Training encoder, CRN, and prior flow simultaneously for {JOINT_TRAINING_EPOCHS} epochs")
        print()
    else:
        # Run single stage based on TRAINING_STAGE
        stages_to_run = [TRAINING_STAGE]
    
    # Run training for each stage
    for stage_idx, current_stage in enumerate(stages_to_run):
        if len(stages_to_run) > 1:
            print("=" * 80)
            print(f"STARTING STAGE {current_stage} of {len(stages_to_run)}")
            print("=" * 80)
            print()
        
        # Determine training mode and num_epochs based on current stage
        if current_stage == 0:
            # Joint training: train everything simultaneously
            update_prior_only = False
            freeze_prior = False
            num_epochs = JOINT_TRAINING_EPOCHS
            print("Training mode: JOINT (encoder + CRN + prior flow all trained simultaneously)")
        elif current_stage == 1:
            update_prior_only = False
            freeze_prior = True
            num_epochs = STAGE_1_EPOCHS
            print("Training mode: Stage 1 (encoder + CRN, prior flow FROZEN)")
        elif current_stage == 2:
            update_prior_only = True
            freeze_prior = False
            num_epochs = STAGE_2_EPOCHS
            print("Training mode: Stage 2 (ONLY prior flow, main model FROZEN)")
        else:
            raise ValueError(f"Invalid TRAINING_STAGE: {current_stage}")
        print()
        
        # If this is Stage 2 and we just finished Stage 1, load the Stage 1 checkpoint
        if current_stage == 2 and stage_idx > 0:
            # Load Stage 1 final checkpoint
            stage1_checkpoint_path = output_dir / f"checkpoint_epoch_{STAGE_1_EPOCHS}.pkl"
            if not stage1_checkpoint_path.exists():
                stage1_checkpoint_path = output_dir / "final_model.pkl"
            
            if stage1_checkpoint_path.exists():
                print(f"Loading Stage 1 checkpoint from {stage1_checkpoint_path}...")
                with open(stage1_checkpoint_path, 'rb') as f:
                    checkpoint = pickle.load(f)
                
                params = checkpoint['params']
                # Reinitialize optimizer for Stage 2 (switching stages)
                optimizer = optax.adam(LEARNING_RATE, b1=BETA1, b2=BETA2)
                opt_state = optimizer.init(params)
                print("✓ Reinitialized optimizer for Stage 2")
                
                # Reset epoch counter for Stage 2
                start_epoch = 0
                # Keep loss histories from Stage 1
                loss_history = checkpoint.get('loss_history', [])
                chamfer_history = checkpoint.get('chamfer_history', [])
                flow_loss_history = checkpoint.get('flow_loss_history', [])
                prior_flow_loss_history = checkpoint.get('prior_flow_loss_history', [])
                vae_kl_history = checkpoint.get('vae_kl_history', [])
                marginal_kl_history = checkpoint.get('marginal_kl_history', [])
                print()
            else:
                raise FileNotFoundError(f"Stage 1 checkpoint not found at {stage1_checkpoint_path}")
        else:
            # Use existing initialization logic for Stage 1 or manual resume
            TRAINING_STAGE = current_stage  # Set for checkpoint saving
        
        # Initialize or load from checkpoint for this stage
        if current_stage == 0 or current_stage == 1 or (current_stage == 2 and stage_idx == 0):
            # Joint training, Stage 1, or first time through: use normal initialization
            if current_stage == 0 or current_stage == 1:
                key = jax.random.PRNGKey(SEED)
            elif current_stage == 2 and stage_idx == 0:
                # Stage 2 but first iteration - initialize key
                key = jax.random.PRNGKey(SEED)
            
            if RESUME_FROM_CHECKPOINT is not None and Path(RESUME_FROM_CHECKPOINT).exists():
                # Load from checkpoint (works for both Stage 1 and Stage 2)
                print(f"Loading checkpoint from {RESUME_FROM_CHECKPOINT}...")
                with open(RESUME_FROM_CHECKPOINT, 'rb') as f:
                    checkpoint = pickle.load(f)
                
                params = checkpoint['params']
                checkpoint_training_stage = checkpoint.get('training_stage', None)
                checkpoint_epoch = checkpoint.get('epoch', 0)
                
                # For Stage 2, if checkpoint is from Stage 1, start from epoch 0
                # Otherwise, resume from the checkpoint epoch
                if current_stage == 2 and checkpoint_training_stage == 1:
                    start_epoch = 0  # Starting Stage 2 fresh from Stage 1 checkpoint
                    print(f"  Checkpoint was from Stage 1, starting Stage 2 from epoch 0")
                else:
                    start_epoch = checkpoint_epoch + 1  # Resume from next epoch
                
                loss_history = checkpoint.get('loss_history', [])
                chamfer_history = checkpoint.get('chamfer_history', [])
                # Load individual loss component histories
                flow_loss_history = checkpoint.get('flow_loss_history', [])
                prior_flow_loss_history = checkpoint.get('prior_flow_loss_history', [])
                vae_kl_history = checkpoint.get('vae_kl_history', [])
                marginal_kl_history = checkpoint.get('marginal_kl_history', [])
                
                print(f"✓ Loaded checkpoint from epoch {checkpoint_epoch}")
                if checkpoint_training_stage is not None:
                    print(f"  Checkpoint was from training stage {checkpoint_training_stage}")
                print(f"  Resuming training from epoch {start_epoch}/{num_epochs}")
                print(f"  Previous loss history: {len(loss_history)} epochs")
                print()
                
                # Count parameters
                param_count = sum(x.size for x in jax.tree_util.tree_leaves(params['params']))
                print(f"Model parameters: {param_count:,}")
                print()
                
                # Optimizer: reinitialize if switching training stages or if opt_state is None
                optimizer = optax.adam(LEARNING_RATE, b1=BETA1, b2=BETA2)
                opt_state_loaded = checkpoint.get('opt_state', None)
                
                # Reinitialize optimizer if:
                # 1. opt_state is None or missing
                # 2. Switching training stages (checkpoint stage != current stage)
                should_reinit_optimizer = (
                    opt_state_loaded is None or
                    (checkpoint_training_stage is not None and checkpoint_training_stage != current_stage)
                )
                
                if should_reinit_optimizer:
                    if checkpoint_training_stage is not None and checkpoint_training_stage != current_stage:
                        print(f"⚠️  Reinitializing optimizer state (switching from stage {checkpoint_training_stage} to {current_stage})")
                    else:
                        print(f"⚠️  Reinitializing optimizer state (missing or invalid state)")
                    opt_state = optimizer.init(params)
                else:
                    opt_state = opt_state_loaded
                    print(f"✓ Using optimizer state from checkpoint")
            else:
                # Initialize from scratch
                if RESUME_FROM_CHECKPOINT is not None and current_stage == 1:
                    print(f"⚠️  Checkpoint not found at {RESUME_FROM_CHECKPOINT}, starting from scratch")
                
                key, k_init = jax.random.split(key)
                dummy_x = X_train[:2]
                params = model.init(k_init, dummy_x, k_init)
                
                # Count parameters
                param_count = sum(x.size for x in jax.tree_util.tree_leaves(params['params']))
                print(f"Model parameters: {param_count:,}")
                print()
                
                # Optimizer
                optimizer = optax.adam(LEARNING_RATE, b1=BETA1, b2=BETA2)
                opt_state = optimizer.init(params)
                
                start_epoch = 0
                loss_history = []
                chamfer_history = []
                # Track individual loss components
                flow_loss_history = []
                prior_flow_loss_history = []
                vae_kl_history = []
                marginal_kl_history = []
        
        # JIT compile training step for this stage
        @jax.jit
        def train_step_jit(p, o, x, k):
            return train_step(model, p, o, x, k, optimizer, 
                             update_prior_only=update_prior_only, 
                             freeze_prior=freeze_prior,
                             use_grid_mask=USE_GRID_MASK,
                             grid_size=GRID_SIZE,
                             mask_prob=GRID_MASK_PROB)
        
        # Training loop for this stage
        if start_epoch == 0:
            print(f"Starting Stage {current_stage} training...")
        else:
            print(f"Resuming Stage {current_stage} training from epoch {start_epoch}...")
        print()
        
        # Track the actual final epoch number for checkpoint saving
        final_epoch = start_epoch + num_epochs - 1 if start_epoch > 0 else num_epochs - 1
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            epoch_losses = []
            epoch_metrics = {}
            
            # Training
            key, k_train = jax.random.split(key)
            # Don't convert to list - keep as generator to avoid materializing all batches in memory
            # Compute total batches for progress bar
            num_batches = (len(X_train) + BATCH_SIZE - 1) // BATCH_SIZE
            # No shuffling for speed - simplified batching
            batches = get_batches(X_train, BATCH_SIZE, shuffle=False)
            
            for batch_idx, x_batch in enumerate(tqdm(batches, total=num_batches, desc=f"Epoch {epoch+1}/{start_epoch + num_epochs}", ncols=100, mininterval=0.5)):
                key, k_step = jax.random.split(k_train)
                params, opt_state, loss, metrics = train_step_jit(params, opt_state, x_batch, k_step)
                
                epoch_losses.append(float(loss))
                for k, v in metrics.items():
                    if k not in epoch_metrics:
                        epoch_metrics[k] = []
                    epoch_metrics[k].append(float(v))
            
            # Average metrics
            avg_loss = np.mean(epoch_losses)
            avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
            loss_history.append(avg_loss)
            
            # Track individual loss components
            flow_loss_history.append(avg_metrics.get('flow_loss', 0.0))
            prior_flow_loss_history.append(avg_metrics.get('prior_flow_loss', 0.0))
            vae_kl_history.append(avg_metrics.get('vae_kl', 0.0))
            marginal_kl_history.append(avg_metrics.get('marginal_kl', 0.0))
            
            # Evaluation (every 10 epochs)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                key, k_eval = jax.random.split(key)
                chamfer = evaluate(model, params, X_test, k_eval, batch_size=BATCH_SIZE, num_samples=100)
                chamfer_history.append(chamfer)
                
                print(f"Epoch {epoch+1}/{start_epoch + num_epochs}:")
                print(f"  Loss: {avg_loss:.4f}")
                for k, v in avg_metrics.items():
                    print(f"  {k}: {v:.4f}")
                print(f"  Chamfer: {chamfer:.4f}")
                print()
                
                # Plot conditional samples
                key, k_plot = jax.random.split(key)
                plot_samples(model, params, X_test, k_plot, output_dir, num_samples=16,
                            use_grid_mask=USE_GRID_MASK, grid_size=GRID_SIZE, mask_prob=GRID_MASK_PROB)
                
                # Plot unconditional samples
                key, k_uncond = jax.random.split(key)
                plot_unconditional_samples(model, params, k_uncond, output_dir, num_samples=36)
            
            # Save checkpoint (if interval is specified)
            if CHECKPOINT_INTERVAL is not None and (epoch + 1) % CHECKPOINT_INTERVAL == 0:
                checkpoint = {
                    'params': params,
                    'opt_state': opt_state,
                    'epoch': epoch,
                    'training_stage': current_stage,  # Save training stage for proper resume
                    'loss_history': loss_history,
                    'chamfer_history': chamfer_history,
                    'flow_loss_history': flow_loss_history,
                    'prior_flow_loss_history': prior_flow_loss_history,
                    'vae_kl_history': vae_kl_history,
                    'marginal_kl_history': marginal_kl_history,
                }
                checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pkl"
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(checkpoint, f)
                print(f"✓ Saved checkpoint to {checkpoint_path}")
    
        # Final evaluation for this stage
        print(f"\nStage {current_stage} final evaluation...")
        key, k_final = jax.random.split(key)
        final_chamfer = evaluate(model, params, X_test, k_final, batch_size=BATCH_SIZE, num_samples=len(X_test))
        print(f"Stage {current_stage} Final Chamfer distance: {final_chamfer:.4f}")
        
        # Save checkpoint at end of stage
        stage_final_checkpoint = {
            'params': params,
            'opt_state': opt_state,
            'epoch': final_epoch,
            'training_stage': current_stage,
            'loss_history': loss_history,
            'chamfer_history': chamfer_history,
            'flow_loss_history': flow_loss_history,
            'prior_flow_loss_history': prior_flow_loss_history,
            'vae_kl_history': vae_kl_history,
            'marginal_kl_history': marginal_kl_history,
        }
        stage_checkpoint_path = output_dir / f"checkpoint_epoch_{final_epoch + 1}.pkl"
        with open(stage_checkpoint_path, 'wb') as f:
            pickle.dump(stage_final_checkpoint, f)
        print(f"✓ Saved Stage {current_stage} checkpoint to {stage_checkpoint_path}")
        print()
    
    # Final evaluation and plotting (after all stages)
    print("\n" + "=" * 80)
    print("FINAL EVALUATION (All Stages Complete)")
    print("=" * 80)
    key, k_final = jax.random.split(key)
    final_chamfer = evaluate(model, params, X_test, k_final, batch_size=BATCH_SIZE, num_samples=len(X_test))
    print(f"Final Chamfer distance: {final_chamfer:.4f}")
    
    # Plot loss trajectory with all components
    plot_loss_trajectory(loss_history, chamfer_history, flow_loss_history,
                         prior_flow_loss_history, vae_kl_history, marginal_kl_history,
                         output_dir)
    
    # Save final checkpoint (always saved, even if checkpointing is off)
    # Use the actual final epoch number (accounting for resume)
    actual_final_epoch = start_epoch + num_epochs - 1 if start_epoch > 0 else num_epochs - 1
    final_checkpoint = {
        'params': params,
        'opt_state': opt_state,
        'epoch': actual_final_epoch,  # Actual final epoch index
        'training_stage': stages_to_run[-1] if len(stages_to_run) > 0 else TRAINING_STAGE,  # Save final training stage
        'loss_history': loss_history,
        'chamfer_history': chamfer_history,
        'flow_loss_history': flow_loss_history,
        'prior_flow_loss_history': prior_flow_loss_history,
        'vae_kl_history': vae_kl_history,
        'marginal_kl_history': marginal_kl_history,
    }
    # Save as both final_model.pkl and checkpoint_epoch_{num_epochs}.pkl
    final_path = output_dir / "final_model.pkl"
    with open(final_path, 'wb') as f:
        pickle.dump(final_checkpoint, f)
    print(f"✓ Saved final model to {final_path}")
    
    # Also save as epoch checkpoint for consistency
    checkpoint_path = output_dir / f"checkpoint_epoch_{actual_final_epoch + 1}.pkl"
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(final_checkpoint, f)
    print(f"✓ Saved final epoch checkpoint to {checkpoint_path}")
    
    # Generate final unconditional samples
    print("\nGenerating final unconditional samples...")
    print(f"  Final epoch: {actual_final_epoch + 1}")
    key, k_final_uncond = jax.random.split(key)
    plot_unconditional_samples(model, params, k_final_uncond, output_dir, num_samples=36)
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()


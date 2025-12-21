# Current Model Configuration

## Training Parameters (`src/train_all_digits.py`)

### Data Settings
- **Dataset**: Full MNIST 2D point cloud dataset (`data/mnist_2d_full_dataset.npz`)
- **Max Samples**: None (uses all 60,000 samples)
- **Train/Test Split**: 0.9 (90% train, 10% test)
- **Batch Size**: 128
- **Seed**: 42

### Training Settings
- **Epochs**: 50
- **Learning Rate**: 1e-3 (0.001)
- **Optimizer**: Adam
- **Evaluation Frequency**: Every 5 epochs
- **Checkpoint Frequency**: Every 10 epochs

### Output Directory
- **Path**: `artifacts/all_digits_pointnet_adaln_target`

---

## Model Parameters (`src/models/mnist_flow_2d.py`)

### Architecture
- **Latent Dimension**: 64
- **Spatial Dimension**: 2 (2D point clouds)
- **Encoder Type**: `pointnet`
- **Encoder Output Type**: `global`
- **CRN Type**: `adaln_mlp` (GlobalAdaLNMLPCRN)

### Flow Matching Settings
- **Prediction Target**: `TARGET` (CRN predicts x_0, the data point)
- **Loss Targets**: `(TARGET,)` (train on target loss only)
- **Time Sampling**: Uniform(0.01, 0.99) - avoids boundary issues at t=0 and t=1
- **Optimal Reweighting**: `True`
  - When enabled with TARGET prediction, applies weight: `(1-t)/t^3`
  - This increases weight for small t (easier cases) and decreases for large t (harder cases)

### VAE Settings
- **Use VAE**: `False` (disabled)
- **VAE KL Weight**: 0.000001 (default, not used when VAE is disabled)

### Prior Flow Settings
- **Use Prior Flow**: `False` (disabled)

### Default Parameters (not explicitly set)
- **Encoder Kwargs**: `None` (uses defaults)
- **CRN Kwargs**: `None` (uses defaults)
- **Prior Flow Kwargs**: `None` (not used)

---

## Loss Function Details

### Current Loss Computation
1. **Base Squared Error**: Computed in CRN's native prediction space
   - Since `prediction_target == TARGET`, computes: `(crn_output - x_0)^2`
   - Mean over feature dimension: `(B, N, D) -> (B, N)`

2. **Loss Weight**: 
   - Base weight: `1.0` (for TARGET loss)
   - With optimal reweighting: `1.0 * (1-t)/t^3`

3. **Mask Handling** (if present):
   - Sum over spatial dimension N, normalized by mask sum: `(B, N) -> (B,)`
   - Otherwise, mean over N: `(B, N) -> (B,)`

4. **Final Loss**:
   - `flow_loss = mean(sq_err * loss_weight)` where both are `(B,)`
   - `total_loss = flow_loss` (no VAE KL, no prior flow)

### Metrics Tracked
- `flow_loss`: The main flow matching loss
- `vae_kl`: Always 0.0 (VAE disabled)

---

## Summary

**Current Configuration**:
- Simple setup: PointNet encoder → GlobalAdaLNMLPCRN
- CRN predicts TARGET (x_0) directly
- Training on TARGET loss with optimal time-dependent reweighting
- No VAE, no prior flow
- Time sampling avoids boundaries (0.01 to 0.99)
- Standard Adam optimizer with learning rate 1e-3

**Key Feature**: Optimal reweighting should help balance the difficulty of predicting x_0 across different time values, giving more weight to easier cases (small t) and less to harder cases (large t).





# Training Script for Digit 4

## Overview

This script trains a simple flow model to generate the digit 4 using:
- **Encoder**: PointNet (global encoder)
- **CRN**: GlobalAdaLNMLPCRN
- **Data**: MNIST digit 4 only (filtered from full MNIST dataset)

## Quick Start

```bash
# First, create the digit 4 dataset (one-time setup)
python src/data/create_digit_4_dataset.py

# Then run training
python src/train_digit_4.py
```

## Configuration

The script uses the following default configuration:

```python
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_SAMPLES = 5000  # Number of digit 4 samples
POINTS_PER_DIGIT = 500
SEED = 42
```

### Model Configuration

```python
model = MnistFlow2D(
    latent_dim=64,
    encoder_type='pointnet',
    encoder_output_type='global',
    crn_type='adaln_mlp',
    prediction_target=PredictionTarget.VELOCITY,
    use_vae=False,
    use_prior_flow=False,
)
```

## What the Script Does

1. **Data Loading**:
   - Loads pre-generated digit 4 point cloud dataset (`data/mnist_2d_just_4s.npz`)
   - Contains 6000 digit 4 point clouds (500 points each)
   - Already normalized to [-1, 1] range
   - Splits into train/test (90/10)

2. **Model Initialization**:
   - Creates PointNet encoder (global, outputs `(B, 64)`)
   - Creates GlobalAdaLNMLPCRN
   - Initializes parameters

3. **Training**:
   - Uses Adam optimizer (lr=1e-3)
   - JIT-compiled training step
   - Computes loss and metrics
   - Saves checkpoints every 10 epochs

4. **Evaluation**:
   - Computes Chamfer distance on test set
   - Evaluates every 5 epochs
   - Generates sample visualizations

5. **Outputs**:
   - Loss trajectory plot
   - Sample visualizations (ground truth vs generated)
   - Model checkpoints
   - Final model

## Output Directory

All outputs are saved to: `artifacts/digit_4_pointnet_adaln/`

- `checkpoint_epoch_*.pkl`: Model checkpoints
- `samples.png`: Generated samples (updated every 5 epochs)
- `loss_trajectory.png`: Training loss and Chamfer distance plots
- `final_model.pkl`: Final trained model

## Expected Results

For a simple single-digit model:
- **Training loss**: Should decrease steadily
- **Chamfer distance**: Should reach < 0.1 for good quality
- **Generated samples**: Should look like digit 4

## Customization

You can modify the script to:
- Change number of epochs
- Adjust batch size
- Use different encoder/CRN types
- Enable VAE mode
- Enable prior flow
- Change prediction target (velocity/noise/target)

## Dataset Creation

The digit 4 dataset is created from the full MNIST 2D dataset using standard MNIST ordering:
- Digit 4 is at indices 24000-29999 in the full dataset
- This gives us 6000 digit 4 samples

To create the dataset:
```bash
python src/data/create_digit_4_dataset.py
```

This creates `data/mnist_2d_just_4s.npz` with shape `(6000, 500, 2)`.

## Troubleshooting

### Dataset Not Found

If you get `FileNotFoundError` for `data/mnist_2d_just_4s.npz`:
```bash
# Create the dataset first
python src/data/create_digit_4_dataset.py
```

### Memory Issues

If you run out of memory:
- Reduce `BATCH_SIZE`
- Reduce `NUM_SAMPLES`
- Reduce `POINTS_PER_DIGIT`

### Slow Training

- Ensure JAX is using GPU (if available)
- Reduce number of evaluation samples
- Increase `BATCH_SIZE` if memory allows

## Next Steps

After training:
1. Load the final model: `pickle.load(open('artifacts/digit_4_pointnet_adaln/final_model.pkl', 'rb'))`
2. Generate samples: Use `model.sample()` method
3. Try other digits: Modify the filter to use different digits
4. Experiment with architectures: Try different encoders/CRNs

## Architecture Details

### PointNet Encoder
- Input: `(B, N, 2)` point clouds
- Output: `(B, 64)` global context
- Architecture: MLP + max pooling

### GlobalAdaLNMLPCRN
- Input: `x: (B, N, 2)`, `c: (B, 64)`, `t: scalar`
- Output: `(B, N, 2)` velocity field
- Architecture: MLP with Adaptive Layer Normalization

### Flow Matching
- Predicts velocity field: `v = x_1 - x_0`
- Integrates from noise (t=1) to data (t=0)
- Uses Euler integration (20 steps)

## Files

- `src/train_digit_4.py`: Main training script
- `TRAIN_DIGIT_4_README.md`: This file


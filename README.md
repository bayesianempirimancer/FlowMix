# Flow-Mix: Flow Matching for Point Clouds

A JAX-based implementation of **Flow Matching** for generating point clouds of arbitrary dimension. While the provided examples focus on 2D point clouds (MNIST digits), the codebase is fully general and supports point clouds in any spatial dimension (2D, 3D, or higher). This project implements a flexible, modular architecture combining state-of-the-art encoders with Conditional ResNets (CRNs) for simulation-free generative modeling of point cloud data.

## Overview

Flow-Mix implements **Flow Matching** (also known as Continuous Normalizing Flows) for point cloud generation. Unlike diffusion models that require multiple denoising steps, flow matching learns a continuous vector field that directly transports samples from a simple prior distribution to the data distribution. This enables fast, single-pass generation with high-quality results.  For simplicity this repo assumes a simple linear transport model to connect the source and target distributions.  

**Key Design Principle**: The architecture is dimension-agnostic. The `spatial_dim` parameter controls the dimensionality of point coordinates (e.g., 2 for 2D, 3 for 3D), while all other components (encoders, CRNs, flow models) operate generically on point clouds of any dimension.

### Key Features

- **Flexible Loss Formulations**: Three mathematically equivalent loss functions (velocity, noise, target prediction) that can be used individually or in combination
- **Modular Encoder Architecture**: Support for both global and local encoders including:
  - **Global Encoders**: PointNet, pooled variants
  - **Local Encoders**: Transformer Set, DGCNN, Slot Attention, Cross-Attention, GMM Featurizer, and more
- **Conditional ResNet (CRN) Framework**: Three specialized CRN types optimized for different context structures:
  - **Global CRNs**: Single global conditioning vector
  - **Local CRNs**: One-to-one point-latent correspondence as in point transformer models
  - **Structured CRNs**: Many points, many fewer abstract latents that correspond to 'objects' made of many points
- **Optional VAE Mode**: Optional KL regularization for latent variabes in the style of a VAE
- **Prior Flow Learning**: Optional learnable prior distribution over latents (option include a gmm or another flow model)  
- **Grid Masking**: Training-time masking strategy for improved generalization
- **Modern Architecture Components**: Pre-normalization, SwiGLU activations, multi-query attention, etc.

## Methodology

### Flow Matching

Flow matching learns a vector field `v_t(x)` that defines a continuous transformation from a prior distribution `p_0(x)` to the data distribution `p_1(x)`. The flow is defined by the ODE:

```
dx/dt = v_t(x)
```

During training, we learn to predict the velocity field at any time `t ∈ [0,1]` given:
- The current point `x_t = (1-t)·x_0 + t·x_1` (linear interpolation)
- A latent code `z` encoding the target shape
- The time `t`

### Inference Architecture Pipeline

```
Input Point Cloud (B, N, D_spatial)
    ↓
Encoder (PointNet, Transformer, etc.)
    ↓
Latent Code z which can be global (B, D_latent) or structured (B, K, D_latent)
    ↓
Conditional ResNet (CRN)
    ↓
Velocity Field v_t(x_t, z, t) (B, N, D_spatial)
    ↓
ODE Solver (Euler/RK4)
    ↓
Generated Point Cloud (B, N, D_spatial)
```

Where `D_spatial` is the spatial dimension (2 for 2D, 3 for 3D, etc.) and is configurable via the `spatial_dim` parameter.

### Loss Functions

The model supports three loss formulations related by an affine transformation:

1. **Velocity Prediction** (standard flow matching):
   ```
   L = ||v_θ(x_t, z, t) - (x_1 - x_0)||²
   ```

2. **Noise Prediction** (diffusion-style):
   ```
   L = ||ε_θ(x_t, z, t) - x_1||²
   ```
   where `x_t = (1-t)·x_0 + t·ε`

3. **Target Prediction** (direct):
   ```
   L = ||x̂_0 - x_0||²
   ```

All three loss functions are mathematically related via affine transformations and can be used individually or combined for multi-objective training.  This is because, $x_t = x_0*(1-t) + x_1*t$ and $v = x_1 - x_0$.  As a result, different loss functions are related through a time dependent scaling factor.  None-the-less, prediction target can have a big imact, i.e. it is generally more stable to have the network learn to predict velocity rather than noise or target.  

### Conditional ResNets (CRNs)

CRNs condition the velocity field prediction on latent codes. The architecture adapts based on the encoder output structure:

- **Global CRNs**: When encoder produces a single global vector `(B, D)`, all points receive the same conditioning via Adaptive Layer Normalization (AdaLN)
- **Structured CRNs**: When encoder produces K abstract latents `(B, K, D)` (e.g., slots, mixture components), either:
  - Pool K latents to global (efficient, O(K) → O(1))
  - Cross-attend each point to K latents (expressive, O(N×K))
- **Local CRNs**: Special case where N=K, enabling one-to-one point-latent correspondence

## Installation

### Prerequisites

- Python 3.8+
- JAX (with appropriate backend: CPU, GPU, or TPU)
- NumPy, Matplotlib, tqdm
- TensorFlow Datasets (for MNIST data)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/bayesianempirimancer/FlowMix.git
cd FlowMix
```

2. Install dependencies (using conda environment recommended):
```bash
conda create -n numpyro python=3.9
conda activate numpyro
pip install jax jaxlib optax flax tensorflow-datasets matplotlib tqdm numpy
```

3. (Optional) Generate the MNIST 2D point cloud datasets:
```bash
# Generate all datasets and visualizations
python src/data/generate_all_datasets.py

# Or generate specific datasets using the data generation module
python -m src.data.mnist_point_cloud_data --dataset_type single --num_points 500
python -m src.data.mnist_point_cloud_data --dataset_type multi --num_scenes 20000
```

The `generate_all_datasets.py` script creates various datasets in `data/` including:
- **Single digit datasets** (60,000 samples each):
  - `mnist_2d_single.npz` - No transformations
  - `mnist_2d_single_rotated.npz` - With rotation (0 to 2π)
  - `mnist_2d_single_translated.npz` - With translation (-1.5 to 1.5)
  - `mnist_2d_single_rotated_translated.npz` - With rotation and translation
- **Multi-digit scene datasets** (20,000 scenes each):
  - `multi_mnist_2d.npz` - No transformations
  - `multi_mnist_2d_rotated.npz` - With rotation
  - `multi_mnist_2d_rotated_scaled.npz` - With rotation and scaling

All visualizations are automatically saved to `data/visualizations/` when using `generate_all_datasets.py`.

**Note**: The codebase works with point clouds of any dimension. Simply provide your own dataset with shape `(num_samples, num_points, spatial_dim)` where `spatial_dim` can be 2, 3, or any positive integer.

## Usage

### Training

Train a model on the MNIST 2D example:

```bash
python src/train_all_digits.py
```

#### Training Configuration Options

The training script (`src/train_all_digits.py`) supports extensive configuration options. Edit the configuration section in the `main()` function to customize training:

**Basic Training Parameters:**
- `BATCH_SIZE`: Batch size (default: 256)
- `LEARNING_RATE`: Learning rate (default: 1e-3)
- `JOINT_TRAINING_EPOCHS`: Number of epochs for joint training (default: 100)
- `MAX_SAMPLES`: Maximum number of training samples (None = use all 60,000)
- `SEED`: Random seed for reproducibility (default: 42)

**Training Protocols:**

1. **Joint Training** (default):
   ```python
   TRAINING_STAGE = 0  # Train encoder, CRN, and prior flow simultaneously
   JOINT_TRAINING_EPOCHS = 100
   ```

2. **Two-Stage Training** (automated):
   ```python
   ENABLE_TWO_STAGE_TRAINING = True  # Automatically runs Stage 1 then Stage 2
   STAGE_1_EPOCHS = 100  # Train encoder + CRN (prior frozen)
   STAGE_2_EPOCHS = 100  # Train only prior flow (main model frozen)
   ```

3. **Manual Stage Training**:
   ```python
   ENABLE_TWO_STAGE_TRAINING = False
   TRAINING_STAGE = 1  # Stage 1: train main model, freeze prior
   # or
   TRAINING_STAGE = 2  # Stage 2: train only prior, freeze main model
   ```

**Grid Masking** (for improved generalization):
```python
USE_GRID_MASK = True   # Enable grid masking
GRID_SIZE = 3         # Grid size (e.g., 3x3 grid)
GRID_MASK_PROB = 0.33 # Probability of masking each grid cell
```
- Encoder sees masked points (encourages robust representations)
- Flow model sees full points (learns complete generation)
- Each batch element gets a unique mask pattern

**Optimizer Configuration:**
```python
MOMENTUM_CONFIG = "default"  # Options: "default" or "doubled"
# "default": beta1=0.9, beta2=0.999
# "doubled": beta1=0.95, beta2=0.9995 (longer gradient averaging window)
```

**Resuming Training:**
```python
RESUME_FROM_CHECKPOINT = "artifacts/.../checkpoint_epoch_50.pkl"
# Set to checkpoint path to resume training from a specific epoch
```

**Checkpointing:**
```python
CHECKPOINT_INTERVAL = 10  # Save checkpoint every N epochs (None = only final checkpoint)
```

**Loss Configuration** (in model initialization):
- `vae_kl_weight`: Weight for VAE KL divergence loss (default: 0.0)
- `marginal_kl_weight`: Weight for marginal KL divergence loss (default: 0.0)
- `use_vae`: Enable VAE mode with KL regularization (default: False)

**Prior Flow Configuration:**
```python
use_prior_flow = True
prior_flow_kwargs = {'hidden_dims': (128, 128, 128, 128, 128)}  # 5 layers of size 128
```

#### Training Outputs

Training artifacts are saved to `artifacts/` directory:
- `loss_trajectory.png`: Comprehensive loss plots (total loss, components, Chamfer distance)
- `samples.png`: Conditional samples (ground truth vs generated)
- `unconditional_samples.png`: Unconditional samples from prior
- `final_model.pkl`: Final model checkpoint
- `checkpoint_epoch_N.pkl`: Intermediate checkpoints (if enabled)

### Model Configuration

The flow model supports extensive configuration and works with point clouds of any dimension:

```python
from src.models.mnist_flow_2d import MnistFlow2D, PredictionTarget

# Example: 2D point clouds (MNIST)
model_2d = MnistFlow2D(
    latent_dim=32,                    # Latent code dimension
    spatial_dim=2,                    # 2D point clouds
    encoder_type='pointnet',          # Encoder type
    encoder_output_type='global',      # 'global' or 'local'
    crn_type='adaln_mlp',             # CRN type
    prediction_target=PredictionTarget.VELOCITY,  # What to predict
    loss_targets=(PredictionTarget.VELOCITY,),    # Loss components
    use_vae=False,                     # Enable VAE mode
    use_prior_flow=True,               # Learn prior with flow
    optimal_reweighting=False,         # Time-dependent reweighting
)

# Example: 3D point clouds
model_3d = MnistFlow2D(
    latent_dim=64,
    spatial_dim=3,                    # 3D point clouds
    encoder_type='pointnet',
    encoder_output_type='global',
    crn_type='adaln_mlp',
    prediction_target=PredictionTarget.VELOCITY,
    loss_targets=(PredictionTarget.VELOCITY,),
    use_vae=False,
    use_prior_flow=True,
)
```

**Key Parameter**: `spatial_dim` controls the dimensionality of point coordinates. The rest of the architecture (encoders, CRNs, flow models) operates generically regardless of this dimension.

### Supported Encoders

**Global Encoders:**
- `pointnet`: PointNet with max/mean pooling
- Pooled variants of local encoders (via `MaxPoolingEncoder`, `MeanPoolingEncoder`, `AttentionPoolingEncoder`)

**Local Encoders:**
- `transformer`: Transformer Set Encoder
- `dgcnn`: Dynamic Graph CNN
- `slot_attention`: Slot Attention Encoder
- `cross_attention`: Perceiver-style Cross-Attention Encoder
- `gmm`: Gaussian Mixture Model Featurizer
- And more (see `src/encoders/local_encoders/`)

### Supported CRN Types

- `adaln_mlp`: MLP with Adaptive Layer Normalization
- `dit`: Diffusion Transformer (DiT-style)
- `cross_attention`: Cross-attention based CRN

## Project Structure

```
OC-Flow-Mix/
├── src/
│   ├── models/
│   │   ├── mnist_flow_2d.py          # Main flow model (general, not 2D-specific)
│   │   ├── global_crn.py             # Global CRNs
│   │   ├── local_crn.py              # Local CRNs (N=K)
│   │   ├── structured_crn.py         # Structured CRNs (N>>K)
│   │   └── simple_latent_flow.py     # Prior flow model
│   ├── encoders/
│   │   ├── global_encoders/          # Global encoders
│   │   │   ├── pointnet.py
│   │   │   └── pooling.py
│   │   └── local_encoders/          # Local encoders
│   │       ├── transformer_set.py
│   │       ├── dgcnn.py
│   │       ├── slot_attention_encoder.py
│   │       └── ...
│   ├── layers/
│   │   ├── self_attention.py         # Modern self-attention
│   │   ├── cross_attention.py        # Modern cross-attention
│   │   └── concat_squash.py          # Concat-squash layers
│   ├── data/
│   │   ├── mnist_point_cloud_data.py  # Dataset generation and loading
│   │   ├── visualize_mnist_data.py    # Dataset visualization
│   │   └── generate_all_datasets.py  # Script to generate all datasets
│   ├── utils/
│   │   ├── training.py                # Training utilities (Chamfer distance, etc.)
│   │   └── viz.py                    # General visualization utilities
│   ├── plotting/                      # Plotting utilities for training and analysis
│   │   ├── samples.py                # Sample generation plots
│   │   ├── latents.py                # Latent space analysis (t-SNE)
│   │   ├── losses.py                 # Loss trajectory plots
│   │   ├── masking.py                # Masked training visualization
│   │   └── training.py               # Legacy plotting utilities
│   ├── gmm/                          # GMM components
│   ├── train_all_digits.py           # Main training script
│   └── analyze_encoder_latents.py   # Latent space analysis script
├── docs/                              # Documentation
│   ├── CRN_ARCHITECTURE_GUIDE.md
│   ├── CRN_FINAL_STRUCTURE.md
│   ├── ENCODER_COMPARISON.md
│   └── ...
├── data/                              # Generated datasets (gitignored)
└── artifacts/                         # Training outputs (gitignored)
```

**Note on Plotting Utilities**: All plotting and figure generation code has been organized into the `src/plotting/` directory. The training script and analysis scripts import plotting functions from this centralized location. Dataset visualization functions remain in `src/data/visualize_mnist_data.py` to keep data-related code together.

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **`CRN_ARCHITECTURE_GUIDE.md`**: Detailed guide to Conditional ResNet architectures
- **`CRN_FINAL_STRUCTURE.md`**: Complete reference for CRN types and usage
- **`ENCODER_COMPARISON.md`**: Comparison of different encoder architectures
- **`MNIST_FLOW_2D_UPDATE.md`**: Model architecture and loss function details
- **`MODERNIZATION_SUMMARY.md`**: Overview of modern architecture improvements

## Key Technical Details

### Dimension-Agnostic Design

The codebase is designed to work with point clouds of **any spatial dimension**. This is achieved through:

1. **Configurable Spatial Dimension**: The `spatial_dim` parameter in the model controls the dimensionality of point coordinates. All components (encoders, CRNs, flow models) operate generically on `(B, N, spatial_dim)` tensors.

2. **Generic Encoders**: All encoders process point clouds regardless of spatial dimension:
   - PointNet operates on `(B, N, D)` where D can be 2, 3, or any dimension
   - Graph-based encoders (DGCNN) build graphs based on point neighborhoods, independent of dimension
   - Attention-based encoders operate on point features, not coordinates

3. **Dimension-Independent CRNs**: Conditional ResNets predict velocity fields `(B, N, spatial_dim)` where the spatial dimension is a parameter, not hardcoded.

4. **Flexible Loss Functions**: All loss formulations work identically for any spatial dimension since they operate on coordinate differences.

**Example Use Cases**:
- **2D**: MNIST digits, 2D shapes, planar point sets
- **3D**: 3D meshes, point cloud scans, molecular structures, CAD models
- **4D+**: Time-varying point clouds, higher-dimensional embeddings

### Flow Matching vs Diffusion

Flow matching offers several advantages over diffusion models:
- **Faster generation**: Single ODE solve vs. multiple denoising steps
- **Straightforward training**: Direct velocity prediction vs. noise prediction
- **Flexible paths**: Can use optimal transport paths vs. fixed noise schedule

### Analysis and Visualization

After training, analyze the learned latent representations:

```bash
python src/analyze_encoder_latents.py
```

This script:
- Computes latent space statistics (mean, variance per dimension)
- Generates per-digit statistics
- Creates t-SNE visualizations colored by digit identity
- Saves results to the model's artifact directory

**Configuration:**
Edit the `checkpoint_path` in `analyze_encoder_latents.py` to point to your trained model.

### Training Protocols

The codebase supports multiple training protocols:

#### Joint Training (Default)
Train encoder, CRN, and prior flow simultaneously. This is the simplest approach and works well for most cases.

#### Two-Stage Training
A two-stage protocol that can improve training stability:
1. **Stage 1**: Train encoder + main CRN (prior flow frozen)
   - Allows the main model to learn good representations first
   - Prior flow parameters are frozen (no gradients)
2. **Stage 2**: Train only prior flow (main model frozen)
   - Learns the prior distribution after the main model is trained
   - Main model parameters are frozen (no gradients)

**Usage:**
- Set `ENABLE_TWO_STAGE_TRAINING = True` for automated two-stage training
- Or manually run stages by setting `TRAINING_STAGE = 1` or `2`

#### Grid Masking
A regularization technique that improves generalization:
- Points are masked in a grid pattern (configurable grid size)
- **Encoder** sees masked points (encourages robust latent representations)
- **Flow model** sees full points (learns complete generation)
- Each batch element receives a unique mask pattern

**Configuration:**
```python
USE_GRID_MASK = True
GRID_SIZE = 3          # 3x3 grid
GRID_MASK_PROB = 0.33 # Probability of masking each cell
```

This technique is particularly useful for improving robustness to partial observations.

## Results

The example implementation generates high-quality 2D point cloud representations of MNIST digits. Training artifacts (samples, loss trajectories, etc.) are saved to the `artifacts/` directory.

**General Applicability**: The same codebase can be used for:
- **2D point clouds**: MNIST digits, 2D shapes, planar point sets
- **3D point clouds**: 3D meshes, point cloud scans, molecular structures
- **Higher dimensions**: Point sets in 4D+ spaces (e.g., time-varying 3D point clouds)

Simply set `spatial_dim` to the desired dimension and provide appropriately shaped data.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{flowmix2025,
  title={OC-Flow-Mix: Flow Matching for Point Clouds},
  author={Bayesian Empirimancer},
  year={2025},
  url={https://github.com/bayesianempirimancer/FlowMix}
}
```

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This implementation builds on:
- Flow Matching (Lipman et al., 2023)
- Conditional ResNets for generative modeling
- Modern transformer architectures (DiT, Perceiver, etc.)
- JAX/Flax ecosystem

---

For questions or issues, please open an issue on GitHub.


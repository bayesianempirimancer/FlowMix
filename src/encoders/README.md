# Encoders for Point Cloud Generative Models

This directory contains encoder architectures for processing 2D/3D point clouds, organized into **global**, **local**, and **structured** encoders.

## Directory Structure

```
encoders/
├── global_encoders/     # Output single vectors (B, dim)
├── local_encoders/      # Output per-point features (B, N, dim)
├── structured_encoders/ # Output reduced features (B, K, dim) where K < N
├── embeddings.py        # Positional and time embeddings
└── __init__.py          # Main exports (backward compatible)
```

## Encoder Types

### Global Encoders
Output: Single vector per batch
- **Output shape**: `(B, dim)`

### Local Encoders
Output: Per-point features, preserving all input points
- **Output shape**: `(B, N, dim)` where N is number of input points

### Structured Encoders
Output: Reduced dimensionality from N to K (number of objects/features)
- **Output shape**: `(B, K, dim)` where K < N

## Local Encoders

Local encoders output sequences of features, preserving spatial/sequential structure:
- **Output shape**: `(B, N, dim)` where N is number of input points

### Available Local Encoders

1. **TransformerSetEncoder** (`local_encoders/transformer_set.py`)
   - Self-attention transformer with CLS token
   - Output: `(B, N+1, embed_dim)`

2. **DGCNN** (`local_encoders/dgcnn.py`)
   - Dynamic Graph CNN using k-nearest neighbors
   - Output: `(B, N, embed_dim)`

3. **VN_DGCNN** (`local_encoders/vn_dgcnn.py`)
   - Vector Neurons DGCNN (equivariant)
   - Output: `(B, N, 3, embed_dim)`

4. **KPConv** (`local_encoders/kpconv.py`)
   - Kernel Point Convolution
   - Output: `(B, N, embed_dim)`

5. **EGNN** (`local_encoders/egnn.py`)
   - E(n)-Equivariant Graph Neural Network
   - Output: `(B, N, embed_dim)`

## Structured Encoders

Structured encoders reduce dimensionality from N (input points) to K (objects/features):
- **Output shape**: `(B, K, dim)` where K < N

### Available Structured Encoders

1. **SlotAttentionEncoder** (`structured_encoders/slot_attention_encoder.py`)
   - Iterative attention mechanism binding input points to K slots
   - Output: `(B, K, slot_dim)`

2. **CrossAttentionEncoder** (`structured_encoders/cross_attention_encoder.py`)
   - Perceiver-style encoder with M learnable latents
   - Output: `(B, M, latent_dim)`

3. **GMMFeaturizer** (`structured_encoders/gmm_featurizer.py`)
   - Variational Bayesian GMM with geometric feature extraction
   - Output: `(B, L, D)` where L = K * (1 + 2*D)

4. **PointNetPlusPlus** (`structured_encoders/pointnet_plus_plus.py`)
   - Hierarchical point cloud processing with set abstraction
   - Output: `(B, M, embed_dim)` where M decreases with each layer

5. **PointNeXt** (`structured_encoders/pointnext.py`)
   - Improved PointNet++ with better normalization
   - Output: `(B, M, embed_dim)` where M < N

6. **PointMLP** (`structured_encoders/pointmlp.py`)
   - Pure MLP-based encoder with hierarchical sampling
   - Output: `(B, M, embed_dim)` where M < N

7. **Set2Set** (`structured_encoders/set2set.py`)
   - LSTM with attention mechanism for set processing
   - Output: `(B, T, embed_dim)` where T is number of processing steps

## Global Encoders

Global encoders output single vectors per batch, typically wrapping local encoders with pooling strategies:
- **Output shape**: `(B, latent_dim)`

### Available Global Encoders

1. **PointNetEncoder** (`pointnet.py`)
   - MLP + Max Pooling
   - Wraps: Direct PointNet implementation

2. **TransformerEncoder** (`transformer.py`)
   - TransformerSetEncoder + CLS token pooling
   - Wraps: `TransformerSetEncoder`

3. **SlotAttentionEncoder** (`slot_attention.py`)
   - SlotAttentionEncoder + Max Pooling
   - Wraps: `SlotAttentionEncoder`

4. **CrossAttentionEncoder** (`cross_attention.py`)
   - CrossAttentionEncoder + Flattening
   - Wraps: `CrossAttentionEncoder`

5. **GMMEncoder** (`gmm.py`)
   - GMMFeaturizer + PointNet Pooling
   - Wraps: `GMMFeaturizer`

### Pooling Strategies (`pooling.py`)

These can wrap any local encoder:

1. **MeanPoolingEncoder**
   - Mean pooling over local features
   - Masked mean if mask provided

2. **MaxPoolingEncoder**
   - Max pooling over local features
   - Masked max if mask provided

3. **AttentionPoolingEncoder**
   - Learned attention-based pooling
   - Learns to attend to important features

4. **Set2SetPoolingEncoder**
   - LSTM with attention for set processing
   - Multiple processing steps

## Usage

### Using Global Encoders (Backward Compatible)

```python
from src.encoders import PointNetEncoder, TransformerEncoder

encoder = PointNetEncoder(latent_dim=128)
z_mu, z_logvar = encoder(x, mask=mask, key=key)  # (B, 128)
```

### Using Local Encoders

```python
from src.encoders.local_encoders import DGCNN, TransformerSetEncoder

local_encoder = DGCNN(embed_dim=64)
features = local_encoder(x, mask=mask, key=key)  # (B, N, 64)
```

### Using Structured Encoders

```python
from src.encoders.structured_encoders import SlotAttentionEncoder, CrossAttentionEncoder

# Slot Attention: reduces N points to K slots
slot_encoder = SlotAttentionEncoder(num_slots=8, slot_dim=64)
slots = slot_encoder(x, mask=mask, key=key)  # (B, 8, 64)

# Cross Attention: reduces N points to M latents
cross_encoder = CrossAttentionEncoder(num_latents=32, latent_dim=64)
latents = cross_encoder(x, mask=mask, key=key)  # (B, 32, 64)
```

### Creating Custom Global Encoders with Pooling

```python
from src.encoders.local_encoders import DGCNN
from src.encoders.structured_encoders import SlotAttentionEncoder
from src.encoders.global_encoders.pooling import AttentionPoolingEncoder

# Pool from local encoder
local_encoder = DGCNN(embed_dim=64)
global_encoder = AttentionPoolingEncoder(
    local_encoder=local_encoder,
    latent_dim=128
)
z = global_encoder(x, mask=mask, key=key)  # (B, 128)

# Pool from structured encoder
structured_encoder = SlotAttentionEncoder(num_slots=8, slot_dim=64)
global_encoder = AttentionPoolingEncoder(
    local_encoder=structured_encoder,
    latent_dim=128
)
z = global_encoder(x, mask=mask, key=key)  # (B, 128)
```

## Architecture Composition

All encoders follow a general pattern:

**Local Encoders:**
```
Input Points (B, N, D) -> [Local Encoder] -> Features (B, N, dim)
```

**Structured Encoders:**
```
Input Points (B, N, D) -> [Structured Encoder] -> Features (B, K, dim) where K < N
```

**Global Encoders:**
```
Input Points (B, N, D) -> [Local/Structured Encoder] -> Features (B, N/K, dim) -> [Pooling] -> Latent (B, dim)
```

## Design Philosophy

- **Separation of Concerns**: Local encoders focus on feature extraction, global encoders focus on aggregation
- **Composability**: Global encoders wrap local encoders, allowing flexible combinations
- **Backward Compatibility**: Existing code using global encoders continues to work
- **Extensibility**: Easy to add new local encoders or pooling strategies

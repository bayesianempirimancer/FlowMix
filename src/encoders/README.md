# Encoders for Point Cloud Generative Models

This directory contains encoder architectures for processing 2D/3D point clouds, organized into **local** and **global** encoders.

## Directory Structure

```
encoders/
├── local_encoders/     # Output sequences (B, N, dim) or (B, K, dim)
├── global_encoders/    # Output single vectors (B, dim)
├── embeddings.py       # Positional and time embeddings
└── __init__.py         # Main exports (backward compatible)
```

## Local Encoders

Local encoders output sequences of features, preserving spatial/sequential structure:
- **Output shape**: `(B, N, dim)` or `(B, K, dim)` where N is number of input points, K is number of learned features

### Available Local Encoders

1. **TransformerSetEncoder** (`transformer_set.py`)
   - Self-attention transformer with CLS token
   - Output: `(B, N+1, embed_dim)`

2. **SlotAttentionEncoder** (`slot_attention_encoder.py`)
   - Iterative attention mechanism binding input points to K slots
   - Output: `(B, K, slot_dim)`

3. **CrossAttentionEncoder** (`cross_attention_encoder.py`)
   - Perceiver-style encoder with M learnable latents
   - Output: `(B, M, latent_dim)`

4. **GMMFeaturizer** (`gmm_featurizer.py`)
   - Variational Bayesian GMM with geometric feature extraction
   - Output: `(B, L, D)` where L = K * (1 + 2*D)

5. **DGCNN** (`dgcnn.py`)
   - Dynamic Graph CNN using k-nearest neighbors
   - Output: `(B, N, embed_dim)`

6. **PointNetPlusPlus** (`pointnet_plus_plus.py`)
   - Hierarchical point cloud processing with set abstraction
   - Output: `(B, M, embed_dim)` where M decreases with each layer

7. **Set2Set** (`set2set.py`)
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

### Creating Custom Global Encoders with Pooling

```python
from src.encoders.local_encoders import DGCNN
from src.encoders.global_encoders.pooling import AttentionPoolingEncoder

local_encoder = DGCNN(embed_dim=64)
global_encoder = AttentionPoolingEncoder(
    local_encoder=local_encoder,
    latent_dim=128
)
z_mu, z_logvar = global_encoder(x, mask=mask, key=key)  # (B, 128)
```

## Architecture Composition

All encoders follow a general pattern:

**Local Encoders:**
```
Input Points (B, N, D) -> [Local Encoder] -> Features (B, N, dim) or (B, K, dim)
```

**Global Encoders:**
```
Input Points (B, N, D) -> [Local Encoder] -> Features (B, N, dim) -> [Pooling] -> Latent (B, dim)
```

## Design Philosophy

- **Separation of Concerns**: Local encoders focus on feature extraction, global encoders focus on aggregation
- **Composability**: Global encoders wrap local encoders, allowing flexible combinations
- **Backward Compatibility**: Existing code using global encoders continues to work
- **Extensibility**: Easy to add new local encoders or pooling strategies

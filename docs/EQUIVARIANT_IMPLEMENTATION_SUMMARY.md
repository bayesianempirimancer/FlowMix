# Equivariant Encoder Implementation Summary

## Implemented Encoders

### 1. VN-DGCNN (Vector Neuron DGCNN)
**File**: `src/encoders/local_encoders/vn_dgcnn.py`

**Key Components**:
- `VNLinear`: Linear layer for 3D vector features
- `VNReLU`: ReLU that preserves vector direction
- `VNLayerNorm`: Layer normalization for vector features
- `VNEdgeConv`: Equivariant edge convolution
- `VN_DGCNN`: Main encoder (outputs equivariant features)
- `VN_DGCNN_Invariant`: Variant that outputs invariant features (takes norms)

**Properties**:
- **Equivariance**: SO(3) (rotations)
- **Output**: `(B, N, 3, embed_dim)` - each feature is a 3D vector
- **Invariant version**: `(B, N, embed_dim)` - scalar features
- **Speed**: Similar to regular DGCNN
- **Memory**: 3× more (vector vs scalar features)

**Usage**:
```python
# Equivariant features (rotate with input)
encoder = VN_DGCNN(embed_dim=64, k=20, num_layers=4)
features = encoder(x)  # (B, N, 3, 64)

# Invariant features (for classification)
encoder = VN_DGCNN_Invariant(embed_dim=64, k=20, num_layers=4)
features = encoder(x)  # (B, N, 64)
```

### 2. EGNN (E(n) Equivariant Graph Neural Network)
**File**: `src/encoders/local_encoders/egnn.py`

**Key Components**:
- `EGNNLayer`: Single E(n)-equivariant message passing layer
- `EGNN`: Main encoder (outputs invariant features)
- `EGNN_Coords`: Variant that also returns updated coordinates

**Properties**:
- **Equivariance**: E(n) (rotations + translations + reflections)
- **Output**: `(B, N, embed_dim)` - invariant features
- **Coordinate updates**: Can also return equivariant coordinate updates
- **Speed**: Similar to DGCNN (graph-based)
- **Memory**: Similar to non-equivariant encoders

**Usage**:
```python
# Invariant features only
encoder = EGNN(embed_dim=64, hidden_dim=128, num_layers=4, k=20)
features = encoder(x)  # (B, N, 64)

# Features + updated coordinates
encoder = EGNN_Coords(embed_dim=64, hidden_dim=128, num_layers=4, k=20)
features, coords = encoder(x)  # (B, N, 64), (B, N, D)
```

## Comparison: VN-DGCNN vs EGNN

| Property | VN-DGCNN | EGNN |
|----------|----------|------|
| **Equivariance** | SO(3) (rotations) | E(n) (rotations + translations + reflections) |
| **Feature Type** | Vector (3D) | Scalar (invariant) |
| **Output** | Equivariant or Invariant | Invariant (+ optional equivariant coords) |
| **Memory** | 3× (vectors) | 1× (scalars) |
| **Speed** | Fast | Fast |
| **Coordinate Updates** | No | Yes |
| **Best For** | When you need equivariant features | When you need invariant features + coord updates |

## When to Use Each

### Use VN-DGCNN when:
- You need **equivariant intermediate features** (e.g., for equivariant flow models)
- You want features that **rotate with the input**
- You're okay with **3× memory** for vector features
- You only need **SO(3)** (rotations), not full E(n)

### Use EGNN when:
- You need **invariant features** for classification/recognition
- You want to **update coordinates** equivariantly (e.g., for denoising, refinement)
- You need **full E(n) equivariance** (rotations + translations + reflections)
- You want **lower memory** usage (scalar features)

## Integration with Global Encoders

Both can be wrapped with pooling strategies:

```python
from src.encoders.global_encoders.pooling import MaxPoolingEncoder

# VN-DGCNN (invariant version) + pooling
local_encoder = VN_DGCNN_Invariant(embed_dim=64)
global_encoder = MaxPoolingEncoder(local_encoder=local_encoder, latent_dim=128)
z_mu, z_logvar = global_encoder(x)  # (B, 128)

# EGNN + pooling
local_encoder = EGNN(embed_dim=64)
global_encoder = MaxPoolingEncoder(local_encoder=local_encoder, latent_dim=128)
z_mu, z_logvar = global_encoder(x)  # (B, 128)
```

## Benefits for Multi-MNIST

### Expected Improvements:
1. **Rotation Invariance**: Digits at any orientation recognized equally well
2. **Data Efficiency**: Less need for rotation augmentation
3. **Generalization**: Better performance on unseen orientations
4. **Robustness**: Natural handling of transformed inputs

### Which to Use:
- **VN-DGCNN**: If you want the flow model itself to be equivariant
- **EGNN**: If you just want rotation-invariant recognition

## Implementation Notes

### VN-DGCNN:
- Features are **3D vectors**: `(B, N, 3, D)` instead of `(B, N, D)`
- Each "feature" is actually a 3D vector that rotates with the input
- Final pooling can either:
  - Keep vectors (equivariant)
  - Take norms (invariant)

### EGNN:
- Maintains both **features** (invariant) and **coordinates** (equivariant)
- Message passing updates both
- Coordinate updates are optional (useful for generative tasks)
- Features are always invariant (don't change under transformations)

## Testing

```bash
✓ VN-DGCNN imports successfully
✓ EGNN imports successfully
✓ Can be wrapped with pooling strategies
```

## Future Extensions

### Easy:
1. **VN-PointNet**: Vector neuron version of PointNet (simpler than VN-DGCNN)
2. **VN-PointNet++**: Hierarchical vector neurons

### Medium:
3. **SE(3)-Transformer**: Equivariant self-attention
4. **Steerable CNNs**: Another approach to equivariance

### Hard:
5. **EPN/E2PN**: Full SE(3) convolutions
6. **Quaternion Capsules**: Explicit pose disentanglement

## References

1. **Vector Neurons**: Deng et al., "Vector Neurons: A General Framework for SO(3)-Equivariant Networks" (NeurIPS 2021)
2. **EGNN**: Satorras et al., "E(n) Equivariant Graph Neural Networks" (ICML 2021)


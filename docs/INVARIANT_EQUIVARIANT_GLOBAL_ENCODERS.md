# Equivariant Global Encoder

## Overview

This document describes the **Equivariant Deep Sets encoder** - a global encoder that respects geometric symmetries for point cloud processing.

## Why Equivariant Encoders?

### Equivariant Encoders
- **Output transforms** in the same way as the input
- Useful for **generation tasks** where we want to preserve geometric relationships
- Example: Generating a point cloud that rotates with the input

### Invariant vs Equivariant
- **Invariant**: Output doesn't change under rotation/translation (useful for recognition)
- **Equivariant**: Output transforms with input (useful for generation)

## EquivariantDeepSetsEncoder

**File**: `src/encoders/global_encoders/invariant_equivariant.py`

### Properties
- **E(n) equivariant**: Rotations + translations + reflections in n-dimensional space
- **Flexible output**: Can output invariant or equivariant features
- **Dimension-agnostic**: Works for 2D, 3D, or any dimension
- **Message passing**: Uses distance-based attention for equivariant updates

### Architecture

1. **Center points** (translation invariant)
2. **Initialize features** from centered coordinates
3. **Equivariant processing layers**:
   - Compute pairwise distances (invariant)
   - Distance-based attention (message passing)
   - Update features equivariantly
4. **Final aggregation**:
   - **Invariant mode**: Pool features to get invariant global vector
   - **Equivariant mode**: Use coordinates as basis for equivariant output

### Usage

#### Invariant Output (Default)
```python
from src.encoders.global_encoders import EquivariantDeepSetsEncoder

encoder = EquivariantDeepSetsEncoder(
    latent_dim=128,
    hidden_dims=(64, 128, 256),
    output_type='invariant',  # Default
)
z = encoder(x, mask=mask)  # (B, 128) - invariant global features
```

#### Equivariant Output
```python
encoder = EquivariantDeepSetsEncoder(
    latent_dim=128,
    hidden_dims=(64, 128, 256),
    output_type='equivariant',
)
z = encoder(x, mask=mask)  # (B, D, 128) - equivariant global features
```

### Parameters

- `latent_dim` (int): Dimension of output latent code (default: 128)
- `hidden_dims` (Sequence[int]): Hidden dimensions for processing layers (default: (64, 128, 256))
- `output_type` (Literal['invariant', 'equivariant']): Type of output (default: 'invariant')

### Output Shapes

- **Invariant mode**: `(B, latent_dim)` - Single global vector
- **Equivariant mode**: `(B, D, latent_dim)` - D vectors (one per spatial dimension)

## Integration with Flow Models

### Invariant Output → Global CRN
```python
from src.encoders.global_encoders import EquivariantDeepSetsEncoder
from src.models.global_crn import GlobalAdaLNMLPCRN

# Invariant encoder
encoder = EquivariantDeepSetsEncoder(
    latent_dim=128,
    output_type='invariant',
)
z = encoder(x, mask=mask)  # (B, 128) - invariant

# Use with Global CRN
crn = GlobalAdaLNMLPCRN()
dx = crn(x_t, c=z, t=t)  # All points get same invariant conditioning
```

### Equivariant Output → Structured CRN
```python
# Equivariant encoder
encoder = EquivariantDeepSetsEncoder(
    latent_dim=128,
    output_type='equivariant',
)
z = encoder(x, mask=mask)  # (B, D, 128) - equivariant

# Use with Structured CRN (treat as K=D abstract latents)
crn = StructuredAdaLNMLPCRN()
dx = crn(x_t, c=z, t=t)  # Points conditioned on equivariant features
```

## When to Use

### Use Invariant Output When:
- You need **rotation-invariant** representations
- Orientation doesn't matter (e.g., digit recognition)
- You want **data efficiency** (less augmentation needed)

### Use Equivariant Output When:
- You need to **preserve geometric relationships**
- Output should **transform with input** (e.g., equivariant flow models)
- You want **geometric interpretability**

## Implementation Notes

### Dimension Handling
- **Dimension-agnostic**: Works for 2D, 3D, or any spatial dimension
- Automatically adapts to input dimension `D`

### Masking Support
- Fully supports **masking** (variable number of points)
- Masking is handled properly for centering and pooling operations

### Performance
- **Speed**: Fast (O(N²) for attention, but efficient)
- **Memory**: Moderate (stores attention matrix)

## References

1. Deep Sets (2017): https://arxiv.org/abs/1703.06114
2. E(n) Equivariant Graph Neural Networks (2021): https://arxiv.org/abs/2102.09844

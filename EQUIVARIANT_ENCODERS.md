# Equivariant Point Cloud Encoders

## Why Equivariance Matters

**Equivariance**: If you rotate/translate the input, the output features rotate/translate in a predictable way.
**Invariance**: Output doesn't change under transformations (special case of equivariance).

For multi-MNIST with digits at different positions and orientations, equivariant encoders can:
1. **Generalize better** - Same digit at different orientations produces related features
2. **Require less data** - Don't need to learn the same pattern at every orientation
3. **Be more robust** - Natural handling of transformations

## Key Equivariant Architectures

### 1. **Vector Neurons (VN-Layers)**
- **Type**: Rotation-equivariant neurons
- **Key Idea**: Neurons are 3D vectors instead of scalars
- **Equivariance**: SO(3) (rotations)
- **Why Important**: Simple, can be added to any architecture
- **Applications**: VN-DGCNN, VN-PointNet
- **Output**: `(B, N, 3×D)` where features are 3D vectors
- **Speed**: Similar to non-equivariant versions
- **Paper**: "Vector Neurons: A General Framework for SO(3)-Equivariant Networks" (2021)

### 2. **Equivariant Point Network (EPN)**
- **Type**: SE(3)-equivariant convolution
- **Key Idea**: Separable point convolutions on SE(3)
- **Equivariance**: SE(3) (rotations + translations)
- **Why Important**: Full SE(3) equivariance, not just rotations
- **Output**: `(B, N, D)` with equivariant features
- **Speed**: Medium (more expensive than VN)
- **Paper**: "Equivariant Point Network for 3D Point Cloud Analysis" (2021)

### 3. **E2PN (Efficient EPN)**
- **Type**: Efficient SE(3)-equivariant network
- **Key Idea**: Convolution on S²×ℝ³ (homogeneous space)
- **Equivariance**: SE(3)
- **Why Important**: More efficient than EPN
- **Output**: `(B, N, D)` with equivariant features
- **Speed**: Faster than EPN, still slower than VN
- **Paper**: "E2PN: Efficient SE(3)-Equivariant Point Network" (2022)

### 4. **SE(3)-Transformer**
- **Type**: Equivariant self-attention
- **Key Idea**: Attention mechanism that respects SE(3)
- **Equivariance**: SE(3)
- **Why Important**: Combines transformers with equivariance
- **Output**: `(B, N, D)` with equivariant features
- **Speed**: Slow (O(N²) like transformers)
- **Paper**: "SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks" (2020)

### 5. **Quaternion Equivariant Capsule Networks**
- **Type**: Capsule network with quaternions
- **Key Idea**: Capsules represent pose with quaternions
- **Equivariance**: SO(3) via quaternions
- **Why Important**: Explicitly disentangles geometry from pose
- **Output**: `(B, K, capsule_dim)` where K is number of capsules
- **Speed**: Medium
- **Paper**: "3D Point Capsule Networks" (2019)

### 6. **EGNN (E(n) Equivariant Graph Neural Networks)**
- **Type**: Graph neural network
- **Key Idea**: Message passing that preserves E(n) symmetry
- **Equivariance**: E(n) (Euclidean group in n dimensions)
- **Why Important**: Simple, general framework
- **Output**: `(B, N, D)` with equivariant features
- **Speed**: Fast (similar to DGCNN)
- **Paper**: "E(n) Equivariant Graph Neural Networks" (2021)

## Comparison Matrix

| Encoder | Equivariance | Complexity | Speed | Memory | Implementation Difficulty |
|---------|--------------|------------|-------|--------|---------------------------|
| **VN-Layers** | SO(3) | O(N) to O(N²) | Fast | Low | ⭐ Easy (drop-in replacement) |
| **EGNN** | E(n) | O(N²) | Fast | Medium | ⭐⭐ Medium (graph-based) |
| **EPN** | SE(3) | O(N×K) | Medium | Medium | ⭐⭐⭐ Hard (SE(3) convolutions) |
| **E2PN** | SE(3) | O(N×K) | Medium | Medium | ⭐⭐⭐⭐ Very Hard (S²×ℝ³) |
| **SE(3)-Transformer** | SE(3) | O(N²) | Slow | High | ⭐⭐⭐ Hard (equivariant attention) |
| **Quaternion Capsules** | SO(3) | O(N×K) | Medium | Medium | ⭐⭐⭐ Hard (capsule routing) |

## Recommended for Multi-MNIST

### High Priority (Easy to Implement, High Value):

1. **VN-DGCNN** (Vector Neuron + DGCNN)
   - Easiest to implement (modify existing DGCNN)
   - SO(3) equivariance (rotations)
   - Fast, proven effective
   - **Implementation**: Replace scalar features with 3D vector features

2. **EGNN** (E(n) Equivariant GNN)
   - Simple, elegant framework
   - E(n) equivariance (rotations + translations + reflections)
   - Graph-based like DGCNN
   - **Implementation**: Message passing with equivariant updates

### Medium Priority:

3. **VN-PointNet++** (Vector Neuron + PointNet++)
   - Hierarchical + equivariant
   - Combines benefits of both

### Lower Priority (More Complex):

4. **SE(3)-Transformer** - If you want equivariant attention
5. **EPN/E2PN** - If you need full SE(3) convolutions
6. **Quaternion Capsules** - If you want explicit pose disentanglement

## Key Design Choices

### Equivariance vs Invariance:
- **Equivariant features**: Useful for intermediate layers (preserve spatial relationships)
- **Invariant features**: Useful for final classification (rotation doesn't matter)
- **Strategy**: Use equivariant layers, then pool to invariant features

### What to Make Equivariant:
- **Local encoders**: Should be equivariant (preserve geometry)
- **Global encoders**: Can be invariant (just need to recognize what it is)
- **Our case**: Local encoders equivariant → pool to invariant global features

### Computational Trade-offs:
- **VN-Layers**: 3× memory (3D vectors vs scalars), same speed
- **SE(3) operations**: More expensive than SO(3)
- **Full equivariance**: Not always necessary (SO(3) often sufficient)

## Implementation Strategy

### Phase 1: Vector Neurons (Easiest)
1. Implement `VNLinear`, `VNReLU`, `VNLayerNorm` (vector neuron layers)
2. Create `VN_DGCNN` by replacing DGCNN layers with VN layers
3. Create `VN_PointNet` as simpler baseline

### Phase 2: EGNN (Medium)
1. Implement EGNN message passing
2. Create `EGNN` encoder

### Phase 3: Advanced (Optional)
1. SE(3)-Transformer if needed
2. EPN/E2PN if full SE(3) needed

## Expected Benefits for Multi-MNIST

1. **Better generalization**: Digits at any orientation recognized equally well
2. **Data efficiency**: Less augmentation needed
3. **Robustness**: Natural handling of rotated/translated digits
4. **Interpretability**: Features have geometric meaning

## Literature Notes

- **VN-Layers** are the "easiest win" - simple to implement, proven effective
- **EGNN** is elegant and general - good for research/comparison
- **SE(3)-Transformer** is powerful but expensive - only if you need it
- Most tasks don't need full SE(3) - **SO(3) (rotations) is often sufficient**

## References

1. Vector Neurons (2021): https://arxiv.org/abs/2104.12229
2. EGNN (2021): https://arxiv.org/abs/2102.09844
3. EPN (2021): https://arxiv.org/abs/2103.14635
4. E2PN (2022): https://arxiv.org/abs/2201.13164
5. SE(3)-Transformer (2020): https://arxiv.org/abs/2006.10503
6. Quaternion Capsules (2019): https://arxiv.org/abs/1912.12098


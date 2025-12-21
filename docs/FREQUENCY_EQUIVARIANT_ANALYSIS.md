# Frequency-Based Equivariant Networks Analysis

## What Are They?

Frequency-based equivariant networks use **frequency-domain representations** (like spherical harmonics, Fourier features) to achieve equivariance. Instead of operating directly on spatial coordinates, they work in frequency space.

## Key Approaches

### 1. **Spherical Harmonics (3D)**
- **Examples**: Tensor Field Networks (TFN), e3nn, SE3-Transformer
- **Mechanism**: Use spherical harmonics as basis functions
- **Equivariance**: SO(3) or SE(3)
- **Best for**: 3D molecular data, full 3D rotations
- **Complexity**: High (spherical harmonic computations)

### 2. **Fourier Features (2D/3D)**
- **Examples**: FER (Frequency-based Equivariant Representation), Fourier Feature Networks
- **Mechanism**: Map to frequency space, apply operations, map back
- **Equivariance**: Can achieve SO(2), SO(3) depending on implementation
- **Best for**: Capturing multi-scale patterns
- **Complexity**: Medium (FFT operations)

### 3. **Steerable Filters**
- **Examples**: Steerable CNNs, Harmonic Networks
- **Mechanism**: Filters that can be rotated in frequency domain
- **Equivariance**: SO(2) or SO(3)
- **Best for**: Regular grids, images
- **Complexity**: Medium-High

## Do We Have Them?

**Current Status**: ❌ **No**, we don't have frequency-based equivariant encoders yet.

**What we have**:
- ✅ VN-DGCNN (spatial domain, vector neurons)
- ✅ EGNN (spatial domain, message passing)

**What we're missing**:
- ❌ Spherical harmonics-based (TFN, e3nn)
- ❌ Fourier feature-based (FER)
- ❌ Steerable filters

## Are They Appropriate for Multi-MNIST?

### Short Answer: **Probably NOT the best choice**

### Detailed Analysis:

#### ✅ **Pros**:
1. **Strong theoretical guarantees** - Mathematically rigorous equivariance
2. **Multi-scale features** - Frequency domain naturally captures different scales
3. **Expressive** - Can represent complex patterns

#### ❌ **Cons for Your Use Case**:

1. **2D Data, Not 3D**
   - Multi-MNIST is **2D point clouds** (x, y coordinates)
   - Most frequency-based equivariant methods are designed for **3D** (spherical harmonics)
   - For 2D, you'd need **circular harmonics** or **2D Fourier** (less common, less mature)

2. **Computational Overhead**
   - Spherical harmonics: Expensive to compute
   - FFT operations: Add overhead
   - Your goal is **minimal network** - frequency methods add complexity

3. **Overkill for the Task**
   - Multi-MNIST has **simple geometric structure** (digits)
   - VN-DGCNN or EGNN already provide rotation equivariance
   - Frequency methods are better for **complex 3D shapes** (molecules, proteins)

4. **Implementation Complexity**
   - Spherical harmonics: Very complex to implement correctly
   - Requires specialized libraries (e3nn, etc.)
   - Harder to debug and understand

5. **Point Cloud Irregularity**
   - Frequency methods work best on **regular grids** or **dense sampling**
   - Point clouds are **irregular** - frequency methods less natural

## When ARE They Appropriate?

### ✅ **Good Use Cases**:
1. **3D molecular data** - Atoms, proteins (TFN, e3nn)
2. **Full 3D rotations** - Need SO(3) equivariance in 3D space
3. **Multi-scale analysis** - Need to capture patterns at different frequencies
4. **Dense 3D shapes** - Meshes, voxels with regular structure
5. **Theoretical guarantees needed** - Research requiring provable equivariance

### ❌ **Poor Use Cases** (like yours):
1. **2D point clouds** - Better methods available (VN, EGNN)
2. **Simple shapes** - Spatial methods sufficient
3. **Minimal networks** - Frequency adds complexity
4. **Sparse/irregular data** - Frequency domain less natural

## Recommendations for Multi-MNIST

### **Best Choices** (in order):

1. **EGNN** ⭐⭐⭐
   - E(n) equivariant (handles 2D naturally)
   - Simple, fast, effective
   - Lower complexity than frequency methods
   - **Recommended**: Start here

2. **VN-DGCNN** ⭐⭐
   - SO(3) equivariant (works for 2D as subset)
   - Proven effective
   - More memory but still reasonable

3. **Standard (non-equivariant) encoders** ⭐
   - With rotation augmentation
   - Simpler, might be sufficient
   - Baseline comparison

### **Skip** (for your use case):

4. ❌ **Frequency-based methods**
   - Overkill for 2D digits
   - High complexity
   - Better suited for 3D molecular/protein data

## If You Still Want Frequency Methods

If you decide frequency methods are needed, here's what to implement:

### Option 1: **2D Fourier Features** (Simplest)
```python
# Add Fourier features as additional input features
def fourier_features_2d(x, num_frequencies=10):
    # x: (B, N, 2)
    freqs = 2**jnp.arange(num_frequencies) * jnp.pi
    features = []
    for freq in freqs:
        features.append(jnp.sin(freq * x))
        features.append(jnp.cos(freq * x))
    return jnp.concatenate(features, axis=-1)  # (B, N, 2*2*num_frequencies)
```
- **Pros**: Easy to implement, adds multi-scale info
- **Cons**: Not inherently equivariant (just features)

### Option 2: **Circular Harmonics** (2D Equivariant)
- Analogous to spherical harmonics but for 2D
- Achieves SO(2) equivariance
- **Complexity**: High
- **Benefit**: Rigorous 2D rotation equivariance

### Option 3: **e3nn** (Full Framework)
- Use e3nn library (PyTorch-based, would need JAX port)
- Spherical harmonics framework
- **Complexity**: Very High
- **Benefit**: Mature, well-tested
- **Problem**: Designed for 3D, overkill for 2D

## Comparison Table

| Method | Equivariance | Complexity | Speed | Best For | Your Use Case |
|--------|--------------|------------|-------|----------|---------------|
| **EGNN** | E(n) | Medium | Fast | 2D/3D point clouds | ⭐⭐⭐ Excellent |
| **VN-DGCNN** | SO(3) | Medium | Fast | 3D point clouds | ⭐⭐ Good |
| **Fourier Features** | None (features only) | Low | Fast | Multi-scale | ⭐ Okay (as addition) |
| **Circular Harmonics** | SO(2) | High | Medium | 2D rotation | ⭐ Possible but complex |
| **Spherical Harmonics** | SO(3) | Very High | Slow | 3D molecules | ❌ Overkill |
| **e3nn/TFN** | SE(3) | Very High | Slow | 3D molecules | ❌ Wrong domain |

## Conclusion

**For Multi-MNIST (2D point clouds of digits):**

1. ✅ **Use EGNN or VN-DGCNN** - They provide equivariance without frequency-domain complexity
2. ✅ **Optionally add Fourier features** - As input features (easy, low overhead)
3. ❌ **Skip spherical harmonics** - Designed for 3D, overkill for your task
4. ❌ **Skip circular harmonics** - High complexity, marginal benefit over EGNN

**Bottom line**: Frequency-based equivariant methods are powerful but **not appropriate** for your use case. EGNN gives you E(n) equivariance with much lower complexity, which is perfect for 2D point clouds.

## References

1. **TFN**: Thomas et al., "Tensor Field Networks" (2018)
2. **e3nn**: Geiger & Smidt, "e3nn: Euclidean Neural Networks" (2022)
3. **FER**: "Frequency-based Equivariant Representation" (2024)
4. **EGNN**: Satorras et al., "E(n) Equivariant Graph Neural Networks" (2021) ← **Recommended for you**


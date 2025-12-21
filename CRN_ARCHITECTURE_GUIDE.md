# CRN Architecture Guide

## Overview

Conditional ResNets (CRNs) come in two flavors based on context structure:

1. **Global CRNs** - Single global vector per batch `(B, Dc)`
2. **Structured CRNs** - Set of K abstract latent representations `(B, K, Dc)`

## Key Insight: What is K?

The K dimension in Structured CRNs `(B, K, Dc)` represents **abstract latent representations**, NOT per-point features:

- **Slots** from Slot Attention (K=8-16) - Object representations
- **Latent queries** from Perceiver (K=32-64) - Learned abstract features  
- **Mixture components** from GMM (K=5-20) - Distributional modes
- **Abstract "super points"** or **collocation points** - Learned basis functions

**Important:** K is typically much smaller than N (number of input points):
- N = 100-1000 (input points)
- K = 8-64 (abstract representations)

## Architecture Types

### 1. Global CRNs (`global_crn.py`)

**Context:** `(B, Dc)` - Single global vector

**Use Case:** Encoders that produce global latent codes
- PointNet → `(B, 128)`
- MaxPoolingEncoder → `(B, 128)`
- MeanPoolingEncoder → `(B, 128)`

**Classes:**
- `GlobalAdaLNMLPCRN` - MLP with global AdaLN
- `GlobalDiTCRN` - Transformer with global AdaLN
- `GlobalCrossAttentionCRN` - Projects global to M latents, cross-attend
- `GlobalSimpleConcatCRN` - Simple concatenation

**Conditioning Flow:**
```
Time (B, 1) + Global Context (B, Dc)
  ↓
MLP → Global Conditioning (B, cond_dim)
  ↓
Broadcast to all N points
  ↓
Apply AdaLN to each point
```

### 2. Structured CRNs (`structured_crn.py`)

**Context:** `(B, K, Dc)` - K abstract latent representations

**Use Case:** Encoders that produce structured representations
- Slot Attention → `(B, 8, 64)` - 8 object slots
- Perceiver → `(B, 32, 64)` - 32 latent queries
- GMM → `(B, K, D)` - K mixture components

**Three Approaches:**

#### A. Pool-Based Structured CRNs

Pool K latents to global, then apply AdaLN.

**Classes:**
- `StructuredAdaLNMLPCRN`
- `StructuredDiTCRN`

**Conditioning Flow:**
```
Time (B, 1) + Structured Context (B, K, Dc)
  ↓
Process each of K latents with time
  ↓
Pool K latents → Global Conditioning (B, cond_dim)
  [max pooling aggregates information from all K latents]
  ↓
Broadcast to all N points
  ↓
Apply AdaLN to each point
```

**Rationale:** AdaLN requires a single conditioning vector per point. By pooling K latents, we aggregate their information into a global conditioning signal.

#### B. Attention-Based Structured CRNs

Keep K latents, use cross-attention for each point to attend to all K.

**Classes:**
- `StructuredCrossAttentionCRN`
- `StructuredSimpleConcatCRN` (attentive pooling)

**Conditioning Flow:**
```
Time (B, 1) + Structured Context (B, K, Dc)
  ↓
Process K latents with time → Context Features (B, K, latent_dim)
  ↓
For each of N input points:
    Cross-attend to all K context features
  ↓
Each point gets weighted combination of K latents
```

**Rationale:** Cross-attention allows each point to selectively attend to relevant latents (e.g., attend to the slot representing the object that point belongs to).

#### C. Direct Structured CRNs

Sample exactly K points (one per latent), with 1-to-1 conditioning.

**Classes:**
- `DirectStructuredAdaLNMLPCRN`
- `DirectStructuredDiTCRN`

**Conditioning Flow:**
```
Time (B, 1) + Structured Context (B, K, Dc)
  ↓
Process K latents with time → Per-Latent Conditioning (B, K, cond_dim)
  ↓
For each of K input points:
    Apply AdaLN using its corresponding latent (1-to-1)
  ↓
(Optional) Self-attention among K points
```

**Rationale:** When sampling one point per latent (e.g., one point per object slot), we have a natural 1-to-1 correspondence. Each point is directly conditioned by its corresponding latent without pooling or attention overhead.

## When to Use Which

### Use Global CRNs When:
- ✅ Context is a single global latent code
- ✅ All points should receive identical conditioning
- ✅ Simpler, more efficient models desired
- ✅ Encoder outputs `(B, D)`

**Examples:**
- PointNet encoder
- Any pooled encoder (max/mean/attention pooling)

### Use Structured CRNs (Pool-Based) When:
- ✅ Context is K abstract representations
- ✅ Want to aggregate information from all K latents
- ✅ AdaLN-style conditioning desired
- ✅ Encoder outputs `(B, K, D)` where K is small (8-64)

**Examples:**
- Slot Attention encoder (K=8 slots)
- Small Perceiver (K=32 latents)
- GMM featurizer (K=10 components)

### Use Structured CRNs (Attention-Based) When:
- ✅ Context is K abstract representations
- ✅ Want points to selectively attend to relevant latents
- ✅ More expressive, point-specific conditioning desired
- ✅ Encoder outputs `(B, K, D)` where K is small-medium (8-128)
- ✅ N >> K (many points, few latents)

**Examples:**
- Slot Attention (points attend to their object's slot)
- Perceiver (points attend to relevant latent queries)
- GMM (points attend to nearby mixture components)

### Use Structured CRNs (Direct) When:
- ✅ Context is K abstract representations
- ✅ Sampling exactly K points (one per latent)
- ✅ Natural 1-to-1 correspondence between points and latents
- ✅ N = K (same number of points and latents)
- ✅ Want most efficient conditioning (no pooling or cross-attention)

**Examples:**
- Sample one point per slot (K=8 slots → 8 points)
- Sample one point per mixture component (K=10 components → 10 points)
- Generate K representative points from K latent queries

## Comparison Matrix

| Feature | Global CRN | Structured (Pool) | Structured (Attention) | Structured (Direct) |
|---------|------------|-------------------|------------------------|---------------------|
| Context Shape | `(B, Dc)` | `(B, K, Dc)` | `(B, K, Dc)` | `(B, K, Dc)` |
| Input Shape | `(B, N, D)` | `(B, N, D)` | `(B, N, D)` | `(B, K, D)` |
| Relationship | - | K << N | K << N | **N = K** |
| Conditioning | Global | Global (pooled) | Point-specific | 1-to-1 per-latent |
| Complexity | O(1) | O(K) → O(1) | O(N × K) | O(K) |
| Expressiveness | Low | Medium | High | Medium-High |
| Efficiency | Highest | High | Medium | High |
| Use Case | Global latents | Aggregated structured | Selective attention | One point per latent |

## Implementation Details

### Global CRN Example

```python
from src.models.global_crn import GlobalAdaLNMLPCRN

# PointNet encoder produces global latent
z_mu, z_logvar = pointnet_encoder(x)  # (B, 128)

# Use global CRN
crn = GlobalAdaLNMLPCRN()
dx = crn(x, c=z_mu, t=t)  # c is (B, 128)
```

### Structured CRN (Pool-Based) Example

```python
from src.models.structured_crn import StructuredAdaLNMLPCRN

# Slot Attention produces K=8 slots
slots = slot_attention_encoder(x)  # (B, 8, 64)

# Use structured CRN with pooling
crn = StructuredAdaLNMLPCRN()
dx = crn(x, c=slots, t=t)  # c is (B, 8, 64)
# Internally: pools 8 slots → global cond → AdaLN
```

### Structured CRN (Attention-Based) Example

```python
from src.models.structured_crn import StructuredCrossAttentionCRN

# Perceiver produces K=32 latent queries
latents = perceiver_encoder(x)  # (B, 32, 64)

# Use structured CRN with cross-attention
crn = StructuredCrossAttentionCRN()
dx = crn(x, c=latents, t=t)  # c is (B, 32, 64)
# Each of N points cross-attends to all 32 latents
```

### Structured CRN (Direct) Example

```python
from src.models.structured_crn import DirectStructuredAdaLNMLPCRN

# Slot Attention produces K=8 slots
slots = slot_attention_encoder(x)  # (B, 8, 64)

# Sample exactly K=8 points (one per slot)
x_t = sample_initial_points(batch_size=B, num_points=8)  # (B, 8, 2)

# Use direct structured CRN (N = K = 8)
crn = DirectStructuredAdaLNMLPCRN()
dx = crn(x_t, c=slots, t=t)  # x_t is (B, 8, 2), c is (B, 8, 64)
# Each of 8 points gets conditioned by its corresponding slot (1-to-1)
```

## Design Rationale

### Why Pool in Some Structured CRNs?

**Problem:** AdaLN requires a single conditioning vector to compute scale/shift parameters.

**Solution:** Pool K latents using max/mean/attention pooling.

**Benefit:** Aggregates information from all K latents while maintaining AdaLN's efficiency.

### Why Cross-Attention in Others?

**Problem:** Pooling loses the structured information - which latent is relevant for which point?

**Solution:** Use cross-attention so each point can selectively attend to relevant latents.

**Benefit:** More expressive - points can focus on their relevant latent (e.g., object slot).

### Trade-offs

**Pool-Based vs Attention-Based (when K << N):**
- **Pool-Based:** More efficient (O(K) → O(1)), less expressive, aggregates all K latents
- **Attention-Based:** More expressive (O(N × K)), higher compute, point-specific conditioning

**Direct (when N = K):**
- **Most efficient for N=K:** O(K) with 1-to-1 mapping, no pooling or cross-attention overhead
- **Natural for per-latent sampling:** Each latent generates its own point
- **Expressiveness:** Medium-High (per-latent conditioning + optional self-attention)

Choose based on your needs:
- **N >> K** (many points, few latents):
  - Small K (≤16) + need expressiveness → Attention-Based
  - Larger K or efficiency priority → Pool-Based
- **N = K** (one point per latent):
  - Direct (most natural and efficient)

## Masking Support

All Structured CRNs support masking:

```python
# Mask for K latents (e.g., some slots are empty)
mask = jnp.array([[1, 1, 1, 1, 0, 0, 0, 0]])  # (B, K) - only 4 slots active

dx = crn(x, c=slots, t=t, mask=mask)
```

Masking is applied:
1. To conditioning (masked latents don't contribute)
2. In attention (masked latents are ignored)
3. To output (if needed)

## Future Extensions

Potential enhancements:
- [ ] Learnable pooling (instead of max pooling)
- [ ] Hierarchical structured CRNs (multi-scale K)
- [ ] Adaptive K (learn number of latents)
- [ ] Hybrid CRNs (combine global + structured)

## Conclusion

The key insight is that **K represents abstract latent representations**, not per-point features:

- **Global CRNs:** Single global vector `(B, Dc)`
- **Structured CRNs:** K abstract representations `(B, K, Dc)`
  - **Pool-Based (K << N):** Aggregate K → global → AdaLN
  - **Attention-Based (K << N):** Cross-attend to K for point-specific conditioning
  - **Direct (N = K):** 1-to-1 correspondence between points and latents

Choose based on:
1. **Encoder output shape:** `(B, Dc)` → Global, `(B, K, Dc)` → Structured
2. **Number of points to sample:** N >> K → Pool/Attention, N = K → Direct
3. **Expressiveness vs efficiency:** Pool (efficient) vs Attention (expressive) vs Direct (efficient + expressive for N=K)


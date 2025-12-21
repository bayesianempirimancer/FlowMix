# Conditional ResNets (CRN) for Flow Models

## Quick Start

```python
from src.models.global_crn import GlobalAdaLNMLPCRN
from src.models.local_crn import LocalAdaLNMLPCRN
from src.models.structured_crn import StructuredAdaLNMLPCRN

# Choose based on your encoder output and sampling strategy
```

## File Structure

```
src/models/
├── global_crn.py       # (B, Dc) context → (B, N, D) points
├── local_crn.py        # (B, K, Dc) context → (B, K, D) points (N=K)
└── structured_crn.py   # (B, K, Dc) context → (B, N, D) points (N>>K)
```

## The Three CRN Types

### 1. Global CRN - Single Global Vector

**File:** `global_crn.py`  
**Context:** `(B, Dc)` - One vector for entire batch  
**Input:** `(B, N, D)` - Any number of points

```python
from src.models.global_crn import GlobalAdaLNMLPCRN

z = pointnet(x)  # (B, 128) - global latent
x_t = sample(B, 100)  # (B, 100, 2)
crn = GlobalAdaLNMLPCRN()
dx = crn(x_t, z, t)  # All 100 points get same global conditioning
```

---

### 2. Local CRN - One Point Per Latent

**File:** `local_crn.py`  
**Context:** `(B, K, Dc)` - K abstract latents  
**Input:** `(B, K, D)` - **Exactly K points (N = K)**

**Key:** 1-to-1 correspondence between points and latents

```python
from src.models.local_crn import LocalAdaLNMLPCRN

slots = slot_attention(x)  # (B, 8, 64) - 8 slots
x_t = sample(B, 8)  # (B, 8, 2) - 8 points (one per slot)
crn = LocalAdaLNMLPCRN()
dx = crn(x_t, slots, t)  # Point i ← Slot i (1-to-1)
```

---

### 3. Structured CRN - Many Points, Few Latents

**File:** `structured_crn.py`  
**Context:** `(B, K, Dc)` - K abstract latents  
**Input:** `(B, N, D)` - **Many points (N >> K)**

**Two variants:**
- **Pool-Based:** Pool K latents → global → condition all N points (efficient)
- **Attention-Based:** Each point cross-attends to K latents (expressive)

```python
from src.models.structured_crn import StructuredAdaLNMLPCRN

slots = slot_attention(x)  # (B, 8, 64) - 8 slots
x_t = sample(B, 100)  # (B, 100, 2) - 100 points
crn = StructuredAdaLNMLPCRN()
dx = crn(x_t, slots, t)  # Pool 8 slots → condition 100 points
```

---

## Decision Tree

```
What's your context shape?
│
├─ (B, Dc) - Single global vector
│  └─> global_crn.py
│
└─ (B, K, Dc) - K abstract latents
   │
   └─ How many points?
      │
      ├─ N = K (one per latent)
      │  └─> local_crn.py
      │
      └─ N >> K (many points)
         └─> structured_crn.py
```

---

## Key Insight

**K is NOT the number of input points!**

K represents **abstract latent representations**:
- 🎰 Slots (Slot Attention) - object representations
- 🔍 Latent queries (Perceiver) - learned features
- 📊 Mixture components (GMM) - distributional modes

**Typical sizes:**
- N (points) = 100-1000
- K (latents) = 8-64

**Exception:** Local CRNs where N = K

---

## Comparison Table

| File | Context | Input | Relationship | Use Case |
|------|---------|-------|--------------|----------|
| `global_crn.py` | `(B, Dc)` | `(B, N, D)` | - | PointNet, pooled encoders |
| `local_crn.py` | `(B, K, Dc)` | `(B, K, D)` | **N = K** | One point per latent |
| `structured_crn.py` | `(B, K, Dc)` | `(B, N, D)` | **N >> K** | Many points, few latents |

---

## Available Classes

### Global CRN (`global_crn.py`)
- `GlobalAdaLNMLPCRN` - MLP with global AdaLN
- `GlobalDiTCRN` - Transformer with global AdaLN
- `GlobalCrossAttentionCRN` - Projects global to M latents
- `GlobalSimpleConcatCRN` - Simple concatenation

### Local CRN (`local_crn.py`)
- `LocalAdaLNMLPCRN` - MLP with per-latent AdaLN (1-to-1)
- `LocalDiTCRN` - Transformer with per-latent AdaLN + self-attention

### Structured CRN (`structured_crn.py`)
- `StructuredAdaLNMLPCRN` - Pool-based (efficient)
- `StructuredCrossAttentionCRN` - Attention-based (expressive)

---

## Documentation

- **`CRN_FINAL_STRUCTURE.md`** - Complete reference with examples
- **`CRN_PATTERNS_SUMMARY.md`** - Quick reference guide
- **`CRN_ARCHITECTURE_GUIDE.md`** - Detailed architecture guide

---

## Status

✅ **Complete and Verified**
- All modules tested and working
- `crn.py` deleted (no longer needed)
- Clean file structure
- Comprehensive documentation

**Date:** December 16, 2025






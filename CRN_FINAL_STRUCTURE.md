# CRN Final Structure

## File Organization

```
src/models/
├── global_crn.py       # Global CRNs: (B, Dc) context
├── local_crn.py        # Local CRNs: (B, K, Dc) context, N=K (one-to-one)
└── structured_crn.py   # Structured CRNs: (B, K, Dc) context, N>>K
```

**Note:** `crn.py` has been **DELETED** - all functionality migrated to the three specialized files above.

---

## The Three CRN Types

### 1. Global CRN (`global_crn.py`)

**Context:** `(B, Dc)` - Single global vector  
**Input:** `(B, N, D)` - Any number of points  
**Use Case:** Encoders that produce a single global latent

**Classes:**
- `GlobalAdaLNMLPCRN` - MLP with global AdaLN
- `GlobalDiTCRN` - Transformer with global AdaLN  
- `GlobalCrossAttentionCRN` - Projects global to M latents, cross-attend
- `GlobalSimpleConcatCRN` - Simple concatenation

**Example:**
```python
from src.models.global_crn import GlobalAdaLNMLPCRN

# PointNet produces global latent
z = pointnet_encoder(x)  # (B, 128)

# Sample 100 points
x_t = jnp.ones((B, 100, 2))

# Use Global CRN
crn = GlobalAdaLNMLPCRN()
dx = crn(x_t, c=z, t=t)  # (B, 100, 2)
```

---

### 2. Local CRN (`local_crn.py`)

**Context:** `(B, K, Dc)` - K abstract latents  
**Input:** `(B, K, D)` - **Exactly K points (N = K)**  
**Use Case:** Sample one point per latent (1-to-1 correspondence)

**Classes:**
- `LocalAdaLNMLPCRN` - MLP with per-latent AdaLN (1-to-1)
- `LocalDiTCRN` - Transformer with per-latent AdaLN + self-attention

**Key Property:** **N = K** (one point per latent)
- Point 0 ← Latent 0
- Point 1 ← Latent 1
- ...
- Point K-1 ← Latent K-1

**Example:**
```python
from src.models.local_crn import LocalAdaLNMLPCRN

# Slot Attention produces K=8 slots
slots = slot_attention_encoder(x)  # (B, 8, 64)

# Sample K=8 points (one per slot, N=K)
x_t = jnp.ones((B, 8, 2))

# Use Local CRN (1-to-1 conditioning)
crn = LocalAdaLNMLPCRN()
dx = crn(x_t, c=slots, t=t)  # (B, 8, 2)
# Each point conditioned by its corresponding slot
```

---

### 3. Structured CRN (`structured_crn.py`)

**Context:** `(B, K, Dc)` - K abstract latents  
**Input:** `(B, N, D)` - **Many points (N >> K)**  
**Use Case:** Sample many points conditioned on K latents

**Two Patterns:**

#### A. Pool-Based (Efficient)
Pool K latents → global conditioning → all N points

**Classes:**
- `StructuredAdaLNMLPCRN` - Pool + MLP with AdaLN

**Example:**
```python
from src.models.structured_crn import StructuredAdaLNMLPCRN

# Slot Attention produces K=8 slots
slots = slot_attention_encoder(x)  # (B, 8, 64)

# Sample N=100 points (N >> K)
x_t = jnp.ones((B, 100, 2))

# Use Structured CRN (Pool-Based)
crn = StructuredAdaLNMLPCRN()
dx = crn(x_t, c=slots, t=t)  # (B, 100, 2)
# Pools 8 slots → global → conditions all 100 points
```

#### B. Attention-Based (Expressive)
Each of N points cross-attends to K latents

**Classes:**
- `StructuredCrossAttentionCRN` - Cross-attention to K latents

**Example:**
```python
from src.models.structured_crn import StructuredCrossAttentionCRN

# Slot Attention produces K=8 slots
slots = slot_attention_encoder(x)  # (B, 8, 64)

# Sample N=100 points (N >> K)
x_t = jnp.ones((B, 100, 2))

# Use Structured CRN (Attention-Based)
crn = StructuredCrossAttentionCRN()
dx = crn(x_t, c=slots, t=t)  # (B, 100, 2)
# Each of 100 points cross-attends to 8 slots
```

---

## Decision Tree

```
What's your context shape?
│
├─ (B, Dc) - Single global vector
│  └─> global_crn.py
│     └─> GlobalAdaLNMLPCRN, GlobalDiTCRN, etc.
│
└─ (B, K, Dc) - K abstract latents
   │
   └─ How many points to sample?
      │
      ├─ N = K (one point per latent)
      │  └─> local_crn.py
      │     └─> LocalAdaLNMLPCRN, LocalDiTCRN
      │        (1-to-1 correspondence)
      │
      └─ N >> K (many points, few latents)
         └─> structured_crn.py
            ├─> StructuredAdaLNMLPCRN (Pool-Based, efficient)
            └─> StructuredCrossAttentionCRN (Attention-Based, expressive)
```

---

## Comparison Table

| File | Context | Input | Relationship | Use Case |
|------|---------|-------|--------------|----------|
| `global_crn.py` | `(B, Dc)` | `(B, N, D)` | - | Single global latent |
| `local_crn.py` | `(B, K, Dc)` | `(B, K, D)` | **N = K** | One point per latent |
| `structured_crn.py` | `(B, K, Dc)` | `(B, N, D)` | **N >> K** | Many points, few latents |

---

## Key Insight

**K is NOT the number of input points!**

K represents **abstract latent representations**:
- 🎰 **Slots** from Slot Attention (object representations)
- 🔍 **Latent queries** from Perceiver (learned features)
- 📊 **Mixture components** from GMM (distributional modes)
- ⭐ **Abstract "super points"** or collocation points

**Typical sizes:**
- N (input points) = 100-1000
- K (abstract latents) = 8-64

**Exception:** Local CRNs where N = K (one point per latent)

---

## Example Scenarios

### Scenario 1: PointNet → 100 points
```python
from src.models.global_crn import GlobalAdaLNMLPCRN

z = pointnet(x)  # (B, 128) - global
x_t = sample(B, 100)  # (B, 100, 2)
crn = GlobalAdaLNMLPCRN()
dx = crn(x_t, z, t)
```

### Scenario 2: Slot Attention (K=8) → 8 points (one per slot)
```python
from src.models.local_crn import LocalAdaLNMLPCRN

slots = slot_attention(x)  # (B, 8, 64)
x_t = sample(B, 8)  # (B, 8, 2) - N=K!
crn = LocalAdaLNMLPCRN()
dx = crn(x_t, slots, t)  # 1-to-1 conditioning
```

### Scenario 3: Slot Attention (K=8) → 100 points
```python
from src.models.structured_crn import StructuredAdaLNMLPCRN

slots = slot_attention(x)  # (B, 8, 64)
x_t = sample(B, 100)  # (B, 100, 2) - N>>K
crn = StructuredAdaLNMLPCRN()
dx = crn(x_t, slots, t)  # Pool 8 slots → condition 100 points
```

### Scenario 4: GMM (K=10) → 10 points (one per component)
```python
from src.models.local_crn import LocalAdaLNMLPCRN

components = gmm(x)  # (B, 10, 32)
x_t = sample(B, 10)  # (B, 10, 2) - N=K!
crn = LocalAdaLNMLPCRN()
dx = crn(x_t, components, t)  # 1-to-1 conditioning
```

---

## Migration from Old Code

### Old (DEPRECATED, crn.py deleted)
```python
from src.models.crn import AdaLNMLPCRN  # ❌ NO LONGER EXISTS
```

### New
```python
# Choose based on context shape and sampling strategy:

# If c is (B, Dc) - global
from src.models.global_crn import GlobalAdaLNMLPCRN

# If c is (B, K, Dc) and sampling K points (N=K)
from src.models.local_crn import LocalAdaLNMLPCRN

# If c is (B, K, Dc) and sampling many points (N>>K)
from src.models.structured_crn import StructuredAdaLNMLPCRN
```

---

## Verification

All modules tested and working:

```bash
✓ Global CRN: (2, 64) -> (2, 100, 2) -> (2, 100, 2)
✓ Local CRN: (2, 8, 64) -> (2, 8, 2) -> (2, 8, 2)
✓ Structured CRN (Pool): (2, 8, 64) -> (2, 100, 2) -> (2, 100, 2)
✓ Structured CRN (Attention): (2, 8, 64) -> (2, 100, 2) -> (2, 100, 2)
✓ mnist_flow_2d.py imports successfully
```

---

## Summary

✅ **Clean file structure:**
- `global_crn.py` - Global context `(B, Dc)`
- `local_crn.py` - Local context `(B, K, Dc)` with **N=K** (1-to-1)
- `structured_crn.py` - Structured context `(B, K, Dc)` with **N>>K**

✅ **Deprecated `crn.py` deleted** - no longer needed

✅ **Clear naming:**
- "Local" = one point per latent (N=K)
- "Structured" = many points, few latents (N>>K)
- "Global" = single global vector

✅ **All tests passing**

**Date:** December 16, 2025  
**Status:** ✅ Complete and Verified






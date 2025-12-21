# CRN Patterns: Quick Reference

## The Three Patterns

### Pattern 1: Global CRN
**Context:** `(B, Dc)` - Single global vector  
**Input:** `(B, N, D)` - Any number of points  
**Relationship:** One vector for all points

```
Encoder → (B, Dc) global latent
                ↓
         GlobalAdaLNMLPCRN
                ↓
         (B, N, D) output
```

**Use Case:** PointNet, pooled encoders

---

### Pattern 2a: Structured CRN (Pool-Based)
**Context:** `(B, K, Dc)` - K abstract latents  
**Input:** `(B, N, D)` - Many points (N >> K)  
**Relationship:** K latents pooled to global

```
Encoder → (B, K, Dc) K slots/latents
                ↓
         Pool K → (B, Dc)
                ↓
    StructuredAdaLNMLPCRN (Pool)
                ↓
         (B, N, D) output
```

**Use Case:** Slot Attention (K=8), Perceiver (K=32)  
**Efficiency:** O(K) → O(1)

---

### Pattern 2b: Structured CRN (Attention-Based)
**Context:** `(B, K, Dc)` - K abstract latents  
**Input:** `(B, N, D)` - Many points (N >> K)  
**Relationship:** Each point attends to K latents

```
Encoder → (B, K, Dc) K slots/latents
                ↓
  StructuredCrossAttentionCRN
    (each point cross-attends to K)
                ↓
         (B, N, D) output
```

**Use Case:** Slot Attention (K=8), Perceiver (K=32)  
**Efficiency:** O(N × K)  
**Expressiveness:** High (point-specific conditioning)

---

### Pattern 3: Local CRN
**Context:** `(B, K, Dc)` - K abstract latents  
**Input:** `(B, K, D)` - Exactly K points (N = K)  
**Relationship:** 1-to-1 correspondence

```
Encoder → (B, K, Dc) K slots/latents
                ↓
   LocalAdaLNMLPCRN
    (point i ← latent i, 1-to-1)
                ↓
         (B, K, D) output
```

**Use Case:** Sample one point per slot/component  
**Efficiency:** O(K) - most efficient for N=K  
**Expressiveness:** Medium-High (per-latent conditioning)

---

## Decision Tree

```
What shape is your context?
│
├─ (B, Dc) - Single global vector
│  └─> Use Global CRN
│
└─ (B, K, Dc) - K abstract latents
   │
   └─ How many points to sample?
      │
      ├─ N >> K (many points, few latents)
      │  │
      │  └─ Need point-specific conditioning?
      │     │
      │     ├─ Yes → Structured CRN (Attention-Based)
      │     │        O(N × K), high expressiveness
      │     │
      │     └─ No  → Structured CRN (Pool-Based)
      │              O(K) → O(1), efficient
      │
      └─ N = K (one point per latent)
         └─> Local CRN
             O(K), most efficient for N=K
```

---

## Examples

### Example 1: PointNet Encoder
```python
# Encoder produces global latent
z = pointnet_encoder(x)  # (B, 128)

# Sample 100 points
x_t = jnp.ones((B, 100, 2))

# Use Global CRN
crn = GlobalAdaLNMLPCRN()
dx = crn(x_t, c=z, t=t)  # (B, 100, 2)
```

### Example 2: Slot Attention → Many Points
```python
# Encoder produces K=8 slots
slots = slot_attention_encoder(x)  # (B, 8, 64)

# Sample 100 points (N >> K)
x_t = jnp.ones((B, 100, 2))

# Option A: Pool-based (efficient)
crn = StructuredAdaLNMLPCRN()
dx = crn(x_t, c=slots, t=t)  # (B, 100, 2)

# Option B: Attention-based (expressive)
crn = StructuredCrossAttentionCRN()
dx = crn(x_t, c=slots, t=t)  # (B, 100, 2)
```

### Example 3: Slot Attention → One Point Per Slot
```python
# Encoder produces K=8 slots
slots = slot_attention_encoder(x)  # (B, 8, 64)

# Sample K=8 points (one per slot, N = K)
x_t = jnp.ones((B, 8, 2))

# Use Local CRN (1-to-1)
crn = LocalAdaLNMLPCRN()
dx = crn(x_t, c=slots, t=t)  # (B, 8, 2)
# Point 0 ← Slot 0, Point 1 ← Slot 1, etc.
```

---

## Key Insight

**K is NOT the number of input points!**

K represents:
- 🎰 **Slots** from Slot Attention (object representations)
- 🔍 **Latent queries** from Perceiver (learned features)
- 📊 **Mixture components** from GMM (modes)
- ⭐ **Abstract "super points"** or collocation points

K is typically much smaller than N:
- N = 100-1000 (input points)
- K = 8-64 (abstract representations)

**Exception:** Direct CRNs where N = K (one point per latent)

---

## Performance Comparison

| Pattern | Context | Input | Complexity | Best For |
|---------|---------|-------|------------|----------|
| Global | `(B, Dc)` | `(B, N, D)` | O(1) | Simple global conditioning |
| Structured (Pool) | `(B, K, Dc)` | `(B, N, D)` | O(K) → O(1) | Efficient structured conditioning |
| Structured (Attention) | `(B, K, Dc)` | `(B, N, D)` | O(N × K) | Expressive point-specific conditioning |
| Structured (Direct) | `(B, K, Dc)` | `(B, K, D)` | O(K) | One point per latent (N=K) |

---

## Implementation Files

- **`src/models/global_crn.py`** - Global CRNs
  - `GlobalAdaLNMLPCRN`
  - `GlobalDiTCRN`
  - `GlobalCrossAttentionCRN`

- **`src/models/local_crn.py`** - Local CRNs (N=K)
  - `LocalAdaLNMLPCRN`
  - `LocalDiTCRN`

- **`src/models/structured_crn.py`** - Structured CRNs (N>>K)
  - Pool-Based:
    - `StructuredAdaLNMLPCRN`
  - Attention-Based:
    - `StructuredCrossAttentionCRN`

**Note:** `crn.py` has been **DELETED** (no longer needed)

---

## Migration Guide

### Old Code (DELETED - crn.py no longer exists)
```python
from src.models.crn import AdaLNMLPCRN  # ❌ NO LONGER EXISTS

crn = AdaLNMLPCRN()
dx = crn(x, c, t)
```

### New Code
```python
# If c is (B, Dc) - global
from src.models.global_crn import GlobalAdaLNMLPCRN
crn = GlobalAdaLNMLPCRN()
dx = crn(x, c, t)

# If c is (B, K, Dc) and x is (B, N, D) with N >> K
from src.models.structured_crn import StructuredAdaLNMLPCRN
crn = StructuredAdaLNMLPCRN()
dx = crn(x, c, t)

# If c is (B, K, Dc) and x is (B, K, D) with N = K
from src.models.local_crn import LocalAdaLNMLPCRN
crn = LocalAdaLNMLPCRN()
dx = crn(x, c, t)
```

---

For detailed information, see `CRN_ARCHITECTURE_GUIDE.md`.


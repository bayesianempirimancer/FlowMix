# CRN Refactoring: Complete Summary

## What Changed

The monolithic `crn.py` has been refactored into two specialized files based on context structure:

1. **`global_crn.py`** - For global context `(B, Dc)`
2. **`structured_crn.py`** - For structured context `(B, K, Dc)`

The old `crn.py` is marked as **DEPRECATED** but kept for backward compatibility.

---

## Key Insight: What is K?

**K is NOT the number of input points!**

K represents **abstract latent representations**:
- 🎰 **Slots** from Slot Attention (object representations)
- 🔍 **Latent queries** from Perceiver (learned features)
- 📊 **Mixture components** from GMM (distributional modes)
- ⭐ **Abstract "super points"** or **collocation points** (learned basis)

**Typical sizes:**
- N (input points) = 100-1000
- K (abstract latents) = 8-64

**Exception:** Direct CRNs where N = K (one point per latent)

---

## The Four CRN Patterns

### Pattern 1: Global CRN
- **Context:** `(B, Dc)` - Single global vector
- **Input:** `(B, N, D)` - Any number of points
- **File:** `global_crn.py`
- **Use:** PointNet, pooled encoders

### Pattern 2a: Structured CRN (Pool-Based)
- **Context:** `(B, K, Dc)` - K abstract latents
- **Input:** `(B, N, D)` - Many points (N >> K)
- **Mechanism:** Pool K latents → global conditioning
- **File:** `structured_crn.py`
- **Complexity:** O(K) → O(1)
- **Use:** Efficient structured conditioning

### Pattern 2b: Structured CRN (Attention-Based)
- **Context:** `(B, K, Dc)` - K abstract latents
- **Input:** `(B, N, D)` - Many points (N >> K)
- **Mechanism:** Each point cross-attends to K latents
- **File:** `structured_crn.py`
- **Complexity:** O(N × K)
- **Use:** Expressive point-specific conditioning

### Pattern 3: Structured CRN (Direct)
- **Context:** `(B, K, Dc)` - K abstract latents
- **Input:** `(B, K, D)` - Exactly K points (N = K)
- **Mechanism:** 1-to-1 correspondence between points and latents
- **File:** `structured_crn.py`
- **Complexity:** O(K)
- **Use:** Sample one point per latent (most efficient for N=K)

---

## File Structure

```
src/models/
├── global_crn.py           # NEW: Global CRNs (B, Dc)
│   ├── GlobalAdaLNMLPCRN
│   ├── GlobalDiTCRN
│   ├── GlobalCrossAttentionCRN
│   └── GlobalSimpleConcatCRN
│
├── structured_crn.py       # NEW: Structured CRNs (B, K, Dc)
│   ├── Pool-Based (K << N):
│   │   ├── StructuredAdaLNMLPCRN
│   │   └── StructuredDiTCRN
│   ├── Attention-Based (K << N):
│   │   └── StructuredCrossAttentionCRN
│   └── Direct (N = K):
│       ├── DirectStructuredAdaLNMLPCRN
│       └── DirectStructuredDiTCRN
│
└── crn.py                  # DEPRECATED (backward compatibility)
```

---

## Migration Examples

### Example 1: PointNet (Global)

**Before:**
```python
from src.models.crn import AdaLNMLPCRN

z = pointnet_encoder(x)  # (B, 128)
x_t = jnp.ones((B, 100, 2))

crn = AdaLNMLPCRN()
dx = crn(x_t, z, t)
```

**After:**
```python
from src.models.global_crn import GlobalAdaLNMLPCRN

z = pointnet_encoder(x)  # (B, 128)
x_t = jnp.ones((B, 100, 2))

crn = GlobalAdaLNMLPCRN()
dx = crn(x_t, z, t)  # Same API!
```

### Example 2: Slot Attention → Many Points (Structured Pool)

**Before:**
```python
from src.models.crn import AdaLNMLPCRN

slots = slot_attention_encoder(x)  # (B, 8, 64)
x_t = jnp.ones((B, 100, 2))

# Had to manually pool or use first slot
crn = AdaLNMLPCRN()
dx = crn(x_t, slots[:, 0, :], t)  # Only used first slot!
```

**After:**
```python
from src.models.structured_crn import StructuredAdaLNMLPCRN

slots = slot_attention_encoder(x)  # (B, 8, 64)
x_t = jnp.ones((B, 100, 2))

# Automatically pools all 8 slots
crn = StructuredAdaLNMLPCRN()
dx = crn(x_t, slots, t)  # Uses all 8 slots!
```

### Example 3: Slot Attention → One Point Per Slot (NEW!)

**Before:**
```python
# This pattern wasn't well-supported!
```

**After:**
```python
from src.models.structured_crn import DirectStructuredAdaLNMLPCRN

slots = slot_attention_encoder(x)  # (B, 8, 64)
x_t = jnp.ones((B, 8, 2))  # Sample 8 points (one per slot)

# Direct 1-to-1 conditioning
crn = DirectStructuredAdaLNMLPCRN()
dx = crn(x_t, slots, t)  # Point i ← Slot i
```

---

## Decision Tree

```
What shape is your encoder output?
│
├─ (B, Dc) - Single global vector
│  └─> Use global_crn.py
│     └─> GlobalAdaLNMLPCRN, GlobalDiTCRN, etc.
│
└─ (B, K, Dc) - K abstract latents
   │
   └─ How many points to sample?
      │
      ├─ N >> K (many points, few latents)
      │  │
      │  └─ Need point-specific conditioning?
      │     │
      │     ├─ Yes → structured_crn.py
      │     │        └─> StructuredCrossAttentionCRN
      │     │            (Attention-Based, O(N × K))
      │     │
      │     └─ No  → structured_crn.py
      │              └─> StructuredAdaLNMLPCRN
      │                  (Pool-Based, O(K) → O(1))
      │
      └─ N = K (one point per latent)
         └─> structured_crn.py
             └─> DirectStructuredAdaLNMLPCRN
                 (Direct, O(K), most efficient for N=K)
```

---

## Benefits of Refactoring

### 1. **Clarity**
- Explicit separation between global and structured context
- Clear naming: `Global*` vs `Structured*` vs `Direct*`
- Self-documenting code

### 2. **Type Safety**
- Global CRNs expect `(B, Dc)` - enforced by assertions
- Structured CRNs expect `(B, K, Dc)` - enforced by assertions
- Direct CRNs enforce `N = K` - enforced by assertions

### 3. **Functionality**
- **Pool-Based:** Properly aggregates all K latents (not just first!)
- **Attention-Based:** Point-specific conditioning via cross-attention
- **Direct:** NEW pattern for N=K case (most efficient)

### 4. **Masking Support**
- All Structured CRNs support optional masking `(B, K)`
- Properly handles variable number of valid latents
- Masked latents don't contribute to conditioning

### 5. **Modularity**
- Easy to add new CRN variants
- Clear file organization
- Backward compatibility maintained

---

## Testing

All patterns tested and verified:

```bash
cd /home/jebeck/GitHub/OC-Flow-Mix
python3 -c "
from src.models.global_crn import GlobalAdaLNMLPCRN
from src.models.structured_crn import (
    StructuredAdaLNMLPCRN,
    StructuredCrossAttentionCRN,
    DirectStructuredAdaLNMLPCRN,
    DirectStructuredDiTCRN
)
# ... all tests pass ✓
"
```

---

## Documentation

Three comprehensive guides created:

1. **`CRN_ARCHITECTURE_GUIDE.md`** - Detailed architecture guide
   - Design rationale
   - Implementation details
   - Trade-offs and comparisons

2. **`CRN_PATTERNS_SUMMARY.md`** - Quick reference
   - Visual diagrams
   - Decision tree
   - Code examples

3. **`CRN_REFACTORING_COMPLETE.md`** - This document
   - Migration guide
   - What changed and why
   - Benefits

---

## Backward Compatibility

The old `crn.py` is **DEPRECATED** but still available:

```python
# Still works (but deprecated)
from src.models.crn import AdaLNMLPCRN

# Prefer new imports
from src.models.global_crn import GlobalAdaLNMLPCRN
from src.models.structured_crn import StructuredAdaLNMLPCRN
```

**Recommendation:** Migrate to new imports for:
- Better clarity
- Type safety
- Access to new patterns (Direct CRNs)
- Proper handling of structured context

---

## Summary

✅ **Refactoring Complete!**

- Split `crn.py` → `global_crn.py` + `structured_crn.py`
- Clarified that K = abstract latents (NOT input points)
- Added three patterns: Pool-Based, Attention-Based, Direct
- Proper handling of structured context `(B, K, Dc)`
- Masking support for variable K
- Comprehensive documentation
- All tests passing
- Backward compatibility maintained

**Key Takeaway:** K represents abstract latent representations (slots, queries, components), not per-point features. Choose your CRN pattern based on context shape and sampling strategy (N >> K vs N = K).

---

## Next Steps

1. ✅ Update `mnist_flow_2d.py` to use `global_crn.py` (DONE)
2. ⏭️ Consider removing deprecated `crn.py` in future release
3. ⏭️ Experiment with Direct CRNs for per-slot sampling
4. ⏭️ Add learnable pooling (vs max pooling) in Pool-Based CRNs
5. ⏭️ Benchmark performance: Pool vs Attention vs Direct

---

**Date:** December 16, 2025  
**Status:** ✅ Complete






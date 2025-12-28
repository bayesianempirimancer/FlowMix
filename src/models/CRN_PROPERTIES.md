# CRN Properties Summary

Complete list of all Conditional ResNets (CRNs) in the models directory, their conditioning variable types, and whether they allow information mixing across points.

## Key Definitions

- **Conditioning Variable Type**:
  - **Global**: `c` has shape `(B, Dc)` - single vector per batch
  - **Local**: `c` has shape `(B, K, Dc)` where `N = K` (one-to-one correspondence)
  - **Structured**: `c` has shape `(B, K, Dc)` where `K << N` (many points, few latents)

- **Cross-Point Mixing**: Whether points with shape `(B, N, dim)` are coupled across the N dimension
  - **Yes**: Information flows between different points (e.g., via attention)
  - **No**: Points are processed independently (e.g., per-point MLPs)

---

## Global CRNs
**Location**: `src/models/global_crn.py`  
**Conditioning**: `c` has shape `(B, Dc)` - global context vector

### 1. GlobalAdaLNMLPCRN
- **Conditioning Type**: Global `(B, Dc)`
- **Cross-Point Mixing**: **No**
- **Architecture**: MLP with Adaptive Layer Normalization (AdaLN)
- **Processing**: Each point processed independently with shared global conditioning
- **Use Case**: Efficient, no inter-point communication needed

### 2. GlobalDiTCRN
- **Conditioning Type**: Global `(B, Dc)`
- **Cross-Point Mixing**: **Yes**
- **Architecture**: Transformer blocks with self-attention + AdaLN
- **Processing**: Self-attention allows all points to communicate
- **Use Case**: When inter-point relationships matter

### 3. GlobalCrossAttentionCRN
- **Conditioning Type**: Global `(B, Dc)`
- **Cross-Point Mixing**: **Yes**
- **Architecture**: Projects global context to M latent points, then cross-attention
- **Processing**: Each input point cross-attends to M latent points (indirect coupling)
- **Use Case**: Perceiver-style architecture with learned latent queries

### 4. GlobalSimpleConcatCRN
- **Conditioning Type**: Global `(B, Dc)`
- **Cross-Point Mixing**: **No**
- **Architecture**: Simple MLP with concatenated conditioning
- **Processing**: Each point processed independently
- **Use Case**: Simple baseline, no inter-point communication

### 5. GlobalGMFlowCRN
- **Conditioning Type**: Global `(B, Dc)`
- **Cross-Point Mixing**: **No**
- **Architecture**: Similar to GlobalAdaLNMLPCRN but outputs GMM parameters
- **Processing**: Each point processed independently, outputs mixture components
- **Output**: Dictionary with `logits`, `means`, `logvars` for Gaussian Mixture Model
- **Use Case**: Multi-modal velocity distributions

---

## Local CRNs
**Location**: `src/models/local_crn.py`  
**Conditioning**: `c` has shape `(B, K, Dc)` where `N = K` (one-to-one correspondence)

### 1. LocalAdaLNMLPCRN
- **Conditioning Type**: Local `(B, K, Dc)` where `N = K`
- **Cross-Point Mixing**: **No**
- **Architecture**: MLP with per-latent AdaLN (1-to-1 conditioning)
- **Processing**: Point i gets conditioned by latent i only
- **Use Case**: Sampling one point per slot/latent/component

### 2. LocalDiTCRN
- **Conditioning Type**: Local `(B, K, Dc)` where `N = K`
- **Cross-Point Mixing**: **Yes**
- **Architecture**: Transformer blocks with self-attention + per-latent AdaLN
- **Processing**: Self-attention among K points, each with its own conditioning
- **Use Case**: When K points need to interact but each has distinct conditioning

---

## Structured CRNs
**Location**: `src/models/structured_crn.py`  
**Conditioning**: `c` has shape `(B, K, Dc)` where `K << N` (many points, few latents)

### 1. StructuredAdaLNMLPCRN
- **Conditioning Type**: Structured `(B, K, Dc)` where `K << N`
- **Cross-Point Mixing**: **No**
- **Architecture**: Soft selection of K object embeddings per point → per-point AdaLN
- **Processing**: For each of N points, computes softmax over K object embeddings (based on point-object affinity), then uses weighted combination for conditioning
- **Pattern**: Soft selection (each point selects which object embedding to use)
- **Use Case**: When each point should be conditioned by one of K object embeddings based on similarity

### 2. StructuredCrossAttentionCRN
- **Conditioning Type**: Structured `(B, K, Dc)` where `K << N`
- **Cross-Point Mixing**: **Yes**
- **Architecture**: Cross-attention from N points to K latents
- **Processing**: Each of N points cross-attends to K structured latents
- **Pattern**: Attention-based (expressive, allows selective attention)
- **Use Case**: When each point should selectively attend to different latents

---

## Summary Table

| CRN | Conditioning Type | Cross-Point Mixing | Architecture | File |
|-----|------------------|-------------------|--------------|------|
| **Global CRNs** |
| GlobalAdaLNMLPCRN | Global `(B, Dc)` | No | MLP + AdaLN | `global_crn.py` |
| GlobalDiTCRN | Global `(B, Dc)` | **Yes** | Transformer + AdaLN | `global_crn.py` |
| GlobalCrossAttentionCRN | Global `(B, Dc)` | **Yes** | Cross-attention to latents | `global_crn.py` |
| GlobalSimpleConcatCRN | Global `(B, Dc)` | No | MLP + Concat | `global_crn.py` |
| GlobalGMFlowCRN | Global `(B, Dc)` | No | MLP + AdaLN (GMM output) | `gmflow_crn.py` |
| **Local CRNs** |
| LocalAdaLNMLPCRN | Local `(B, K, Dc)`, N=K | No | MLP + per-latent AdaLN | `local_crn.py` |
| LocalDiTCRN | Local `(B, K, Dc)`, N=K | **Yes** | Transformer + per-latent AdaLN | `local_crn.py` |
| **Structured CRNs** |
| StructuredAdaLNMLPCRN | Structured `(B, K, Dc)`, K<<N | No | Soft Selection → MLP + AdaLN | `structured_crn.py` |
| StructuredCrossAttentionCRN | Structured `(B, K, Dc)`, K<<N | **Yes** | Cross-attention to K latents | `structured_crn.py` |

---

## Cross-Point Mixing Mechanisms

### Mechanisms that Enable Mixing:
1. **Self-Attention** (GlobalDiTCRN, LocalDiTCRN): Points attend to each other
2. **Cross-Attention** (GlobalCrossAttentionCRN, StructuredCrossAttentionCRN): Points attend to shared latents/queries

### Mechanisms that Prevent Mixing:
1. **Per-Point MLPs** (GlobalAdaLNMLPCRN, LocalAdaLNMLPCRN, StructuredAdaLNMLPCRN): Each point processed independently
2. **Broadcast Conditioning**: Global conditioning broadcasted to all points without interaction

---

## Factory Usage

The `crn_factory.py` currently supports:
- **Global CRNs**: `'adaln_mlp'`, `'dit'`, `'cross_attention'`, `'gmflow'`
- **Structured CRNs**: `'adaln_mlp'`, `'cross_attention'`
- **Local CRNs**: Not yet exposed via factory (use directly)


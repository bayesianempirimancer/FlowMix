# Encoder Organization Summary

## Encoder Categorization by Output Type

### Global Encoders (B × dim)
**Location**: `src/encoders/global_encoders/`
**Output**: Single vector per batch

| Encoder | File | Output Shape |
|---------|------|--------------|
| PointNetEncoder | `pointnet.py` | (B, dim) |

### Local Encoders (B × N × dim)
**Location**: `src/encoders/local_encoders/`
**Output**: Per-point features, where N = number of input points

| Encoder | File | Output Shape |
|---------|------|--------------|
| DGCNN | `dgcnn.py` | (B, N, embed_dim) |
| TransformerSetEncoder | `transformer_set.py` | (B, N+1, embed_dim) |
| VN_DGCNN | `vn_dgcnn.py` | (B, N, 3, embed_dim) |
| KPConv | `kpconv.py` | (B, N, embed_dim) |
| EGNN | `egnn.py` | (B, N, embed_dim) |

### Structured Encoders (B × K × dim)
**Location**: `src/encoders/structured_encoders/`
**Output**: Reduced dimensionality from N to K, where K < N (number of objects/features)

| Encoder | File | Output Shape | Reduction Method |
|---------|------|--------------|------------------|
| SlotAttentionEncoder | `slot_attention_encoder.py` | (B, K, slot_dim) | Attention-based slot binding |
| CrossAttentionEncoder | `cross_attention_encoder.py` | (B, M, latent_dim) | Perceiver-style cross-attention |
| GMMFeaturizer | `gmm_featurizer.py` | (B, L, D) where L = K*(1+2D) | GMM clustering |
| PointNetPlusPlus | `pointnet_plus_plus.py` | (B, M, embed_dim) | Hierarchical FPS sampling |
| PointNeXt | `pointnext.py` | (B, M, embed_dim) | Hierarchical FPS sampling |
| PointMLP | `pointmlp.py` | (B, M, embed_dim) | Hierarchical FPS sampling |
| Set2Set | `set2set.py` | (B, T, embed_dim) | LSTM processing steps |

## Key Distinction

- **Local encoders**: Preserve all input points (N → N)
- **Structured encoders**: Reduce dimensionality (N → K where K < N)
- **Global encoders**: Aggregate to single vector (N → 1)

## Migration Notes

The following encoders were moved from `local_encoders/` to `structured_encoders/`:
- `slot_attention_encoder.py`
- `cross_attention_encoder.py`
- `gmm_featurizer.py`
- `pointnet_plus_plus.py`
- `pointnext.py`
- `pointmlp.py`
- `set2set.py`

All imports have been updated in `encoder_factory.py` to use the new locations.


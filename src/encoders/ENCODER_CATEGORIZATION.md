# Encoder Categorization by Output Type

## Global Encoders (B × dim)
Output: Single vector per batch
- **PointNetEncoder** (`global_encoders/pointnet.py`) - (B, dim)

## Local Encoders (B × N × dim)
Output: Per-point features, where N = number of input points
- **DGCNN** (`local_encoders/dgcnn.py`) - (B, N, embed_dim)
- **TransformerSetEncoder** (`local_encoders/transformer_set.py`) - (B, N+1, embed_dim) - includes CLS token
- **VN_DGCNN** (`local_encoders/vn_dgcnn.py`) - (B, N, 3, embed_dim) - vector neurons
- **KPConv** (`local_encoders/kpconv.py`) - (B, N, embed_dim)
- **EGNN** (`local_encoders/egnn.py`) - (B, N, embed_dim)

## Structured Encoders (B × K × dim)
Output: Reduced dimensionality from N to K, where K < N (number of objects/features)
- **SlotAttentionEncoder** (`structured_encoders/slot_attention_encoder.py`) - (B, K, slot_dim) - reduces N to K slots
- **CrossAttentionEncoder** (`structured_encoders/cross_attention_encoder.py`) - (B, M, latent_dim) - reduces N to M latents
- **GMMFeaturizer** (`structured_encoders/gmm_featurizer.py`) - (B, L, D) where L = K * (1 + 2*D) - reduces N to K clusters
- **PointNetPlusPlus** (`structured_encoders/pointnet_plus_plus.py`) - (B, M, embed_dim) where M < N - hierarchical sampling
- **PointNeXt** (`structured_encoders/pointnext.py`) - (B, M, embed_dim) where M < N - hierarchical sampling
- **PointMLP** (`structured_encoders/pointmlp.py`) - (B, M, embed_dim) where M < N - hierarchical sampling
- **Set2Set** (`structured_encoders/set2set.py`) - (B, T, embed_dim) where T = num_steps - reduces N to T processing steps


"""
GMM Featurizer - outputs sequence of geometric features (B, L, D).
Extracts features from GMM clusters including means and eigen-axes.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Optional
from src.gmm.vb_gmm import fit_vb_gmm


class GMMFeaturizer(nn.Module):
    """
    GMM featurizer that outputs a sequence of features per cluster.
    Outputs (features, valid_mask) where features: (B, L, D) and valid_mask: (B, L).
    L = K * (1 + 2*D) where K is num_clusters.
    """
    num_clusters: int = 10
    hidden_dim: int = 128
    backbone_type: str = 'transformer' # 'transformer' or 'pointnet'
    global_scale: float = 0.3  # Empirical std of MNIST point clouds
    init_method: str = 'fps'
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, key: Optional[jax.random.PRNGKey] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Returns (features, valid_mask)
        # features: (B, L, D)
        # valid_mask: (B, L)
        
        # x: (B, N, D)
        # mask: (B, N) or None
        B, N, D = x.shape
        
        # 1. Fit VB-GMM (Batched)
        if key is not None:
            seed_int = jax.random.randint(key, (), 0, 1000000)
        else:
            seed_int = 42
            
        # Dynamic N_eff: min(N_valid, 40 * K)
        if mask is not None:
            current_N = jnp.sum(mask, axis=1) # (B,)
        else:
            current_N = jnp.full((B,), N)
            
        N_eff_val = jnp.minimum(current_N, 40.0 * self.num_clusters)
            
        means, covs, weights, valid_mask, _ = fit_vb_gmm(
            x, 
            self.num_clusters, 
            mask=mask, 
            prior_counts=None, # Use default 1/K sparsity
            beta_0=1.0,
            num_iters=10, # Updated default
            seed=seed_int,
            lr=0.5, # Updated default
            N_eff=N_eff_val, # Dynamic N_eff
            global_scale=self.global_scale,
            init_method=self.init_method
        )
        
        # 2. Extract Geometric Features
        # Eigendecomposition:  e[..,i] is the i-th eigenvalue, v[..,:,i] is the i-th eigenvector.
        e, v = jnp.linalg.eigh(covs) # e: (B, K, D), v: (B, K, D, D)
        
        # sigma = sqrt(lambda)
        sigma = jnp.sqrt(jnp.maximum(e, 0.0)) # (B, K, D)
        
        scaled_vecs = v * sigma[..., None, :] # (B, K, D, D)
                
        axes_pos = means[..., :, None] + scaled_vecs # (B, K, D, D)
        axes_neg = means[..., :, None] - scaled_vecs # (B, K, D, D)

        # Now since it is the second to last dimensions that holds the vectors of interest, 
        # we need to transpose and then concatenate along the second to last dimension to
        # get the desired shape (B, K, 1 + 2D, D).

        collocation_points = jnp.concatenate([means[..., :, None], axes_pos, axes_neg], axis=-1)
        collocation_points = jnp.swapaxes(collocation_points, -1, -2)

        # Flatten K and points per cluster to get a long sequence L = K * (1+2D)
        # (B, K * (1+2D), D)
        L = self.num_clusters * (1 + 2*D)
        points_seq = collocation_points.reshape(B, L, D)

        # Valid Mask expansion
        # valid_mask: (B, K)
        # We need to expand this to the new sequence length
        # (B, K, 1) -> (B, K, 1+2D) -> (B, L)
        valid_mask_expanded = jnp.repeat(valid_mask[:, :, None], 1 + 2*D, axis=2)
        valid_mask_seq = valid_mask_expanded.reshape(B, L)
        
        # Zero out invalid points (using mask)
        points_seq = points_seq * valid_mask_seq[:, :, None]
        
        # Return sequence of points (B, L, D)
        return points_seq, valid_mask_seq


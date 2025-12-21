"""Hierarchical Variational Bayesian GMM with Overlap-Based Merging.

This module implements a bottom-up hierarchical VB-GMM that:
1. Starts with a base-level GMM (many clusters)
2. Iteratively merges overlapping clusters into higher-level "objects"
3. Stops when clusters are sufficiently separated (or max levels reached)
4. Returns GMM parameters at each level of the hierarchy

The merge criterion is based on Gaussian overlap (Bhattacharyya coefficient),
which considers both mean distance AND covariance overlap.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import digamma
from typing import Tuple, Optional, NamedTuple, List
from src.gmm.fps import batched_fps


class LevelParams(NamedTuple):
    """GMM parameters at one level of the hierarchy."""
    means: jnp.ndarray       # (B, K_level, D)
    covariances: jnp.ndarray # (B, K_level, D, D)
    weights: jnp.ndarray     # (B, K_level)
    valid_mask: jnp.ndarray  # (B, K_level)
    num_clusters: jnp.ndarray  # (B,) number of valid clusters at this level


class HierarchicalGMMOutput(NamedTuple):
    """Output of hierarchical GMM fitting."""
    levels: List[LevelParams]  # List of parameters at each level (fine to coarse)
    num_levels: int            # Actual number of levels created
    merge_history: Optional[jnp.ndarray]  # (num_merges, B, 2) pairs that were merged


class ClusterState(NamedTuple):
    """Internal state for hierarchical clustering."""
    # Sufficient statistics
    N_k: jnp.ndarray        # (B, K) cluster counts
    sum_x: jnp.ndarray      # (B, K, D) sum of points
    sum_xx: jnp.ndarray     # (B, K, D, D) sum of outer products
    
    # Variational parameters (Normal-Wishart)
    beta_k: jnp.ndarray     # (B, K) precision scale
    m_k: jnp.ndarray        # (B, K, D) mean
    W_k_inv: jnp.ndarray    # (B, K, D, D) inverse scale matrix
    nu_k: jnp.ndarray       # (B, K) degrees of freedom
    alpha_k: jnp.ndarray    # (B, K) Dirichlet concentration
    
    # Active mask (1.0 = active, 0.0 = merged/inactive)
    active: jnp.ndarray     # (B, K)


def fit_hierarchical_vb_gmm(
    x: jnp.ndarray,
    max_levels: int = 5,
    max_clusters_base: int = 32,
    mask: Optional[jnp.ndarray] = None,
    overlap_threshold: float = 0.3,
    min_clusters: int = 1,
    prior_counts: float = 0.1,
    beta_0: float = 1.0,
    nu_0: Optional[float] = None,
    seed: int = 42,
    global_scale: float = 1.0,
    base_em_iters: int = 10,
    return_merge_history: bool = False,
) -> HierarchicalGMMOutput:
    """
    Fit hierarchical VB-GMM with overlap-based merging.
    
    Algorithm:
    1. Fit base-level GMM with max_clusters_base components
       (Bayesian pruning will eliminate unused clusters)
    2. Compute pairwise Bhattacharyya overlap between active clusters
    3. Merge all pairs with overlap > threshold (greedy, by highest overlap)
    4. Record this as a new level
    5. Repeat until no pairs exceed threshold or min_clusters reached
    
    Args:
        x: (B, N, D) data points
        max_levels: Maximum hierarchy depth
        max_clusters_base: Initial number of clusters for base level
        mask: (B, N) boolean/float mask. 1.0 = valid, 0.0 = ignored.
        overlap_threshold: Bhattacharyya coefficient threshold for merging (0-1)
                          Higher = merge more aggressively
        min_clusters: Minimum clusters to keep (stop merging if reached)
        prior_counts: Dirichlet prior (smaller = more pruning)
        beta_0: Gaussian prior parameter
        nu_0: Wishart degrees of freedom (default: D + 2)
        seed: Random seed
        global_scale: Global scale of the data
        base_em_iters: EM iterations for base level fitting
        return_merge_history: Whether to record merge pairs
        
    Returns:
        HierarchicalGMMOutput with parameters at each level
    """
    B, N, D = x.shape
    
    # Default parameters
    if mask is None:
        mask = jnp.ones((B, N), dtype=jnp.float32)
    else:
        mask = mask.astype(jnp.float32)
        if mask.ndim == 3:
            mask = mask.squeeze(-1)
            
    if nu_0 is None:
        nu_0 = float(D) + 2.0
    
    # Prior parameters
    m_0 = jnp.zeros(D)
    scale_factor = max_clusters_base ** (2.0 / D)
    W_0_inv = jnp.eye(D) * nu_0 / scale_factor * (global_scale ** 2)
    
    # === Phase 1: Fit base-level GMM ===
    state = _fit_base_level(
        x, mask, max_clusters_base,
        m_0, W_0_inv, beta_0, nu_0, prior_counts,
        seed, base_em_iters
    )
    
    # Extract base level parameters
    levels = []
    merge_history = [] if return_merge_history else None
    
    base_params = _extract_level_params(state)
    levels.append(base_params)
    
    # === Phase 2: Iterative overlap-based merging ===
    for level_idx in range(max_levels - 1):
        # Count active clusters
        num_active = jnp.sum(state.active, axis=-1)  # (B,)
        
        # Check stopping conditions
        if jnp.all(num_active <= min_clusters):
            break
            
        # Compute overlap matrix
        overlap = _compute_bhattacharyya_overlap(state)  # (B, K, K)
        
        # Find pairs to merge (overlap > threshold)
        merge_mask = (overlap > overlap_threshold) & (state.active[:, :, None] > 0.5) & (state.active[:, None, :] > 0.5)
        
        # Exclude self-merges and ensure we only count each pair once (upper triangle)
        K = state.N_k.shape[1]
        upper_tri = jnp.triu(jnp.ones((K, K)), k=1)
        merge_mask = merge_mask & (upper_tri[None, :, :] > 0.5)
        
        # Check if any merges needed
        if not jnp.any(merge_mask):
            break
            
        # Perform all merges for this level
        state, level_merges = _merge_overlapping_clusters(
            state, overlap, merge_mask,
            m_0, W_0_inv, beta_0, nu_0, prior_counts,
            min_clusters
        )
        
        if return_merge_history and level_merges is not None:
            merge_history.append(level_merges)
        
        # Extract parameters for this level
        level_params = _extract_level_params(state)
        levels.append(level_params)
        
        # Check if we made progress
        new_num_active = jnp.sum(state.active, axis=-1)
        if jnp.all(new_num_active >= num_active):
            break  # No merges happened
    
    # Convert merge history to array if recorded
    if return_merge_history and merge_history:
        merge_history = jnp.stack(merge_history, axis=0)
    else:
        merge_history = None
    
    return HierarchicalGMMOutput(
        levels=levels,
        num_levels=len(levels),
        merge_history=merge_history
    )


def _fit_base_level(
    x: jnp.ndarray,
    mask: jnp.ndarray,
    K: int,
    m_0: jnp.ndarray,
    W_0_inv: jnp.ndarray,
    beta_0: float,
    nu_0: float,
    prior_counts: float,
    seed: int,
    num_iters: int
) -> ClusterState:
    """Fit base-level GMM using VB-EM."""
    B, N, D = x.shape
    
    # FPS initialization
    indices = batched_fps(x, K, mask=mask, seed=seed)  # (B, K)
    centroids = jnp.take_along_axis(x, indices[:, :, None], axis=1)  # (B, K, D)
    
    # Soft initialization based on distance
    dist_sq = jnp.sum((x[:, :, None, :] - centroids[:, None, :, :]) ** 2, axis=-1)
    r_nk = jax.nn.softmax(-0.5 * dist_sq, axis=-1)  # (B, N, K)
    r_nk = r_nk * mask[:, :, None]
    
    # Initialize parameters
    beta_k = jnp.ones((B, K)) * beta_0
    m_k = centroids
    W_k_inv = jnp.tile(W_0_inv[None, None, :, :], (B, K, 1, 1))
    nu_k = jnp.ones((B, K)) * nu_0
    alpha_k = jnp.ones((B, K)) * prior_counts
    
    # Precompute outer products
    X_outer = x[:, :, :, None] * x[:, :, None, :]  # (B, N, D, D)
    
    # VB-EM iterations
    def em_step(carry, _):
        r_nk, beta_k, m_k, W_k_inv, nu_k, alpha_k = carry
        
        # E-step
        W_k = _safe_inverse(W_k_inv)
        log_det_W_k = -jnp.linalg.slogdet(W_k_inv + 1e-6 * jnp.eye(D))[1]
        
        # Expected log determinant of precision
        d_range = jnp.arange(1, D + 1)
        E_ln_det_lambda = (
            jnp.sum(digamma(0.5 * (nu_k[:, :, None] + 1 - d_range)), axis=-1)
            + D * jnp.log(2.0) + log_det_W_k
        )
        
        # Expected log pi
        E_ln_pi = digamma(alpha_k) - digamma(jnp.sum(alpha_k, axis=1, keepdims=True))
        
        # Mahalanobis distance
        diff_x = x[:, :, None, :] - m_k[:, None, :, :]
        mahal = jnp.einsum('bnkd,bkde,bnke->bnk', diff_x, W_k, diff_x)
        E_mahal = D / beta_k[:, None, :] + nu_k[:, None, :] * mahal
        
        # Log responsibilities
        ln_rho = E_ln_pi[:, None, :] + 0.5 * E_ln_det_lambda[:, None, :] - 0.5 * E_mahal
        
        # Softmax
        new_r_nk = jax.nn.softmax(ln_rho, axis=-1) * mask[:, :, None]
        
        # M-step
        N_k = jnp.sum(new_r_nk, axis=1) + 1e-10
        sum_x = jnp.einsum('bnk,bnd->bkd', new_r_nk, x)
        sum_xx = jnp.einsum('bnk,bnij->bkij', new_r_nk, X_outer)
        
        # Update parameters
        new_beta_k = beta_0 + N_k
        new_nu_k = nu_0 + N_k
        new_alpha_k = prior_counts + N_k
        
        new_m_k = (beta_0 * m_0 + sum_x) / new_beta_k[:, :, None]
        
        x_bar = sum_x / N_k[:, :, None]
        S_k = sum_xx - N_k[:, :, None, None] * (x_bar[:, :, :, None] * x_bar[:, :, None, :])
        
        diff_0 = x_bar - m_0
        outer_0 = diff_0[:, :, :, None] * diff_0[:, :, None, :]
        factor = (beta_0 * N_k) / (beta_0 + N_k)
        
        new_W_k_inv = W_0_inv[None, None, :, :] + S_k + factor[:, :, None, None] * outer_0
        new_W_k_inv = 0.5 * (new_W_k_inv + jnp.swapaxes(new_W_k_inv, -1, -2))
        
        return (new_r_nk, new_beta_k, new_m_k, new_W_k_inv, new_nu_k, new_alpha_k), None
    
    init_carry = (r_nk, beta_k, m_k, W_k_inv, nu_k, alpha_k)
    (r_nk, beta_k, m_k, W_k_inv, nu_k, alpha_k), _ = jax.lax.scan(
        em_step, init_carry, None, length=num_iters
    )
    
    # Compute final sufficient statistics
    N_k = jnp.sum(r_nk, axis=1)
    sum_x = jnp.einsum('bnk,bnd->bkd', r_nk, x)
    sum_xx = jnp.einsum('bnk,bnij->bkij', r_nk, X_outer)
    
    # Determine active clusters (Bayesian pruning via weight threshold)
    weights = alpha_k / jnp.sum(alpha_k, axis=-1, keepdims=True)
    active = (weights > 0.01).astype(jnp.float32)  # Prune clusters with < 1% weight
    
    return ClusterState(
        N_k=N_k, sum_x=sum_x, sum_xx=sum_xx,
        beta_k=beta_k, m_k=m_k, W_k_inv=W_k_inv, nu_k=nu_k, alpha_k=alpha_k,
        active=active
    )


def _compute_bhattacharyya_overlap(state: ClusterState) -> jnp.ndarray:
    """
    Compute pairwise Bhattacharyya coefficient (overlap measure) between clusters.
    
    BC(p, q) = exp(-D_B) where D_B is the Bhattacharyya distance.
    
    For Gaussians:
    D_B = (1/8)(μ₁-μ₂)ᵀΣ⁻¹(μ₁-μ₂) + (1/2)ln(|Σ|/√(|Σ₁||Σ₂|))
    where Σ = (Σ₁+Σ₂)/2
    
    Returns:
        overlap: (B, K, K) pairwise overlap coefficients (0 = no overlap, 1 = identical)
    """
    B, K = state.N_k.shape
    D = state.m_k.shape[-1]
    
    m_k = state.m_k  # (B, K, D)
    W_k_inv = state.W_k_inv  # (B, K, D, D)
    nu_k = state.nu_k  # (B, K)
    
    # Expected covariance: E[Σ] = W_k_inv / nu_k
    cov_k = W_k_inv / (nu_k[:, :, None, None] + 1e-10)  # (B, K, D, D)
    
    # Pairwise mean difference
    diff = m_k[:, :, None, :] - m_k[:, None, :, :]  # (B, K, K, D)
    
    # Average covariance: Σ_avg = (Σ_i + Σ_j) / 2
    cov_avg = 0.5 * (cov_k[:, :, None, :, :] + cov_k[:, None, :, :, :])  # (B, K, K, D, D)
    
    # Inverse of average covariance
    cov_avg_inv = _safe_inverse(cov_avg.reshape(B * K * K, D, D)).reshape(B, K, K, D, D)
    
    # Mahalanobis term: (1/8)(μ₁-μ₂)ᵀΣ_avg⁻¹(μ₁-μ₂)
    mahal_term = 0.125 * jnp.einsum('bijd,bijde,bije->bij', diff, cov_avg_inv, diff)
    
    # Determinant term: (1/2)ln(|Σ_avg|/√(|Σ₁||Σ₂|))
    log_det_avg = jnp.linalg.slogdet(cov_avg.reshape(B * K * K, D, D) + 1e-8 * jnp.eye(D))[1].reshape(B, K, K)
    log_det_k = jnp.linalg.slogdet(cov_k + 1e-8 * jnp.eye(D))[1]  # (B, K)
    
    det_term = 0.5 * (log_det_avg - 0.5 * (log_det_k[:, :, None] + log_det_k[:, None, :]))
    
    # Bhattacharyya distance
    D_B = mahal_term + det_term
    
    # Bhattacharyya coefficient (overlap)
    overlap = jnp.exp(-D_B)
    
    # Mask inactive clusters
    active_mask = state.active[:, :, None] * state.active[:, None, :]
    overlap = overlap * active_mask
    
    # Set diagonal to 0 (self-overlap not meaningful for merging)
    overlap = overlap * (1.0 - jnp.eye(K)[None, :, :])
    
    return overlap


def _merge_overlapping_clusters(
    state: ClusterState,
    overlap: jnp.ndarray,
    merge_mask: jnp.ndarray,
    m_0: jnp.ndarray,
    W_0_inv: jnp.ndarray,
    beta_0: float,
    nu_0: float,
    prior_counts: float,
    min_clusters: int
) -> Tuple[ClusterState, Optional[jnp.ndarray]]:
    """
    Merge all cluster pairs that exceed the overlap threshold.
    Uses greedy strategy: merge highest overlap pair first.
    """
    B, K = state.N_k.shape
    D = state.m_k.shape[-1]
    
    # We'll iterate, merging one pair at a time (greedy by overlap)
    def merge_step(carry, _):
        state, any_merged = carry
        
        # Recompute overlap for current active clusters
        overlap = _compute_bhattacharyya_overlap(state)
        
        # Find best merge candidate
        # Mask: upper triangle, both active
        upper_tri = jnp.triu(jnp.ones((K, K)), k=1)
        valid_merge = (state.active[:, :, None] > 0.5) & (state.active[:, None, :] > 0.5)
        valid_merge = valid_merge & (upper_tri[None, :, :] > 0.5)
        
        # Check min_clusters constraint
        num_active = jnp.sum(state.active, axis=-1)  # (B,)
        can_merge = num_active > min_clusters  # (B,)
        valid_merge = valid_merge & can_merge[:, None, None]
        
        # Masked overlap
        masked_overlap = jnp.where(valid_merge, overlap, -jnp.inf)
        
        # Find max overlap per batch
        flat_overlap = masked_overlap.reshape(B, -1)
        best_flat = jnp.argmax(flat_overlap, axis=-1)  # (B,)
        best_overlap = jnp.take_along_axis(flat_overlap, best_flat[:, None], axis=-1).squeeze(-1)  # (B,)
        
        i = best_flat // K
        j = best_flat % K
        
        # Ensure i < j
        i_new = jnp.minimum(i, j)
        j_new = jnp.maximum(i, j)
        
        # Check if this merge is valid (overlap > 0 means it wasn't masked)
        should_merge = best_overlap > 0  # (B,)
        
        # Merge cluster j into cluster i
        batch_idx = jnp.arange(B)
        
        # Get statistics
        N_i = state.N_k[batch_idx, i_new]
        N_j = state.N_k[batch_idx, j_new]
        sum_x_i = state.sum_x[batch_idx, i_new]
        sum_x_j = state.sum_x[batch_idx, j_new]
        sum_xx_i = state.sum_xx[batch_idx, i_new]
        sum_xx_j = state.sum_xx[batch_idx, j_new]
        
        # Merged statistics
        N_merged = N_i + N_j
        sum_x_merged = sum_x_i + sum_x_j
        sum_xx_merged = sum_xx_i + sum_xx_j
        
        # Conditional update (only if should_merge)
        new_N_k = jnp.where(
            should_merge[:, None],
            state.N_k.at[batch_idx, i_new].set(N_merged).at[batch_idx, j_new].set(0.0),
            state.N_k
        )
        
        new_sum_x = jnp.where(
            should_merge[:, None, None],
            state.sum_x.at[batch_idx, i_new].set(sum_x_merged).at[batch_idx, j_new].set(jnp.zeros(D)),
            state.sum_x
        )
        
        new_sum_xx = jnp.where(
            should_merge[:, None, None, None],
            state.sum_xx.at[batch_idx, i_new].set(sum_xx_merged).at[batch_idx, j_new].set(jnp.zeros((D, D))),
            state.sum_xx
        )
        
        new_active = jnp.where(
            should_merge[:, None],
            state.active.at[batch_idx, j_new].set(0.0),
            state.active
        )
        
        # Recompute posterior parameters
        beta_k, m_k, W_k_inv, nu_k, alpha_k = _compute_posterior_params(
            new_N_k, new_sum_x, new_sum_xx, m_0, W_0_inv, beta_0, nu_0, prior_counts
        )
        
        new_state = ClusterState(
            N_k=new_N_k, sum_x=new_sum_x, sum_xx=new_sum_xx,
            beta_k=beta_k, m_k=m_k, W_k_inv=W_k_inv, nu_k=nu_k, alpha_k=alpha_k,
            active=new_active
        )
        
        new_any_merged = any_merged | jnp.any(should_merge)
        
        return (new_state, new_any_merged), jnp.stack([i_new, j_new], axis=-1)
    
    # Run enough iterations to potentially merge all pairs
    max_merges = K // 2
    (final_state, _), merge_pairs = jax.lax.scan(
        merge_step, (state, False), None, length=max_merges
    )
    
    return final_state, merge_pairs


def _compute_posterior_params(
    N_k: jnp.ndarray,
    sum_x: jnp.ndarray,
    sum_xx: jnp.ndarray,
    m_0: jnp.ndarray,
    W_0_inv: jnp.ndarray,
    beta_0: float,
    nu_0: float,
    prior_counts: float
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute posterior parameters from sufficient statistics."""
    B, K, D = sum_x.shape
    
    beta_k = beta_0 + N_k
    nu_k = nu_0 + N_k
    alpha_k = prior_counts + N_k
    
    m_k = (beta_0 * m_0 + sum_x) / (beta_k[:, :, None] + 1e-10)
    
    x_bar = sum_x / (N_k[:, :, None] + 1e-10)
    S_k = sum_xx - N_k[:, :, None, None] * (x_bar[:, :, :, None] * x_bar[:, :, None, :])
    
    diff_0 = x_bar - m_0
    outer_0 = diff_0[:, :, :, None] * diff_0[:, :, None, :]
    factor = (beta_0 * N_k) / (beta_0 + N_k + 1e-10)
    
    W_k_inv = W_0_inv[None, None, :, :] + S_k + factor[:, :, None, None] * outer_0
    W_k_inv = 0.5 * (W_k_inv + jnp.swapaxes(W_k_inv, -1, -2))
    
    return beta_k, m_k, W_k_inv, nu_k, alpha_k


def _extract_level_params(state: ClusterState) -> LevelParams:
    """Extract GMM parameters for current hierarchy level."""
    B, K = state.N_k.shape
    D = state.m_k.shape[-1]
    
    # Expected covariance
    W_k_inv = 0.5 * (state.W_k_inv + jnp.swapaxes(state.W_k_inv, -1, -2))
    covariances = W_k_inv / (state.nu_k[:, :, None, None] + 1e-10)
    
    # Mixing weights (only among active clusters for this level)
    active_alpha = state.alpha_k * state.active
    weights = active_alpha / (jnp.sum(active_alpha, axis=-1, keepdims=True) + 1e-10)
    
    # Valid mask
    N_total = jnp.sum(state.N_k * state.active, axis=-1)
    threshold = 0.005 * N_total  # 0.5% threshold
    valid_mask = (state.active > 0.5) & (state.N_k > threshold[:, None])
    
    num_clusters = jnp.sum(valid_mask, axis=-1)
    
    return LevelParams(
        means=state.m_k,
        covariances=covariances,
        weights=weights,
        valid_mask=valid_mask.astype(jnp.float32),
        num_clusters=num_clusters
    )


def _safe_inverse(A: jnp.ndarray) -> jnp.ndarray:
    """Compute matrix inverse with numerical stability."""
    D = A.shape[-1]
    A_stable = A + 1e-6 * jnp.eye(D)
    return jnp.linalg.inv(A_stable)


# === Convenience functions ===

def get_level_gmm(output: HierarchicalGMMOutput, level: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Get GMM parameters at a specific hierarchy level.
    
    Args:
        output: HierarchicalGMMOutput from fit_hierarchical_vb_gmm
        level: Hierarchy level (0 = finest/base, higher = coarser)
        
    Returns:
        means, covariances, weights, valid_mask
    """
    if level >= len(output.levels):
        level = len(output.levels) - 1
        
    params = output.levels[level]
    return params.means, params.covariances, params.weights, params.valid_mask


def get_all_valid_clusters(output: HierarchicalGMMOutput) -> List[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Get valid clusters at each level.
    
    Returns list of (means, covariances, weights) for each level,
    filtered to only valid clusters.
    """
    results = []
    for params in output.levels:
        # This returns variable-length arrays per batch, so we return the full arrays
        # and the valid_mask for downstream filtering
        results.append((params.means, params.covariances, params.weights))
    return results


# === Legacy API compatibility ===

def fit_vb_gmm_hierarchical(
    x: jnp.ndarray,
    num_clusters: int,
    mask: Optional[jnp.ndarray] = None,
    prior_counts: Optional[float] = None,
    beta_0: float = 1.0,
    nu_0: Optional[float] = None,
    num_iters: int = 20,
    seed: int = 42,
    lr: float = 1.0,
    N_eff: float = None,
    global_scale: float = 1.0,
    init_method: str = 'fps',
    compute_elbo: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Legacy API wrapper - returns finest level that has approximately num_clusters.
    """
    if prior_counts is None:
        prior_counts = 0.1
        
    output = fit_hierarchical_vb_gmm(
        x=x,
        max_levels=5,
        max_clusters_base=max(num_clusters * 2, 16),
        mask=mask,
        overlap_threshold=0.3,
        min_clusters=max(1, num_clusters // 2),
        prior_counts=prior_counts,
        beta_0=beta_0,
        nu_0=nu_0,
        seed=seed,
        global_scale=global_scale,
        base_em_iters=num_iters
    )
    
    # Find level closest to target num_clusters
    best_level = 0
    best_diff = float('inf')
    for i, params in enumerate(output.levels):
        avg_clusters = float(jnp.mean(params.num_clusters))
        diff = abs(avg_clusters - num_clusters)
        if diff < best_diff:
            best_diff = diff
            best_level = i
    
    params = output.levels[best_level]
    
    # Reorder to put valid clusters first
    B, K = params.means.shape[:2]
    D = params.means.shape[-1]
    
    # Sort by weight (descending)
    sort_idx = jnp.argsort(-params.weights * params.valid_mask, axis=-1)
    batch_idx = jnp.arange(B)[:, None]
    
    means = params.means[batch_idx, sort_idx][:, :num_clusters]
    covariances = params.covariances[batch_idx, sort_idx][:, :num_clusters]
    weights = params.weights[batch_idx, sort_idx][:, :num_clusters]
    valid_mask = params.valid_mask[batch_idx, sort_idx][:, :num_clusters]
    
    # Re-normalize weights
    weights = weights / (jnp.sum(weights, axis=-1, keepdims=True) + 1e-10)
    
    return means, covariances, weights, valid_mask, None

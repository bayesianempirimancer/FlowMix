"""Variational Bayesian GMM with Normal-Wishart Priors."""

import jax
import jax.numpy as jnp
from jax.scipy.special import digamma, gammaln
from jax.nn import logsumexp
from typing import Tuple, Any, NamedTuple, Optional
from src.gmm.fps import batched_fps


def fit_vb_gmm(
    x: jnp.ndarray,
    num_clusters: int,
    mask: Optional[jnp.ndarray] = None,
    prior_counts: Optional[float] = None,
    beta_0: float = 1.0,
    nu_0: float = None,
    num_iters: int = 20,
    seed: int = 42,
    lr: float = 1.0,
    N_eff: float = None,
    global_scale: float = 1.0,
    init_method: str = 'fps', # 'fps' or 'random'
    compute_elbo: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Fit Variational Bayesian GMM with Normal-Wishart priors using soft updates.
    Handles batched inputs (B, N, D) using broadcasting and einsum.
    
    Args:
        x: (B, N, D) data points
        num_clusters: Number of components K
        mask: (B, N) or (B, N, 1) boolean/float mask. 1.0 = valid, 0.0 = ignored.
        prior_counts: Dirichlet prior parameter alpha_0. Defaults to 1.0/num_clusters if None.
        beta_0: Gaussian prior parameter
        nu_0: Wishart degrees of freedom
        num_iters: EM iterations (Default: 20)
        seed: Random seed
        lr: Learning rate for soft updates
        N_eff: Effective number of data points for scaling. If None, uses sum(mask).
        global_scale: Global scale of the data. Scales prior expected variance (W_0_inv).
        init_method: Initialization method for centroids. 'fps' (Farthest Point Sampling) or 'random'.
        compute_elbo: Whether to compute and return the ELBO history.
        
    Returns:
        means: (B, K, D)
        covariances: (B, K, D, D)
        weights: (B, K)
        valid_mask: (B, K)
        elbo_history: (num_iters, B) - ELBO at each step (or None if compute_elbo=False)
    """
    B, N, D = x.shape
    
    if mask is None:
        mask = jnp.ones((B, N), dtype=jnp.float32)
    else:
        mask = mask.astype(jnp.float32)
        if mask.ndim == 3:
            mask = mask.squeeze(-1)
            
    if nu_0 is None:
        nu_0 = float(D) + 2.0 
        
    if prior_counts is None:
        prior_counts = 1.0 / num_clusters
        
    if N_eff is None:
        N_eff = jnp.sum(mask, axis=1) # (B,)
        # Avoid division by zero if all masked
        N_eff = jnp.maximum(N_eff, 1e-6)
    else:
        # Broadcast N_eff if scalar
        if isinstance(N_eff, (int, float)):
             N_eff = jnp.full((B,), N_eff)
        
    # Priors
    m_0 = jnp.zeros(D)
    
    # Scale factor for variance/precision
    scale_factor = num_clusters**(2.0 / D)
    
    # W_0: prior Wishart scale matrix
    # E[Lambda] = nu_0 * W_0
    # We want E[Lambda] = scale_factor * I -> W_0 = scale_factor/nu_0 * I
    # W_0_inv = nu_0/scale_factor * I
    # If global_scale != 1.0, we want to scale expected variance by global_scale**2.
    # W_0_inv corresponds to (expected precision)^-1 ~ expected variance.
    # So scale W_0_inv by global_scale**2.
    W_0_inv = jnp.eye(D) * nu_0 / scale_factor * (global_scale**2)
    
    # Initialization
    key = jax.random.PRNGKey(seed)
    
    if init_method == 'fps':
        # Use efficient batched FPS from src.gmm.fps
        indices = batched_fps(x, num_clusters, mask=mask, seed=seed)
        m_k = jnp.take_along_axis(x, indices[:, :, None], axis=1)
        
    elif init_method == 'random':
        log_p = jnp.where(mask > 0.5, 0.0, -1e9)
        gumbel = jax.random.gumbel(key, shape=(B, N))
        
        scores = log_p + gumbel        
        # Top-K indices
        # indices: (B, K)
        _, indices = jax.lax.top_k(scores, num_clusters)        
        m_k = jnp.take_along_axis(x, indices[:, :, None], axis=1)
        
    else:
        raise ValueError(f"Unknown init_method: {init_method}")
    
    dist_sq = jnp.sum((x[:, :, None, :] - m_k[:, None, :, :])**2, axis=-1)
    
    # Initialize r_nk using softmax of negative distance
    # This is effectively "hard assignment" via softmax temperature
    # But for GMM we usually start with soft.
    r_nk = stable_softmax(-0.5 * dist_sq, axis=2)
    
    # Apply Mask to initialization responsibilities
    # Valid points have sum(r_nk) = 1. Invalid points have sum(r_nk) = 0.
    r_nk = r_nk * mask[:, :, None]
    
    # Initial Parameters
    beta_k = jnp.ones((B, num_clusters)) * beta_0
    # Initialize W_k_inv directly
    W_k_inv = jnp.tile(W_0_inv[None, None, :, :], (B, num_clusters, 1, 1))
    nu_k = jnp.ones((B, num_clusters)) * nu_0
    alpha_k = jnp.ones((B, num_clusters)) * prior_counts
    
    current_N_valid = jnp.sum(mask, axis=1) + 1e-10
    N_scale = N_eff / current_N_valid # (B,)

    # ELBO Computation Setup
    if compute_elbo:
        alpha_p = jnp.broadcast_to(prior_counts, (B, num_clusters))
        beta_p = jnp.broadcast_to(beta_0,(B, num_clusters))
        m_p = jnp.broadcast_to(m_0, (B, num_clusters, D))
        
        W_p_val = jnp.eye(D) / (nu_0 / scale_factor * (global_scale**2)) # Inverse of diagonal is 1/diag
        W_p = jnp.broadcast_to(W_p_val, (B, num_clusters, D, D))
        
        # Precompute W_p inverse and logdet
        # W_p is diagonal, so inverse is easy, but general case:
        W_p_inv = jnp.linalg.inv(W_p)
        _, log_det_W_p = jnp.linalg.slogdet(W_p)
        
        nu_p = jnp.broadcast_to(nu_0,(B, num_clusters))
    else:
        # Dummy values for scanning if not computing ELBO
        alpha_p = beta_p = m_p = W_p_inv = log_det_W_p = nu_p = None

    # Optimization: Precompute Outer Product of X
    # X_outer: (B, N, D, D)
    X_outer = x[:, :, :, None] * x[:, :, None, :] 
    
    def em_step(val, _):
        # Unpack current parameters (from previous step)
        r_nk, beta_k, m_k, W_k_inv, nu_k, alpha_k = val
        
        # --- E-Step ---
        # Compute precision expectations from current parameters
        # W_k is the Scale Matrix W = inv(W_k_inv)
        W_k, log_det_W_k = compute_inverse_and_logdet(W_k_inv)
        
        ln_rho = compute_log_prob(x, m_k, W_k, log_det_W_k, alpha_k, beta_k, nu_k, D)
        
        # Softmax to get responsibilities
        # Calculated for ALL points (valid and invalid)
        new_r_nk = stable_softmax(ln_rho, axis=2)
        
        # Apply Mask
        # Zero out responsibilities for invalid points
        # This ensures they don't contribute to the next M-step
        new_r_nk = new_r_nk * mask[:, :, None]
        
        if compute_elbo:
            log_norm = logsumexp(ln_rho, axis=2) # (B, N)
            ll_term = jnp.sum(log_norm * mask, axis=1) # (B,)
        
            kl_nw = kl_normal_wishart(beta_k, m_k, W_k, log_det_W_k, nu_k, beta_p, m_p, W_p_inv, log_det_W_p, nu_p) # (B,)
            kl_dir = kl_dirichlet(alpha_k, alpha_p) # (B,)
            elbo = ll_term - kl_dir - kl_nw # (B,)
        else:
            elbo = None
    
        # --- M-Step (Updates) ---
        N_k_like = jnp.sum(new_r_nk, axis=1) * N_scale[:, None]
        
        # Correct einsum for batched x_bar
        x_sum_k_like = jnp.einsum('bnk,bnd->bkd', new_r_nk, x) * N_scale[:, None, None]
        x_bar_k_like = x_sum_k_like / (N_k_like[:, :, None] + 1e-10)
        
        # S_k calculation
        # Optimized S_k calculation (Expanded Variance)
        # S_k = sum_n r_nk x_n x_n^T - N_k x_bar x_bar^T
        sum_r_xx = jnp.einsum('bnk,bnij->bkij', new_r_nk, X_outer) * N_scale[:, None, None, None]
        term2 = N_k_like[:, :, None, None] * (x_bar_k_like[:, :, :, None] * x_bar_k_like[:, :, None, :])
        S_k_like = sum_r_xx - term2
        
        # 2. Update Parameters (Soft Updates)
        # Update beta_k
        beta_k_target = beta_0 + N_k_like
        beta_k = (1 - lr) * beta_k + lr * beta_k_target
        
        # Update m_k
        kappa_mu_current = beta_k[:, :, None] * m_k
        kappa_mu_target = beta_0 * m_0 + x_sum_k_like
        kappa_mu_new = (1 - lr) * kappa_mu_current + lr * kappa_mu_target
        m_k = kappa_mu_new / beta_k[:, :, None]
        
        # Update W_k_inv
        diff_0 = x_bar_k_like - m_0
        outer_0 = diff_0[:, :, :, None] * diff_0[:, :, None, :]
        factor = (beta_0 * N_k_like) / (beta_0 + N_k_like)
        
        # W_0_inv is (D, D). Broadcast to (B, K, D, D)
        W_k_inv_target = W_0_inv[None, None, :, :] + S_k_like + factor[:, :, None, None] * outer_0
        W_k_inv = (1 - lr) * W_k_inv + lr * W_k_inv_target
        
        # Update nu_k
        nu_k_target = nu_0 + N_k_like
        nu_k = (1 - lr) * nu_k + lr * nu_k_target
        
        # Update alpha_k
        alpha_k_target = prior_counts + N_k_like
        alpha_k = (1 - lr) * alpha_k + lr * alpha_k_target
        
        return (new_r_nk, beta_k, m_k, W_k_inv, nu_k, alpha_k), elbo

    init_val = (r_nk, beta_k, m_k, W_k_inv, nu_k, alpha_k)
    final_val, elbo_history = jax.lax.scan(em_step, init_val, None, length=num_iters)
    
    r_nk, beta_k, m_k, W_k_inv, nu_k, alpha_k = final_val
    
    # Compute Features
    # W_k_inv is already carried over
    # E[Sigma] approx W^{-1} / nu
    # Ensure symmetry
    W_k_inv = 0.5 * (W_k_inv + jnp.swapaxes(W_k_inv, -1, -2))
    covariances = W_k_inv / (nu_k[:, :, None, None] + 1e-10)
    mixing_weights = alpha_k / jnp.sum(alpha_k, axis=1, keepdims=True)
    
    # Stricter valid mask: Cluster must have > 1% of total data points
    # N_k: (B, K)
    N_k = jnp.sum(r_nk, axis=1)
    
    # Calculate N_total per batch item
    # mask: (B, N)
    N_total = jnp.sum(mask, axis=1) # (B,)
    
    threshold = 0.01 * N_total # (B,)
    
    # Broadcast threshold to (B, K)
    valid_mask = N_k > threshold[:, None]
    
    return m_k, covariances, mixing_weights, valid_mask, elbo_history


# --- Helper Functions ---

def stable_softmax(x, axis=-1):
    """Compute softmax in a numerically stable way."""
    max_x = jnp.max(x, axis=axis, keepdims=True)
    exp_x = jnp.exp(x - max_x)
    return exp_x / jnp.sum(exp_x, axis=axis, keepdims=True)

def compute_inverse_and_logdet(W_k_inv):
    """
    Computes:
    1. Scale Matrix W = inv(W_k_inv)
    2. Log determinant of W (log|W|).
    
    Uses Cholesky of W_k_inv for efficiency.
    W_k_inv = L L^T
    W = inv(W_k_inv) = inv(L^T) inv(L)
    log|W| = -log|W_k_inv| = -2 sum(log(diag(L)))
    
    Args:
        W_k_inv: (..., D, D) Inverse Scale Matrix (Parameter)
        
    Returns:
        W: (..., D, D) Scale Matrix
        log_det_W: (...) Log determinant of W
    """
    D = W_k_inv.shape[-1]
    # Add jitter for stability
    W_k_inv_stable = W_k_inv + 1e-6 * jnp.eye(D)[None, None, :, :]
    
    # Cholesky: L s.t. L @ L.T = W_k_inv
    L = jnp.linalg.cholesky(W_k_inv_stable)
    
    # Log determinant of W_k_inv
    log_det_W_k_inv = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)), axis=-1)
    # Log determinant of W (scale matrix)
    log_det_W = -log_det_W_k_inv
    
    # Compute W (scale matrix) = inv(W_k_inv)
    # Solve L X = I  -> X = L^{-1}
    # W = X^T X = (L^{-1})^T L^{-1}
    # Identity matrix
    I = jnp.eye(D)
    # Broadcast I to (B, K, D, D)
    I_broad = jnp.broadcast_to(I, W_k_inv.shape)
    
    # Solve L Y = I (find L inverse)
    L_inv = jax.scipy.linalg.solve_triangular(L, I_broad, lower=True)
    
    # W = L_inv.T @ L_inv
    # Matmul: (B,K,D,D) @ (B,K,D,D). Transpose last two dims for first op.
    W = jnp.matmul(jnp.swapaxes(L_inv, -1, -2), L_inv)
    
    return W, log_det_W

def expected_log_det_lambda_from_logdet(logdet_W, nu, D):
    """E[ln |Lambda|] under Wishart(W, nu) using precomputed log|W|."""
    d_range = jnp.arange(1, D + 1)
    psi_terms = jnp.sum(digamma(0.5 * (nu[..., None] + 1 - d_range)), axis=-1)
    return psi_terms + D * jnp.log(2.0) + logdet_W

def compute_log_prob(x, m_k, W_k, logdet_W_k, alpha_k, beta_k, nu_k, D):
    """
    Compute ln_rho_nk (unnormalized log responsibility).
    
    Args:
        x: (B, N, D) Data
        m_k: (B, K, D) Means
        W_k: (B, K, D, D) Scale Matrices
        logdet_W_k: (B, K) Log determinant of W_k
        alpha_k, beta_k, nu_k: (B, K) Parameters
        D: Dimension
        
    Returns:
        ln_rho: (B, N, K) Unnormalized log responsibilities
    """
    E_ln_pi = digamma(alpha_k) - digamma(jnp.sum(alpha_k, axis=1, keepdims=True))
    E_ln_det_lambda = expected_log_det_lambda_from_logdet(logdet_W_k, nu_k, D)
    
    # Mahalanobis Distance
    # x: (B, N, D), m_k: (B, K, D)
    # diff_x: (B, N, K, D)
    diff_x = x[:, :, None, :] - m_k[:, None, :, :]
    
    # W_k: (B, K, D, D)
    # mahal: (x-m)T W (x-m)
    # einsum: 'bnkd,bkde,bnke->bnk'
    mahal = jnp.einsum('bnkd,bkde,bnke->bnk', diff_x, W_k, diff_x)
    E_mahal = D / beta_k[:, None, :] + nu_k[:, None, :] * mahal
    
    ln_rho = E_ln_pi[:, None, :] + 0.5 * E_ln_det_lambda[:, None, :] - 0.5 * D * jnp.log(2 * jnp.pi) - 0.5 * E_mahal
    return ln_rho

def kl_dirichlet(alpha_q, alpha_p):
    """
    KL(Dir(alpha_q) || Dir(alpha_p))
    alpha_q, alpha_p: (B, K)
    """
    term1 = gammaln(jnp.sum(alpha_q, axis=1)) - gammaln(jnp.sum(alpha_p, axis=1))
    term2 = jnp.sum(gammaln(alpha_p), axis=1) - jnp.sum(gammaln(alpha_q), axis=1)
    
    digamma_sum_q = digamma(jnp.sum(alpha_q, axis=1, keepdims=True))
    term3 = jnp.sum((alpha_q - alpha_p) * (digamma(alpha_q) - digamma_sum_q), axis=1)
    
    return term1 + term2 + term3

def kl_normal_wishart(beta_q, m_q, W_q, log_det_W_q, nu_q, beta_p, m_p, W_p_inv, log_det_W_p, nu_p):
    """
    KL(NW(beta_q, m_q, W_q, nu_q) || NW(beta_p, m_p, W_p, nu_p))
    Inputs are (B, K) or (B, K, D) or (B, K, D, D)
    Summed over K.
    
    W_q is Scale Matrix. W_p_inv is Inverse Scale Matrix of prior (Parameter).
    log_det_W_q is ln|W_q|. log_det_W_p is ln|W_p|.
    """
    # Based on Bishop PRML B.53
    D = m_q.shape[-1]
    
    # Expected Log Det Lambda using precomputed log_det_W_q
    E_ln_det_L = expected_log_det_lambda_from_logdet(log_det_W_q, nu_q, D)
    
    def log_norm_wishart(log_det_W, nu):
        # log Z(W, nu) = -nu/2 ln|W| - ...
        return -0.5 * nu * log_det_W - (nu * D / 2.0) * jnp.log(2.0) - jax.scipy.special.multigammaln(nu / 2.0, D)
    
    ln_B_p = log_norm_wishart(log_det_W_p, nu_p)
    ln_B_q = log_norm_wishart(log_det_W_q, nu_q)
    
    # Part 1: Wishart KL
    # KL(W_q || W_p)
    # Trace term: Tr(W_p^{-1} W_q). W_p^{-1} is W_p_inv (parameter).
    tr_term = 0.5 * nu_q * jnp.einsum('bkij,bkji->bk', W_p_inv, W_q)
    
    # KL(W_q || W_p) = ln B_q - ln B_p + 0.5(nu_q - nu_p)E[ln|L|] + 0.5 nu_q (Tr(W_p^{-1} W_q) - D)
    kl_wishart = (ln_B_q - ln_B_p) + 0.5 * (nu_q - nu_p) * E_ln_det_L + (tr_term - 0.5 * nu_q * D)
    
    # Part 2: Normal KL
    diff = m_q - m_p # (B, K, D)
    quad_term = 0.5 * beta_p * nu_q * jnp.einsum('bkd,bkde,bke->bk', diff, W_q, diff)
    log_term = 0.5 * D * (jnp.log(beta_q) - jnp.log(beta_p))
    trace_term = 0.5 * D * (beta_p / beta_q - 1.0)
    
    kl_normal = quad_term + trace_term - log_term
    
    return jnp.sum(kl_wishart + kl_normal, axis=1) # Sum over K

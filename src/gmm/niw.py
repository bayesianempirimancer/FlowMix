"""
Variational Bayesian Gaussian Mixture Model with NIW Priors.

This module provides an efficient implementation of VB-GMM using:
- Normal-Inverse-Wishart (NIW) priors for component parameters (μ, Λ)
- Dirichlet prior for mixing weights (π)
- Natural gradient updates in exponential family form

Key design principles:
1. Properly scaled sufficient statistics T = (Λμ, -½μᵀΛμ, -½Λ, ½log|Λ|)
2. Gradients of ELBO w.r.t. E[T] directly give data sufficient statistics
3. Natural gradient update: η_post = η_prior + data_stats

All functions support arbitrary batch shapes via [..., ] indexing.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import digamma, polygamma
from typing import Tuple, NamedTuple
from src.gmm.dirichlet import DirichletNaturalParams, natural_to_expected_stats as dir_nat_to_stats
from src.gmm.dirichlet import DirichletParams, kl_divergence as dirichlet_kl

# =============================================================================
# Data Structures
# =============================================================================

class NIWNaturalParams(NamedTuple):
    """Natural parameters η of NIW distribution."""
    eta1: jnp.ndarray  # κm, shape (..., D)
    eta2: jnp.ndarray  # κ, shape (...,)
    eta3: jnp.ndarray  # W⁻¹ + κmm', shape (..., D, D)
    eta4: jnp.ndarray  # ν - D - 1, shape (...,)


class NIWStandardParams(NamedTuple):
    """Standard parameters of NIW distribution."""
    m: jnp.ndarray     # mean, shape (..., D)
    kappa: jnp.ndarray # mean precision scaling, shape (...,)
    W: jnp.ndarray     # Wishart scale matrix, shape (..., D, D)
    nu: jnp.ndarray    # degrees of freedom, shape (...,)


class NIWSufficientStats(NamedTuple):
    """
    Expected sufficient statistics E[T] with proper exponential family scaling.
    
    T = (Λμ, -½μᵀΛμ, -½Λ, ½log|Λ|)
    
    With this convention, ⟨η, E[T]⟩ computes the expected inner product,
    and ∂ELBO/∂E[T] directly gives data sufficient statistics.
    """
    E_T1: jnp.ndarray  # E[Λμ], shape (..., D)
    E_T2: jnp.ndarray  # E[-½μᵀΛμ], shape (...,)
    E_T3: jnp.ndarray  # E[-½Λ], shape (..., D, D)
    E_T4: jnp.ndarray  # E[½log|Λ|], shape (...,)


# =============================================================================
# Special Functions
# =============================================================================

def multivariate_digamma(a: jnp.ndarray, D: int) -> jnp.ndarray:
    """Multivariate digamma: ψ_D(a) = Σᵢ ψ(a + (1-i)/2)"""
    a = jnp.asarray(a)
    offsets = (1 - jnp.arange(1, D + 1)) / 2
    return jnp.sum(digamma(a[..., None] + offsets), axis=-1)


def multivariate_trigamma(a: jnp.ndarray, D: int) -> jnp.ndarray:
    """Multivariate trigamma: ψ'_D(a) = Σᵢ ψ'(a + (1-i)/2)"""
    a = jnp.asarray(a)
    offsets = (1 - jnp.arange(1, D + 1)) / 2
    return jnp.sum(polygamma(1, a[..., None] + offsets), axis=-1)


def multivariate_log_gamma(a: jnp.ndarray, D: int) -> jnp.ndarray:
    """Multivariate log-gamma: log Γ_D(a)"""
    a = jnp.asarray(a)
    offsets = (1 - jnp.arange(1, D + 1)) / 2
    return (D * (D - 1) / 4) * jnp.log(jnp.pi) + jnp.sum(
        jax.scipy.special.gammaln(a[..., None] + offsets), axis=-1
    )


# =============================================================================
# Cholesky Utilities
# =============================================================================

def _cholesky_inverse_and_logdet(A: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute A⁻¹ and log|A| efficiently via Cholesky."""
    L = jnp.linalg.cholesky(A)
    log_det = 2 * jnp.sum(jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)), axis=-1)
    D = A.shape[-1]
    I = jnp.broadcast_to(jnp.eye(D), A.shape)
    L_inv = jax.scipy.linalg.solve_triangular(L, I, lower=True)
    A_inv = jnp.swapaxes(L_inv, -2, -1) @ L_inv
    return A_inv, log_det


def _cholesky_logdet(A: jnp.ndarray) -> jnp.ndarray:
    """Compute log|A| via Cholesky (when inverse not needed)."""
    L = jnp.linalg.cholesky(A)
    return 2 * jnp.sum(jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)), axis=-1)


# =============================================================================
# Parameter Conversions
# =============================================================================

def standard_to_natural(params: NIWStandardParams) -> NIWNaturalParams:
    """Convert standard parameters (m, κ, W, ν) to natural parameters.
    
    NIW natural parameters:
        η1 = κ * m
        η2 = κ
        η3 = W⁻¹ + κ * m * m'
        η4 = ν - D - 1
    """
    m, kappa, W, nu = params
    D = m.shape[-1]
    W_inv, _ = _cholesky_inverse_and_logdet(W)
    # η3 = W⁻¹ + κ * m * m'
    kappa_mm = kappa[..., None, None] * jnp.einsum('...d,...e->...de', m, m)
    return NIWNaturalParams(
        eta1=kappa[..., None] * m,
        eta2=kappa,
        eta3=W_inv + kappa_mm,
        eta4=nu - D - 1
    )


def natural_to_standard(eta: NIWNaturalParams) -> NIWStandardParams:
    """Convert natural parameters to standard parameters.
    
    Recovers (m, κ, W, ν) from (η1, η2, η3, η4):
        κ = η2
        m = η1 / κ
        W⁻¹ = η3 - κ * m * m'
        ν = η4 + D + 1
    """
    eta1, eta2, eta3, eta4 = eta
    D = eta1.shape[-1]
    kappa = eta2
    m = eta1 / kappa[..., None]
    # W⁻¹ = η3 - κ * m * m'
    kappa_mm = kappa[..., None, None] * jnp.einsum('...d,...e->...de', m, m)
    W_inv = eta3 - kappa_mm
    W, _ = _cholesky_inverse_and_logdet(W_inv)
    nu = eta4 + D + 1
    return NIWStandardParams(m=m, kappa=kappa, W=W, nu=nu)


# =============================================================================
# Expected Sufficient Statistics
# =============================================================================

def expected_stats(eta: NIWNaturalParams) -> NIWSufficientStats:
    """Compute expected sufficient statistics E[T] from natural parameters.
    
    With η3 = W⁻¹ + κmm', we first extract W⁻¹, then compute E[T].
    """
    eta1, eta2, eta3, eta4 = eta
    D = eta1.shape[-1]
    
    kappa = eta2
    m = eta1 / kappa[..., None]
    nu = eta4 + D + 1
    
    # Extract W⁻¹ from η3 = W⁻¹ + κmm'
    kappa_mm = kappa[..., None, None] * jnp.einsum('...d,...e->...de', m, m)
    W_inv = eta3 - kappa_mm
    W, log_det_W_inv = _cholesky_inverse_and_logdet(W_inv)
    log_det_W = -log_det_W_inv
    
    E_Lambda = nu[..., None, None] * W
    E_log_det_Lambda = multivariate_digamma(nu / 2, D) + D * jnp.log(2.0) + log_det_W
    E_Lambda_mu = jnp.einsum('...ij,...j->...i', E_Lambda, m)
    mT_E_Lambda_m = jnp.einsum('...i,...ij,...j->...', m, E_Lambda, m)
    E_muT_Lambda_mu = D / kappa + mT_E_Lambda_m
    
    return NIWSufficientStats(
        E_T1=E_Lambda_mu,
        E_T2=-0.5 * E_muT_Lambda_mu,
        E_T3=-0.5 * E_Lambda,
        E_T4=0.5 * E_log_det_Lambda
    )


def sufficient_stats_to_natural(
    stats: NIWSufficientStats,
    max_iter: int = 5
) -> NIWNaturalParams:
    """
    Recover natural parameters from E[T] via Newton's method.
    
    Uses asymptotic expansion for initialization and Newton on log(ν).
    """
    E_T1, E_T2, E_T3, E_T4 = stats
    D = E_T1.shape[-1]
    
    # Recover raw expectations
    E_Lambda_mu = E_T1
    E_muT_Lambda_mu = -2 * E_T2
    E_Lambda = -2 * E_T3
    E_log_det_Lambda = 2 * E_T4
    
    # Solve for m and κ
    E_Lambda_inv, log_det_E_Lambda = _cholesky_inverse_and_logdet(E_Lambda)
    m = jnp.einsum('...ij,...j->...i', E_Lambda_inv, E_Lambda_mu)
    mT_E_Lambda_m = jnp.einsum('...i,...ij,...j->...', m, E_Lambda, m)
    kappa = D / (E_muT_Lambda_mu - mT_E_Lambda_m)
    
    # Solve for ν via Newton on log(ν)
    nu_min = D
    delta = jnp.maximum(log_det_E_Lambda - E_log_det_Lambda, 1e-6)
    log_nu = jnp.log(jnp.clip(2.0 * D / delta, nu_min, 1e6))
    log_nu_min = jnp.log(nu_min)
    
    def newton_step(log_nu, _):
        nu = jnp.exp(log_nu)
        f = multivariate_digamma(nu/2, D) + D*jnp.log(2.0) + log_det_E_Lambda - D*log_nu - E_log_det_Lambda
        df = 0.5 * multivariate_trigamma(nu/2, D) - D/nu
        log_nu_new = log_nu - 0.95 * jnp.clip(f / (df * nu + 1e-10), -2.0, 2.0)
        return jnp.maximum(log_nu_new, log_nu_min), None
    
    log_nu, _ = jax.lax.scan(newton_step, log_nu, None, length=max_iter)
    nu = jnp.exp(log_nu)
    W = E_Lambda / nu[..., None, None]
    
    return standard_to_natural(NIWStandardParams(m=m, kappa=kappa, W=W, nu=nu))


# =============================================================================
# Log Partition and KL Divergence
# =============================================================================

def log_partition(eta: NIWNaturalParams) -> jnp.ndarray:
    """Compute log partition function A(η) from natural parameters.
    
    With η3 = W⁻¹ + κmm', we must extract W⁻¹ first.
    """
    eta1, eta2, eta3, eta4 = eta
    D = eta1.shape[-1]
    
    # Extract standard params for computation
    kappa = eta2
    m = eta1 / kappa[..., None]
    nu = eta4 + D + 1
    
    # Extract W⁻¹ from η3 = W⁻¹ + κmm'
    kappa_mm = kappa[..., None, None] * jnp.einsum('...d,...e->...de', m, m)
    W_inv = eta3 - kappa_mm
    log_det_W_inv = _cholesky_logdet(W_inv)
    log_det_W = -log_det_W_inv
    
    return (
        (nu / 2) * log_det_W +
        (nu * D / 2) * jnp.log(2.0) +
        multivariate_log_gamma(nu / 2, D) +
        (D / 2) * jnp.log(2 * jnp.pi / kappa)
    )


def kl_divergence(eta_q: NIWNaturalParams, eta_p: NIWNaturalParams) -> jnp.ndarray:
    """Compute KL(q || p) between NIW distributions from natural parameters."""
    E_T = expected_stats(eta_q)
    
    inner = (
        jnp.sum((eta_q.eta1 - eta_p.eta1) * E_T.E_T1, axis=-1) +
        (eta_q.eta2 - eta_p.eta2) * E_T.E_T2 +
        jnp.sum((eta_q.eta3 - eta_p.eta3) * E_T.E_T3, axis=(-2, -1)) +
        (eta_q.eta4 - eta_p.eta4) * E_T.E_T4
    )
    return inner - log_partition(eta_q) + log_partition(eta_p)


# =============================================================================
# GMM Likelihood Functions
# =============================================================================

def expected_log_likelihood_gaussian(x: jnp.ndarray, stats: NIWSufficientStats) -> jnp.ndarray:
    """
    Compute E[log N(x | μ, Λ⁻¹)] for each point and component.
    
    With properly scaled E[T]:
        E[log N(x)] = xᵀE_T1 + E_T2 + xᵀE_T3 x + E_T4 - D/2 log(2π)
    
    Args:
        x: data, shape (N, D)
        stats: expected sufficient stats, shapes (K, ...)
        
    Returns:
        log-likelihood, shape (N, K)
    """
    E_T1, E_T2, E_T3, E_T4 = stats
    D = E_T1.shape[-1]
    
    x_E_T3_x = jnp.einsum('nd,kde,ne->nk', x, E_T3, x)
    x_E_T1 = jnp.einsum('nd,kd->nk', x, E_T1)
    
    return x_E_T1 + E_T2[None, :] + x_E_T3_x + E_T4[None, :] - 0.5 * D * jnp.log(2 * jnp.pi)


def gmm_expected_log_likelihood(
    x: jnp.ndarray,
    stats: NIWSufficientStats,
    resp: jnp.ndarray,
    log_pi: jnp.ndarray = None
) -> jnp.ndarray:
    """
    Compute total expected log-likelihood weighted by responsibilities.
    
    L = Σₙₖ rₙₖ [E[log N(xₙ|μₖ,Λₖ)] + E[log πₖ]]
    """
    log_lik = expected_log_likelihood_gaussian(x, stats)
    if log_pi is not None:
        log_lik = log_lik + log_pi[None, :]
    return jnp.sum(resp * log_lik)


def gmm_responsibilities(
    x: jnp.ndarray,
    stats: NIWSufficientStats,
    log_pi: jnp.ndarray = None
) -> tuple:
    """
    Compute responsibilities (E-step) and log normalizer.
    
    rₙₖ ∝ exp(E[log πₖ] + E[log N(xₙ|μₖ,Λₖ)])
    
    Returns:
        resp: responsibilities, shape (N, K)
        logZ: log normalizer, shape (N,)
    """
    log_rho = expected_log_likelihood_gaussian(x, stats)
    if log_pi is not None:
        log_rho = log_rho + log_pi[None, :]
    logZ = jnp.max(log_rho, axis=-1, keepdims=True)
    log_rho = log_rho - logZ
    rho = jnp.exp(log_rho)
    sum_rho = jnp.sum(rho, axis=-1, keepdims=True)
    logZ = logZ + jnp.log(sum_rho)
    return rho / sum_rho, logZ[..., 0]


# =============================================================================
# ELBO and Gradient-Based Updates
# =============================================================================

def gmm_elbo(
    x: jnp.ndarray,
    stats: NIWSufficientStats,
    log_pi: jnp.ndarray,
    resp: jnp.ndarray,
    niw_eta_post: NIWNaturalParams,
    niw_eta_prior: NIWNaturalParams,
    dirichlet_post_alpha: jnp.ndarray,
    dirichlet_prior_alpha: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute ELBO for monitoring convergence.
    
    ELBO = E[log p(X,Z|θ,π)] - E[log q(Z)] - KL(q(π)||p(π)) - Σₖ KL(q(μₖ,Λₖ)||p(μₖ,Λₖ))
         = E[log p(X|Z,θ)] + E[log p(Z|π)] + H[q(Z)] - KL(q(π)||p(π)) - KL(q(θ)||p(θ))
    
    The entropy term H[q(Z)] = -Σₙₖ rₙₖ log rₙₖ is essential for proper ELBO.
    """
    
    E_log_lik = gmm_expected_log_likelihood(x, stats, resp, log_pi)
    
    # Entropy of responsibilities: H[q(Z)] = -Σₙₖ rₙₖ log rₙₖ
    resp_entropy = -jnp.sum(resp * jnp.log(resp + 1e-10))
    
    kl_dir = dirichlet_kl(DirichletParams(dirichlet_post_alpha), DirichletParams(dirichlet_prior_alpha))
    kl_niw = jnp.sum(kl_divergence(niw_eta_post, niw_eta_prior))
    
    return E_log_lik + resp_entropy - kl_dir - kl_niw


def gmm_gradient_step(
    x: jnp.ndarray,
    niw_eta: NIWNaturalParams,
    dirichlet_eta: jnp.ndarray,
    resp: jnp.ndarray,
    niw_eta_prior: NIWNaturalParams,
    dirichlet_prior_alpha: jnp.ndarray,
    lr_e: float = 1.0,
    lr_m: float = 1.0
) -> tuple:
    """
    Single gradient-based VB-EM step.
    
    Both E-step and M-step derived from gradients of ELBO:
    - E-step: log rₙₖ = ∂ELBO/∂rₙₖ → softmax
    - M-step: η_new = (1-lr)η_curr + lr(η_prior + ∂ELBO/∂E[T])
    
    Args:
        x: data, shape (N, D)
        niw_eta: current NIW natural parameters
        dirichlet_eta: current Dirichlet natural parameter (α - 1)
        resp: current responsibilities, shape (N, K)
        niw_eta_prior: NIW prior (natural params, can broadcast)
        dirichlet_prior_alpha: Dirichlet prior α₀
        lr_e: E-step learning rate
        lr_m: M-step learning rate
        
    Returns:
        (niw_eta_new, dirichlet_eta_new, resp_new, E_log_lik)
    """
    
    # Current expected stats
    stats = expected_stats(niw_eta)
    log_pi = dir_nat_to_stats(DirichletNaturalParams(eta=dirichlet_eta)).E_log_pi
    
    # Compute value and gradients in single pass
    def ell(r, t1, t2, t3, t4, lp):
        return gmm_expected_log_likelihood(x, NIWSufficientStats(t1, t2, t3, t4), r, lp)
    
    E_log_lik, grads = jax.value_and_grad(ell, argnums=(0, 1, 2, 3, 4, 5))(
        resp, stats.E_T1, stats.E_T2, stats.E_T3, stats.E_T4, log_pi
    )
    d_resp, d_T1, d_T2, d_T3, d_T4, d_log_pi = grads
    
    # E-step: gradient gives log probabilities
    resp_new = jax.nn.softmax(jnp.log(resp + 1e-10) + lr_e * d_resp, axis=-1)
    
    # M-step: η_new = (1-lr)η_curr + lr(η_prior + data_stats)
    niw_eta_new = NIWNaturalParams(
        (1-lr_m) * niw_eta.eta1 + lr_m * (niw_eta_prior.eta1 + d_T1),
        (1-lr_m) * niw_eta.eta2 + lr_m * (niw_eta_prior.eta2 + d_T2),
        (1-lr_m) * niw_eta.eta3 + lr_m * (niw_eta_prior.eta3 + d_T3),
        (1-lr_m) * niw_eta.eta4 + lr_m * (niw_eta_prior.eta4 + d_T4),
    )
    
    dirichlet_eta_prior = dirichlet_prior_alpha - 1
    dirichlet_eta_new = (1-lr_m) * dirichlet_eta + lr_m * (dirichlet_eta_prior + d_log_pi)
    
    return niw_eta_new, dirichlet_eta_new, resp_new, E_log_lik


def gmm_param_gradient_step(
    x: jnp.ndarray,
    niw_eta: NIWNaturalParams,
    dirichlet_eta: jnp.ndarray,
    resp: jnp.ndarray,
    niw_eta_prior: NIWNaturalParams,
    dirichlet_prior_alpha: jnp.ndarray,
    lr_m: float = 1.0
) -> tuple:
    """
    M-step only: update parameters given fixed responsibilities.
    
    Uses gradient-based natural parameter updates. With the correct NIW natural
    parameterization (η3 = W⁻¹ + κmm'), the gradient approach is equivalent to VB-EM.
    
    Args:
        x: data, shape (N, D)
        niw_eta: current NIW natural parameters
        dirichlet_eta: current Dirichlet natural parameter (α - 1)
        resp: responsibilities (fixed), shape (N, K)
        niw_eta_prior: NIW prior (natural params, can broadcast)
        dirichlet_prior_alpha: Dirichlet prior α₀
        lr_m: M-step learning rate
        
    Returns:
        (niw_eta_new, dirichlet_eta_new, E_log_lik)
    """
    from src.gmm.dirichlet import DirichletNaturalParams, natural_to_expected_stats as dir_nat_to_stats
    
    stats = expected_stats(niw_eta)
    log_pi = dir_nat_to_stats(DirichletNaturalParams(eta=dirichlet_eta)).E_log_pi
    
    # Compute value and gradients for all params in single pass
    def ell(t1, t2, t3, t4, lp):
        return gmm_expected_log_likelihood(x, NIWSufficientStats(t1, t2, t3, t4), resp, lp)
    
    E_log_lik, grads = jax.value_and_grad(ell, argnums=(0, 1, 2, 3, 4))(
        stats.E_T1, stats.E_T2, stats.E_T3, stats.E_T4, log_pi
    )
    d_T1, d_T2, d_T3, d_T4, d_log_pi = grads
    
    # M-step: η_new = (1-lr_m)*η + lr_m*(η_prior + data_stats)
    niw_eta_new = NIWNaturalParams(
        (1-lr_m) * niw_eta.eta1 + lr_m * (niw_eta_prior.eta1 + d_T1),
        (1-lr_m) * niw_eta.eta2 + lr_m * (niw_eta_prior.eta2 + d_T2),
        (1-lr_m) * niw_eta.eta3 + lr_m * (niw_eta_prior.eta3 + d_T3),
        (1-lr_m) * niw_eta.eta4 + lr_m * (niw_eta_prior.eta4 + d_T4),
    )
    
    dirichlet_eta_prior = dirichlet_prior_alpha - 1
    dirichlet_eta_new = (1-lr_m) * dirichlet_eta + lr_m * (dirichlet_eta_prior + d_log_pi)
    
    return niw_eta_new, dirichlet_eta_new, E_log_lik


# =============================================================================
# High-Level Fitting Function
# =============================================================================

def fit_gmm(
    x: jnp.ndarray,
    K: int,
    n_iter: int = 50,
    lr_e: float = 1.0,
    lr_m: float = 1.0,
    scale: float = 1.0,
    prior_kappa: float = 0.1,
    prior_nu_offset: float = 0.5,
    prior_alpha: float = 0.5,
    em_iteration: bool = False,
    key: jax.random.PRNGKey = None
) -> tuple:
    """
    Fit a Variational Bayesian GMM.
    
    Args:
        x: data, shape (N, D)
        K: number of components
        n_iter: number of VB-EM iterations
        lr_e: E-step learning rate (1.0 = full update, only used when em_iteration=False)
        lr_m: M-step learning rate (1.0 = full update)
        scale: expected scale of data (default 1.0)
        prior_kappa: NIW prior κ (weak prior on mean)
        prior_nu_offset: NIW prior ν = D + prior_nu_offset (default 0.5, just above minimum)
        prior_alpha: Dirichlet prior α (< 1 encourages sparsity)
        em_iteration: if True, use standard EM (E-step then M-step separately);
                      if False, use simultaneous gradient update
        key: random key for initialization
        
    Returns:
        (niw_posterior, dirichlet_alpha, resp, elbo_history)
    """
    
    N, D = x.shape
    if key is None:
        key = jax.random.PRNGKey(0)
    
    # W controls expected precision: E[Λ] = ν * W
    # For K clusters in D dimensions, each cluster covers ~1/K of volume
    # Cluster linear scale ~ scale / K^(1/D), so variance ~ scale^2 / K^(2/D)
    # Expected precision ~ K^(2/D) / scale^2
    # With ν ≈ D + 0.5, set W = K^(2/D) / (scale^2 * ν) * I
    nu_prior = D + prior_nu_offset
    tiling_factor = K ** (2.0 / D)
    W_scale = tiling_factor / (scale ** 2 * nu_prior)
    
    # Set up priors - use broadcasting instead of tiling to save memory
    # Dirichlet needs explicit (K,) shape for KL computation
    dirichlet_prior_alpha = jnp.full(K, prior_alpha)
    niw_prior = NIWStandardParams(
        m=jnp.zeros(D),                    # (D,) broadcasts to (K, D)
        kappa=jnp.array(prior_kappa),      # scalar broadcasts to (K,)
        W=jnp.eye(D) * W_scale,            # (D, D) broadcasts to (K, D, D)
        nu=jnp.array(nu_prior)             # scalar broadcasts to (K,)
    )
    
    # Initialize means from random data points
    key, k1 = jax.random.split(key)
    idx = jax.random.choice(k1, N, shape=(K,), replace=False)
    init_means = x[idx]
    
    niw_init = NIWStandardParams(
        m=init_means,                                    # (K, D)
        kappa=jnp.full(K, prior_kappa + 1.0),           # (K,)
        W=jnp.tile(niw_prior.W[None], (K, 1, 1)),       # (K, D, D)
        nu=jnp.full(K, nu_prior + 1.0)                  # (K,)
    )
    
    # Work entirely in natural parameter space
    niw_eta_prior = standard_to_natural(niw_prior)
    niw_eta_init = standard_to_natural(niw_init)
    dirichlet_eta_init = jnp.full(K, dirichlet_prior_alpha - 1)  # (K,) shaped
    
    from src.gmm.dirichlet import (
        DirichletParams as DirParams, DirichletNaturalParams,
        natural_to_expected_stats as dir_nat_to_stats,
        kl_divergence as dirichlet_kl
    )
    
    if em_iteration:
        # Standard EM using lax.scan
        def em_step(carry, _):
            niw_eta, dirichlet_eta = carry
            
            # E-step: compute optimal responsibilities
            stats = expected_stats(niw_eta)
            log_pi = dir_nat_to_stats(DirichletNaturalParams(eta=dirichlet_eta)).E_log_pi
            resp, logZ = gmm_responsibilities(x, stats, log_pi)
            
            # Compute ELBO (E_log_lik + resp_entropy = sum(logZ))
            elbo = jnp.sum(logZ) - dirichlet_kl(DirParams(dirichlet_eta + 1), DirParams(dirichlet_prior_alpha)) - jnp.sum(kl_divergence(niw_eta, niw_eta_prior))
            
            # M-step: update parameters
            niw_eta_new, dirichlet_eta_new, _ = gmm_param_gradient_step(
                x, niw_eta, dirichlet_eta, resp,
                niw_eta_prior, dirichlet_prior_alpha, lr_m
            )
            
            return (niw_eta_new, dirichlet_eta_new), (elbo, resp)
        
        init_carry = (niw_eta_init, dirichlet_eta_init)
        (niw_eta, dirichlet_eta), (elbo_history, resp_history) = jax.lax.scan(
            em_step, init_carry, None, length=n_iter
        )
        resp = resp_history[-1]  # Final responsibilities
        
    else:
        # Simultaneous gradient update using lax.scan
        resp_init, _ = gmm_responsibilities(
            x, expected_stats(niw_eta_init), 
            dir_nat_to_stats(DirichletNaturalParams(eta=dirichlet_eta_init)).E_log_pi
        )
        
        def simul_step(carry, _):
            niw_eta, dirichlet_eta, resp = carry
            
            # Simultaneous gradient update
            niw_eta_new, dirichlet_eta_new, resp_new, _ = gmm_gradient_step(
                x, niw_eta, dirichlet_eta, resp,
                niw_eta_prior, dirichlet_prior_alpha, lr_e, lr_m
            )
            
            # Compute ELBO after update
            stats = expected_stats(niw_eta_new)
            log_pi = dir_nat_to_stats(DirichletNaturalParams(eta=dirichlet_eta_new)).E_log_pi
            E_log_lik = gmm_expected_log_likelihood(x, stats, resp_new, log_pi)
            resp_entropy = -jnp.sum(resp_new * jnp.log(resp_new + 1e-10))
            kl_dir = dirichlet_kl(DirParams(dirichlet_eta_new + 1), DirParams(dirichlet_prior_alpha))
            kl_niw = jnp.sum(kl_divergence(niw_eta_new, niw_eta_prior))
            elbo = E_log_lik + resp_entropy - kl_dir - kl_niw
            
            return (niw_eta_new, dirichlet_eta_new, resp_new), elbo
        
        init_carry = (niw_eta_init, dirichlet_eta_init, resp_init)
        (niw_eta, dirichlet_eta, resp), elbo_history = jax.lax.scan(
            simul_step, init_carry, None, length=n_iter
        )
    
    niw_post = natural_to_standard(niw_eta)
    return niw_post, dirichlet_eta + 1, resp, elbo_history

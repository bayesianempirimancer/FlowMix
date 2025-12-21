"""
Dirichlet distribution in exponential family form.

The Dirichlet is the conjugate prior for categorical/multinomial distributions,
used for mixing weights π in a GMM.

Exponential Family Form:
    p(π | α) ∝ ∏ₖ πₖ^(αₖ - 1) = exp(⟨η, T(π)⟩ - A(η))

where:
    Standard parameter: α = (α₁, ..., αₖ), αₖ > 0
    Natural parameter: η = α - 1
    Sufficient statistic: T(π) = log π
    Expected sufficient statistic: E[log πₖ] = ψ(αₖ) - ψ(α₀) where α₀ = Σₖ αₖ
    Log partition: A(η) = Σₖ log Γ(αₖ) - log Γ(α₀)

All functions support arbitrary batch shapes via [..., ] indexing.
The K components are always the last dimension.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import digamma, gammaln
from typing import Tuple, NamedTuple


class DirichletParams(NamedTuple):
    """Standard parameters of Dirichlet distribution."""
    alpha: jnp.ndarray  # concentration parameters, shape (..., K), all > 0


class DirichletNaturalParams(NamedTuple):
    """Natural parameters of Dirichlet distribution."""
    eta: jnp.ndarray  # η = α - 1, shape (..., K)


class DirichletSufficientStats(NamedTuple):
    """Expected sufficient statistics of Dirichlet distribution."""
    E_log_pi: jnp.ndarray  # E[log πₖ] = ψ(αₖ) - ψ(α₀), shape (..., K)


# =============================================================================
# Parameter Conversions
# =============================================================================

def standard_to_natural(params: DirichletParams) -> DirichletNaturalParams:
    """
    Convert standard Dirichlet parameters to natural parameters.
    
    η = α - 1
    
    Args:
        params: standard parameters with alpha shape (..., K)
        
    Returns:
        Natural parameters with same shape
    """
    return DirichletNaturalParams(eta=params.alpha - 1)


def natural_to_standard(eta: DirichletNaturalParams) -> DirichletParams:
    """
    Convert natural Dirichlet parameters to standard parameters.
    
    α = η + 1
    
    Args:
        eta: natural parameters with eta shape (..., K)
        
    Returns:
        Standard parameters with same shape
    """
    return DirichletParams(alpha=eta.eta + 1)


# =============================================================================
# Expected Sufficient Statistics
# =============================================================================

def expected_sufficient_stats(params: DirichletParams) -> DirichletSufficientStats:
    """
    Compute expected sufficient statistics E[T(π)] given standard parameters.
    
    E[log πₖ] = ψ(αₖ) - ψ(α₀)
    
    where α₀ = Σₖ αₖ and ψ is the digamma function.
    
    Args:
        params: standard parameters with alpha shape (..., K)
        
    Returns:
        Expected sufficient statistics with same shape
    """
    alpha = params.alpha
    alpha0 = jnp.sum(alpha, axis=-1, keepdims=True)  # (..., 1)
    E_log_pi = digamma(alpha) - digamma(alpha0)  # (..., K)
    return DirichletSufficientStats(E_log_pi=E_log_pi)


def natural_to_expected_stats(eta: DirichletNaturalParams) -> DirichletSufficientStats:
    """
    Compute expected sufficient statistics directly from natural parameters.
    
    This avoids creating intermediate DirichletParams objects.
    
    E[log πₖ] = ψ(ηₖ + 1) - ψ(Σₖ(ηₖ + 1))
    
    This is equivalent to ∇_η A(η) where A(η) is the log partition function.
    
    Args:
        eta: natural parameters with eta shape (..., K)
        
    Returns:
        Expected sufficient statistics with same batch shape
    """
    alpha = eta.eta + 1  # α = η + 1
    alpha0 = jnp.sum(alpha, axis=-1, keepdims=True)  # (..., 1)
    E_log_pi = digamma(alpha) - digamma(alpha0)  # (..., K)
    return DirichletSufficientStats(E_log_pi=E_log_pi)


# Alias for backwards compatibility
def expected_sufficient_stats_from_natural(eta: DirichletNaturalParams) -> DirichletSufficientStats:
    """Alias for natural_to_expected_stats for backwards compatibility."""
    return natural_to_expected_stats(eta)


def sufficient_stats_to_natural(
    stats: DirichletSufficientStats,
    alpha_init: float = 1.0,
    max_iter: int = 20,
    tol: float = 1e-8
) -> DirichletNaturalParams:
    """
    Recover natural parameters from expected sufficient statistics.
    
    Given E[log π], find α such that ψ(αₖ) - ψ(α₀) = E[log πₖ].
    
    This requires solving a nonlinear system. We use fixed-point iteration:
    α_new = invψ(E[log π] + ψ(α₀))
    
    where invψ is the inverse digamma function (approximated).
    
    Args:
        stats: expected sufficient statistics with E_log_pi shape (..., K)
        alpha_init: initial guess for α (scalar, broadcast to all components)
        max_iter: maximum iterations
        tol: convergence tolerance
        
    Returns:
        Natural parameters with same batch shape
    """
    E_log_pi = stats.E_log_pi
    K = E_log_pi.shape[-1]
    batch_shape = E_log_pi.shape[:-1]
    
    # Initialize α
    alpha = jnp.full(E_log_pi.shape, alpha_init)
    
    # Inverse digamma approximation (Newton's method inline)
    def inv_digamma(y):
        """Approximate inverse of digamma function."""
        # Initial guess from asymptotic expansion: ψ(x) ≈ log(x) - 1/(2x) for large x
        # So x ≈ exp(y) for y > -2.22
        # For small y: ψ(x) ≈ -1/x - γ, so x ≈ -1/(y + γ)
        x = jnp.where(
            y >= -2.22,
            jnp.exp(y) + 0.5,
            -1.0 / (y + 0.5772156649)
        )
        x = jnp.maximum(x, 1e-8)
        
        # Newton refinement: x_new = x - (ψ(x) - y) / ψ'(x)
        for _ in range(5):
            psi_x = digamma(x)
            # Trigamma ψ'(x) ≈ 1/x + 1/(2x²) for large x
            psi_prime_x = jax.scipy.special.polygamma(1, x)
            x = x - (psi_x - y) / psi_prime_x
            x = jnp.maximum(x, 1e-8)
        
        return x
    
    # Fixed-point iteration
    def fp_step(alpha, _):
        alpha0 = jnp.sum(alpha, axis=-1, keepdims=True)
        target = E_log_pi + digamma(alpha0)
        alpha_new = inv_digamma(target)
        return alpha_new, None
    
    alpha, _ = jax.lax.scan(fp_step, alpha, None, length=max_iter)
    
    return DirichletNaturalParams(eta=alpha - 1)


# =============================================================================
# Log Partition Function
# =============================================================================

def log_partition(params: DirichletParams) -> jnp.ndarray:
    """
    Compute log partition function A(α) of Dirichlet.
    
    A(α) = Σₖ log Γ(αₖ) - log Γ(α₀)
    
    where α₀ = Σₖ αₖ.
    
    Args:
        params: standard parameters with alpha shape (..., K)
        
    Returns:
        A(α), shape (...,)
    """
    alpha = params.alpha
    alpha0 = jnp.sum(alpha, axis=-1)  # (...,)
    return jnp.sum(gammaln(alpha), axis=-1) - gammaln(alpha0)


def log_partition_from_natural(eta: DirichletNaturalParams) -> jnp.ndarray:
    """Compute log partition from natural parameters."""
    return log_partition(natural_to_standard(eta))


# =============================================================================
# KL Divergence
# =============================================================================

def kl_divergence(
    q_params: DirichletParams,
    p_params: DirichletParams
) -> jnp.ndarray:
    """
    Compute KL divergence KL(q || p) between two Dirichlet distributions.
    
    KL(q || p) = log B(αₚ) - log B(αq) + Σₖ (αqₖ - αpₖ)(ψ(αqₖ) - ψ(αq₀))
    
    where B(α) = ∏ₖ Γ(αₖ) / Γ(Σₖ αₖ) is the multivariate beta function.
    
    Equivalently:
    KL(q || p) = A(αₚ) - A(αq) + ⟨αq - αp, E_q[log π]⟩
    
    Args:
        q_params: query distribution parameters, alpha shape (..., K)
        p_params: prior distribution parameters, same shape (broadcastable)
        
    Returns:
        KL divergence, shape (...,)
    """
    alpha_q = q_params.alpha
    alpha_p = p_params.alpha
    
    # Expected sufficient stats under q
    E_log_pi = expected_sufficient_stats(q_params).E_log_pi  # (..., K)
    
    # KL = A(p) - A(q) + <α_q - α_p, E_q[log π]>
    A_q = log_partition(q_params)
    A_p = log_partition(p_params)
    
    inner = jnp.sum((alpha_q - alpha_p) * E_log_pi, axis=-1)
    
    return A_p - A_q + inner


# =============================================================================
# GMM-specific functions
# =============================================================================

def expected_log_mixing_weights(params: DirichletParams) -> jnp.ndarray:
    """
    Compute E[log πₖ] for use in GMM.
    
    This is just the expected sufficient statistics.
    
    Args:
        params: Dirichlet parameters with alpha shape (..., K)
        
    Returns:
        E[log πₖ], shape (..., K)
    """
    return expected_sufficient_stats(params).E_log_pi


def update_from_responsibilities(
    prior: DirichletParams,
    responsibilities: jnp.ndarray
) -> DirichletParams:
    """
    Update Dirichlet parameters given responsibilities (M-step).
    
    In natural parameter form:
        η_post = η_prior + Σₙ rₙₖ = η_prior + Nₖ
    
    In standard form:
        α_post = α_prior + Nₖ
    
    Args:
        prior: prior Dirichlet parameters, alpha shape (..., K) or (K,)
        responsibilities: assignment probabilities, shape (..., N, K)
        
    Returns:
        Posterior Dirichlet parameters
    """
    # Nₖ = Σₙ rₙₖ
    N_k = jnp.sum(responsibilities, axis=-2)  # (..., K)
    
    # α_post = α_prior + Nₖ
    alpha_post = prior.alpha + N_k
    
    return DirichletParams(alpha=alpha_post)


def update_natural_from_responsibilities(
    prior_eta: DirichletNaturalParams,
    responsibilities: jnp.ndarray
) -> DirichletNaturalParams:
    """
    Update natural parameters given responsibilities (M-step).
    
    η_post = η_prior + Nₖ
    
    Args:
        prior_eta: prior natural parameters, eta shape (..., K) or (K,)
        responsibilities: assignment probabilities, shape (..., N, K)
        
    Returns:
        Posterior natural parameters
    """
    N_k = jnp.sum(responsibilities, axis=-2)  # (..., K)
    return DirichletNaturalParams(eta=prior_eta.eta + N_k)


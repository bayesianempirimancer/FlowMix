"""
MNIST Flow 2D Model with Flexible Loss Functions.

This model implements a simulation-free flow matching approach with three
equivalent loss formulations:
1. Velocity prediction (standard flow matching)
2. Noise prediction (diffusion-style)
3. Target prediction (direct)

All three are related via time-dependent affine transformations.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Sequence, Tuple, Any, Optional, Literal
from enum import Enum

# Encoders
from src.encoders.global_encoders.pointnet import PointNetEncoder
from src.encoders.local_encoders.transformer_set import TransformerSetEncoder
from src.encoders.local_encoders.slot_attention_encoder import SlotAttentionEncoder
from src.encoders.local_encoders.cross_attention_encoder import CrossAttentionEncoder
from src.encoders.local_encoders.gmm_featurizer import GMMFeaturizer
from src.encoders.local_encoders.dgcnn import DGCNN

# Pooling strategies to convert local -> global
from src.encoders.global_encoders.pooling import (
    MaxPoolingEncoder,
    MeanPoolingEncoder,
    AttentionPoolingEncoder,
)

from src.encoders.embeddings import SinusoidalTimeEmbedding

# CRNs
from src.models.global_crn import (
    GlobalAdaLNMLPCRN,
    GlobalDiTCRN,
    GlobalCrossAttentionCRN,
)
from src.models.local_crn import (
    LocalAdaLNMLPCRN,
    LocalDiTCRN,
)
from src.models.structured_crn import (
    StructuredAdaLNMLPCRN,
    StructuredCrossAttentionCRN,
)
from src.models.simple_latent_flow import SimpleLatentFlow


class PredictionTarget(str, Enum):
    """What quantity the network predicts."""
    VELOCITY = "velocity"  # Predict v(t, x_t) directly
    NOISE = "noise"        # Predict ε (like diffusion models)
    TARGET = "target"      # Predict x_1 (the data target)


class MnistFlow2D(nn.Module):
    """
    Flow model for 2D point clouds with flexible loss functions.
    
    Architecture:
    1. Encoder: x -> z (or x -> q(z|x) = N(μ, σ²) if use_vae=True)
    2. Sample: z ~ q(z|x) (if use_vae=True) or z directly
    3. CRN: Predicts velocity/noise/target given (x_t, z, t)
    
    Loss Functions (all equivalent via affine transformations):
    - velocity: v = x_1 - x_0
    - noise: ε where x_t = (1-t)x_0 + t·ε
    - target: x_1 (the data point)
    
    Args:
        latent_dim: Dimension of latent code z
        spatial_dim: Dimension of point coordinates (2 for 2D)
        encoder_type: Type of encoder ('pointnet', 'transformer', 'dgcnn', etc.)
        encoder_output_type: 'global' or 'local' (determines CRN type)
        encoder_kwargs: Kwargs for encoder
        crn_type: Type of CRN ('adaln_mlp', 'dit', 'cross_attention')
        crn_kwargs: Kwargs for CRN
        prediction_target: What the network predicts ('velocity', 'noise', 'target')
        loss_targets: List of targets to include in loss (for multi-objective)
        use_vae: If True, encoder outputs 2*latent_dim, split into (mu, logvar), compute KL loss
        use_prior_flow: Whether to learn p(z) with flow model
        prior_flow_kwargs: Kwargs for prior flow CRN
    """
    latent_dim: int = 128
    spatial_dim: int = 2
    encoder_type: str = 'pointnet'
    encoder_output_type: Literal['global', 'local'] = 'global'
    encoder_kwargs: dict = None
    crn_type: str = 'adaln_mlp'
    crn_kwargs: dict = None
    prediction_target: PredictionTarget = PredictionTarget.VELOCITY
    loss_targets: Sequence[PredictionTarget] = (PredictionTarget.VELOCITY,)
    use_vae: bool = False  # Enable VAE mode (mu/logvar split and KL loss)
    vae_kl_weight: float = 0.0  # Weight for VAE KL loss term
    marginal_kl_weight: float = 0.01  # Weight for marginal KL loss term
    use_prior_flow: bool = False
    prior_flow_kwargs: dict = None
    optimal_reweighting: bool = False  # Apply optimal time-dependent reweighting
    
    def setup(self):
        # Encoder
        enc_kwargs = self.encoder_kwargs or {}
        
        # Determine encoder output dimension
        # If use_vae=True, encoder outputs 2*latent_dim for (mu, logvar) split
        encoder_output_dim = 2 * self.latent_dim if self.use_vae else self.latent_dim
        
        if self.encoder_type == 'pointnet':
            # PointNet is inherently global
            self.encoder = PointNetEncoder(latent_dim=encoder_output_dim, **enc_kwargs)
            self._encoder_is_global = True
        elif self.encoder_type == 'transformer':
            # TransformerSetEncoder is local, wrap if needed
            base_encoder = TransformerSetEncoder(embed_dim=self.latent_dim, **enc_kwargs)
            if self.encoder_output_type == 'global':
                self.encoder = MaxPoolingEncoder(base_encoder, encoder_output_dim)
                self._encoder_is_global = True
            else:
                self.encoder = base_encoder
                self._encoder_is_global = False
        elif self.encoder_type == 'dgcnn':
            base_encoder = DGCNN(embed_dim=self.latent_dim, **enc_kwargs)
            if self.encoder_output_type == 'global':
                self.encoder = MaxPoolingEncoder(base_encoder, encoder_output_dim)
                self._encoder_is_global = True
            else:
                self.encoder = base_encoder
                self._encoder_is_global = False
        elif self.encoder_type == 'slot_attention':
            # Slot Attention is local (outputs K slots)
            base_encoder = SlotAttentionEncoder(slot_dim=encoder_output_dim, **enc_kwargs)
            if self.encoder_output_type == 'global':
                self.encoder = MaxPoolingEncoder(base_encoder, encoder_output_dim)
                self._encoder_is_global = True
            else:
                self.encoder = base_encoder
                self._encoder_is_global = False
        elif self.encoder_type == 'cross_attention':
            # CrossAttentionEncoder is local (outputs K latent queries)
            base_encoder = CrossAttentionEncoder(latent_dim=encoder_output_dim, **enc_kwargs)
            if self.encoder_output_type == 'global':
                self.encoder = MaxPoolingEncoder(base_encoder, encoder_output_dim)
                self._encoder_is_global = True
            else:
                self.encoder = base_encoder
                self._encoder_is_global = False
        elif self.encoder_type == 'gmm':
            # GMM is local (outputs K components)
            base_encoder = GMMFeaturizer(latent_dim=encoder_output_dim, **enc_kwargs)
            if self.encoder_output_type == 'global':
                self.encoder = MaxPoolingEncoder(base_encoder, encoder_output_dim)
                self._encoder_is_global = True
            else:
                self.encoder = base_encoder
                self._encoder_is_global = False
        else:
            raise ValueError(f"Unknown encoder type: {self.encoder_type}")
        
        # CRN (choose based on encoder output type)
        crn_kwargs = self.crn_kwargs or {}
        
        if self._encoder_is_global:
            # Global CRNs: context is (B, Dc)
            if self.crn_type == 'adaln_mlp':
                self.crn = GlobalAdaLNMLPCRN(**crn_kwargs)
            elif self.crn_type == 'dit':
                self.crn = GlobalDiTCRN(**crn_kwargs)
            elif self.crn_type == 'cross_attention':
                self.crn = GlobalCrossAttentionCRN(**crn_kwargs)
            else:
                raise ValueError(f"Unknown CRN type: {self.crn_type}")
        else:
            # Local/Structured CRNs: context is (B, K, Dc)
            # For now, use Structured (pool-based) as default
            # TODO: Add option to use Local CRNs when sampling K points
            if self.crn_type == 'adaln_mlp':
                self.crn = StructuredAdaLNMLPCRN(**crn_kwargs)
            elif self.crn_type == 'cross_attention':
                self.crn = StructuredCrossAttentionCRN(**crn_kwargs)
            else:
                raise ValueError(f"CRN type {self.crn_type} not supported for local encoders")
        
        # Prior Flow (optional)
        # Use SimpleLatentFlow for prior flow (simple MLP designed for latent vectors)
        if self.use_prior_flow:
            prior_kwargs = dict(self.prior_flow_kwargs or {})
            # Default hidden dims if not specified
            if 'hidden_dims' not in prior_kwargs:
                prior_kwargs['hidden_dims'] = (256, 256, 256)
            self.prior_vector_field = SimpleLatentFlow(**prior_kwargs)
    
    def __call__(self, x: jnp.ndarray, key: jax.random.PRNGKey, 
                 mask: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, dict]:
        return self.compute_loss(x, key, mask=mask)
    
    def reparameterize(self, mu: jnp.ndarray, logvar: jnp.ndarray, 
                       key: jax.random.PRNGKey) -> jnp.ndarray:
        """Sample z ~ N(μ, σ²) using reparameterization trick."""
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, mu.shape)
        return mu + eps * std
    
    @nn.compact
    def encode(self, x: jnp.ndarray, key: jax.random.PRNGKey,
               mask: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray]]:
        """
        Encode input to latent code.
        
        If use_vae=True:
            Encoder outputs 2*latent_dim, split into (mu, logvar)
            Returns: z ~ N(mu, exp(logvar)), mu, logvar
        
        If use_vae=False:
            Encoder outputs latent_dim directly
            Returns: z, None, None
        """
        # Some encoders don't take key parameter
        try:
            z_encoded = self.encoder(x, mask=mask, key=key)
        except TypeError:
            try:
                z_encoded = self.encoder(x, mask=mask)
            except TypeError:
                z_encoded = self.encoder(x)
        
        if self.use_vae:
            # Split into mu and logvar
            # Shape: (B, 2*D) -> (B, D), (B, D)
            # or: (B, K, 2*D) -> (B, K, D), (B, K, D)
            z_mu, z_logvar = jnp.split(z_encoded, 2, axis=-1)
            # Normalize z_mu to unit vector with numerical stability
            z_mu_norm = jnp.sqrt(jnp.sum(z_mu**2, axis=-1, keepdims=True) + 1e-8)
            z_mu = z_mu / z_mu_norm
            z = self.reparameterize(z_mu, z_logvar, key)
            return z, z_mu, z_logvar
        else:
            # No VAE: use encoder output directly
            # Normalize to unit vector with numerical stability
            if z_encoded.ndim == 3:
                # Local: (B, K, D) - normalize across last dimension
                z_encoded_norm = jnp.sqrt(jnp.sum(z_encoded**2, axis=-1, keepdims=True) + 1e-8)
                z_encoded = z_encoded / z_encoded_norm
            else:
                # Global: (B, D) - normalize across last dimension
                z_encoded_norm = jnp.sqrt(jnp.sum(z_encoded**2, axis=-1, keepdims=True) + 1e-8)
                z_encoded = z_encoded / z_encoded_norm
            return z_encoded, None, None
    
    def convert_crn_output_to_velocity(self, crn_output: jnp.ndarray, 
                                       x_t: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        Convert CRN output to velocity field.
        
        Flow matching interpolation: x_t = (1-t)·x_0 + t·x_1
        Velocity field: v = dx/dt = x_1 - x_0
        
        From x_t and CRN output, we can compute velocity:
        - If predicting v: return v directly
        - If predicting noise ε (where ε = x_1): v = (ε - x_t) / (1-t)
        - If predicting target x_0: v = (x_t - x_0) / t
        
        Args:
            crn_output: CRN output, shape (B, N, D)
            x_t: Interpolated point at time t, shape (B, N, D)
            t: Time, shape (B, 1, 1) or (B, 1) or (B,)
        
        Returns:
            Velocity v, shape (B, N, D)
        """
        # Ensure t has correct shape for broadcasting
        if t.ndim == 1:
            t = t[:, None, None]  # (B,) -> (B, 1, 1)
        elif t.ndim == 2 and t.shape[-1] == 1:
            t = t[:, :, None]  # (B, 1) -> (B, 1, 1)
        
        if self.prediction_target == PredictionTarget.VELOCITY:
            # Already velocity
            return crn_output
        elif self.prediction_target == PredictionTarget.NOISE:
            # crn_output = ε = x_1
            # x_t = (1-t)·x_0 + t·ε
            # v = ε - x_0 = (ε - x_t) / (1-t)
            return (crn_output - x_t) / (1 - t + 1e-8)
        elif self.prediction_target == PredictionTarget.TARGET:
            # crn_output = x_0
            # x_t = (1-t)·x_0 + t·x_1
            # x_t = x_0 + t·v  (since x_1 = x_0 + v)
            # v = (x_t - x_0) / t
            return (x_t - crn_output) / (t + 1e-8)
        else:
            raise ValueError(f"Unknown prediction target: {self.prediction_target}")
    
    def convert_crn_output_to_noise(self, crn_output: jnp.ndarray,
                                    x_t: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        Convert CRN output to noise (ε = x_1).
        
        Args:
            crn_output: CRN output, shape (B, N, D)
            x_t: Interpolated point at time t, shape (B, N, D)
            t: Time, shape (B, 1, 1) or (B, 1) or (B,)
        
        Returns:
            Noise ε = x_1, shape (B, N, D)
        """
        # Ensure t has correct shape for broadcasting
        if t.ndim == 1:
            t = t[:, None, None]
        elif t.ndim == 2 and t.shape[-1] == 1:
            t = t[:, :, None]
        
        if self.prediction_target == PredictionTarget.NOISE:
            # Already noise
            return crn_output
        elif self.prediction_target == PredictionTarget.VELOCITY:
            # crn_output = v = x_1 - x_0
            # x_t = (1-t)·x_0 + t·x_1
            # x_1 = x_t + (1-t)·v
            return x_t + (1 - t) * crn_output
        elif self.prediction_target == PredictionTarget.TARGET:
            # crn_output = x_0
            # x_t = (1-t)·x_0 + t·x_1
            # x_1 = (x_t - (1-t)·x_0) / t
            return (x_t - (1 - t) * crn_output) / (t + 1e-8)
        else:
            raise ValueError(f"Unknown prediction target: {self.prediction_target}")
    
    def convert_crn_output_to_target(self, crn_output: jnp.ndarray,
                                     x_t: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        Convert CRN output to target (x_0, the data point).
        
        Args:
            crn_output: CRN output, shape (B, N, D)
            x_t: Interpolated point at time t, shape (B, N, D)
            t: Time, shape (B, 1, 1) or (B, 1) or (B,)
        
        Returns:
            Target x_0 (data point), shape (B, N, D)
        """
        # Ensure t has correct shape for broadcasting
        if t.ndim == 1:
            t = t[:, None, None]
        elif t.ndim == 2 and t.shape[-1] == 1:
            t = t[:, :, None]
        
        if self.prediction_target == PredictionTarget.TARGET:
            # Already target (x_0)
            return crn_output
        elif self.prediction_target == PredictionTarget.VELOCITY:
            # crn_output = v = x_1 - x_0 
            # x_t = (1-t)·x_0 + t·x_1
            # x_t = (1-t)·x_0 + t·(x_0 + v) = x_0 + t·v
            # x_0 = x_t - t·v
            return x_t - t * crn_output
        elif self.prediction_target == PredictionTarget.NOISE:
            # crn_output = ε = x_1
            # x_t = (1-t)·x_0 + t·ε
            # x_0 = (x_t - t·ε) / (1-t)
            return (x_t - t * crn_output) / (1 - t + 1e-8)
        else:
            raise ValueError(f"Unknown prediction target: {self.prediction_target}")
    
    def compute_crn_jacobian_trace(self, x_t: jnp.ndarray, z: jnp.ndarray, 
                                   t: jnp.ndarray, mask: Optional[jnp.ndarray] = None,
                                   hutchinson_samples: int = 1) -> jnp.ndarray:
        """
        Compute trace of Jacobian of CRN output w.r.t. x_t using Hutchinson's estimator.
        
        Hutchinson's trace estimator: trace(J) ≈ E[v^T J v] where v ~ Rademacher(±1)
        
        This is much more efficient than computing the full Jacobian:
        - Full Jacobian: O(D²) time and memory per point
        - Hutchinson: O(D) time and memory per point
        
        For 2D (D=2), both are fast. For higher dimensions, Hutchinson is essential.
        
        Args:
            x_t: Input points, shape (B, N, D)
            z: Latent context, shape (B, Dc) or (B, K, Dc)
            t: Time, scalar or (B,) or (B, 1)
            mask: Optional mask, shape (B, N)
            hutchinson_samples: Number of random samples (1 is usually sufficient)
        
        Returns:
            Trace of Jacobian for each point, shape (B, N)
        """
        B, N, D = x_t.shape
        
        # Define CRN function for a single point
        def crn_single(x_point, z_ctx, t_val, mask_val):
            """
            CRN output for a single point.
            x_point: (D,), z_ctx: (Dc,) or (K, Dc), t_val: (), mask_val: ()
            Returns: (D,)
            """
            x_reshaped = x_point[None, None, :]  # (1, 1, D)
            t_reshaped = jnp.array([[t_val]])  # (1, 1)
            
            if self._encoder_is_global:
                z_reshaped = z_ctx[None, :]  # (1, Dc)
                out = self.crn(x_reshaped, z_reshaped, t_reshaped)
            else:
                z_reshaped = z_ctx[None, :, :] if z_ctx.ndim == 2 else z_ctx[None, None, :]
                mask_reshaped = jnp.array([[mask_val]]) if mask_val is not None else None
                out = self.crn(x_reshaped, z_reshaped, t_reshaped, mask=mask_reshaped)
            
            return out[0, 0, :]  # (D,)
        
        # Hutchinson's trace estimator
        def hutchinson_trace(x_point, z_ctx, t_val, mask_val, key):
            """Estimate trace(J) using Hutchinson's method."""
            def single_sample(subkey):
                # Random vector with ±1 entries (Rademacher distribution)
                v = jax.random.rademacher(subkey, (D,), dtype=x_point.dtype)
                
                # Compute Jacobian-vector product: J @ v using forward-mode AD
                _, jvp_result = jax.jvp(
                    lambda x: crn_single(x, z_ctx, t_val, mask_val),
                    (x_point,),
                    (v,)
                )
                
                # Estimate: v^T (J @ v) = sum(v * (J @ v))
                return jnp.sum(v * jvp_result)
            
            # Average over multiple samples
            subkeys = jax.random.split(key, hutchinson_samples)
            traces = jax.vmap(single_sample)(subkeys)
            return jnp.mean(traces)
        
        # Prepare inputs for vmap
        if jnp.isscalar(t):
            t_array = jnp.full((B, N), t)
        elif t.ndim == 1:
            t_array = jnp.tile(t[:, None], (1, N))
        elif t.ndim == 2 and t.shape[-1] == 1:
            t_array = jnp.tile(t, (1, N))
        else:
            t_array = t
        
        # Prepare z for vmap
        if z.ndim == 2:  # Global (B, Dc)
            z_array = jnp.tile(z[:, None, :], (1, N, 1))
        else:  # Local/Structured (B, K, Dc)
            z_array = jnp.tile(z[:, None, :, :], (1, N, 1, 1))
        
        # Prepare mask
        mask_array = jnp.ones((B, N)) if mask is None else mask
        
        # Generate random keys for each point
        key = jax.random.PRNGKey(0)  # Fixed seed for deterministic estimation
        keys = jax.random.split(key, B * N).reshape(B, N, -1)
        
        # Vmap over batch and points
        trace_vmap = jax.vmap(
            jax.vmap(hutchinson_trace, in_axes=(0, 0, 0, 0, 0)),
            in_axes=(0, 0, 0, 0, 0)
        )
        
        all_traces = trace_vmap(x_t, z_array, t_array, mask_array, keys)
        return all_traces  # (B, N)
    
    def compute_velocity_divergence(self, x_t: jnp.ndarray, z: jnp.ndarray,
                                    t: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Compute divergence of the velocity field.
        
        The velocity field is v(x_t, t) and its divergence depends on what the CRN predicts:
        
        1. If predicting velocity directly:
           v = crn_output
           div(v) = trace(J_crn)
        
        2. If predicting noise ε:
           v = (ε - x_t) / (1-t)
           div(v) = D / (1-t) - trace(J_crn) / (1-t)
                  = (D - trace(J_crn)) / (1-t)
        
        3. If predicting target x_0:
           v = (x_t - x_0) / t
           div(v) = D / t - trace(J_crn) / t
                  = (D - trace(J_crn)) / t
        
        Args:
            x_t: Input points, shape (B, N, D)
            z: Latent context, shape (B, Dc) or (B, K, Dc)
            t: Time, scalar or (B,) or (B, 1)
            mask: Optional mask, shape (B, N)
        
        Returns:
            Divergence of velocity field, shape (B, N)
        """
        B, N, D = x_t.shape
        
        # Compute trace of CRN Jacobian
        trace_jac_crn = self.compute_crn_jacobian_trace(x_t, z, t, mask)  # (B, N)
        
        # Ensure t has correct shape for broadcasting
        if jnp.isscalar(t):
            t_broadcast = jnp.full((B, N), t)
        elif t.ndim == 1:
            t_broadcast = jnp.tile(t[:, None], (1, N))  # (B, N)
        elif t.ndim == 2 and t.shape[-1] == 1:
            t_broadcast = jnp.tile(t, (1, N))  # (B, N)
        else:
            t_broadcast = t
        
        if self.prediction_target == PredictionTarget.VELOCITY:
            # v = crn_output
            # div(v) = trace(J_crn)
            return trace_jac_crn
        
        elif self.prediction_target == PredictionTarget.NOISE:
            # v = (ε - x_t) / (1-t)
            # dv_i/dx_i = (dε_i/dx_i - 1) / (1-t)
            # div(v) = sum_i dv_i/dx_i = (trace(J_ε) - D) / (1-t)
            # Since ε doesn't depend on x_t directly through identity:
            # div(v) = (trace(J_crn) - D) / (1-t)
            return (trace_jac_crn - D) / (1 - t_broadcast + 1e-8)
        
        elif self.prediction_target == PredictionTarget.TARGET:
            # v = (x_t - x_0) / t
            # dv_i/dx_i = (1 - dx0_i/dx_i) / t
            # div(v) = (D - trace(J_x0)) / t
            return (D - trace_jac_crn) / (t_broadcast + 1e-8)
        
        else:
            raise ValueError(f"Unknown prediction target: {self.prediction_target}")
    
    def compute_loss(self, x: jnp.ndarray, key: jax.random.PRNGKey,
                    mask: Optional[jnp.ndarray] = None,
                    encoder_mask: Optional[jnp.ndarray] = None,
                    compute_prior_loss: bool = True,
                    prior_only_mode: bool = False) -> Tuple[jnp.ndarray, dict]:
        """
        Compute loss with flexible target formulations.
        
        Args:
            x: Input point cloud, shape (B, N, D) - full point cloud for flow training
            key: Random key
            mask: Optional mask for flow training, shape (B, N) (currently unused, flow uses full x)
            encoder_mask: Optional mask for encoder, shape (B, N) - masked points generate z
        
        Returns:
            loss: Total loss
            metrics: Dictionary of metrics
        """
        if self.use_prior_flow:
            k_sample, k_flow, k_prior_flow = jax.random.split(key, 3)
        else:
            k_sample, k_flow = jax.random.split(key)
        
        # 1. Encode using encoder_mask (masked points generate z)
        # If encoder_mask is provided, use it; otherwise use mask (for backward compatibility)
        enc_mask = encoder_mask if encoder_mask is not None else mask
        z, mu_z, logvar_z = self.encode(x, k_sample, mask=enc_mask)
        
        # 2. Flow Matching Setup
        B, N, D = x.shape
        x_0 = x  # Data
        # Split key to get independent randomness for noise and time
        k_noise, k_time = jax.random.split(k_flow, 2)
        x_1 = jax.random.normal(k_noise, x.shape)  # Noise/target
        
        # Sample time
        t = jax.random.uniform(k_time, (B,))
        
        # Interpolate: x_t = (1-t)·x_0 + t·x_1
        t_exp = t[:, None, None]  # (B, 1, 1)
        x_t = (1 - t_exp) * x_0 + t_exp * x_1
                
        # 3. CRN Output
        # Global CRNs don't take mask, Structured CRNs do
        if self._encoder_is_global:
            crn_output = self.crn(x_t, z, t)
        else:
            crn_output = self.crn(x_t, z, t, mask=mask)
        
        # 4. Compute losses efficiently using affine relationships
        # Compute base loss in CRN's prediction space, then apply time-dependent weights
        # for additional loss terms if requested.
        

        loss_weight = 0.0
        
        # Compute base squared error in CRN's native prediction space (per point)
        # Mean over feature dimension to save memory: (B, N, D) -> (B, N)
        if self.prediction_target == PredictionTarget.VELOCITY:
            sq_err = jnp.mean((crn_output - (x_1 - x_0)) ** 2, axis=-1)  # (B, N)
            for target_type in self.loss_targets:
                if target_type == PredictionTarget.VELOCITY:
                    loss_weight = loss_weight + 1.0
                elif target_type == PredictionTarget.NOISE:
                    loss_weight = loss_weight + (1 - t) ** 2
                elif target_type == PredictionTarget.TARGET:
                    loss_weight = loss_weight + t ** 2

        elif self.prediction_target == PredictionTarget.NOISE:
            sq_err = jnp.mean((crn_output - x_1) ** 2, axis=-1)  # (B, N)
            for target_type in self.loss_targets:
                if target_type == PredictionTarget.VELOCITY:
                    loss_weight = loss_weight + 1.0 / ((1 - t) ** 2 + 1e-8)
                elif target_type == PredictionTarget.NOISE:
                    loss_weight = loss_weight + 1.0
                elif target_type == PredictionTarget.TARGET:
                    loss_weight = loss_weight + (t / (1 - t + 1e-8)) ** 2

        elif self.prediction_target == PredictionTarget.TARGET:
            sq_err = jnp.mean((crn_output - x_0) ** 2, axis=-1)  # (B, N)
            for target_type in self.loss_targets:
                if target_type == PredictionTarget.VELOCITY:
                    loss_weight = loss_weight + 1.0 / (t ** 2 + 1e-8)
                elif target_type == PredictionTarget.NOISE:
                    loss_weight = loss_weight + (1 - t)**2 / (t**2 + 1e-8)
                elif target_type == PredictionTarget.TARGET:
                    loss_weight = loss_weight + 1.0

        else:
            raise ValueError(f"Unknown prediction target: {self.prediction_target}")
        
        # If VAE is enabled, ensure target loss is included
        if self.use_vae:
            if PredictionTarget.TARGET not in self.loss_targets:
                if self.prediction_target == PredictionTarget.VELOCITY:
                    loss_weight = loss_weight + t ** 2
                elif self.prediction_target == PredictionTarget.NOISE:
                    loss_weight = loss_weight + ((1 - t) / (t + 1e-8)) ** 2
                elif self.prediction_target == PredictionTarget.TARGET:
                    loss_weight = loss_weight + 1.0

        # Sum over spatial dimension first, then apply weight
        if mask is not None:
            sq_err = jnp.sum(sq_err * mask, axis=-1) / (jnp.sum(mask, axis=-1) + 1e-10)  # (B, N) -> (B,)
        else:
            sq_err = jnp.mean(sq_err, axis=-1)  # (B, N) -> (B,)
        
        # Apply time-dependent weight per sample (optimal reweighting for flow matching)
        if self.optimal_reweighting:
            if self.prediction_target == PredictionTarget.VELOCITY:
                # Optimal weight: (1-t)/t for velocity prediction
                loss_weight = loss_weight * (1 - t) / (t + 1e-8)
            elif self.prediction_target == PredictionTarget.NOISE:
                # Optimal weight: 1/((1-t)*t) for noise prediction
                loss_weight = loss_weight / ((1 - t + 1e-8) * (t + 1e-8))
            elif self.prediction_target == PredictionTarget.TARGET:
                # Optimal weight: (1-t)/t^3 for target prediction
                loss_weight = loss_weight * (1 - t) / (t**3 + 1e-8)
            else:
                raise ValueError(f"Unknown prediction target: {self.prediction_target}")

        flow_loss = jnp.mean(sq_err * loss_weight)  # (B,) * (B,) -> (B,) -> scalar

        
        # 5. VAE KL: KL(q(z|x) || N(0,I)) - only if use_vae=True
        if self.use_vae:
            # Shape depends on encoder type:
            # - Global: (B, Dc) -> (B,)
            # - Local: (B, K, Dc) -> (B, K) -> (B,)
            vae_kl = jnp.exp(logvar_z) + jnp.square(mu_z) - 1.0 - logvar_z            
            vae_kl = 0.5*jnp.mean(vae_kl)
        else:
            vae_kl = 0.0
        
        # 5b. Marginal KL: KL(p(z) || N(0,I)) where p(z) = E_x[q(z|x)]
        # This ensures the marginal distribution of z is unit normal
        # Compute this even when use_vae=False to encourage unit normal marginals
        # Compute empirical mean and variance of z in batch
        # z shape: (B, D) for global, (B, K, D) for local
        if z.ndim == 3:
            # Local: average over batch and K
            z_mean_empirical = jnp.mean(z, axis=(0, 1))  # (D,)
            z_var_empirical = jnp.var(z, axis=(0, 1))    # (D,)
        else:
            # Global: average over batch
            z_mean_empirical = jnp.mean(z, axis=0)  # (D,)
            z_var_empirical = jnp.var(z, axis=0)    # (D,)
        
        # KL divergence: KL(N(μ_emp, σ²_emp) || N(0, I))
        # = 0.5 * [tr(Σ) + μ^T μ - D - log|Σ|]
        # where Σ is diagonal with σ²_emp
        
        marginal_kl = (z_var_empirical + z_mean_empirical**2 - 1.0 - jnp.log(z_var_empirical + 1e-8))
        marginal_kl = 0.5 * jnp.mean(marginal_kl)
        
        # 6. Prior Flow (optional) - only compute if needed for gradients
        if self.use_prior_flow and compute_prior_loss:
            # Train prior flow: mu_z -> N(0, I)
            # Use SimpleLatentFlow which takes (mu_z, t) directly
            # IMPORTANT: Use mu_z (mean) instead of z (sampled) for prior flow training
            
            # Get mu_z - if VAE is enabled, use mu_z; otherwise use z as mu_z
            if self.use_vae and mu_z is not None:
                mu_z_for_prior = mu_z
            else:
                # If not VAE, z is deterministic, so use it as mu_z
                mu_z_for_prior = z
            
            # Split key to get independent randomness for noise and time
            k_noise, k_time_prior = jax.random.split(k_prior_flow, 2)
            z_prior_noise = jax.random.normal(k_noise, mu_z_for_prior.shape)
            time_prior = jax.random.uniform(k_time_prior, (B,))
            
            # Stop gradient on mu_z to prevent backprop through encoder
            # The prior flow should only train the prior_vector_field, not the encoder
            mu_z_stopped = jax.lax.stop_gradient(mu_z_for_prior)
            
            # Interpolate mu_z (using stopped gradient version)
            if mu_z_stopped.ndim == 3:
                # Local: (B, K, Dc)
                time_prior_exp = time_prior[:, None, None]  # (B, 1, 1)
            else:
                # Global: (B, Dc)
                time_prior_exp = time_prior[:, None]  # (B, 1)
            
            mu_z_t = (1 - time_prior_exp) * mu_z_stopped + time_prior_exp * z_prior_noise
            
            # SimpleLatentFlow takes (mu_z, t) directly - no context needed
            crn_output_prior = self.prior_vector_field(mu_z_t, time_prior)  # (B, D) or (B, K, D)
            target_v_prior = z_prior_noise - mu_z_stopped
            
            prior_flow_loss = jnp.mean((crn_output_prior - target_v_prior) ** 2)
            
            # In prior_only_mode, exclude flow_loss from total loss since encoder/CRN are frozen
            if prior_only_mode:
                loss = prior_flow_loss + self.vae_kl_weight * vae_kl + self.marginal_kl_weight * marginal_kl
            else:
                loss = flow_loss + prior_flow_loss + self.vae_kl_weight * vae_kl + self.marginal_kl_weight * marginal_kl
            
            metrics = {"flow_loss": flow_loss, 
                      "prior_flow_loss": prior_flow_loss, 
                      "vae_kl": vae_kl,
                      "marginal_kl": marginal_kl}
        elif self.use_prior_flow and not compute_prior_loss:
            # Prior flow is enabled but we're not computing loss (prior is frozen)
            loss = flow_loss + self.vae_kl_weight * vae_kl + self.marginal_kl_weight * marginal_kl
            metrics = {"flow_loss": flow_loss, 
                      "prior_flow_loss": 0.0,  # Set to 0 to indicate it wasn't computed
                      "vae_kl": vae_kl,
                      "marginal_kl": marginal_kl}
        else:
            loss = flow_loss + self.vae_kl_weight * vae_kl + self.marginal_kl_weight * marginal_kl
            metrics = {"flow_loss": flow_loss, "vae_kl": vae_kl, "marginal_kl": marginal_kl}
        
        return loss, metrics
    
    def sample(self, num_points: int, key: jax.random.PRNGKey,
               z: Optional[jnp.ndarray] = None, num_steps: int = 20) -> jnp.ndarray:
        """
        Sample points by integrating the flow from t=1 (noise) to t=0 (data).
        
        Args:
            num_points: Number of points to sample
            key: Random key
            z: Optional latent code (if None, sample from prior)
            num_steps: Number of integration steps
        
        Returns:
            Sampled points, shape (num_points, spatial_dim)
        """
        if z is None:
            if self.use_prior_flow:
                # Sample from learned prior using GlobalAdaLNMLPCRN
                key, k_prior = jax.random.split(key)
                if self._encoder_is_global:
                    z_prior_noise = jax.random.normal(k_prior, (1, self.latent_dim))
                else:
                    # For local encoders, need to know K
                    # TODO: Make this configurable
                    K = 8  # Default
                    z_prior_noise = jax.random.normal(k_prior, (1, K, self.latent_dim))
                
                dt_prior = -1.0 / num_steps
                
                def prior_step(z_t, step_idx):
                    t = 1.0 + step_idx * dt_prior
                    
                    # SimpleLatentFlow takes (z, t) directly - no context needed
                    # t should be scalar for single sample
                    t_scalar = t if isinstance(t, (int, float)) else (t[0] if t.ndim > 0 else t)
                    t_batch = jnp.array([t_scalar])  # (1,)
                    
                    v = self.prior_vector_field(z_t, t_batch)  # (1, D) or (1, K, D)
                    
                    return z_t + dt_prior * v, None
                
                z, _ = jax.lax.scan(prior_step, z_prior_noise, jnp.arange(num_steps))
            else:
                # Sample from standard normal
                if self._encoder_is_global:
                    z = jax.random.normal(key, (1, self.latent_dim))
                else:
                    K = 8  # Default
                    z = jax.random.normal(key, (1, K, self.latent_dim))

        z_sample = z/jnp.sqrt(jnp.sum(z**2, axis=-1, keepdims=True) + 1e-8)
        
        # Initialize from noise
        key, k_points = jax.random.split(key)
        x_init = jax.random.normal(k_points, (1, num_points, self.spatial_dim))
        
        # Integrate from t=1 to t=0
        dt = -1.0 / num_steps
        
        def euler_step(x_t, step_idx):
            t = 1.0 + step_idx * dt
            # Get CRN output
            if self._encoder_is_global:
                crn_output = self.crn(x_t, z_sample, t)
            else:
                crn_output = self.crn(x_t, z_sample, t)
            
            # Convert CRN output to velocity for Euler integration
            # The CRN may predict velocity, noise, or target - we need velocity for integration
            v = self.convert_crn_output_to_velocity(crn_output, x_t, t)
            
            return x_t + dt * v, None
        
        x_final, _ = jax.lax.scan(euler_step, x_init, jnp.arange(num_steps))
        
        return x_final[0]

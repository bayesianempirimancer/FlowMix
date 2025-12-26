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

# Encoder factory
from src.encoders.encoder_factory import create_encoder

from src.encoders.embeddings import SinusoidalTimeEmbedding

# CRN factory
from src.models.crn_factory import create_crn
from src.models.simple_latent_flow import AdaLNLatentFlow


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
    latent_dim: int = 64
    num_latents: int = 1
    spatial_dim: int = 2
    encoder_type: str = 'pointnet'
    encoder_output_type: Literal['global', 'local'] = 'global'
    encoder_kwargs: dict = None
    crn_type: str = 'adaln_mlp'
    crn_kwargs: dict = None
    prediction_target: PredictionTarget = PredictionTarget.VELOCITY
    loss_targets: Sequence[PredictionTarget] = (PredictionTarget.VELOCITY,)
    normalize_z: bool = True
    use_vae: bool = False  # Enable VAE mode (mu/logvar split and KL loss)
    vae_kl_weight: float = 0.0  # Weight for VAE KL loss term
    marginal_kl_weight: float = 0.0  # Weight for marginal KL loss term
    use_prior_flow: bool = False
    prior_flow_kwargs: dict = None
    optimal_reweighting: bool = False  # Apply optimal time-dependent reweighting
    
    def setup(self):
        # Encoder
        # Determine encoder output dimension
        # If use_vae=True, encoder outputs 2*latent_dim for (mu, logvar) split
        encoder_output_dim = 2 * self.latent_dim if self.use_vae else self.latent_dim
        
        # Use encoder factory to create encoder
        # Pass latent_dim separately for encoders that use embed_dim internally (transformer, dgcnn)
        self.encoder, self._encoder_is_global = create_encoder(
            encoder_type=self.encoder_type,
            encoder_output_type=self.encoder_output_type,
            encoder_output_dim=encoder_output_dim,
            encoder_kwargs=self.encoder_kwargs,
            pooling_type='max',  # Default to max pooling for local->global conversion
            latent_dim=self.latent_dim  # For encoders that use embed_dim internally
        )
        
        # CRN (choose based on encoder output type)
        self.crn = create_crn(
            encoder_is_global=self._encoder_is_global,
            crn_type=self.crn_type,
            crn_kwargs=self.crn_kwargs
        )
        
        # Prior Flow (optional)
        # Use AdaLNLatentFlow for prior flow (MLP with time-dependent AdaLN for latent vectors)
        if self.use_prior_flow:
            prior_kwargs = dict(self.prior_flow_kwargs or {})
            # Default hidden dims if not specified
            if 'hidden_dims' not in prior_kwargs:
                prior_kwargs['hidden_dims'] = (128, 128, 128, 128)
            # Default time_embed_dim if not specified
            if 'time_embed_dim' not in prior_kwargs:
                prior_kwargs['time_embed_dim'] = 128
            self.prior_vector_field = AdaLNLatentFlow(**prior_kwargs)
    
    def __call__(self, x: jnp.ndarray, key: jax.random.PRNGKey, 
                 enc_mask: Optional[jnp.ndarray] = None,
                 point_mask: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, dict]:
        return self.compute_loss(x, key, enc_mask=enc_mask, point_mask=point_mask)
    
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
            if self.normalize_z:
                z_mu_norm = jnp.sqrt(jnp.sum(z_mu**2, axis=-1, keepdims=True) + 1e-8)
                z_mu = z_mu / z_mu_norm
            z = self.reparameterize(z_mu, z_logvar, key)
            return z, z_mu, z_logvar
        else:
            if self.normalize_z:
                z_encoded_norm = jnp.sqrt(jnp.sum(z_encoded**2, axis=-1, keepdims=True) + 1e-8)
                z_encoded = z_encoded / z_encoded_norm
            return z_encoded, None, None
    
    
    def convert_crn_output_to_velocity(self, crn_output: Any, 
                                       x_t: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        Convert CRN output to velocity field.
        
        Flow matching interpolation: x_t = (1-t)·x_0 + t·x_1
        Velocity field: v = dx/dt = x_1 - x_0
        
        From x_t and CRN output, we can compute velocity:
        - If predicting v: return v directly
        - If predicting noise ε (where ε = x_1): v = (ε - x_t) / (1-t)
        - If predicting target x_0: v = (x_t - x_0) / t
        
        For GMFlow (returns dict):
        - Compute expected velocity v = sum(pi_k * mu_k)
        
        Args:
            crn_output: CRN output, shape (B, N, D) OR dict for GMFlow
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
            
        # Handle GMFlow output (dict)
        if isinstance(crn_output, dict):
            # Compute expected velocity: E[v] = sum(pi_k * mu_k)
            logits = crn_output['logits']  # (B, N, K) or (B, K)
            means = crn_output['means']    # (B, N, K, D) or (B, K, D)
            
            # Normalize logits to probabilities
            probs = jax.nn.softmax(logits, axis=-1)
            
            # Broadcast probs if needed to match means
            # If probs is (B, K) and means is (B, K, D) -> (B, K, 1)
            # If probs is (B, N, K) and means is (B, N, K, D) -> (B, N, K, 1)
            probs_expanded = probs[..., None]
            
            # Compute expectation
            # Sum over component dimension (K is -2)
            v_expected = jnp.sum(probs_expanded * means, axis=-2)
            
            # If GMFlow output was global (B, K, D), v_expected is (B, D)
            # Broadcast to (B, N, D) if x_t has N points
            if v_expected.ndim == 2 and x_t.ndim == 3:
                v_expected = v_expected[:, None, :]
                
            # Assume GMFlow always predicts VELOCITY for now
            return v_expected
        
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
    
    def convert_crn_output_to_noise(self, crn_output: Any,
                                    x_t: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        Convert CRN output to noise (ε = x_1).
        
        Args:
            crn_output: CRN output, shape (B, N, D) OR dict
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
            
        # Handle GMFlow
        if isinstance(crn_output, dict):
            # Get velocity first
            v = self.convert_crn_output_to_velocity(crn_output, x_t, t)
            # x_1 = x_t + (1-t)v
            return x_t + (1 - t) * v
        
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
    
    def convert_crn_output_to_target(self, crn_output: Any,
                                     x_t: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        Convert CRN output to target (x_0, the data point).
        
        Args:
            crn_output: CRN output, shape (B, N, D) OR dict
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
            
        # Handle GMFlow
        if isinstance(crn_output, dict):
            # Get velocity first
            v = self.convert_crn_output_to_velocity(crn_output, x_t, t)
            # x_0 = x_t - t*v
            return x_t - t * v
        
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
                                   t: jnp.ndarray, point_mask: Optional[jnp.ndarray] = None,
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
            point_mask: Optional mask for valid points, shape (B, N)
            hutchinson_samples: Number of random samples (1 is usually sufficient)
        
        Returns:
            Trace of Jacobian for each point, shape (B, N)
        """
        B, N, D = x_t.shape
        
        # Define CRN function for a single point
        def crn_single(x_point, z_ctx, t_val, point_mask_val):
            """
            CRN output for a single point.
            x_point: (D,), z_ctx: (Dc,) or (K, Dc), t_val: (), point_mask_val: ()
            Returns: (D,)
            """
            x_reshaped = x_point[None, None, :]  # (1, 1, D)
            t_reshaped = jnp.array([[t_val]])  # (1, 1)
            
            if self._encoder_is_global:
                z_reshaped = z_ctx[None, :]  # (1, Dc)
                out = self.crn(x_reshaped, z_reshaped, t_reshaped)
            else:
                z_reshaped = z_ctx[None, :, :] if z_ctx.ndim == 2 else z_ctx[None, None, :]
                point_mask_reshaped = jnp.array([[point_mask_val]]) if point_mask_val is not None else None
                out = self.crn(x_reshaped, z_reshaped, t_reshaped, mask=point_mask_reshaped)
            
            return out[0, 0, :]  # (D,)
        
        # Hutchinson's trace estimator
        def hutchinson_trace(x_point, z_ctx, t_val, point_mask_val, key):
            """Estimate trace(J) using Hutchinson's method."""
            def single_sample(subkey):
                # Random vector with ±1 entries (Rademacher distribution)
                v = jax.random.rademacher(subkey, (D,), dtype=x_point.dtype)
                
                # Compute Jacobian-vector product: J @ v using forward-mode AD
                _, jvp_result = jax.jvp(
                    lambda x: crn_single(x, z_ctx, t_val, point_mask_val),
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
        
        # Prepare point_mask
        point_mask_array = jnp.ones((B, N)) if point_mask is None else point_mask
        
        # Generate random keys for each point
        key = jax.random.PRNGKey(0)  # Fixed seed for deterministic estimation
        keys = jax.random.split(key, B * N).reshape(B, N, -1)
        
        # Vmap over batch and points
        trace_vmap = jax.vmap(
            jax.vmap(hutchinson_trace, in_axes=(0, 0, 0, 0, 0)),
            in_axes=(0, 0, 0, 0, 0)
        )
        
        all_traces = trace_vmap(x_t, z_array, t_array, point_mask_array, keys)
        return all_traces  # (B, N)
    
    def compute_velocity_divergence(self, x_t: jnp.ndarray, z: jnp.ndarray,
                                    t: jnp.ndarray, point_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Compute divergence of the velocity field.
        
        The velocity field is v(x_t, t) and its divergence depends on what the CRN predicts:
        
        1. If predicting velocity directly:
           v = crn_output
           div(v) = trace(J_crn)
        
        2. If predicting noise ε:
           v = (ε - x_t) / (1-t)
           div(v) = D / (1-t) - trace(J_crn) / (1-t)
                  = (trace(J_crn) - D) / (1-t)
        
        3. If predicting target x_0:
           v = (x_t - x_0) / t
           div(v) = (D - trace(J_crn)) / t
        
        Args:
            x_t: Input points, shape (B, N, D)
            z: Latent context, shape (B, Dc) or (B, K, Dc)
            t: Time, scalar or (B,) or (B, 1)
            point_mask: Optional mask for valid points, shape (B, N)
        
        Returns:
            Divergence of velocity field, shape (B, N)
        """
        B, N, D = x_t.shape
        
        # Compute trace of CRN Jacobian
        trace_jac_crn = self.compute_crn_jacobian_trace(x_t, z, t, point_mask)  # (B, N)
        
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
            # div(v) = (trace(J_crn) - D) / (1-t)
            return (trace_jac_crn - D) / (1 - t_broadcast + 1e-8)
        
        elif self.prediction_target == PredictionTarget.TARGET:
            # v = (x_t - x_0) / t
            # div(v) = (D - trace(J_x0)) / t
            return (D - trace_jac_crn) / (t_broadcast + 1e-8)
        
        else:
            raise ValueError(f"Unknown prediction target: {self.prediction_target}")
    
    def compute_loss(self, x: jnp.ndarray, key: jax.random.PRNGKey,
                    enc_mask: Optional[jnp.ndarray] = None,
                    point_mask: Optional[jnp.ndarray] = None,
                    compute_prior_loss: bool = True,
                    prior_only_mode: bool = False) -> Tuple[jnp.ndarray, dict]:
        """
        Compute loss with flexible target formulations.
        
        Args:
            x: Input point cloud, shape (B, N, D) - full point cloud for flow training
            key: Random key
            enc_mask: Optional mask for encoder, shape (B, N) - used for masked VAE/grid masking.
                     When provided, encoder sees masked points (for robust encoding).
                     Flow training always uses full x regardless of enc_mask.
            point_mask: Optional mask for loss computation, shape (B, N) - used for variable-length
                       point clouds. When provided, masks out invalid/padding points in loss computation.
                       True = valid point, False = invalid/padding point.
        
        Returns:
            loss: Total loss
            metrics: Dictionary of metrics
        """
        if self.use_prior_flow:
            k_sample, k_flow, k_prior_flow = jax.random.split(key, 3)
        else:
            k_sample, k_flow = jax.random.split(key)
        
        # 1. Encode using enc_mask (encoder handles mask=None)
        # When enc_mask is provided (grid masking), encoder sees masked points, but flow uses full x
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
                
        # 4. CRN Output
        # Global CRNs don't take mask, Structured CRNs do (use point_mask for CRN if needed)
        if self._encoder_is_global:
            crn_output = self.crn(x_t, z, t)
        else:
            crn_output = self.crn(x_t, z, t, mask=point_mask)
        
        
        # 4. Compute losses efficiently using affine relationships
        
        # Handle GMFlow (dict output)
        if isinstance(crn_output, dict):
            # GMFlow uses NLL loss: -log sum(pi * N(v; mu, sigma))
            # The target is the velocity field v.
            # In general, according to Eq (2) of the paper: u = (x_t - x_0) / sigma_t
            # For our linear schedule (sigma_t = t, alpha_t = 1-t), this simplifies to x_1 - x_0.
            # We use the simplified form here for numerical stability (avoids division by small t).
            v_target = x_1 - x_0
            
            # Extract parameters
            logits = crn_output['logits']   # (B, N, K) or (B, K)
            means = crn_output['means']     # (B, N, K, D) or (B, K, D)
            logvars = crn_output['logvars'] # (B, 1, 1, 1) or (B, 1, 1)
            
            # Broadcast global outputs to per-point if needed
            if logits.ndim == 2: # (B, K)
                logits = logits[:, None, :] # (B, 1, K)
            
            if means.ndim == 3: # (B, K, D)
                means = means[:, None, :, :] # (B, 1, K, D)
                
            if logvars.ndim == 3: # (B, 1, 1)
                logvars = logvars[:, None, :, :] # (B, 1, 1, 1)
            
            # Target needs to be broadcast to (B, N, 1, D) for K components
            v_target_expanded = v_target[:, :, None, :]
            
            # Compute log likelihood per component: log N(v; mu, sigma)
            # = -0.5 * (D * log(2pi) + sum(logvar) + sum((v-mu)^2 / exp(logvar)))
            neg_half_log_2pi = -0.5 * jnp.log(2 * jnp.pi)
            
            # (B, N, K, D) -> sum over D -> (B, N, K)
            log_prob_component = jnp.sum(
                neg_half_log_2pi - 0.5 * logvars - 0.5 * jnp.square(v_target_expanded - means) * jnp.exp(-logvars),
                axis=-1
            )
            
            # Combine with mixture weights using LogSumExp for stability
            # log p(v) = log sum(exp(logits) * exp(log_prob_comp)) - log sum(exp(logits))
            #          = log sum(exp(logits + log_prob_comp)) - log sum(exp(logits))
            #          = LSE(logits + log_prob_comp) - LSE(logits)
            
            log_likelihood = jax.nn.logsumexp(logits + log_prob_component, axis=-1) - jax.nn.logsumexp(logits, axis=-1)
            
            # NLL = -log_likelihood
            nll = -log_likelihood # (B, N)
            
            # Apply standard flow matching weighting if requested (usually 1.0 for velocity)
            # Default to uniform weighting for GMFlow unless specified
            loss_weight = 1.0
            
            sq_err = nll
            
        else:
            loss_weight = 0.0
            # Standard Regression Loss (MSE)
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
        # Use point_mask to handle variable-length point clouds (mask out invalid/padding points)
        if point_mask is not None:
             # sq_err is (B, N) - apply mask and average
            sq_err = jnp.sum(sq_err * point_mask, axis=-1) / (jnp.sum(point_mask, axis=-1) + 1e-10)  # (B, N) -> (B,)
        else:
            sq_err = jnp.mean(sq_err, axis=-1)  # (B, N) -> (B,)
        
        # Apply time-dependent weight per sample (optimal reweighting for flow matching)
        if self.optimal_reweighting and not isinstance(crn_output, dict):
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
                
        # For GMFlow, we trust the NLL formulation and don't apply extra reweighting by default yet
        
        flow_loss = jnp.mean(sq_err * loss_weight)  # (B,) * (B,) -> (B,) -> scalar
        
        # 5. VAE KL: KL(q(z|x) || N(0,I)) - only if use_vae=True
        if self.use_vae:
            # Shape depends on encoder type:
            # - Global: (B, Dc) -> (B,)
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
        prior_flow_loss = 0.0
        if self.use_prior_flow and compute_prior_loss:
            # Train prior flow: mu_z -> N(0, I)
            # Use AdaLNLatentFlow which takes (mu_z, t) directly
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
            
            # AdaLNLatentFlow takes (mu_z, t) directly - no context needed
            crn_output_prior = self.prior_vector_field(mu_z_t, time_prior)  # (B, D) or (B, K, D)
            target_v_prior = z_prior_noise - mu_z_stopped
            
            prior_flow_loss = jnp.mean((crn_output_prior - target_v_prior) ** 2)
            
            # In prior_only_mode, exclude flow_loss from total loss since encoder/CRN are frozen
            if prior_only_mode:
                loss = prior_flow_loss + self.vae_kl_weight * vae_kl + self.marginal_kl_weight * marginal_kl
            else:
                loss = flow_loss + prior_flow_loss + self.vae_kl_weight * vae_kl + self.marginal_kl_weight * marginal_kl
            
        elif self.use_prior_flow and not compute_prior_loss:
            # Prior flow is enabled but we're not computing loss (prior is frozen)
            loss = flow_loss + self.vae_kl_weight * vae_kl + self.marginal_kl_weight * marginal_kl
        else:
            loss = flow_loss + self.vae_kl_weight * vae_kl + self.marginal_kl_weight * marginal_kl
        
        metrics = {"flow_loss": flow_loss, 
                    "prior_flow_loss": prior_flow_loss,  # Set to 0 to indicate it wasn't computed
                    "vae_kl": vae_kl,
                    "marginal_kl": marginal_kl}
        return loss, metrics
    
    def sample(self, num_points: int, key: jax.random.PRNGKey,
               z: Optional[jnp.ndarray] = None, num_steps: int = 20,
               batch_size: Optional[int] = None) -> jnp.ndarray:
        """
        Sample points by integrating the flow from t=1 (noise) to t=0 (data).
        
        Args:
            num_points: Number of points to sample per batch element
            key: Random key
            z: Optional latent code (if None, sample from prior)
                - If provided, can have shape (B, D) for global or (B, K, D) for local
                - If None, samples from prior (batch size determined by batch_size parameter)
            num_steps: Number of integration steps
            batch_size: Optional batch size when z=None. If None and z=None, defaults to 1.
                       Ignored if z is provided (uses z's batch size instead).
        
        Returns:
            Sampled points:
            - If batch size B > 1: returns (B, num_points, spatial_dim)
            - If batch size B == 1: returns (num_points, spatial_dim)
        """
        if z is None:
            # Determine batch size: use batch_size parameter or default to 1
            B = batch_size if batch_size is not None else 1
            
            if self.use_prior_flow:
                # Sample from learned prior using AdaLNLatentFlow
                key, k_prior = jax.random.split(key)
                if self._encoder_is_global:
                    z_prior_noise = jax.random.normal(k_prior, (B, self.latent_dim))
                else:
                    # For local encoders, need to know K
                    # TODO: Make this configurable
                    z_prior_noise = jax.random.normal(k_prior, (B, self.num_latents, self.latent_dim))
                
                dt_prior = -1.0 / num_steps
                
                def prior_step(z_t, t):
                    
                    # AdaLNLatentFlow takes (z, t) directly - no context needed
                    # Handle batch dimension: t should match z's batch dimension
                    B_current = z_t.shape[0]
                    if B_current == 1:
                        t_batch = jnp.array([t])  # (1,)
                    else:
                        t_batch = jnp.full((B_current,), t)  # (B,)
                    
                    v = self.prior_vector_field(z_t, t_batch)  # (B, D) or (B, K, D)
                    
                    return z_t + dt_prior * v, t + dt_prior
                t = jnp.asarray(1.0)
                z, _ = jax.lax.scan(prior_step, z_prior_noise, t)
            else:
                # Sample from standard normal
                if self._encoder_is_global:
                    z = jax.random.normal(key, (B, self.latent_dim))
                else:
                    K = 8  # Default
                    z = jax.random.normal(key, (B, K, self.latent_dim))

        # Normalize z to unit vector
        if self.normalize_z:
            z = z / jnp.sqrt(jnp.sum(z**2, axis=-1, keepdims=True) + 1e-8)
        
        # Determine batch size from z
        B = z.shape[0]
        
        # Initialize from noise (match batch size of z)
        key, k_points = jax.random.split(key)
        x_init = jax.random.normal(k_points, (B, num_points, self.spatial_dim))
        t_init = 1.0
        # Integrate from t=1 to t=0
        dt = -1.0 / num_steps
        
        def euler_step(x_t, t):
            # Broadcast t to batch dimension
            t_batch = jnp.full((B,), t) if B > 1 else t
            
            # Get CRN output
            if self._encoder_is_global:
                crn_output = self.crn(x_t, z, t_batch)
            else:
                crn_output = self.crn(x_t, z, t_batch)
            
            # Convert CRN output to velocity for Euler integration
            # The CRN may predict velocity, noise, or target - we need velocity for integration
            v = self.convert_crn_output_to_velocity(crn_output, x_t, t_batch)
            
            return x_t + dt * v, None
        
        x_final, _ = jax.lax.scan(euler_step, x_init, jnp.arange(num_steps))
        
        # Return shape: (B, num_points, spatial_dim) if B > 1, else (num_points, spatial_dim)
        if B == 1:
            return x_final[0]  # Remove batch dimension for single sample
        else:
            return x_final  # Keep batch dimension for batch sampling

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional

# =============================================================================
# Time Embeddings (1D)
# =============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for scalar time values."""
    dim: int = 32
    
    @nn.compact
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        # t: (B, 1) or (B,)
        if t.ndim == 1: t = t[:, None]
        half_dim = self.dim // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = t * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb


# =============================================================================
# 2D Positional Embeddings
# =============================================================================

class SinusoidalPositionEmbedding2D(nn.Module):
    """
    Sinusoidal positional embedding for 2D coordinates.
    
    Applies log-spaced frequencies to each coordinate dimension independently,
    similar to transformer positional encoding but for continuous 2D space.
    
    Output dim = num_frequencies * 2 * 2 = num_frequencies * 4
    (sin + cos for each of x and y)
    """
    num_frequencies: int = 16
    max_freq: float = 10.0
    min_freq: float = 1.0
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (..., 2) - 2D coordinates
        *batch_dims, D = x.shape
        assert D == 2, f"Expected 2D coordinates, got {D}D"
        
        # Log-spaced frequencies
        freqs = jnp.exp(jnp.linspace(
            jnp.log(self.min_freq), 
            jnp.log(self.max_freq), 
            self.num_frequencies
        ))  # (num_frequencies,)
        
        # Apply to each coordinate: x[..., 0] and x[..., 1]
        # Shape: (..., num_frequencies)
        x_scaled = x[..., 0:1] * freqs  # (..., num_frequencies)
        y_scaled = x[..., 1:2] * freqs  # (..., num_frequencies)
        
        # Sin and cos for each
        emb = jnp.concatenate([
            jnp.sin(x_scaled), jnp.cos(x_scaled),
            jnp.sin(y_scaled), jnp.cos(y_scaled)
        ], axis=-1)  # (..., num_frequencies * 4)
        
        return emb


class FourierFeatures2D(nn.Module):
    """
    Random Fourier Features for 2D coordinates (NeRF-style).
    
    Uses fixed random frequencies sampled from N(0, scale²).
    Good for learning high-frequency spatial variations.
    
    Output dim = num_frequencies * 2 (sin + cos)
    """
    num_frequencies: int = 32
    scale: float = 10.0
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (..., 2) - 2D coordinates
        *batch_dims, D = x.shape
        assert D == 2, f"Expected 2D coordinates, got {D}D"
        
        # Fixed random projection matrix (initialized once)
        B_matrix = self.param(
            'B_matrix',
            lambda rng, shape: jax.random.normal(rng, shape) * self.scale,
            (D, self.num_frequencies)
        )  # (2, num_frequencies)
        
        # Project coordinates: x @ B -> (..., num_frequencies)
        projected = x @ B_matrix  # (..., num_frequencies)
        
        # Apply sin and cos
        emb = jnp.concatenate([
            jnp.sin(2 * jnp.pi * projected),
            jnp.cos(2 * jnp.pi * projected)
        ], axis=-1)  # (..., num_frequencies * 2)
        
        return emb


class LearnedFourierFeatures2D(nn.Module):
    """
    Learned Fourier Features for 2D coordinates.
    
    Like FourierFeatures2D but frequencies are learned during training.
    More flexible but may overfit on small datasets.
    
    Output dim = num_frequencies * 2 (sin + cos)
    """
    num_frequencies: int = 32
    init_scale: float = 1.0
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (..., 2) - 2D coordinates
        *batch_dims, D = x.shape
        assert D == 2, f"Expected 2D coordinates, got {D}D"
        
        # Learnable frequency matrix
        B_matrix = self.param(
            'B_matrix',
            nn.initializers.normal(self.init_scale),
            (D, self.num_frequencies)
        )  # (2, num_frequencies)
        
        # Learnable phase shifts (optional, helps flexibility)
        phase = self.param(
            'phase',
            nn.initializers.zeros,
            (self.num_frequencies,)
        )
        
        # Project and add phase
        projected = x @ B_matrix + phase  # (..., num_frequencies)
        
        # Apply sin and cos
        emb = jnp.concatenate([
            jnp.sin(2 * jnp.pi * projected),
            jnp.cos(2 * jnp.pi * projected)
        ], axis=-1)  # (..., num_frequencies * 2)
        
        return emb


class GaussianFourierFeatures2D(nn.Module):
    """
    Gaussian Random Fourier Features for 2D coordinates.
    
    Uses frequencies sampled from isotropic Gaussian, which approximates
    an RBF kernel. Scale controls the "bandwidth" of the kernel.
    
    - Small scale: smooth, low-frequency features
    - Large scale: sharp, high-frequency features
    
    Output dim = num_frequencies * 2 (sin + cos)
    """
    num_frequencies: int = 32
    scale: float = 1.0
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (..., 2) - 2D coordinates
        *batch_dims, D = x.shape
        assert D == 2, f"Expected 2D coordinates, got {D}D"
        
        # Fixed Gaussian random frequencies
        freqs = self.param(
            'freqs',
            lambda rng, shape: jax.random.normal(rng, shape) * self.scale,
            (D, self.num_frequencies)
        )  # (2, num_frequencies)
        
        # Project: scale by 2π for proper Fourier basis
        projected = 2 * jnp.pi * (x @ freqs)  # (..., num_frequencies)
        
        # Sin and cos
        emb = jnp.concatenate([
            jnp.sin(projected),
            jnp.cos(projected)
        ], axis=-1)  # (..., num_frequencies * 2)
        
        return emb


class MultiScaleFourierFeatures2D(nn.Module):
    """
    Multi-scale Fourier Features combining multiple frequency bands.
    
    Uses deterministic log-spaced frequencies at multiple scales,
    providing both fine and coarse spatial resolution.
    
    Output dim = num_bands * num_frequencies_per_band * 4
    """
    num_bands: int = 4
    num_frequencies_per_band: int = 8
    min_freq: float = 1.0
    max_freq: float = 64.0
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (..., 2) - 2D coordinates
        *batch_dims, D = x.shape
        assert D == 2, f"Expected 2D coordinates, got {D}D"
        
        embeddings = []
        
        # Create frequency bands
        band_edges = jnp.exp(jnp.linspace(
            jnp.log(self.min_freq),
            jnp.log(self.max_freq),
            self.num_bands + 1
        ))
        
        for i in range(self.num_bands):
            # Frequencies for this band
            freqs = jnp.linspace(
                band_edges[i], 
                band_edges[i + 1], 
                self.num_frequencies_per_band
            )
            
            # Apply to x and y
            x_scaled = x[..., 0:1] * freqs  # (..., num_freq)
            y_scaled = x[..., 1:2] * freqs  # (..., num_freq)
            
            band_emb = jnp.concatenate([
                jnp.sin(x_scaled), jnp.cos(x_scaled),
                jnp.sin(y_scaled), jnp.cos(y_scaled)
            ], axis=-1)
            
            embeddings.append(band_emb)
        
        return jnp.concatenate(embeddings, axis=-1)


class PositionalEmbedding2D(nn.Module):
    """
    Unified interface for 2D positional embeddings.
    
    Combines raw coordinates with chosen embedding type, then projects
    to desired output dimension.
    
    Args:
        embed_type: 'sinusoidal', 'fourier', 'learned_fourier', 
                    'gaussian_fourier', 'multiscale', or 'linear'
        output_dim: Final embedding dimension
        num_frequencies: Number of frequency components
        include_input: Whether to concatenate raw (x, y) with embedding
    """
    embed_type: str = 'sinusoidal'
    output_dim: int = 64
    num_frequencies: int = 16
    scale: float = 10.0
    include_input: bool = True
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (..., 2) - 2D coordinates
        
        if self.embed_type == 'linear':
            # Just linear projection
            return nn.Dense(self.output_dim)(x)
        
        elif self.embed_type == 'sinusoidal':
            emb = SinusoidalPositionEmbedding2D(
                num_frequencies=self.num_frequencies
            )(x)
            
        elif self.embed_type == 'fourier':
            emb = FourierFeatures2D(
                num_frequencies=self.num_frequencies,
                scale=self.scale
            )(x)
            
        elif self.embed_type == 'learned_fourier':
            emb = LearnedFourierFeatures2D(
                num_frequencies=self.num_frequencies
            )(x)
            
        elif self.embed_type == 'gaussian_fourier':
            emb = GaussianFourierFeatures2D(
                num_frequencies=self.num_frequencies,
                scale=self.scale
            )(x)
            
        elif self.embed_type == 'multiscale':
            emb = MultiScaleFourierFeatures2D(
                num_frequencies_per_band=self.num_frequencies // 4
            )(x)
            
        else:
            raise ValueError(f"Unknown embed_type: {self.embed_type}")
        
        # Optionally include raw coordinates
        if self.include_input:
            emb = jnp.concatenate([x, emb], axis=-1)
        
        # Project to output dimension
        return nn.Dense(self.output_dim)(emb)


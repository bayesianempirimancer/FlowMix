"""
Encoder factory for creating encoders based on type and output configuration.

This factory centralizes encoder creation logic to simplify model setup.
"""

from typing import Tuple, Literal, Optional, Dict, Any

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


def create_encoder(
    encoder_type: str,
    encoder_output_type: Literal['global', 'local'],
    encoder_output_dim: int,
    encoder_kwargs: Optional[Dict[str, Any]] = None,
    pooling_type: str = 'max',
    latent_dim: Optional[int] = None
) -> Tuple[Any, bool]:
    """
    Factory function to create an encoder based on type and output configuration.
    
    Args:
        encoder_type: Type of encoder ('pointnet', 'transformer', 'dgcnn', 'slot_attention', 
                     'cross_attention', 'gmm')
        encoder_output_type: 'global' or 'local' - determines if encoder outputs a single vector
                            or per-point/local features
        encoder_output_dim: Output dimension of the encoder (may be 2*latent_dim for VAE)
        encoder_kwargs: Optional kwargs to pass to the encoder
        pooling_type: Type of pooling to use when converting local -> global ('max', 'mean', 'attention')
        latent_dim: Internal dimension for encoders that use embed_dim (transformer, dgcnn).
                   If None, uses encoder_output_dim. This is needed because some encoders use
                   embed_dim internally while output_dim may differ (e.g., for VAE).
        
    Returns:
        Tuple of (encoder, is_global):
        - encoder: The created encoder module
        - is_global: Boolean indicating if encoder outputs global (single vector) or local features
        
    Raises:
        ValueError: If encoder_type is unknown
    """
    enc_kwargs = encoder_kwargs or {}
    
    # For encoders that use embed_dim internally, use latent_dim if provided, otherwise encoder_output_dim
    embed_dim = latent_dim if latent_dim is not None else encoder_output_dim
    
    if encoder_type == 'pointnet':
        # PointNet is inherently global
        encoder = PointNetEncoder(latent_dim=encoder_output_dim, **enc_kwargs)
        is_global = True
        
    elif encoder_type == 'transformer':
        # TransformerSetEncoder is local, wrap if needed
        # Uses embed_dim for internal dimension
        base_encoder = TransformerSetEncoder(embed_dim=embed_dim, **enc_kwargs)
        if encoder_output_type == 'global':
            encoder = _get_pooling_encoder(base_encoder, encoder_output_dim, pooling_type)
            is_global = True
        else:
            encoder = base_encoder
            is_global = False
            
    elif encoder_type == 'dgcnn':
        # Uses embed_dim for internal dimension
        base_encoder = DGCNN(embed_dim=embed_dim, **enc_kwargs)
        if encoder_output_type == 'global':
            encoder = _get_pooling_encoder(base_encoder, encoder_output_dim, pooling_type)
            is_global = True
        else:
            encoder = base_encoder
            is_global = False
            
    elif encoder_type == 'slot_attention':
        # Slot Attention is local (outputs K slots)
        base_encoder = SlotAttentionEncoder(slot_dim=encoder_output_dim, **enc_kwargs)
        if encoder_output_type == 'global':
            encoder = _get_pooling_encoder(base_encoder, encoder_output_dim, pooling_type)
            is_global = True
        else:
            encoder = base_encoder
            is_global = False
            
    elif encoder_type == 'cross_attention':
        # CrossAttentionEncoder is local (outputs K latent queries)
        base_encoder = CrossAttentionEncoder(latent_dim=encoder_output_dim, **enc_kwargs)
        if encoder_output_type == 'global':
            encoder = _get_pooling_encoder(base_encoder, encoder_output_dim, pooling_type)
            is_global = True
        else:
            encoder = base_encoder
            is_global = False
            
    elif encoder_type == 'gmm':
        # GMM is local (outputs K components)
        base_encoder = GMMFeaturizer(latent_dim=encoder_output_dim, **enc_kwargs)
        if encoder_output_type == 'global':
            encoder = _get_pooling_encoder(base_encoder, encoder_output_dim, pooling_type)
            is_global = True
        else:
            encoder = base_encoder
            is_global = False
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    return encoder, is_global


def _get_pooling_encoder(base_encoder, encoder_output_dim: int, pooling_type: str):
    """Helper to get the appropriate pooling encoder."""
    if pooling_type == 'max':
        return MaxPoolingEncoder(base_encoder, encoder_output_dim)
    elif pooling_type == 'mean':
        return MeanPoolingEncoder(base_encoder, encoder_output_dim)
    elif pooling_type == 'attention':
        return AttentionPoolingEncoder(base_encoder, encoder_output_dim)
    else:
        raise ValueError(f"Unknown pooling type: {pooling_type}")


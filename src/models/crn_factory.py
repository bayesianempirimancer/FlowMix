"""
CRN factory for creating Conditional ResNets based on encoder output type and CRN type.

This factory centralizes CRN creation logic to simplify model setup.
"""

from typing import Any, Optional, Dict

# Global CRNs
from src.models.global_crn import (
    GlobalAdaLNMLPCRN,
    GlobalDiTCRN,
    GlobalCrossAttentionCRN,
)
from src.models.gmflow_crn import GlobalGMFlowCRN

# Structured CRNs
from src.models.structured_crn import (
    StructuredAdaLNMLPCRN,
    StructuredCrossAttentionCRN,
)


def create_crn(
    encoder_is_global: bool,
    crn_type: str,
    crn_kwargs: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Factory function to create a CRN based on encoder output type and CRN type.
    
    Args:
        encoder_is_global: Whether the encoder outputs global (single vector) or 
                          local/structured (multiple vectors) features
        crn_type: Type of CRN ('adaln_mlp', 'dit', 'cross_attention', 'gmflow')
        crn_kwargs: Optional kwargs to pass to the CRN
        
    Returns:
        The created CRN module
        
    Raises:
        ValueError: If crn_type is unknown or not supported for the encoder type
    """
    crn_kwargs = crn_kwargs or {}
    
    if encoder_is_global:
        # Global CRNs: context is (B, Dc)
        if crn_type == 'adaln_mlp':
            return GlobalAdaLNMLPCRN(**crn_kwargs)
        elif crn_type == 'dit':
            return GlobalDiTCRN(**crn_kwargs)
        elif crn_type == 'cross_attention':
            return GlobalCrossAttentionCRN(**crn_kwargs)
        elif crn_type == 'gmflow':
            return GlobalGMFlowCRN(**crn_kwargs)
        else:
            raise ValueError(f"Unknown CRN type: {crn_type}")
    else:
        # Local/Structured CRNs: context is (B, K, Dc)
        # For now, use Structured (pool-based) as default
        # TODO: Add option to use Local CRNs when sampling K points
        if crn_type == 'adaln_mlp':
            return StructuredAdaLNMLPCRN(**crn_kwargs)
        elif crn_type == 'cross_attention':
            return StructuredCrossAttentionCRN(**crn_kwargs)
        else:
            raise ValueError(f"CRN type {crn_type} not supported for local encoders. "
                           f"Supported types: 'adaln_mlp', 'cross_attention'")




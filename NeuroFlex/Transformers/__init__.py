"""
NeuroFlex Transformers Module

This module provides transformer models and utilities, supporting various types like Unified, JAX, Flax, and Sonnet.

Recent updates:
- Enhanced support for JAX and Flax transformers
- Improved initialization functions for transformer models
- Added support for Sonnet-based transformers
"""

from .unified_transformer import (
    UnifiedTransformer,
    get_unified_transformer,
    JAXUnifiedTransformer,
    FlaxUnifiedTransformer,
    SonnetUnifiedTransformer,
)

__all__ = [
    "UnifiedTransformer",
    "get_unified_transformer",
    "JAXUnifiedTransformer",
    "FlaxUnifiedTransformer",
    "SonnetUnifiedTransformer",
    "get_transformer_version",
    "SUPPORTED_TRANSFORMER_TYPES",
    "initialize_transformers",
    "create_transformer",
]


def get_transformer_version():
    return "1.0.0"


SUPPORTED_TRANSFORMER_TYPES = ["Unified", "JAX", "Flax", "Sonnet"]


def initialize_transformers():
    print("Initializing Transformers Module...")
    # Add any necessary initialization code here


def create_transformer(transformer_type, *args, **kwargs):
    if transformer_type == "Unified":
        return UnifiedTransformer(*args, **kwargs)
    elif transformer_type == "JAX":
        return JAXUnifiedTransformer(*args, **kwargs)
    elif transformer_type == "Flax":
        return FlaxUnifiedTransformer(*args, **kwargs)
    elif transformer_type == "Sonnet":
        return SonnetUnifiedTransformer(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported transformer type: {transformer_type}")


# Add any other Transformer-specific utility functions or constants as needed

"""
NeuroFlex Transformers Module

This module provides transformer models and utilities, supporting various types like Unified, JAX, Flax, and Sonnet.

Recent updates:
- Enhanced support for JAX and Flax transformers
- Improved initialization functions for transformer models
- Added support for Sonnet-based transformers
- Updated version to match main NeuroFlex version
"""

from .unified_transformer import (
    UnifiedTransformer,
    get_unified_transformer,
    JAXUnifiedTransformer,
    FlaxUnifiedTransformer,
    SonnetUnifiedTransformer
)

__all__ = [
    'UnifiedTransformer',
    'get_unified_transformer',
    'JAXUnifiedTransformer',
    'FlaxUnifiedTransformer',
    'SonnetUnifiedTransformer',
    'get_transformer_version',
    'SUPPORTED_TRANSFORMER_TYPES',
    'initialize_transformers',
    'create_transformer',
    'validate_transformer_config'
]

def get_transformer_version():
    return "0.1.3"  # Updated to match main NeuroFlex version

SUPPORTED_TRANSFORMER_TYPES = ['Unified', 'JAX', 'Flax', 'Sonnet']

def initialize_transformers():
    print("Initializing Transformers Module...")
    # Add any necessary initialization code here

def create_transformer(transformer_type, *args, **kwargs):
    if transformer_type not in SUPPORTED_TRANSFORMER_TYPES:
        raise ValueError(f"Unsupported transformer type: {transformer_type}")

    if transformer_type == 'Unified':
        return UnifiedTransformer(*args, **kwargs)
    elif transformer_type == 'JAX':
        return JAXUnifiedTransformer(*args, **kwargs)
    elif transformer_type == 'Flax':
        return FlaxUnifiedTransformer(*args, **kwargs)
    elif transformer_type == 'Sonnet':
        return SonnetUnifiedTransformer(*args, **kwargs)

def validate_transformer_config(config):
    """
    Validate the configuration for a transformer model.
    """
    required_keys = ['vocab_size', 'd_model', 'num_heads', 'num_layers']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    if config['transformer_type'] not in SUPPORTED_TRANSFORMER_TYPES:
        raise ValueError(f"Unsupported transformer type: {config['transformer_type']}")

    return True

# Add any other Transformer-specific utility functions or constants as needed

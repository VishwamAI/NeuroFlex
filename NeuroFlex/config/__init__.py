"""
NeuroFlex Configuration Module

This module provides configuration settings and utilities for the NeuroFlex library.
"""

from .default_config import DEFAULT_CONFIG
from .config_utils import load_config, save_config, update_config

__all__ = [
    'DEFAULT_CONFIG',
    'load_config',
    'save_config',
    'update_config',
    'get_config_version',
    'SUPPORTED_CONFIG_TYPES',
    'initialize_config',
    'create_config'
]

def get_config_version():
    return "1.0.0"

SUPPORTED_CONFIG_TYPES = [
    "default",
    "custom"
]

def initialize_config():
    print("Initializing Configuration Module...")
    # Add any necessary initialization code here

def create_config(config_type="default", **kwargs):
    if config_type == "default":
        return DEFAULT_CONFIG.copy()
    elif config_type == "custom":
        custom_config = DEFAULT_CONFIG.copy()
        custom_config.update(kwargs)
        return custom_config
    else:
        raise ValueError(f"Unsupported configuration type: {config_type}")

# Add any other Configuration-specific utility functions or constants as needed

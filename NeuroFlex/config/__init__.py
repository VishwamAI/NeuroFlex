# MIT License
# 
# Copyright (c) 2024 VishwamAI
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
NeuroFlex Configuration Module

This module provides configuration settings and utilities for the NeuroFlex library.

Recent updates:
- Updated version to match main NeuroFlex version
- Enhanced configuration creation and validation
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
    'create_config',
    'validate_config'
]

def get_config_version():
    return "0.1.3"  # Updated to match main NeuroFlex version

SUPPORTED_CONFIG_TYPES = [
    "default",
    "custom"
]

def initialize_config():
    print("Initializing Configuration Module...")
    # Add any necessary initialization code here

def create_config(config_type="default", **kwargs):
    if config_type not in SUPPORTED_CONFIG_TYPES:
        raise ValueError(f"Unsupported configuration type: {config_type}")

    if config_type == "default":
        return DEFAULT_CONFIG.copy()
    elif config_type == "custom":
        custom_config = DEFAULT_CONFIG.copy()
        custom_config.update(kwargs)
        return custom_config

def validate_config(config):
    """
    Validate the given configuration against the default configuration structure.
    """
    for key in DEFAULT_CONFIG:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    return True

# Add any other Configuration-specific utility functions or constants as needed

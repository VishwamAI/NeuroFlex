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
NeuroFlex Utilities Module

This module provides utility functions and tools for data handling, preprocessing, and analysis.

Recent updates:
- Enhanced data normalization and JSON handling functions
- Improved support for descriptive statistics and BCI data analysis
- Added utility functions for directory management and list manipulation
- Updated version to match main NeuroFlex version
"""

# __init__.py for utils module

from .utils import (
    load_data,
    save_data,
    normalize_data,
    create_directory,
    load_json,
    save_json,
    flatten_list,
    get_activation_function
)

from .config import Config
from .logging import setup_logging

from .descriptive_statistics import (
    calculate_descriptive_statistics,
    preprocess_data,
    analyze_bci_data
)

__all__ = [
    'load_data',
    'save_data',
    'normalize_data',
    'create_directory',
    'load_json',
    'save_json',
    'flatten_list',
    'calculate_descriptive_statistics',
    'preprocess_data',
    'analyze_bci_data',
    'get_activation_function',
    'get_utils_version',
    'SUPPORTED_UTILS',
    'initialize_utils',
    'create_util_function',
    'validate_util_config',
    'Config',
    'setup_logging'
]

def get_utils_version():
    return "0.1.3"  # Updated to match main NeuroFlex version

SUPPORTED_UTILS = [
    "load_data",
    "save_data",
    "normalize_data",
    "create_directory",
    "load_json",
    "save_json",
    "flatten_list",
    "calculate_descriptive_statistics",
    "preprocess_data",
    "analyze_bci_data",
    "get_activation_function"
]

def initialize_utils():
    print("Initializing Utils Module...")
    print(f"Utils version: {get_utils_version()}")
    # Add any necessary initialization code here

def create_util_function(util_name, *args, **kwargs):
    if util_name not in SUPPORTED_UTILS:
        raise ValueError(f"Unsupported util function: {util_name}")

    if util_name == "load_data":
        return load_data(*args, **kwargs)
    elif util_name == "save_data":
        return save_data(*args, **kwargs)
    elif util_name == "normalize_data":
        return normalize_data(*args, **kwargs)
    elif util_name == "create_directory":
        return create_directory(*args, **kwargs)
    elif util_name == "load_json":
        return load_json(*args, **kwargs)
    elif util_name == "save_json":
        return save_json(*args, **kwargs)
    elif util_name == "flatten_list":
        return flatten_list(*args, **kwargs)
    elif util_name == "calculate_descriptive_statistics":
        return calculate_descriptive_statistics(*args, **kwargs)
    elif util_name == "preprocess_data":
        return preprocess_data(*args, **kwargs)
    elif util_name == "analyze_bci_data":
        return analyze_bci_data(*args, **kwargs)
    elif util_name == "get_activation_function":
        return get_activation_function(*args, **kwargs)

def validate_util_config(config):
    """
    Validate the configuration for a utility function.
    """
    required_keys = ['util_name', 'parameters']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    if config['util_name'] not in SUPPORTED_UTILS:
        raise ValueError(f"Unsupported util function: {config['util_name']}")

    return True

# Add any other Utils-specific utility functions or constants as needed

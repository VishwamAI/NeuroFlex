"""
NeuroFlex Utilities Module

This module provides utility functions and tools for data handling, preprocessing, and analysis.

Recent updates:
- Enhanced data normalization and JSON handling functions
- Improved support for descriptive statistics and BCI data analysis
- Added utility functions for directory management and list manipulation
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
    get_activation_function,
)

from .descriptive_statistics import (
    calculate_descriptive_statistics,
    preprocess_data,
    analyze_bci_data,
)

__all__ = [
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
    "get_activation_function",
    "get_utils_version",
    "SUPPORTED_UTILS",
    "initialize_utils",
    "create_util_function",
]


def get_utils_version():
    return "1.0.0"


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
    "get_activation_function",
]


def initialize_utils():
    print("Initializing Utils Module...")
    # Add any necessary initialization code here


def create_util_function(util_name, *args, **kwargs):
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
    else:
        raise ValueError(f"Unsupported util function: {util_name}")


# Add any other Utils-specific utility functions or constants as needed

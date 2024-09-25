"""
NeuroFlex Advanced Models Module

This module implements advanced models for mathematical solving, time series analysis, and multi-modal learning.

Recent updates:
- Enhanced mathematical solving capabilities with self-healing mechanisms
- Improved time series analysis techniques including ARIMA, SARIMA, and Prophet forecasting
- Integrated multi-modal learning models with support for text, image, tabular, and time series data
- Implemented performance monitoring and self-optimization strategies
- Updated version to match main NeuroFlex version
"""

from .advanced_math_solving import AdvancedMathSolver
from .advanced_time_series_analysis import AdvancedTimeSeriesAnalysis
from .multi_modal_learning import MultiModalLearning

__all__ = [
    'AdvancedMathSolver',
    'AdvancedTimeSeriesAnalysis',
    'MultiModalLearning',
    'get_advanced_models_version',
    'SUPPORTED_ADVANCED_MODELS',
    'initialize_advanced_models',
    'create_advanced_model',
    'validate_advanced_model_config'
]

def get_advanced_models_version():
    return "0.1.3"  # Updated to match main NeuroFlex version

SUPPORTED_ADVANCED_MODELS = [
    "AdvancedMathSolver",
    "AdvancedTimeSeriesAnalysis",
    "MultiModalLearning"
]

def initialize_advanced_models():
    print("Initializing Advanced Models Module...")
    print(f"Advanced Models version: {get_advanced_models_version()}")
    # Add any necessary initialization code here

def create_advanced_model(model_type, *args, **kwargs):
    if model_type not in SUPPORTED_ADVANCED_MODELS:
        raise ValueError(f"Unsupported advanced model type: {model_type}")

    if model_type == "AdvancedMathSolver":
        return AdvancedMathSolver(*args, **kwargs)
    elif model_type == "AdvancedTimeSeriesAnalysis":
        return AdvancedTimeSeriesAnalysis(*args, **kwargs)
    elif model_type == "MultiModalLearning":
        return MultiModalLearning(*args, **kwargs)

def validate_advanced_model_config(config):
    """
    Validate the configuration for an advanced model.
    """
    required_keys = ['model_type', 'parameters']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    if config['model_type'] not in SUPPORTED_ADVANCED_MODELS:
        raise ValueError(f"Unsupported advanced model type: {config['model_type']}")

    return True

# Add any other Advanced Models-specific utility functions or constants as needed

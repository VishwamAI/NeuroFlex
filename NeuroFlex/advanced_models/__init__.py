"""
NeuroFlex Advanced Models Module

This module implements advanced models for mathematical solving, time series analysis, and multi-modal learning.

Recent updates:
- Enhanced mathematical solving capabilities with self-healing mechanisms
- Improved time series analysis techniques including ARIMA, SARIMA, and Prophet forecasting
- Integrated multi-modal learning models with support for text, image, tabular, and time series data
- Implemented performance monitoring and self-optimization strategies
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
    'create_advanced_model'
]

def get_advanced_models_version():
    return "1.0.0"

SUPPORTED_ADVANCED_MODELS = [
    "AdvancedMathSolver",
    "AdvancedTimeSeriesAnalysis",
    "MultiModalLearning"
]

def initialize_advanced_models():
    print("Initializing Advanced Models Module...")
    # Add any necessary initialization code here

def create_advanced_model(model_type, *args, **kwargs):
    if model_type == "AdvancedMathSolver":
        return AdvancedMathSolver(*args, **kwargs)
    elif model_type == "AdvancedTimeSeriesAnalysis":
        return AdvancedTimeSeriesAnalysis(*args, **kwargs)
    elif model_type == "MultiModalLearning":
        return MultiModalLearning(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported advanced model type: {model_type}")

# Add any other Advanced Models-specific utility functions or constants as needed

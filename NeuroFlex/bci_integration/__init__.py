"""
NeuroFlex BCI Integration Module

This module facilitates integration with brain-computer interfaces (BCI), providing tools for BCI data processing and neuroscience model integration.

Recent updates:
- Enhanced BCI data processing capabilities
- Improved integration with neuroscience models
- Supported devices include OpenBCI, Emotiv, and Neurosky
"""

from .bci_processing import BCIProcessor
from .neuro_data_integration import NeuroDataIntegrator
from .neuroscience_models import NeuroscienceModel

__all__ = [
    'BCIProcessor',
    'NeuroDataIntegrator',
    'NeuroscienceModel',
    'get_bci_version',
    'BCI_SUPPORTED_DEVICES',
    'validate_bci_data'
]

def get_bci_version():
    return "1.0.0"

BCI_SUPPORTED_DEVICES = ['OpenBCI', 'Emotiv', 'Neurosky']

def validate_bci_data(data):
    # Implement validation logic here
    pass

# Add any other BCI integration-specific utility functions or constants as needed

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
NeuroFlex BCI Integration Module

This module facilitates integration with brain-computer interfaces (BCI), providing tools for BCI data processing and neuroscience model integration.

Recent updates:
- Enhanced BCI data processing capabilities
- Improved integration with neuroscience models
- Supported devices include OpenBCI, Emotiv, and Neurosky
- Updated version to match main NeuroFlex version
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
    'validate_bci_data',
    'initialize_bci_integration'
]

def get_bci_version():
    return "0.1.3"  # Updated to match main NeuroFlex version

BCI_SUPPORTED_DEVICES = ['OpenBCI', 'Emotiv', 'Neurosky']

def validate_bci_data(data):
    # Implement validation logic here
    if not isinstance(data, dict):
        raise ValueError("BCI data must be a dictionary")
    required_keys = ['timestamp', 'channel_data', 'device_type']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key: {key}")
    if data['device_type'] not in BCI_SUPPORTED_DEVICES:
        raise ValueError(f"Unsupported device type: {data['device_type']}")
    return True

def initialize_bci_integration():
    print("Initializing BCI Integration Module...")
    # Add any necessary initialization code here

# Add any other BCI integration-specific utility functions or constants as needed

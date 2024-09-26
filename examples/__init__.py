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
NeuroFlex Examples Module

This module provides example implementations and usage demonstrations for the NeuroFlex library.

Recent updates:
- Updated version to match main NeuroFlex version
- Added validation function for example configurations
"""

from .basic_usage import basic_neuroflex_example
from .advanced_features import advanced_neuroflex_example
from .quantum_nn_demo import quantum_nn_example
from .bci_integration_demo import bci_integration_example

__all__ = [
    'basic_neuroflex_example',
    'advanced_neuroflex_example',
    'quantum_nn_example',
    'bci_integration_example',
    'get_examples_version',
    'SUPPORTED_EXAMPLES',
    'initialize_examples',
    'run_example',
    'validate_example_config'
]

def get_examples_version():
    return "0.1.3"  # Updated to match main NeuroFlex version

SUPPORTED_EXAMPLES = [
    "basic",
    "advanced",
    "quantum_nn",
    "bci_integration"
]

def initialize_examples():
    print("Initializing Examples Module...")
    print(f"Examples version: {get_examples_version()}")
    # Add any necessary initialization code here

def run_example(example_type, *args, **kwargs):
    if example_type not in SUPPORTED_EXAMPLES:
        raise ValueError(f"Unsupported example type: {example_type}")

    if example_type == "basic":
        return basic_neuroflex_example(*args, **kwargs)
    elif example_type == "advanced":
        return advanced_neuroflex_example(*args, **kwargs)
    elif example_type == "quantum_nn":
        return quantum_nn_example(*args, **kwargs)
    elif example_type == "bci_integration":
        return bci_integration_example(*args, **kwargs)

def validate_example_config(config):
    """
    Validate the configuration for an example.
    """
    required_keys = ['example_type', 'parameters']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    if config['example_type'] not in SUPPORTED_EXAMPLES:
        raise ValueError(f"Unsupported example type: {config['example_type']}")

    return True

# Add any other Examples-specific utility functions or constants as needed

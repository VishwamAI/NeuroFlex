"""
NeuroFlex Examples Module

This module provides example implementations and usage demonstrations for the NeuroFlex library.
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
    'run_example'
]

def get_examples_version():
    return "1.0.0"

SUPPORTED_EXAMPLES = [
    "basic",
    "advanced",
    "quantum_nn",
    "bci_integration"
]

def initialize_examples():
    print("Initializing Examples Module...")
    # Add any necessary initialization code here

def run_example(example_type, *args, **kwargs):
    if example_type == "basic":
        return basic_neuroflex_example(*args, **kwargs)
    elif example_type == "advanced":
        return advanced_neuroflex_example(*args, **kwargs)
    elif example_type == "quantum_nn":
        return quantum_nn_example(*args, **kwargs)
    elif example_type == "bci_integration":
        return bci_integration_example(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported example type: {example_type}")

# Add any other Examples-specific utility functions or constants as needed

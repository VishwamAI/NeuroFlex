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
NeuroFlex Core Neural Networks Module

This module provides core neural network architectures and utilities, supporting frameworks like TensorFlow, PyTorch, and JAX.

Recent updates:
- Enhanced support for TensorFlow and PyTorch submodules
- Improved model creation and initialization functions
- Added support for advanced thinking models like CDSTDP
- Updated version to match main NeuroFlex version
"""

from .jax.pytorch_module_converted import PyTorchModel
from .advanced_thinking import CDSTDP, create_cdstdp
from .model import SelfCuringAlgorithm
# NeuroFlex import removed to avoid circular import
from .cnn import CNN, create_cnn
from .lstm import LSTMModule
from .rnn import LRNN
from .machinelearning import MachineLearning

# Importing TensorFlow and PyTorch submodules
from .tensorflow import *
from .pytorch import *

__all__ = [
    'PyTorchModel',
    'CDSTDP',
    'create_cdstdp',
    'NeuroFlex',
    'SelfCuringAlgorithm',
    'CNN',
    'create_cnn',
    'LSTMModule',
    'LRNN',
    'MachineLearning',
    'get_core_nn_version',
    'SUPPORTED_FRAMEWORKS',
    'initialize_core_nn',
    'create_model',
    'validate_model_config'
]

def get_core_nn_version():
    return "0.1.3"  # Updated to match main NeuroFlex version

SUPPORTED_FRAMEWORKS = ['TensorFlow', 'PyTorch', 'JAX']

def initialize_core_nn():
    print("Initializing Core Neural Networks Module...")
    # Add any necessary initialization code here

def create_model(framework, model_type, *args, **kwargs):
    if framework.lower() not in [f.lower() for f in SUPPORTED_FRAMEWORKS]:
        raise ValueError(f"Unsupported framework: {framework}")

    if framework.lower() == 'tensorflow':
        # Import and use TensorFlow-specific functions
        from .tensorflow import create_tensorflow_model
        return create_tensorflow_model(*args, **kwargs)
    elif framework.lower() == 'pytorch':
        # Import and use PyTorch-specific functions
        from .pytorch import create_pytorch_model
        return create_pytorch_model(*args, **kwargs)
    elif framework.lower() == 'jax':
        # TODO: Implement JAX model creation
        raise NotImplementedError("JAX model creation not yet implemented")

def validate_model_config(config):
    """
    Validate the configuration for a neural network model.
    """
    required_keys = ['framework', 'model_type', 'input_shape', 'output_dim', 'hidden_layers']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    if config['framework'] not in SUPPORTED_FRAMEWORKS:
        raise ValueError(f"Unsupported framework: {config['framework']}")

    return True

# Add any other utility functions or constants as needed

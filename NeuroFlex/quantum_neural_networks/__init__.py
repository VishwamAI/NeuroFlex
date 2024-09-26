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
NeuroFlex Quantum Neural Networks Module

This module implements quantum neural networks with self-healing capabilities
and adaptive algorithms. It integrates concepts from quantum computing with
classical machine learning techniques to create robust and adaptive quantum-classical
hybrid models.

Recent updates:
- Integrated self-healing mechanisms and adaptive learning
- Implemented performance monitoring and diagnosis
- Enhanced quantum circuit with advanced encoding and variational layers
- Updated version to match main NeuroFlex version
"""

from .quantum_nn_module import QuantumNeuralNetwork
from ..cognitive_architectures import (
    PERFORMANCE_THRESHOLD,
    UPDATE_INTERVAL,
    LEARNING_RATE_ADJUSTMENT,
    MAX_HEALING_ATTEMPTS
)

__all__ = [
    'QuantumNeuralNetwork',
    'PERFORMANCE_THRESHOLD',
    'UPDATE_INTERVAL',
    'LEARNING_RATE_ADJUSTMENT',
    'MAX_HEALING_ATTEMPTS',
    'get_quantum_nn_version',
    'SUPPORTED_QUANTUM_MODELS',
    'initialize_quantum_nn',
    'create_quantum_nn',
    'validate_quantum_nn_config'
]

def get_quantum_nn_version():
    return "0.1.3"  # Updated to match main NeuroFlex version

SUPPORTED_QUANTUM_MODELS = [
    "QuantumNeuralNetwork"
]

def initialize_quantum_nn():
    print("Initializing Quantum Neural Networks Module...")
    # Add any necessary initialization code here

def create_quantum_nn(model_type, *args, **kwargs):
    if model_type not in SUPPORTED_QUANTUM_MODELS:
        raise ValueError(f"Unsupported quantum model type: {model_type}")

    if model_type == "QuantumNeuralNetwork":
        return QuantumNeuralNetwork(*args, **kwargs)

def validate_quantum_nn_config(config):
    """
    Validate the configuration for a quantum neural network model.
    """
    required_keys = ['model_type', 'n_qubits', 'n_layers']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    if config['model_type'] not in SUPPORTED_QUANTUM_MODELS:
        raise ValueError(f"Unsupported quantum model type: {config['model_type']}")

    return True

# Add any other Quantum Neural Networks-specific utility functions or constants as needed

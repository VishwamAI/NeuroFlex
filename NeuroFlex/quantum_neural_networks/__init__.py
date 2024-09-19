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
    'create_quantum_nn'
]

def get_quantum_nn_version():
    return "1.0.0"

SUPPORTED_QUANTUM_MODELS = [
    "QuantumNeuralNetwork"
]

def initialize_quantum_nn():
    print("Initializing Quantum Neural Networks Module...")
    # Add any necessary initialization code here

def create_quantum_nn(model_type, *args, **kwargs):
    if model_type == "QuantumNeuralNetwork":
        return QuantumNeuralNetwork(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported quantum model type: {model_type}")

# Add any other Quantum Neural Networks-specific utility functions or constants as needed

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
    'MAX_HEALING_ATTEMPTS'
]

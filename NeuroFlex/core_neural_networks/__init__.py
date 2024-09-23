"""
NeuroFlex Core Neural Networks Module

This module provides core neural network architectures and utilities, supporting frameworks like TensorFlow, PyTorch, and JAX.

Recent updates:
- Enhanced support for TensorFlow and PyTorch submodules
- Improved model creation and initialization functions
- Added support for advanced thinking models like CDSTDP
"""

from .jax.pytorch_module_converted import PyTorchModel
from .advanced_thinking import CDSTDP, create_cdstdp
from .model import NeuroFlex, SelfCuringAlgorithm
from .cnn import CNN, create_cnn
from .lstm import LSTMModule
from .rnn import LRNN
from .machinelearning import MachineLearning

# Importing TensorFlow and PyTorch submodules
from .tensorflow import *
from .pytorch import *

__all__ = [
    "PyTorchModel",
    "CDSTDP",
    "create_cdstdp",
    "NeuroFlex",
    "SelfCuringAlgorithm",
    "CNN",
    "create_cnn",
    "LSTMModule",
    "LRNN",
    "MachineLearning",
    "get_core_nn_version",
    "SUPPORTED_FRAMEWORKS",
    "initialize_core_nn",
    "create_model",
]


def get_core_nn_version():
    return "1.0.0"


SUPPORTED_FRAMEWORKS = ["TensorFlow", "PyTorch", "JAX"]


def initialize_core_nn():
    print("Initializing Core Neural Networks Module...")
    # Add any necessary initialization code here


def create_model(framework, model_type, *args, **kwargs):
    if framework.lower() == "tensorflow":
        # Import and use TensorFlow-specific functions
        from .tensorflow import create_tensorflow_model

        return create_tensorflow_model(*args, **kwargs)
    elif framework.lower() == "pytorch":
        # Import and use PyTorch-specific functions
        from .pytorch import create_pytorch_model

        return create_pytorch_model(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported framework: {framework}")


# Add any other utility functions or constants as needed

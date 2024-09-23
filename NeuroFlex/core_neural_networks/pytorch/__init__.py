# This __init__.py file marks the pytorch directory as a Python module.
# It allows Python to recognize this directory as a package and enables importing from it.

from .pytorch_module import (
    PyTorchModel,
    create_pytorch_model,
    train_pytorch_model,
    pytorch_predict,
)

__all__ = [
    "PyTorchModel",
    "create_pytorch_model",
    "train_pytorch_model",
    "pytorch_predict",
]


def get_pytorch_version():
    import torch

    return torch.__version__


SUPPORTED_PYTORCH_LAYERS = ["Linear", "Conv2d", "LSTM", "GRU"]


def create_pytorch_mlp(input_size, hidden_sizes, output_size):
    import torch.nn as nn

    layers = []
    sizes = [input_size] + hidden_sizes + [output_size]
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def initialize_pytorch():
    print("Initializing PyTorch Module...")
    # Add any necessary initialization code here


# Add any other PyTorch-specific utility functions or constants as needed

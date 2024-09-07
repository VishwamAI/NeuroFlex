import warnings

# This __init__.py file marks the directory as a Python module.
# It allows Python to recognize this directory as a package and enables importing from it.

# DEPRECATION WARNING: This module is deprecated and will be removed in a future version.
# The 'tensorflow' directory now contains PyTorch implementations and should be renamed.
warnings.warn("The 'tensorflow' module is deprecated and contains PyTorch implementations. "
              "It will be removed in a future version. Please use the 'pytorch' module instead.",
              DeprecationWarning, stacklevel=2)

# TODO: Rename this directory to 'legacy_tensorflow' or merge with 'pytorch' directory

from .tensorflow_module import PyTorchModel, create_pytorch_model, train_pytorch_model, pytorch_predict
from .tensorflow_convolutions import PyTorchConvolutions, create_conv_model, train_conv_model, conv_predict

__all__ = [
    'PyTorchModel', 'create_pytorch_model', 'train_pytorch_model', 'pytorch_predict',
    'PyTorchConvolutions', 'create_conv_model', 'train_conv_model', 'conv_predict'
]

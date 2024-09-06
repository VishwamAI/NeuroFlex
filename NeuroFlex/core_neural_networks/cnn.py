import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Callable
from NeuroFlex.utils.utils import get_activation_function

class CNNBlock(nn.Module):
    """
    A Convolutional Neural Network (CNN) block for the NeuroFlexNN.

    This module encapsulates the CNN-specific functionality that was previously
    part of the NeuroFlexNN class.

    Args:
        features (Tuple[int, ...]): The number of filters in each convolutional layer.
        conv_dim (int): The dimension of convolution (2 or 3).
        dtype (jnp.dtype): The data type to use for computations.
        activation (Callable): The activation function to use in the network.
    """
    features: Tuple[int, ...]
    conv_dim: int
    dtype: jnp.dtype
    activation: Callable

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Forward pass of the CNN block.

        Args:
            x (jnp.ndarray): Input tensor.
            train (bool): Whether to run in training mode. Defaults to True.

        Returns:
            jnp.ndarray: Output tensor after passing through the CNN layers.
        """
        kernel_size = (3,) * self.conv_dim
        pool_window = (2,) * self.conv_dim
        pool_strides = (2,) * self.conv_dim

        for feat in self.features:
            x = nn.Conv(features=feat, kernel_size=kernel_size, dtype=self.dtype)(x)
            x = self.activation(x)
            x = nn.max_pool(x, window_shape=pool_window, strides=pool_strides)

        return x

def create_cnn_block(features: Tuple[int, ...], conv_dim: int, dtype: jnp.dtype, activation: Callable) -> CNNBlock:
    """
    Factory function to create a CNNBlock instance.

    Args:
        features (Tuple[int, ...]): The number of filters in each convolutional layer.
        conv_dim (int): The dimension of convolution (2 or 3).
        dtype (jnp.dtype): The data type to use for computations.
        activation (Callable): The activation function to use in the network.

    Returns:
        CNNBlock: An instance of the CNNBlock.
    """
    return CNNBlock(features=features, conv_dim=conv_dim, dtype=dtype, activation=activation)

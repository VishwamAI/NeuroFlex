import jax
import jax.numpy as jnp
import tensorflow as tf
from typing import Dict, Any, Union

def create_backend(backend: str = 'jax'):
    """
    Create a backend object for either JAX or TensorFlow.

    Args:
        backend (str): Either 'jax' or 'tensorflow'.

    Returns:
        Dict[str, Any]: A dictionary containing backend-specific functions and modules.
    """
    if backend == 'jax':
        return {
            'numpy': jnp,
            'random': jax.random,
            'nn': jax.nn,
            'optimizers': jax.experimental.optimizers,
        }
    elif backend == 'tensorflow':
        return {
            'numpy': tf.numpy_function,
            'random': tf.random,
            'nn': tf.keras.layers,
            'optimizers': tf.keras.optimizers,
        }
    else:
        raise ValueError(f"Unsupported backend: {backend}")

def convert_array(array: Union[jnp.ndarray, tf.Tensor], target_backend: str) -> Union[jnp.ndarray, tf.Tensor]:
    """
    Convert an array between JAX and TensorFlow.

    Args:
        array (Union[jnp.ndarray, tf.Tensor]): The input array.
        target_backend (str): The target backend ('jax' or 'tensorflow').

    Returns:
        Union[jnp.ndarray, tf.Tensor]: The converted array.
    """
    if target_backend == 'jax':
        if isinstance(array, tf.Tensor):
            return jnp.array(array.numpy())
        return array
    elif target_backend == 'tensorflow':
        if isinstance(array, jnp.ndarray):
            return tf.convert_to_tensor(array)
        return array
    else:
        raise ValueError(f"Unsupported target backend: {target_backend}")

def get_activation_function(activation_name: str, backend: str):
    """
    Get the activation function for the specified backend.

    Args:
        activation_name (str): Name of the activation function.
        backend (str): Either 'jax' or 'tensorflow'.

    Returns:
        Callable: The activation function.
    """
    if backend == 'jax':
        return getattr(jax.nn, activation_name)
    elif backend == 'tensorflow':
        return getattr(tf.keras.activations, activation_name)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

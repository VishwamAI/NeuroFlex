# Hybrid Neural Network Usage Guide

## Introduction

The HybridNeuralNetwork class in NeuroFlex provides a flexible neural network implementation that supports both PyTorch and JAX/Flax frameworks. This guide will walk you through the usage of this hybrid model, including initialization, forward pass, training, and advanced features.

## Installation

Before using the HybridNeuralNetwork, ensure you have installed the required dependencies:

```bash
pip install torch jax flax optax
```

## Importing the HybridNeuralNetwork

```python
from src.core_neural_networks.neural_networks import HybridNeuralNetwork
```

## Initializing the Model

The HybridNeuralNetwork can be initialized with either PyTorch or JAX as the backend:

```python
# PyTorch initialization
pytorch_model = HybridNeuralNetwork(input_size=10, hidden_size=20, output_size=5, framework='pytorch')

# JAX initialization
jax_model = HybridNeuralNetwork(input_size=10, hidden_size=20, output_size=5, framework='jax')
```

## Forward Pass

Performing a forward pass is straightforward for both frameworks:

```python
import torch
import jax.numpy as jnp

# PyTorch forward pass
pytorch_input = torch.randn(1, 10)
pytorch_output = pytorch_model.forward(pytorch_input)

# JAX forward pass
jax_input = jnp.ones((1, 10))
jax_output = jax_model.forward(jax_input)
```

## Training the Model

The HybridNeuralNetwork provides a unified training interface for both frameworks:

```python
# PyTorch training
pytorch_x = torch.randn(100, 10)
pytorch_y = torch.randn(100, 5)
pytorch_model.train(pytorch_x, pytorch_y, epochs=100, learning_rate=0.01)

# JAX training
jax_x = jnp.ones((100, 10))
jax_y = jnp.ones((100, 5))
jax_model.train(jax_x, jax_y, epochs=100, learning_rate=0.01)
```

## Mixed Precision Operations

The HybridNeuralNetwork supports mixed precision operations:

```python
pytorch_model.mixed_precision_operations()
jax_model.mixed_precision_operations()
```

Note: For JAX, mixed precision is handled automatically in most operations.

## Framework Conversion

The HybridNeuralNetwork class provides methods for converting between PyTorch and JAX frameworks:

```python
pytorch_model.convert_to_jax()  # Convert PyTorch model to JAX
jax_model.convert_to_pytorch()  # Convert JAX model to PyTorch
```

Note: These conversion methods are placeholders and need to be implemented.

## Conclusion

The HybridNeuralNetwork provides a unified interface for working with neural networks in both PyTorch and JAX frameworks. This flexibility allows users to leverage the strengths of both frameworks within a single model architecture.

## Future Improvements

- Implement different activation functions and optimizers
- Add model saving and loading functionality
- Implement methods for model evaluation and performance metrics
- Complete the implementation of framework conversion methods

For more detailed information, refer to the source code and unit tests in the NeuroFlex repository.

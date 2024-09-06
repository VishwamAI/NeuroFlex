import unittest
import torch
import jax
import jax.numpy as jnp
from flax import linen as flax_nn
from NeuroFlex.core_neural_networks import HybridNeuralNetwork

class TestHybridNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.input_size = 10
        self.hidden_size = 20
        self.output_size = 5

    def test_pytorch_initialization(self):
        model = HybridNeuralNetwork(self.input_size, self.hidden_size, self.output_size, framework='pytorch')
        self.assertIsInstance(model, HybridNeuralNetwork)
        self.assertEqual(model.framework, 'pytorch')
        self.assertIsInstance(model.model, torch.nn.Sequential)

    def test_jax_initialization(self):
        model = HybridNeuralNetwork(self.input_size, self.hidden_size, self.output_size, framework='jax')
        self.assertIsInstance(model, HybridNeuralNetwork)
        self.assertEqual(model.framework, 'jax')
        self.assertIsInstance(model.model, flax_nn.Module)
        self.assertIsInstance(model.params, dict)

    def test_pytorch_forward(self):
        model = HybridNeuralNetwork(self.input_size, self.hidden_size, self.output_size, framework='pytorch')
        x = torch.randn(1, self.input_size)
        output = model.forward(x)
        self.assertEqual(output.shape, (1, self.output_size))

    def test_jax_forward(self):
        model = HybridNeuralNetwork(self.input_size, self.hidden_size, self.output_size, framework='jax')
        x = jnp.ones((1, self.input_size))
        output = model.forward(x)
        self.assertEqual(output.shape, (1, self.output_size))

    def test_pytorch_train(self):
        model = HybridNeuralNetwork(self.input_size, self.hidden_size, self.output_size, framework='pytorch')
        x = torch.randn(10, self.input_size)
        y = torch.randn(10, self.output_size)
        model.train(x, y, epochs=10, learning_rate=0.01)
        # Just checking if training runs without errors

    def test_jax_train(self):
        model = HybridNeuralNetwork(self.input_size, self.hidden_size, self.output_size, framework='jax')
        x = jnp.ones((10, self.input_size))
        y = jnp.ones((10, self.output_size))
        model.train(x, y, epochs=10, learning_rate=0.01)
        # Just checking if training runs without errors

    def test_pytorch_mixed_precision(self):
        model = HybridNeuralNetwork(self.input_size, self.hidden_size, self.output_size, framework='pytorch')
        model.mixed_precision_operations()
        self.assertTrue(all(param.dtype == torch.float16 for param in model.model.parameters()))

    def test_jax_mixed_precision(self):
        model = HybridNeuralNetwork(self.input_size, self.hidden_size, self.output_size, framework='jax')
        model.mixed_precision_operations()
        # JAX mixed precision is handled by default, so we just check if the operation doesn't raise an error

    def test_invalid_framework(self):
        with self.assertRaises(ValueError):
            HybridNeuralNetwork(self.input_size, self.hidden_size, self.output_size, framework='invalid')

if __name__ == '__main__':
    unittest.main()

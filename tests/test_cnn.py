import unittest
import jax
import jax.numpy as jnp
import flax.linen as nn
from NeuroFlex.core_neural_networks import CNNBlock, create_cnn_block

class TestCNN(unittest.TestCase):
    def setUp(self):
        self.input_shape = (28, 28, 1)
        self.features = (32, 64)
        self.num_classes = 10
        self.conv_dim = 2
        self.dtype = jnp.float32
        self.activation = nn.relu

    def test_cnn_initialization(self):
        cnn = create_cnn_block(features=self.features, conv_dim=self.conv_dim, dtype=self.dtype, activation=self.activation)
        self.assertIsInstance(cnn, CNNBlock)

    def test_cnn_forward_pass(self):
        cnn = create_cnn_block(features=self.features, conv_dim=self.conv_dim, dtype=self.dtype, activation=self.activation)
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (1,) + self.input_shape)
        params = cnn.init(key, x)
        output = cnn.apply(params, x)

        expected_output_shape = (1, jnp.prod(jnp.array(self.input_shape[:-1]) // (2 ** len(self.features))) * self.features[-1])
        self.assertEqual(output.shape, expected_output_shape)

    def test_cnn_output_shape(self):
        cnn = create_cnn_block(features=self.features, conv_dim=self.conv_dim, dtype=self.dtype, activation=self.activation)
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (32,) + self.input_shape)
        params = cnn.init(key, x)
        output = cnn.apply(params, x)

        expected_output_shape = (32, jnp.prod(jnp.array(self.input_shape[:-1]) // (2 ** len(self.features))) * self.features[-1])
        self.assertEqual(output.shape, expected_output_shape)

    def test_error_handling(self):
        with self.assertRaises(ValueError):
            create_cnn_block(features=(), conv_dim=self.conv_dim, dtype=self.dtype, activation=self.activation)

        with self.assertRaises(ValueError):
            create_cnn_block(features=self.features, conv_dim=4, dtype=self.dtype, activation=self.activation)

    def test_model_training(self):
        cnn = create_cnn_block(features=self.features, conv_dim=self.conv_dim, dtype=self.dtype, activation=self.activation)
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (32,) + self.input_shape)
        params = cnn.init(key, x)

        @jax.jit
        def loss_fn(params, x):
            output = cnn.apply(params, x)
            return jnp.mean(output**2)  # Simple loss for demonstration

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params, x)

        # Check if gradients are non-zero
        self.assertTrue(jax.tree_util.tree_reduce(lambda acc, x: acc or jnp.any(x != 0), grads, False))

if __name__ == '__main__':
    unittest.main()

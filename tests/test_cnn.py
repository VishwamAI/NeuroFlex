import unittest
import jax
import jax.numpy as jnp
import flax.linen as nn
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from NeuroFlex.cnn import CNNBlock, create_cnn_block

class TestCNN(unittest.TestCase):
    def setUp(self):
        self.rng = jax.random.PRNGKey(0)
        self.features = (32, 64)
        self.conv_dim = 2
        self.dtype = jnp.float32
        self.activation = nn.relu

    def test_cnn_block_initialization(self):
        cnn_block = CNNBlock(features=self.features, conv_dim=self.conv_dim, dtype=self.dtype, activation=self.activation)
        self.assertIsInstance(cnn_block, CNNBlock)
        self.assertEqual(cnn_block.features, self.features)
        self.assertEqual(cnn_block.conv_dim, self.conv_dim)
        self.assertEqual(cnn_block.dtype, self.dtype)
        self.assertEqual(cnn_block.activation, self.activation)

    def test_cnn_block_forward_pass(self):
        cnn_block = CNNBlock(features=self.features, conv_dim=self.conv_dim, dtype=self.dtype, activation=self.activation)
        input_shape = (1, 28, 28, 1)
        x = jnp.ones(input_shape)
        params = cnn_block.init(self.rng, x)['params']
        output = cnn_block.apply({'params': params}, x)

        self.assertEqual(output.shape[0], input_shape[0])
        self.assertLess(output.shape[1], input_shape[1])
        self.assertLess(output.shape[2], input_shape[2])
        self.assertEqual(output.shape[3], self.features[-1])
        self.assertEqual(output.dtype, self.dtype)

    def test_create_cnn_block(self):
        cnn_block = create_cnn_block(features=self.features, conv_dim=self.conv_dim, dtype=self.dtype, activation=self.activation)
        self.assertIsInstance(cnn_block, CNNBlock)
        self.assertEqual(cnn_block.features, self.features)
        self.assertEqual(cnn_block.conv_dim, self.conv_dim)
        self.assertEqual(cnn_block.dtype, self.dtype)
        self.assertEqual(cnn_block.activation, self.activation)

    def test_3d_convolution(self):
        features_3d = (16, 32)
        conv_dim_3d = 3
        cnn_block_3d = CNNBlock(features=features_3d, conv_dim=conv_dim_3d, dtype=self.dtype, activation=self.activation)
        input_shape_3d = (1, 16, 16, 16, 1)
        x_3d = jnp.ones(input_shape_3d)
        params_3d = cnn_block_3d.init(self.rng, x_3d)['params']
        output_3d = cnn_block_3d.apply({'params': params_3d}, x_3d)

        self.assertEqual(output_3d.shape[0], input_shape_3d[0])
        self.assertLess(output_3d.shape[1], input_shape_3d[1])
        self.assertLess(output_3d.shape[2], input_shape_3d[2])
        self.assertLess(output_3d.shape[3], input_shape_3d[3])
        self.assertEqual(output_3d.shape[4], features_3d[-1])
        self.assertEqual(output_3d.dtype, self.dtype)

if __name__ == '__main__':
    unittest.main()

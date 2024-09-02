import unittest
import jax
import jax.numpy as jnp
import flax.linen as nn
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from NeuroFlex.rnn import LRNN, create_rnn_block, LRNNCell

class TestRNN(unittest.TestCase):
    def setUp(self):
        self.rng = jax.random.PRNGKey(0)
        self.features = (32, 64)
        self.activation = nn.tanh

    def test_lrnn_initialization(self):
        lrnn = LRNN(features=self.features, activation=self.activation)
        self.assertIsInstance(lrnn, LRNN)
        self.assertEqual(lrnn.features, self.features)
        self.assertEqual(lrnn.activation, self.activation)

    def test_lrnn_forward_pass(self):
        lrnn = LRNN(features=self.features, activation=self.activation)
        input_shape = (1, 10, 32)  # (batch_size, sequence_length, input_size)
        x = jnp.ones(input_shape)
        params = lrnn.init(self.rng, x)['params']
        output, state = lrnn.apply({'params': params}, x)

        self.assertEqual(output.shape, input_shape[:-1] + (self.features[-1],))
        self.assertEqual(state.shape, (input_shape[0], self.features[-1]))

    def test_create_rnn_block(self):
        rnn_block = create_rnn_block(features=self.features, activation=self.activation)
        self.assertIsInstance(rnn_block, LRNN)
        self.assertEqual(rnn_block.features, self.features)
        self.assertEqual(rnn_block.activation, self.activation)

    def test_lrnn_cell(self):
        cell = LRNNCell(features=32, activation=self.activation)
        input_shape = (1, 32)
        state_shape = (1, 32)
        x = jnp.ones(input_shape)
        h = jnp.zeros(state_shape)
        params = cell.init(self.rng, h, x)['params']
        new_h, _ = cell.apply({'params': params}, h, x)

        self.assertEqual(new_h.shape, state_shape)

if __name__ == '__main__':
    unittest.main()

import unittest
import jax
import jax.numpy as jnp
import flax.linen as nn
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from NeuroFlex.rnn import RNNModule, create_rnn_block, RNNCell

class TestRNN(unittest.TestCase):
    def setUp(self):
        self.rng = jax.random.PRNGKey(0)
        self.features = (32, 64)
        self.activation = nn.tanh

    def test_rnn_module_initialization(self):
        rnn_module = RNNModule(features=self.features, activation=self.activation)
        self.assertIsInstance(rnn_module, RNNModule)
        self.assertEqual(rnn_module.features, self.features)
        self.assertEqual(rnn_module.activation, self.activation)

    def test_rnn_module_forward_pass(self):
        rnn_module = RNNModule(features=self.features, activation=self.activation)
        input_shape = (1, 10, 32)  # (batch_size, sequence_length, input_size)
        x = jnp.ones(input_shape)
        params = rnn_module.init(self.rng, x)['params']
        output, state = rnn_module.apply({'params': params}, x)

        self.assertEqual(output.shape, input_shape[:-1] + (self.features[-1],))
        self.assertEqual(state.shape, (input_shape[0], sum(self.features)))

    def test_create_rnn_block(self):
        rnn_block = create_rnn_block(features=self.features, activation=self.activation)
        self.assertIsInstance(rnn_block, RNNModule)
        self.assertEqual(rnn_block.features, self.features)
        self.assertEqual(rnn_block.activation, self.activation)

    def test_rnn_with_initial_state(self):
        rnn_module = RNNModule(features=self.features, activation=self.activation)
        input_shape = (1, 10, 32)  # (batch_size, sequence_length, input_size)
        x = jnp.ones(input_shape)
        initial_state = jnp.zeros((input_shape[0], sum(self.features)))
        params = rnn_module.init(self.rng, x, initial_state)['params']
        output, state = rnn_module.apply({'params': params}, x, initial_state)

        self.assertEqual(output.shape, input_shape[:-1] + (self.features[-1],))
        self.assertEqual(state.shape, (input_shape[0], sum(self.features)))

    def test_rnn_cell(self):
        cell = RNNCell(features=32, activation=self.activation)
        input_shape = (1, 32)
        state_shape = (1, 32)
        x = jnp.ones(input_shape)
        state = (jnp.zeros(state_shape),)
        params = cell.init(self.rng, state, x)['params']
        (new_state,), _ = cell.apply({'params': params}, state, x)

        self.assertEqual(new_state.shape, state_shape)

if __name__ == '__main__':
    unittest.main()

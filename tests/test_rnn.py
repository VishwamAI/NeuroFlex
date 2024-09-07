import unittest
import jax
import jax.numpy as jnp
import flax.linen as nn

from NeuroFlex.core_neural_networks.neural_networks import RNN, create_rnn_block, LRNNCell

class TestRNN(unittest.TestCase):
    def setUp(self):
        self.rng = jax.random.PRNGKey(0)
        self.features = (32, 64)
        self.activation = nn.tanh

    def test_rnn_initialization(self):
        rnn = RNN(input_size=self.features[0], hidden_size=self.features[1], output_size=self.features[-1])
        self.assertIsInstance(rnn, RNN)
        self.assertEqual(rnn.hidden_size, self.features[1])
        self.assertEqual(rnn.output_size, self.features[-1])

    def test_rnn_forward_pass(self):
        rnn = RNN(input_size=self.features[0], hidden_size=self.features[1], output_size=self.features[-1])
        input_shape = (1, 10, self.features[0])  # (batch_size, sequence_length, input_size)
        x = jnp.ones(input_shape)
        params = rnn.init(self.rng, x)['params']
        output = rnn.apply({'params': params}, x)

        self.assertEqual(output.shape, (input_shape[0], self.features[-1]))

    def test_create_rnn_block(self):
        rnn_block = create_rnn_block(features=self.features)
        self.assertIsInstance(rnn_block, RNN)
        self.assertEqual(rnn_block.hidden_size, self.features[1])
        self.assertEqual(rnn_block.output_size, self.features[-1])

    def test_rnn_cell(self):
        # RNN doesn't have a separate cell implementation in this version,
        # so we'll test the RNN's single step behavior
        rnn = RNN(input_size=32, hidden_size=32, output_size=32)
        input_shape = (1, 32)
        x = jnp.ones(input_shape)
        params = rnn.init(self.rng, x)['params']
        output = rnn.apply({'params': params}, x)

        self.assertEqual(output.shape, input_shape)

if __name__ == '__main__':
    unittest.main()

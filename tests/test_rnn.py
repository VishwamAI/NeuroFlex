import unittest
import jax
import jax.numpy as jnp
import flax.linen as nn

from NeuroFlex.core_neural_networks import LRNN, create_rnn_block, LRNNCell

class TestLRNN(unittest.TestCase):
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
        input_shape = (1, 10, self.features[0])  # (batch_size, sequence_length, input_size)
        x = jnp.ones(input_shape)
        variables = lrnn.init(self.rng, x)
        output, state = lrnn.apply(variables, x)

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
        variables = cell.init(self.rng, h, x)
        new_h, _ = cell.apply(variables, h, x)

        self.assertEqual(new_h.shape, state_shape)

    def test_lrnn_multiple_layers(self):
        multi_layer_features = (32, 64, 128)
        lrnn = LRNN(features=multi_layer_features, activation=self.activation)
        input_shape = (1, 10, multi_layer_features[0])
        x = jnp.ones(input_shape)
        variables = lrnn.init(self.rng, x)
        output, state = lrnn.apply(variables, x)

        self.assertEqual(output.shape, input_shape[:-1] + (multi_layer_features[-1],))
        self.assertEqual(state.shape, (input_shape[0], multi_layer_features[-1]))

    def test_lrnn_different_activations(self):
        activations = [nn.relu, nn.sigmoid, nn.swish]
        for activation in activations:
            lrnn = LRNN(features=self.features, activation=activation)
            input_shape = (1, 10, self.features[0])
            x = jnp.ones(input_shape)
            variables = lrnn.init(self.rng, x)
            output, state = lrnn.apply(variables, x)
            self.assertEqual(output.shape, input_shape[:-1] + (self.features[-1],))
            self.assertEqual(state.shape, (input_shape[0], self.features[-1]))

    def test_lrnn_single_layer(self):
        single_layer_features = (64,)
        lrnn = LRNN(features=single_layer_features, activation=self.activation)
        input_shape = (1, 10, single_layer_features[0])
        x = jnp.ones(input_shape)
        variables = lrnn.init(self.rng, x)
        output, state = lrnn.apply(variables, x)
        self.assertEqual(output.shape, input_shape)
        self.assertEqual(state.shape, (input_shape[0], single_layer_features[0]))

    def test_lrnn_zero_length_input(self):
        lrnn = LRNN(features=self.features, activation=self.activation)
        input_shape = (1, 0, self.features[0])  # Zero-length sequence
        x = jnp.ones(input_shape)
        variables = lrnn.init(self.rng, x)
        output, state = lrnn.apply(variables, x)
        self.assertEqual(output.shape, (1, 0, self.features[-1]))
        self.assertEqual(state.shape, (input_shape[0], self.features[-1]))

if __name__ == '__main__':
    unittest.main()

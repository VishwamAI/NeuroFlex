import unittest
import jax
import jax.numpy as jnp
import flax.linen as nn
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from NeuroFlex.lstm import LSTMModule

class TestLSTM(unittest.TestCase):
    def setUp(self):
        self.rng = jax.random.PRNGKey(0)
        self.hidden_size = 64
        self.num_layers = 2
        self.batch_size = 32
        self.seq_len = 10
        self.input_size = 32
        self.dropout_rate = 0.1

    def test_lstm_module_initialization(self):
        lstm = LSTMModule(hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.dropout_rate)
        self.assertIsInstance(lstm, LSTMModule)
        self.assertEqual(lstm.hidden_size, self.hidden_size)
        self.assertEqual(lstm.num_layers, self.num_layers)
        self.assertEqual(lstm.dropout, self.dropout_rate)

    def test_lstm_module_forward_pass(self):
        lstm = LSTMModule(hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.dropout_rate)
        input_shape = (self.batch_size, self.seq_len, self.input_size)
        x = jnp.ones(input_shape)
        params = lstm.init(self.rng, x)['params']
        output, (final_h, final_c) = lstm.apply({'params': params}, x, rngs={'dropout': self.rng})

        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_size))
        self.assertEqual(len(final_h), self.num_layers)
        self.assertEqual(len(final_c), self.num_layers)
        self.assertEqual(final_h[0].shape, (self.batch_size, self.hidden_size))
        self.assertEqual(final_c[0].shape, (self.batch_size, self.hidden_size))

    def test_lstm_module_dropout(self):
        lstm_with_dropout = LSTMModule(hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.dropout_rate)
        lstm_without_dropout = LSTMModule(hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=0.0)

        input_shape = (self.batch_size, self.seq_len, self.input_size)
        x = jnp.ones(input_shape)

        params_with_dropout = lstm_with_dropout.init(self.rng, x)['params']
        params_without_dropout = lstm_without_dropout.init(self.rng, x)['params']

        output_with_dropout, _ = lstm_with_dropout.apply({'params': params_with_dropout}, x, rngs={'dropout': self.rng}, train=True)
        output_without_dropout, _ = lstm_without_dropout.apply({'params': params_without_dropout}, x, rngs={'dropout': self.rng}, train=True)

        # Check that outputs are different when dropout is applied
        self.assertFalse(jnp.allclose(output_with_dropout, output_without_dropout))

        # Check that outputs are the same when not in training mode (dropout should not be applied)
        output_with_dropout_eval, _ = lstm_with_dropout.apply({'params': params_with_dropout}, x, rngs={'dropout': self.rng}, train=False)
        output_without_dropout_eval, _ = lstm_without_dropout.apply({'params': params_without_dropout}, x, rngs={'dropout': self.rng}, train=False)

        self.assertTrue(jnp.allclose(output_with_dropout_eval, output_without_dropout_eval))

if __name__ == '__main__':
    unittest.main()

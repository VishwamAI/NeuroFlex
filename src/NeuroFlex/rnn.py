import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Tuple

class RNNCell(nn.Module):
    features: int
    activation: Callable = nn.tanh

    @nn.compact
    def __call__(self, carry, x):
        h, = carry
        combined = jnp.concatenate([h, x], axis=-1)
        new_h = self.activation(nn.Dense(self.features)(combined))
        return (new_h,), new_h

class RNNModule(nn.Module):
    features: Tuple[int, ...]
    activation: Callable = nn.tanh

    @nn.compact
    def __call__(self, x, initial_state=None):
        batch_size, seq_len, input_dim = x.shape

        if initial_state is None:
            initial_state = self.initialize_carry(batch_size)

        def scan_fn(carry, x):
            new_carry = []
            for i, (h, cell) in enumerate(zip(carry, self.cells)):
                h, y = cell(h, x)
                new_carry.append(h)
                x = y  # Use the output of the cell as input for the next cell
            return tuple(new_carry), x

        self.cells = [RNNCell(feat, self.activation, name=f'rnn_cell_{i}')
                      for i, feat in enumerate(self.features)]

        ScanRNN = nn.scan(
            scan_fn,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )

        final_carry, outputs = ScanRNN()(initial_state, x)
        final_state = jnp.concatenate([c[0] for c in final_carry], axis=-1)

        return outputs, final_state

    def initialize_carry(self, batch_size):
        return tuple((jnp.zeros((batch_size, feat)),) for feat in self.features)

def create_rnn_block(features: Tuple[int, ...], activation: Callable = nn.tanh):
    return RNNModule(features=features, activation=activation)

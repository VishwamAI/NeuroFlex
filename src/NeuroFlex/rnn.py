import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Tuple, Any
import functools

class LRNNCell(nn.Module):
    features: int
    activation: Callable = nn.tanh

    @nn.compact
    def __call__(self, h, x):
        input_size = h.shape[-1] + x.shape[-1]
        combined = jnp.concatenate([h, x], axis=-1)
        new_h = self.activation(nn.Dense(self.features, kernel_init=nn.initializers.xavier_uniform())(combined))
        return new_h, new_h

class LRNN(nn.Module):
    features: Tuple[int, ...]
    activation: Callable = nn.tanh

    @nn.compact
    def __call__(self, x):
        batch_size, seq_len, input_dim = x.shape

        # Create a list of LRNNCell instances, one for each layer
        cells = [LRNNCell(features=feat, activation=self.activation) for feat in self.features]

        # Initial hidden state for each layer
        h = [jnp.zeros((batch_size, feat)) for feat in self.features]

        # Process the input sequence
        for t in range(seq_len):
            x_t = x[:, t, :]
            for i, cell in enumerate(cells):
                h[i], _ = cell(h[i], x_t if i == 0 else h[i-1])
            if t == 0:
                y = h[-1][:, None, :]
            else:
                y = jnp.concatenate([y, h[-1][:, None, :]], axis=1)

        return y, h[-1]

def create_rnn_block(features: Tuple[int, ...], activation: Callable = nn.tanh):
    return LRNN(features=features, activation=activation)

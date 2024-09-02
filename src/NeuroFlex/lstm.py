import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Optional

class LSTMModule(nn.Module):
    hidden_size: int
    num_layers: int = 1
    dropout: float = 0.0
    activation: Callable = nn.tanh
    recurrent_activation: Callable = nn.sigmoid
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, initial_state=None, train: bool = True):
        batch_size, seq_len, input_size = inputs.shape

        if initial_state is None:
            initial_state = self.initialize_state(batch_size)

        def lstm_step(carry, x):
            h, c = carry
            new_h, new_c = [], []
            for layer in range(self.num_layers):
                lstm_state = LSTMCell(
                    hidden_size=self.hidden_size,
                    activation=self.activation,
                    recurrent_activation=self.recurrent_activation,
                    dtype=self.dtype
                )(x, (h[layer], c[layer]))
                x = lstm_state[0]
                if layer < self.num_layers - 1 and train:
                    x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
                new_h.append(lstm_state[0])
                new_c.append(lstm_state[1])
            return (new_h, new_c), x

        initial_carry = initial_state
        (final_h, final_c), outputs = jax.lax.scan(lstm_step, initial_carry, inputs.swapaxes(0, 1))

        return outputs.swapaxes(0, 1), (final_h, final_c)

    def initialize_state(self, batch_size):
        return (
            [jnp.zeros((batch_size, self.hidden_size), dtype=self.dtype) for _ in range(self.num_layers)],
            [jnp.zeros((batch_size, self.hidden_size), dtype=self.dtype) for _ in range(self.num_layers)]
        )

class LSTMCell(nn.Module):
    hidden_size: int
    activation: Callable = nn.tanh
    recurrent_activation: Callable = nn.sigmoid
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, carry):
        h, c = carry
        gates = nn.Dense(4 * self.hidden_size, dtype=self.dtype)(jnp.concatenate([inputs, h], axis=-1))
        i, f, g, o = jnp.split(gates, 4, axis=-1)

        new_c = self.recurrent_activation(f) * c + self.recurrent_activation(i) * self.activation(g)
        new_h = self.recurrent_activation(o) * self.activation(new_c)

        return new_h, new_c

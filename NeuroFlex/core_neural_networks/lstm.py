import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, List
from NeuroFlex.utils.utils import normalize_data
from NeuroFlex.utils.descriptive_statistics import preprocess_data

class LSTMModule(nn.Module):
    hidden_size: int
    num_layers: int = 1
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, initial_state=None, train: bool = True, rngs=None):
        batch_size, seq_len, input_size = inputs.shape

        # Create LSTM cells for each layer
        lstm_cells = [nn.LSTMCell(features=self.hidden_size) for _ in range(self.num_layers)]

        if initial_state is None:
            initial_state = self.initialize_state(batch_size)

        # Process through LSTM layers
        current_input = inputs
        final_states = []

        for layer in range(self.num_layers):
            layer_state = initial_state[layer]
            layer_outputs = []

            for t in range(seq_len):
                # Get input for this time step
                x_t = current_input[:, t, :]

                # Apply LSTM cell
                layer_state, y_t = lstm_cells[layer](layer_state, x_t)

                # Store output
                layer_outputs.append(y_t)

            # Stack time steps
            layer_output = jnp.stack(layer_outputs, axis=1)

            # Store final state for this layer
            final_states.append(layer_state)

            # Set up input for next layer
            current_input = layer_output

            # Apply dropout between layers if specified
            if self.dropout > 0.0 and train and layer < self.num_layers - 1:
                dropout_rng = None if rngs is None else rngs.get('dropout')
                current_input = nn.Dropout(rate=self.dropout, deterministic=not train)(current_input, rng=dropout_rng)

        # Ensure outputs is an array and has the correct shape
        outputs = jnp.asarray(current_input)
        final_h, final_c = zip(*final_states)
        final_h = jnp.stack(final_h)
        final_c = jnp.stack(final_c)

        return outputs, (final_h, final_c)

    def initialize_state(self, batch_size) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
        return [
            (jnp.zeros((batch_size, self.hidden_size), dtype=self.dtype),
             jnp.zeros((batch_size, self.hidden_size), dtype=self.dtype))
            for _ in range(self.num_layers)
        ]

# MIT License
#
# Copyright (c) 2024 VishwamAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

class ASTModel(nn.Module):
    """
    Attention Schema Theory (AST) Model

    This class implements a basic model of Attention Schema Theory,
    which proposes that the brain creates a simplified model of attention
    to help predict and control cognitive processes.
    """

    attention_dim: int
    hidden_dim: int

    def setup(self):
        self.attention_schema = nn.Dense(self.attention_dim)
        self.attention_control = nn.Dense(self.hidden_dim)
        self.output_layer = nn.Dense(self.attention_dim)

    def __call__(self, inputs):
        # Create an attention schema
        schema = nn.relu(self.attention_schema(inputs))

        # Use the schema to control attention
        control = nn.sigmoid(self.attention_control(schema))

        # Apply attention control to inputs
        attended = inputs * control

        # Generate output based on attended inputs
        output = self.output_layer(attended)

        return output, schema

    def update_schema(self, inputs, feedback):
        """
        Update the attention schema based on feedback
        """
        updated_schema = self.attention_schema(inputs) + 0.1 * feedback
        return updated_schema

    def simulate_awareness(self, inputs):
        """
        Simulate awareness of attention processes
        """
        _, schema = self(inputs)
        awareness = jnp.mean(schema, axis=-1)
        return awareness

def create_train_state(rng, model, learning_rate):
    """Create initial training state"""
    params = model.init(rng, jnp.ones([1, 64]))
    tx = optax.adam(learning_rate)
    return optax.InjectHyperparamsState(step=0, params=params, tx=tx, opt_state=tx.init(params))

@jax.jit
def train_step(state, batch):
    """Perform a single training step"""
    def loss_fn(params):
        output, _ = state.apply_fn({'params': params}, batch)
        return jnp.mean((output - batch) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# Example usage
if __name__ == "__main__":
    # Initialize the model and training state
    model = ASTModel(attention_dim=64, hidden_dim=128)
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, model, learning_rate=1e-3)

    # Generate some dummy input data
    inputs = jnp.array(np.random.randn(10, 64))

    # Training loop
    for epoch in range(100):
        state, loss = train_step(state, inputs)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    # Run the trained model
    output, schema = model.apply({'params': state.params}, inputs)
    print("Output shape:", output.shape)
    print("Schema shape:", schema.shape)

    # Simulate awareness
    awareness = model.apply({'params': state.params}, inputs, method=model.simulate_awareness)
    print("Awareness level:", awareness)

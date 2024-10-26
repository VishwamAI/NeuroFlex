import jax.numpy as jnp
import flax.linen as nn
from flax.linen import sigmoid, tanh
from dataclasses import field

class CustomDense(nn.Module):
    features: int
    input_size: int
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs):
        kernel = self.param(
            'kernel',
            nn.initializers.orthogonal(),
            (self.input_size, self.features),  # Shape is correct: (input_size, features)
            self.param_dtype
        )

        y = jnp.dot(inputs, kernel)  # inputs: (batch, input_size) @ kernel: (input_size, features)
        if self.use_bias:
            bias = self.param(
                'bias',
                nn.initializers.zeros,
                (self.features,),
                self.param_dtype
            )
            y = y + bias
        return y

python
class LongTermMemory(nn.Module):
    memory_size: int
    input_size: int

    def setup(self):
        # Input projection maps from input_size to memory_size
        self.input_projection = CustomDense(
            features=self.memory_size,
            input_size=self.input_size,
            use_bias=True,
            dtype=jnp.float32
        )

        # Initialize GRU cell with memory size as features
        self.gru = nn.GRUCell(
            num_features=self.memory_size,  # Specify number of hidden units
            gate_fn=sigmoid,
            activation_fn=tanh,
            kernel_init=nn.initializers.orthogonal(),
            recurrent_kernel_init=nn.initializers.orthogonal(),
            bias_init=nn.initializers.zeros,
            dtype=jnp.float32,
            param_dtype=jnp.float32
        )

        # Output projection maintains memory_size dimensions
        self.output_projection = CustomDense(
            features=self.memory_size,
            input_size=self.memory_size,  # Input comes from GRU output which has memory_size features
            use_bias=True,
            dtype=jnp.float32
        )

    def __call__(self, x, carry=None):
        # Project input to match memory size
        projected_x = self.input_projection(x)  # x shape: (batch_size, input_size) -> (batch_size, memory_size)

        # Initialize carry state if None
        if carry is None:
            carry = jnp.zeros((x.shape[0], self.memory_size))

        # Process with GRU cell - both outputs are the same new carry state
        new_carry, _ = self.gru(carry, projected_x)

        # Project the new carry state through output projection
        y = self.output_projection(new_carry)

        return new_carry, y

def create_long_term_memory(memory_size, input_size):
    return LongTermMemory(memory_size=memory_size, input_size=input_size)

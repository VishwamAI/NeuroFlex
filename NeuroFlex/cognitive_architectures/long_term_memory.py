import jax.numpy as jnp
import flax.linen as nn

class LongTermMemory(nn.Module):
    memory_size: int

    @nn.compact
    def __call__(self, x, state):
        gru = nn.GRUCell()
        new_state, y = gru(x, state)
        return new_state, y

def create_long_term_memory(memory_size):
    return LongTermMemory(memory_size=memory_size)

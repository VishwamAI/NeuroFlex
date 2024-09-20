import jax
import jax.numpy as jnp
import haiku as hk
from .higher_order_theories import HigherOrderTheories

class AttentionMechanism(hk.Module):
    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

    def __call__(self, inputs):
        batch_size, seq_len, _ = inputs.shape
        qkv = hk.Linear(3 * self.num_heads * self.head_dim)(inputs)
        qkv = jnp.reshape(qkv, (batch_size, seq_len, 3, self.num_heads, self.head_dim))
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attention = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) / jnp.sqrt(self.head_dim)
        attention = jax.nn.softmax(attention, axis=-1)

        output = jnp.matmul(attention, v)
        output = jnp.transpose(output, (0, 2, 1, 3))
        output = jnp.reshape(output, (batch_size, seq_len, -1))

        return output

class WorkingMemory(hk.Module):
    def __init__(self, memory_size: int, hidden_dim: int):
        super().__init__()
        self.memory_size = memory_size
        self.hidden_dim = hidden_dim

    def __call__(self, inputs, prev_memory):
        # Reduce sequence dimension by taking the mean across the sequence
        inputs_mean = jnp.mean(inputs, axis=1)

        # Ensure inputs_mean and prev_memory have the same number of dimensions
        if inputs_mean.ndim < prev_memory.ndim:
            inputs_mean = jnp.expand_dims(inputs_mean, axis=1)
        elif prev_memory.ndim < inputs_mean.ndim:
            prev_memory = jnp.expand_dims(prev_memory, axis=1)

        # Concatenate along the last dimension
        combined = jnp.concatenate([inputs_mean, prev_memory], axis=-1)

        gate = jax.nn.sigmoid(hk.Linear(self.memory_size)(combined))
        update = jnp.tanh(hk.Linear(self.memory_size)(combined))
        new_memory = gate * update + (1 - gate) * prev_memory
        return new_memory

class CustomCognitiveModel(hk.Module):
    def __init__(self, num_attention_heads: int, attention_head_dim: int, working_memory_size: int, hidden_dim: int, thalamic_gate_size: int = 64, attention_schema_size: int = 128):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.working_memory_size = working_memory_size
        self.hidden_dim = hidden_dim
        self.thalamic_gate_size = thalamic_gate_size
        self.attention_schema_size = attention_schema_size

    def __call__(self, inputs, prev_memory, prev_attention_state):
        attention = AttentionMechanism(num_heads=self.num_attention_heads, head_dim=self.attention_head_dim)
        working_memory = WorkingMemory(memory_size=self.working_memory_size, hidden_dim=self.hidden_dim)

        attended = attention(inputs)
        new_memory = working_memory(attended, prev_memory)

        # Attention Schema processing
        attention_state = hk.Linear(self.attention_schema_size)(attended)
        attention_control = jax.nn.sigmoid(hk.Linear(self.attention_head_dim)(attention_state))
        # Reshape attention_control to match attended's shape
        print("attended shape:", attended.shape)
        print("attention_control shape before reshape:", attention_control.shape)
        attention_control = jnp.reshape(attention_control, attended.shape[:2] + (self.attention_head_dim,))
        print("attention_control shape after reshape:", attention_control.shape)
        attended = attended * jnp.repeat(attention_control, attended.shape[-1] // self.attention_head_dim, axis=-1)

        # Thalamocortical processing
        thalamic_output = jax.nn.sigmoid(hk.Linear(self.thalamic_gate_size)(new_memory))
        # Ensure thalamic_output and new_memory have compatible shapes
        thalamic_output = hk.Linear(new_memory.shape[-1])(thalamic_output)
        cortical_input = thalamic_output * new_memory
        integrated_output = jax.nn.relu(hk.Linear(self.hidden_dim)(cortical_input))

        # Social cognition processing
        # Ensure integrated_output and attention_state have compatible shapes for concatenation
        integrated_output = jnp.reshape(integrated_output, (integrated_output.shape[0], 1, -1))
        attention_state = jnp.reshape(attention_state, (attention_state.shape[0], 1, -1))
        social_input = jnp.concatenate([integrated_output, attention_state], axis=-1)
        social_output = jax.nn.relu(hk.Linear(self.hidden_dim)(social_input))

        # Higher-Order Theories processing
        hot = HigherOrderTheories(hidden_dim=self.hidden_dim, output_dim=self.hidden_dim)
        hot_output = hot.apply(hot.init(jax.random.PRNGKey(0), integrated_output), None, integrated_output)

        # Combine integrated output, social output, and higher-order output
        combined_output = jnp.concatenate([integrated_output, social_output, hot_output], axis=-1)

        # Reduce sequence dimension by taking the mean across the sequence
        combined_output_mean = jnp.mean(combined_output, axis=1, keepdims=True)
        output = hk.Linear(self.hidden_dim)(combined_output_mean)
        # Ensure output shape is (batch_size, hidden_dim) without using squeeze
        output = jnp.reshape(output, (-1, self.hidden_dim))
        return output, new_memory, attention_state

    def init_memory(self, batch_size):
        return jnp.zeros((batch_size, self.working_memory_size))

    def init_attention_state(self, batch_size):
        return jnp.zeros((batch_size, self.attention_schema_size))

def create_custom_cognitive_model(num_attention_heads=4, attention_head_dim=64, working_memory_size=256, hidden_dim=512, attention_schema_size=128):
    return CustomCognitiveModel(
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        working_memory_size=working_memory_size,
        hidden_dim=hidden_dim,
        attention_schema_size=attention_schema_size
    )

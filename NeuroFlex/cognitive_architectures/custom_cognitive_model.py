import jax
import jax.numpy as jnp
import flax.linen as nn

class AttentionMechanism(nn.Module):
    num_heads: int
    head_dim: int

    @nn.compact
    def __call__(self, inputs):
        batch_size, seq_len, _ = inputs.shape
        qkv = nn.Dense(3 * self.num_heads * self.head_dim)(inputs)
        qkv = jnp.reshape(qkv, (batch_size, seq_len, 3, self.num_heads, self.head_dim))
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attention = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) / jnp.sqrt(self.head_dim)
        attention = nn.softmax(attention, axis=-1)

        output = jnp.matmul(attention, v)
        output = jnp.transpose(output, (0, 2, 1, 3))
        output = jnp.reshape(output, (batch_size, seq_len, -1))

        return output

class WorkingMemory(nn.Module):
    memory_size: int
    hidden_dim: int

    @nn.compact
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

        gate = nn.sigmoid(nn.Dense(self.memory_size)(combined))
        update = nn.tanh(nn.Dense(self.memory_size)(combined))
        new_memory = gate * update + (1 - gate) * prev_memory
        return new_memory

class CustomCognitiveModel(nn.Module):
    num_attention_heads: int
    attention_head_dim: int
    working_memory_size: int
    hidden_dim: int
    thalamic_gate_size: int = 64

    def setup(self):
        self.attention = AttentionMechanism(num_heads=self.num_attention_heads, head_dim=self.attention_head_dim)
        self.working_memory = WorkingMemory(memory_size=self.working_memory_size, hidden_dim=self.hidden_dim)
        self.thalamic_gate = nn.Dense(self.thalamic_gate_size)
        self.cortical_integration = nn.Dense(self.hidden_dim)
        self.output_layer = nn.Dense(self.hidden_dim)

    @nn.compact
    def __call__(self, inputs, prev_memory):
        attended = self.attention(inputs)
        new_memory = self.working_memory(attended, prev_memory)

        # Thalamocortical processing
        thalamic_output = nn.sigmoid(self.thalamic_gate(new_memory))
        # Ensure thalamic_output and new_memory have compatible shapes
        thalamic_output = nn.Dense(features=new_memory.shape[-1])(thalamic_output)
        cortical_input = thalamic_output * new_memory
        integrated_output = nn.relu(self.cortical_integration(cortical_input))

        # Reduce sequence dimension by taking the mean across the sequence
        integrated_output_mean = jnp.mean(integrated_output, axis=1, keepdims=True)
        output = self.output_layer(integrated_output_mean)
        # Ensure output shape is (batch_size, hidden_dim) without using squeeze
        output = jnp.reshape(output, (-1, self.hidden_dim))
        return output, new_memory

    def init_memory(self, batch_size):
        return jnp.zeros((batch_size, self.working_memory_size))

def create_custom_cognitive_model(num_attention_heads=4, attention_head_dim=64, working_memory_size=256, hidden_dim=512):
    return CustomCognitiveModel(
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        working_memory_size=working_memory_size,
        hidden_dim=hidden_dim
    )

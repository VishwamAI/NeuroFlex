import unittest
import jax
import jax.numpy as jnp
from jax import Array
import pytest
from NeuroFlex.cognitive_architectures.custom_cognitive_model import (
    CustomCognitiveModel,
    create_custom_cognitive_model,
    AttentionMechanism,
    WorkingMemory
)
from NeuroFlex.cognitive_architectures import (
    PERFORMANCE_THRESHOLD,
    UPDATE_INTERVAL
)

class TestCognitiveArchitectures(unittest.TestCase):
    def setUp(self):
        self.seed = 42
        self.input_shape = (100, 32)  # Updated to match expected input shape
        self.config = {
            'seed': self.seed,
            'hidden_size': 64,
            'num_layers': 3,
            'working_memory_capacity': 10,
            'prng_key': jax.random.PRNGKey(self.seed)  # Generate prng_key from seed
        }

    def test_custom_cognitive_model(self):
        num_attention_heads = 4
        attention_head_dim = 64
        working_memory_size = 256
        hidden_dim = 512
        attention_schema_size = 128
        model = create_custom_cognitive_model(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            working_memory_size=working_memory_size,
            hidden_dim=hidden_dim,
            attention_schema_size=attention_schema_size
        )
        prng_key = self.config['prng_key']
        batch_size = 1
        seq_len = 10
        input_dim = hidden_dim
        inputs = jax.random.normal(prng_key, (batch_size, seq_len, input_dim))
        prev_memory = jax.random.normal(jax.random.fold_in(prng_key, 1), (batch_size, working_memory_size))
        prev_attention_state = jax.random.normal(jax.random.fold_in(prng_key, 2), (batch_size, attention_schema_size))

        variables = model.init(prng_key, inputs, prev_memory, prev_attention_state)

        # Ensure the variables dictionary has the correct structure
        self.assertIn('params', variables)

        output, new_memory, new_attention_state = model.apply(variables, inputs, prev_memory, prev_attention_state)

        # Check that output, new_memory, and new_attention_state are JAX arrays
        self.assertIsInstance(output, (jnp.ndarray, jax.Array))
        self.assertIsInstance(new_memory, (jnp.ndarray, jax.Array))
        self.assertIsInstance(new_attention_state, (jnp.ndarray, jax.Array))

        # Check the shapes of the outputs
        expected_output_shape = (batch_size, hidden_dim)
        expected_memory_shape = (batch_size, working_memory_size)
        expected_attention_state_shape = (batch_size, 1, 10 * attention_schema_size)
        self.assertEqual(output.shape, expected_output_shape)
        self.assertEqual(new_memory.shape, expected_memory_shape)
        self.assertEqual(new_attention_state.shape, expected_attention_state_shape)

    def test_attention_mechanism(self):
        num_heads = 4
        head_dim = 64
        attention = AttentionMechanism(num_heads=num_heads, head_dim=head_dim)

        batch_size = 1
        seq_len = 10
        input_dim = num_heads * head_dim
        inputs = jax.random.normal(self.config['prng_key'], (batch_size, seq_len, input_dim))

        variables = attention.init(self.config['prng_key'], inputs)
        output = attention.apply(variables, inputs)

        self.assertIsInstance(output, (jnp.ndarray, jax.Array))
        self.assertEqual(output.shape, inputs.shape)

    def test_working_memory(self):
        memory_size = 256
        hidden_dim = 512
        working_memory = WorkingMemory(memory_size=memory_size, hidden_dim=hidden_dim)

        batch_size = 1
        inputs = jax.random.normal(self.config['prng_key'], (batch_size, hidden_dim))
        prev_memory = jax.random.normal(jax.random.fold_in(self.config['prng_key'], 1), (batch_size, memory_size))

        variables = working_memory.init(self.config['prng_key'], inputs, prev_memory)
        new_memory = working_memory.apply(variables, inputs, prev_memory)

        self.assertIsInstance(new_memory, (jnp.ndarray, jax.Array))
        self.assertEqual(new_memory.shape, (batch_size, memory_size))

    def test_custom_cognitive_model_integration(self):
        num_attention_heads = 4
        attention_head_dim = 64
        working_memory_size = 256
        hidden_dim = 512
        attention_schema_size = 128  # Add this line
        model = create_custom_cognitive_model(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            working_memory_size=working_memory_size,
            hidden_dim=hidden_dim,
            attention_schema_size=attention_schema_size  # Add this line
        )

        batch_size = 1
        seq_len = 10
        input_dim = hidden_dim
        inputs = jax.random.normal(self.config['prng_key'], (batch_size, seq_len, input_dim))
        prev_memory = jax.random.normal(jax.random.fold_in(self.config['prng_key'], 1), (batch_size, working_memory_size))
        prev_attention_state = jax.random.normal(jax.random.fold_in(self.config['prng_key'], 2), (batch_size, attention_schema_size))  # Add this line

        variables = model.init(self.config['prng_key'], inputs, prev_memory, prev_attention_state)
        output, new_memory, new_attention_state = model.apply(variables, inputs, prev_memory, prev_attention_state)

        self.assertIsInstance(output, (jnp.ndarray, jax.Array))
        self.assertIsInstance(new_memory, (jnp.ndarray, jax.Array))
        self.assertIsInstance(new_attention_state, (jnp.ndarray, jax.Array))  # Add this line
        self.assertEqual(output.shape, (batch_size, hidden_dim))
        self.assertEqual(new_memory.shape, (batch_size, working_memory_size))
        self.assertEqual(new_attention_state.shape, (batch_size, 1, 10 * attention_schema_size))  # Add this line

    def test_performance_threshold(self):
        self.assertIsInstance(PERFORMANCE_THRESHOLD, float)
        self.assertTrue(0 < PERFORMANCE_THRESHOLD < 1)

    def test_update_interval(self):
        self.assertIsInstance(UPDATE_INTERVAL, int)
        self.assertTrue(UPDATE_INTERVAL > 0)

    def test_performance_threshold(self):
        self.assertIsInstance(PERFORMANCE_THRESHOLD, float)
        self.assertTrue(0 < PERFORMANCE_THRESHOLD < 1)

    def test_update_interval(self):
        self.assertIsInstance(UPDATE_INTERVAL, int)
        self.assertTrue(UPDATE_INTERVAL > 0)

    def test_working_memory(self):
        memory_size = 10
        hidden_dim = 64
        wm = WorkingMemory(memory_size=memory_size, hidden_dim=hidden_dim)

        key = jax.random.PRNGKey(0)
        batch_size = 1
        inputs = jax.random.normal(key, (batch_size, hidden_dim))
        prev_memory = jax.random.normal(jax.random.fold_in(key, 1), (batch_size, memory_size))

        # Initialize the WorkingMemory
        variables = wm.init(key, inputs, prev_memory)

        # First call to WorkingMemory
        new_memory1 = wm.apply(variables, inputs, prev_memory)
        self.assertIsInstance(new_memory1, (jnp.ndarray, jax.Array))
        self.assertEqual(new_memory1.shape, (batch_size, memory_size))

        # Second call to WorkingMemory (should update without error)
        new_memory2 = wm.apply(variables, inputs, new_memory1)
        self.assertIsInstance(new_memory2, (jnp.ndarray, jax.Array))
        self.assertEqual(new_memory2.shape, (batch_size, memory_size))

        # Check that memory has been updated
        self.assertFalse(jnp.array_equal(new_memory1, new_memory2))

    def test_attention_schema_theory(self):
        num_attention_heads = 4
        attention_head_dim = 64
        working_memory_size = 256
        hidden_dim = 512
        attention_schema_size = 128
        model = create_custom_cognitive_model(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            working_memory_size=working_memory_size,
            hidden_dim=hidden_dim,
            attention_schema_size=attention_schema_size
        )

        batch_size = 1
        seq_len = 10
        input_dim = hidden_dim
        inputs = jax.random.normal(self.config['prng_key'], (batch_size, seq_len, input_dim))
        prev_memory = jax.random.normal(jax.random.fold_in(self.config['prng_key'], 1), (batch_size, working_memory_size))
        prev_attention_state = jax.random.normal(jax.random.fold_in(self.config['prng_key'], 2), (batch_size, attention_schema_size))

        variables = model.init(self.config['prng_key'], inputs, prev_memory, prev_attention_state)
        output, new_memory, new_attention_state = model.apply(variables, inputs, prev_memory, prev_attention_state)

        self.assertIsInstance(output, (jnp.ndarray, jax.Array))
        self.assertIsInstance(new_memory, (jnp.ndarray, jax.Array))
        self.assertIsInstance(new_attention_state, (jnp.ndarray, jax.Array))
        self.assertEqual(output.shape, (batch_size, hidden_dim))
        self.assertEqual(new_memory.shape, (batch_size, working_memory_size))
        self.assertEqual(new_attention_state.shape, (batch_size, 1, 10 * attention_schema_size))

    def test_higher_order_theories(self):
        # TODO: Implement test for Higher-Order Theories components
        pass

if __name__ == '__main__':
    unittest.main()

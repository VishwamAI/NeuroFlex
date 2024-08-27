import unittest
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from NeuroFlex.consciousness_simulation import ConsciousnessSimulation, create_consciousness_simulation

class TestConsciousnessSimulation(unittest.TestCase):
    def setUp(self):
        self.features = [64, 32]
        self.output_dim = 16
        self.max_attention_heads = 8
        self.rng = random.PRNGKey(0)
        self.model = create_consciousness_simulation(
            features=self.features,
            output_dim=self.output_dim,
            max_attention_heads=self.max_attention_heads
        )
        self.input_shape = (1, 10)  # Example input shape

    def test_model_initialization(self):
        self.assertIsInstance(self.model, ConsciousnessSimulation)
        self.assertEqual(self.model.features, self.features)
        self.assertEqual(self.model.output_dim, self.output_dim)
        self.assertEqual(self.model.max_attention_heads, self.max_attention_heads)

    def test_forward_pass(self):
        x = random.normal(self.rng, self.input_shape)
        params = self.model.init(self.rng, x)

        # Test deterministic mode
        consciousness_state_det, working_memory_det, info_det = self.model.apply(params, x, deterministic=True)

        # Test non-deterministic mode
        consciousness_state_non_det, working_memory_non_det, info_non_det = self.model.apply(params, x, deterministic=False)

        expected_output_dim = sum(self.features) + self.output_dim * 3 + 2  # +2 for decision and metacognition
        self.assertEqual(consciousness_state_det.shape, (self.input_shape[0], expected_output_dim))
        self.assertEqual(working_memory_det.shape, (self.input_shape[0], self.output_dim))
        self.assertIn('attention_heads', info_det)

        # Check attention mechanism output
        def inspect_attention_output(model, x, params, deterministic):
            return model.apply(params, x, method=lambda m, x: m._apply_dynamic_attention(x, deterministic=deterministic), deterministic=deterministic)

        attention_output_det = inspect_attention_output(self.model, x, params, deterministic=True)
        attention_output_non_det = inspect_attention_output(self.model, x, params, deterministic=False)
        self.assertEqual(attention_output_det.shape, (self.input_shape[0], self.output_dim))
        self.assertEqual(attention_output_non_det.shape, (self.input_shape[0], self.output_dim))

        # Check projected output shape
        def inspect_projected_output(model, x, params, deterministic):
            attention_output = model.apply(params, x, method=lambda m, x: m._apply_dynamic_attention(x, deterministic=deterministic), deterministic=deterministic)
            return model.apply(params, attention_output, method=lambda m, x: m.attention_output_proj(x))

        projected_output_det = inspect_projected_output(self.model, x, params, deterministic=True)
        projected_output_non_det = inspect_projected_output(self.model, x, params, deterministic=False)
        self.assertEqual(projected_output_det.shape, (self.input_shape[0], self.output_dim))
        self.assertEqual(projected_output_non_det.shape, (self.input_shape[0], self.output_dim))

        # Verify attention heads
        self.assertGreaterEqual(info_det['attention_heads'], 1)
        self.assertLessEqual(info_det['attention_heads'], self.model.max_attention_heads)
        self.assertEqual(info_det['attention_heads'], info_non_det['attention_heads'])

        # Verify that deterministic and non-deterministic outputs are different
        self.assertFalse(jnp.allclose(consciousness_state_det, consciousness_state_non_det))
        self.assertFalse(jnp.allclose(working_memory_det, working_memory_non_det))

    def test_consciousness_simulation(self):
        x = random.normal(self.rng, self.input_shape)
        params = self.model.init(self.rng, x)
        consciousness_state, working_memory, info = self.model.apply(params, x)

        # Check shapes
        expected_output_dim = sum(self.features) + self.output_dim * 3 + 2
        self.assertEqual(consciousness_state.shape, (self.input_shape[0], expected_output_dim))
        self.assertEqual(working_memory.shape, (self.input_shape[0], self.output_dim))

        # Check attention heads
        self.assertIn('attention_heads', info)
        self.assertGreaterEqual(info['attention_heads'], 1)
        self.assertLessEqual(info['attention_heads'], self.max_attention_heads)

        # Check consciousness state components
        expected_components = 5  # cognitive_state, attention_output, working_memory, decision, metacognition
        components = jnp.split(consciousness_state, [self.output_dim, self.output_dim * 2, self.output_dim * 3, -1], axis=-1)
        self.assertEqual(len(components), expected_components)

    def test_generate_thought(self):
        x = random.normal(self.rng, self.input_shape)
        params = self.model.init(self.rng, x)
        thought = self.model.apply(params, x, generate_thought=True)

        self.assertEqual(thought.shape, (self.input_shape[0], self.output_dim))
        self.assertTrue(jnp.allclose(jnp.sum(thought, axis=-1), 1.0))  # Check if softmax was applied

    def test_dynamic_attention(self):
        # Test with different input complexities and dimensions
        x_simple = random.normal(self.rng, self.input_shape)
        x_complex = random.normal(self.rng, self.input_shape) * 10  # More complex input
        x_large_dim = random.normal(self.rng, (1, 32))  # Larger input dimension

        params = self.model.init(self.rng, x_simple)

        _, _, info_simple = self.model.apply(params, x_simple)
        _, _, info_complex = self.model.apply(params, x_complex)
        _, _, info_large_dim = self.model.apply(params, x_large_dim)

        # Check if attention heads are adjusted based on input complexity
        self.assertLessEqual(info_simple['attention_heads'], info_complex['attention_heads'])
        self.assertGreater(info_complex['attention_heads'], 1)  # Ensure complex input uses more than 1 head

        # Verify that attention heads are adjusted for larger input dimensions
        self.assertLessEqual(info_simple['attention_heads'], info_large_dim['attention_heads'])

        # Check if attention heads are divisible by input dimension
        self.assertEqual(x_large_dim.shape[1] % info_large_dim['attention_heads'], 0)

        # Test with input dimension not easily divisible by max_attention_heads
        x_odd_dim = random.normal(self.rng, (1, 17))
        _, _, info_odd_dim = self.model.apply(params, x_odd_dim)
        self.assertEqual(x_odd_dim.shape[1] % info_odd_dim['attention_heads'], 0)
        self.assertLess(info_odd_dim['attention_heads'], self.max_attention_heads)

    def test_attention_head_adjustment(self):
        # Create a model with an output_dim that's not easily divisible
        model = create_consciousness_simulation(features=self.features, output_dim=17, max_attention_heads=8)
        x = random.normal(self.rng, self.input_shape)
        params = model.init(self.rng, x)

        with self.assertLogs(level='WARNING') as log:
            _, _, info = model.apply(params, x)

        self.assertTrue(any("Adjusted number of attention heads" in message for message in log.output))
        self.assertLess(info['attention_heads'], 8)  # Should be adjusted to a lower number
        self.assertEqual(17 % info['attention_heads'], 0)  # Should be evenly divisible

    def test_meta_network(self):
        x_low = jnp.ones(self.input_shape)
        x_high = jnp.ones(self.input_shape) * 100

        params = self.model.init(self.rng, x_low)

        _, _, info_low = self.model.apply(params, x_low)
        _, _, info_high = self.model.apply(params, x_high)

        self.assertLessEqual(info_low['attention_heads'], info_high['attention_heads'])

    def test_working_memory_update(self):
        x = random.normal(self.rng, self.input_shape)
        params = self.model.init(self.rng, x)

        # First simulation
        _, working_memory1, _ = self.model.apply(params, x)

        # Second simulation with the same input
        _, working_memory2, _ = self.model.apply(params, x, working_memory_state=working_memory1)

        # Check that working memory has been updated
        self.assertFalse(jnp.allclose(working_memory1, working_memory2))

    def test_gru_input_projection(self):
        x = random.normal(self.rng, self.input_shape)
        params = self.model.init(self.rng, x)

        def inspect_gru_input(model, x, params):
            # Apply the model up to the gru_input_proj layer
            attention_output = model.apply(params, x, method=lambda m, x: m._apply_dynamic_attention(x))
            attention_output = jnp.mean(attention_output, axis=0)
            projected_output = model.apply(params, attention_output, method=lambda m, x: m.attention_output_proj(x))
            gru_input = model.apply(params, projected_output, method=lambda m, x: m.gru_input_proj(x))
            return gru_input

        gru_input = inspect_gru_input(self.model, x, params)
        expected_gru_input_size = self.output_dim
        self.assertEqual(gru_input.shape, (self.input_shape[0], expected_gru_input_size))

        # Test if GRU can process the projected input without errors
        working_memory_state = jnp.zeros((self.input_shape[0], self.output_dim))
        try:
            new_working_memory, _ = self.model.apply(
                params,
                gru_input,
                working_memory_state,
                method=lambda m, x, state: nn.GRUCell(features=m.output_dim)(x, state)
            )
            self.assertEqual(new_working_memory.shape, (self.input_shape[0], self.output_dim))
        except Exception as e:
            self.fail(f"GRU failed to process projected input: {str(e)}")

        # Additional test to ensure GRU input and working memory state have compatible shapes
        self.assertEqual(gru_input.shape[-1], working_memory_state.shape[-1],
                         "GRU input shape should match working memory state shape")

        # Test the shapes of intermediate outputs
        attention_output = self.model.apply(params, x, method=lambda m, x: m._apply_dynamic_attention(x))
        self.assertEqual(attention_output.shape, (self.input_shape[0], self.output_dim))

        projected_output = self.model.apply(params, jnp.mean(attention_output, axis=0), method=lambda m, x: m.attention_output_proj(x))
        self.assertEqual(projected_output.shape, (self.input_shape[0], self.output_dim))

    def test_1d_input_handling(self):
        x_1d = random.normal(self.rng, (10,))  # 1D input
        params = self.model.init(self.rng, x_1d)

        # Test if the model can process 1D input without errors
        try:
            consciousness_state, working_memory, info = self.model.apply(params, x_1d)
            self.assertIsNotNone(consciousness_state)
            self.assertIsNotNone(working_memory)
            self.assertIn('attention_heads', info)
        except Exception as e:
            self.fail(f"Model failed to process 1D input: {str(e)}")

        # Check output shapes
        expected_output_dim = sum(self.features) + self.output_dim * 3 + 2
        self.assertEqual(consciousness_state.shape, (1, expected_output_dim))
        self.assertEqual(working_memory.shape, (1, self.output_dim))

if __name__ == '__main__':
    unittest.main()

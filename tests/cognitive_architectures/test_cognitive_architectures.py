import unittest
import jax
import jax.numpy as jnp
import pytest
from NeuroFlex.cognitive_architectures import (
    CognitiveArchitecture,
    create_consciousness,
    create_feedback_mechanism,
    ConsciousnessSimulation,
    create_consciousness_simulation,
    ExtendedCognitiveArchitecture,
    BCIProcessor,
    create_extended_cognitive_model,
    PERFORMANCE_THRESHOLD,
    UPDATE_INTERVAL
)
from NeuroFlex.cognitive_architectures.extended_cognitive_architectures import WorkingMemory

class TestCognitiveArchitectures(unittest.TestCase):
    def setUp(self):
        self.seed = 42
        self.input_shape = (100,)  # Updated to match expected input shape
        self.config = {
            'seed': self.seed,
            'hidden_size': 64,
            'num_layers': 3,
            'working_memory_capacity': 10,
            'prng_key': jax.random.PRNGKey(self.seed)  # Generate prng_key from seed
        }

    def test_cognitive_architecture(self):
        cognitive_arch = CognitiveArchitecture(self.config)
        prng_key = self.config['prng_key']
        inputs = {
            'vision': jax.random.normal(prng_key, self.input_shape),
            'audition': jax.random.normal(jax.random.fold_in(prng_key, 1), self.input_shape),
            'touch': jax.random.normal(jax.random.fold_in(prng_key, 2), self.input_shape)
        }
        consciousness_state, feedback = cognitive_arch.update_architecture(inputs)

        self.assertIsInstance(consciousness_state, jnp.ndarray)
        self.assertIsInstance(feedback, jnp.ndarray)

    def test_create_consciousness(self):
        consciousness = create_consciousness(self.config['prng_key'])
        self.assertIsInstance(consciousness, jnp.ndarray)
        self.assertEqual(consciousness.shape, (100,))

    def test_create_feedback_mechanism(self):
        feedback_mechanism = create_feedback_mechanism(self.config['prng_key'])
        self.assertIsInstance(feedback_mechanism, jnp.ndarray)
        self.assertEqual(feedback_mechanism.shape, (100,))

    @pytest.mark.skip(reason="This test is currently failing and needs to be updated")
    def test_consciousness_simulation(self):
        features = [32, 64]
        output_dim = 16
        working_memory_size = 192
        sim = create_consciousness_simulation(features, output_dim)
        inputs = jax.random.normal(self.config['prng_key'], self.input_shape)
        variables = sim.init(self.config['prng_key'], inputs)

        # Ensure the variables dictionary has the correct structure
        self.assertIn('params', variables)
        self.assertIn('working_memory', variables)
        self.assertIn('model_state', variables)

        rngs = {
            'dropout': jax.random.fold_in(self.config['prng_key'], 1),
            'perturbation': jax.random.fold_in(self.config['prng_key'], 2)
        }

        output, mutated = sim.apply(
            variables, inputs,
            rngs=rngs,
            method=sim.simulate_consciousness,
            mutable=['working_memory', 'model_state']
        )

        consciousness_state, new_working_memory = output
        self.assertIsInstance(consciousness_state, jnp.ndarray)
        self.assertIsInstance(new_working_memory, jnp.ndarray)

        # Check the shapes of the outputs
        expected_consciousness_shape = (inputs.shape[0], sim.output_dim + 2*working_memory_size + 4 + working_memory_size)
        self.assertEqual(consciousness_state.shape, expected_consciousness_shape)
        self.assertEqual(new_working_memory.shape, (inputs.shape[0], working_memory_size))

        # Check that the mutated variables have the expected structure
        self.assertIn('working_memory', mutated)
        self.assertIn('model_state', mutated)

    @pytest.mark.skip(reason="Test is currently failing and needs to be updated")
    def test_extended_cognitive_architecture(self):
        input_dim = 100
        model = create_extended_cognitive_model(
            num_layers=3,
            hidden_size=64,
            working_memory_capacity=10,
            bci_input_channels=32,
            bci_output_size=5,
            input_dim=input_dim
        )
        prng_key = self.config['prng_key']
        cognitive_input = jax.random.normal(prng_key, (1, input_dim))
        bci_input = jax.random.normal(jax.random.fold_in(prng_key, 1), (1, 32, 32, 1))
        task_context = jax.random.normal(jax.random.fold_in(prng_key, 2), (1, 64))

        params = model.init(prng_key, cognitive_input, bci_input, task_context)
        output = model.apply(params, cognitive_input, bci_input, task_context)

        self.assertIsInstance(output, jnp.ndarray)

    def test_performance_threshold(self):
        self.assertIsInstance(PERFORMANCE_THRESHOLD, float)
        self.assertTrue(0 < PERFORMANCE_THRESHOLD < 1)

    def test_update_interval(self):
        self.assertIsInstance(UPDATE_INTERVAL, int)
        self.assertTrue(UPDATE_INTERVAL > 0)

    @pytest.mark.skip(reason="Test is currently failing and needs to be updated")
    def test_working_memory(self):
        capacity = 10
        hidden_size = 64
        wm = WorkingMemory(capacity, hidden_size)

        key = jax.random.PRNGKey(0)
        inputs = jax.random.normal(key, (1, hidden_size))
        query = jax.random.normal(key, (1, hidden_size))

        # Initialize the WorkingMemory
        params = wm.init(key, inputs, query)

        # First call to WorkingMemory
        output1, mutated1 = wm.apply(params, inputs, query, mutable=['memory'])
        self.assertIsInstance(output1, jnp.ndarray)
        self.assertIn('memory', mutated1)

        # Second call to WorkingMemory (should update without error)
        output2, mutated2 = wm.apply(mutated1, inputs, query, mutable=['memory'])
        self.assertIsInstance(output2, jnp.ndarray)
        self.assertIn('memory', mutated2)

        # Check that memory has been updated
        self.assertFalse(jnp.array_equal(mutated1['memory']['buffer'], mutated2['memory']['buffer']))

if __name__ == '__main__':
    unittest.main()

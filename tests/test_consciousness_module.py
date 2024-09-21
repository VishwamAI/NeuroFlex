import unittest
import jax
import jax.numpy as jnp
import numpy as np
import logging
from NeuroFlex import ConsciousnessSimulation, create_consciousness_simulation
import neurolib
from neurolib.models.aln import ALNModel

# Configure basic logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TestConsciousnessModule(unittest.TestCase):
    def setUp(self):
        logging.info("Setting up TestConsciousnessModule")
        self.features = [64, 32]
        self.output_dim = 16
        self.working_memory_size = 192  # Updated to match the default in create_consciousness_simulation
        self.attention_heads = 4
        self.qkv_features = 64
        self.dropout_rate = 0.1
        self.num_brain_areas = 90
        self.simulation_length = 1.0
        logging.debug(f"Creating ConsciousnessSimulation with features={self.features}, output_dim={self.output_dim}")
        self.consciousness_sim = create_consciousness_simulation(
            features=self.features,
            output_dim=self.output_dim,
            working_memory_size=self.working_memory_size,
            attention_heads=self.attention_heads,
            qkv_features=self.qkv_features,
            dropout_rate=self.dropout_rate,
            num_brain_areas=self.num_brain_areas,
            simulation_length=self.simulation_length
        )
        self.rng = jax.random.PRNGKey(0)
        logging.info("TestConsciousnessModule setup complete")

    def test_consciousness_simulation_initialization(self):
        self.assertIsInstance(self.consciousness_sim, ConsciousnessSimulation)
        self.assertEqual(self.consciousness_sim.features, self.features)
        self.assertEqual(self.consciousness_sim.output_dim, self.output_dim)
        self.assertEqual(self.consciousness_sim.working_memory_size, self.working_memory_size)
        self.assertEqual(self.consciousness_sim.attention_heads, self.attention_heads)
        self.assertEqual(self.consciousness_sim.qkv_features, self.qkv_features)
        self.assertEqual(self.consciousness_sim.dropout_rate, self.dropout_rate)

    def test_consciousness_simulation_forward_pass(self):
        logging.debug("Starting test_consciousness_simulation_forward_pass")
        try:
            x = jax.random.normal(self.rng, (1, self.output_dim))
            logging.debug(f"Input shape: {x.shape}")
            params = self.consciousness_sim.init(self.rng, x)
            logging.debug(f"Initialized params: {jax.tree_util.tree_map(lambda x: x.shape, params)}")

            # First forward pass
            rng_key1 = jax.random.PRNGKey(0)
            rng_keys1 = {
                'dropout': jax.random.fold_in(rng_key1, 0),
                'perturbation': jax.random.fold_in(rng_key1, 1)
            }
            result1 = self.consciousness_sim.apply(
                params, x, rngs=rng_keys1, method=self.consciousness_sim.simulate_consciousness, mutable=['working_memory', 'model_state']
            )
            logging.debug(f"Result1 type: {type(result1)}, length: {len(result1) if isinstance(result1, tuple) else 'N/A'}")
            self.assertIsInstance(result1, tuple, "Result should be a tuple")
            self.assertEqual(len(result1), 3, f"Expected 3 return values, got {len(result1)}")
            consciousness1, new_working_memory1, working_memory_dict1 = result1
            logging.debug(f"Unpacked values - consciousness1: {consciousness1.shape}, new_working_memory1: {new_working_memory1.shape}, working_memory_dict1 keys: {working_memory_dict1.keys()}")

            # Validate the unpacked values
            self.assertIsInstance(consciousness1, jnp.ndarray, "consciousness1 should be a jax.numpy array")
            self.assertIsInstance(new_working_memory1, jnp.ndarray, "new_working_memory1 should be a jax.numpy array")
            self.assertIsInstance(working_memory_dict1, dict, "working_memory_dict1 should be a dictionary")
            self.assertIn('working_memory', working_memory_dict1, "working_memory_dict1 should contain 'working_memory' key")
            self.assertIn('current_state', working_memory_dict1['working_memory'], "working_memory_dict1['working_memory'] should contain 'current_state' key")

            # Second forward pass with the same input but different RNG key
            rng_key2 = jax.random.PRNGKey(1)
            rng_keys2 = {'dropout': jax.random.fold_in(rng_key2, 0), 'perturbation': jax.random.fold_in(rng_key2, 1)}
            result2 = self.consciousness_sim.apply(
                params, x, rngs=rng_keys2, method=self.consciousness_sim.simulate_consciousness, mutable=['working_memory', 'model_state']
            )
            logging.debug(f"Result2 type: {type(result2)}, length: {len(result2) if isinstance(result2, tuple) else 'N/A'}")
            self.assertIsInstance(result2, tuple, "Result2 should be a tuple")
            self.assertEqual(len(result2), 3, f"Expected 3 return values, got {len(result2)}")
            consciousness2, new_working_memory2, working_memory_dict2 = result2
            logging.debug(f"Unpacked Result2 - consciousness2 shape: {consciousness2.shape}, "
                          f"new_working_memory2 shape: {new_working_memory2.shape}, "
                          f"working_memory_dict2 keys: {working_memory_dict2.keys()}")
            self.assertIn('working_memory', working_memory_dict2, "working_memory_dict2 should contain 'working_memory' key")
            self.assertIn('current_state', working_memory_dict2['working_memory'], "working_memory_dict2['working_memory'] should contain 'current_state' key")

            logging.debug(f"Consciousness shape: {consciousness1.shape}")
            logging.debug(f"New working memory shape: {new_working_memory1.shape}")
            logging.debug(f"Working memory dict keys: {working_memory_dict1.keys()}")
            expected_consciousness_shape = (1, self.output_dim + 2*self.working_memory_size + 4 + self.working_memory_size)
            self.assertEqual(consciousness1.shape, expected_consciousness_shape,
                             f"Expected consciousness shape {expected_consciousness_shape}, got {consciousness1.shape}")
            self.assertEqual(new_working_memory1.shape, (1, self.working_memory_size),
                             f"Expected new working memory shape (1, {self.working_memory_size}), got {new_working_memory1.shape}")
            self.assertIsInstance(working_memory_dict1, dict, "working_memory_dict1 should be a dictionary")
            self.assertIn('working_memory', working_memory_dict1, "working_memory_dict1 should contain 'working_memory' key")
            self.assertIn('current_state', working_memory_dict1['working_memory'], "working_memory_dict1['working_memory'] should contain 'current_state' key")
            self.assertEqual(working_memory_dict1['working_memory']['current_state'].shape, (1, self.working_memory_size),
                             f"Expected working memory current state shape (1, {self.working_memory_size}), got {working_memory_dict1['working_memory']['current_state'].shape}")

            # Check that working memory is actually being perturbed
            self.assertFalse(jnp.allclose(working_memory_dict1['working_memory']['current_state'],
                                          working_memory_dict2['working_memory']['current_state']),
                             "Working memory should be different due to perturbation")
            self.assertFalse(jnp.allclose(new_working_memory1, new_working_memory2),
                             "New working memory should be different due to perturbation")

            # Check that consciousness states are different due to different RNG keys
            self.assertFalse(jnp.allclose(consciousness1, consciousness2),
                             "Consciousness states should be different due to different RNG keys")

            # Check for NaN values
            self.assertFalse(jnp.any(jnp.isnan(consciousness1)), "Consciousness output contains NaN values")
            self.assertFalse(jnp.any(jnp.isnan(new_working_memory1)), "New working memory output contains NaN values")
            self.assertFalse(jnp.any(jnp.isnan(working_memory_dict1['working_memory']['current_state'])), "Working memory output contains NaN values")

            logging.debug("All shape and content checks passed successfully")

            logging.debug("Finished test_consciousness_simulation_forward_pass")
        except Exception as e:
            logging.error(f"Error in test_consciousness_simulation_forward_pass: {str(e)}")
            raise

    def test_simulate_consciousness(self):
        logging.debug("Starting test_simulate_consciousness")
        x = jax.random.normal(self.rng, (1, self.output_dim))
        logging.debug(f"Generated input x with shape: {x.shape}")
        params = self.consciousness_sim.init(self.rng, x)
        logging.debug(f"Initialized params: {jax.tree_util.tree_map(lambda x: x.shape, params)}")

        # Create PRNG keys for the simulate_consciousness method
        rng_key1 = jax.random.PRNGKey(0)
        rng_key2 = jax.random.PRNGKey(1)
        rng_keys1 = {'dropout': jax.random.fold_in(rng_key1, 0), 'perturbation': jax.random.fold_in(rng_key1, 1)}
        rng_keys2 = {'dropout': jax.random.fold_in(rng_key2, 0), 'perturbation': jax.random.fold_in(rng_key2, 1)}

        def validate_result(result, rng_key_name):
            self.assertIsInstance(result, tuple, f"Result from {rng_key_name} should be a tuple")
            self.assertEqual(len(result), 3, f"simulate_consciousness with {rng_key_name} should return 3 values, got {len(result)}")
            consciousness_state, new_working_memory, working_memory_dict = result

            # Validate consciousness_state
            self.assertIsInstance(consciousness_state, jnp.ndarray, f"consciousness_state from {rng_key_name} should be a jax.numpy array")
            expected_consciousness_shape = (1, self.output_dim + 2*self.working_memory_size + 4 + self.working_memory_size)
            self.assertEqual(consciousness_state.shape, expected_consciousness_shape,
                             f"Expected consciousness_state shape {expected_consciousness_shape} for {rng_key_name}, got {consciousness_state.shape}")
            self.assertFalse(jnp.any(jnp.isnan(consciousness_state)), f"consciousness_state from {rng_key_name} contains NaN values")

            # Validate new_working_memory
            self.assertIsInstance(new_working_memory, jnp.ndarray, f"new_working_memory from {rng_key_name} should be a jax.numpy array")
            self.assertEqual(new_working_memory.shape, (1, self.working_memory_size),
                             f"Expected new_working_memory shape (1, {self.working_memory_size}) for {rng_key_name}, got {new_working_memory.shape}")
            self.assertFalse(jnp.any(jnp.isnan(new_working_memory)), f"new_working_memory from {rng_key_name} contains NaN values")

            # Validate working_memory_dict
            self.assertIsInstance(working_memory_dict, dict, f"working_memory_dict from {rng_key_name} should be a dictionary")
            self.assertIn('working_memory', working_memory_dict, f"working_memory_dict from {rng_key_name} should contain 'working_memory' key")
            self.assertIsInstance(working_memory_dict['working_memory'], dict, f"working_memory_dict['working_memory'] from {rng_key_name} should be a dictionary")
            self.assertIn('current_state', working_memory_dict['working_memory'], f"working_memory_dict['working_memory'] from {rng_key_name} should contain 'current_state' key")
            current_state = working_memory_dict['working_memory']['current_state']
            self.assertIsInstance(current_state, jnp.ndarray, f"current_state from {rng_key_name} should be a jax.numpy array")
            self.assertEqual(current_state.shape, (1, self.working_memory_size),
                             f"Expected working_memory current_state shape (1, {self.working_memory_size}) for {rng_key_name}, got {current_state.shape}")
            self.assertFalse(jnp.any(jnp.isnan(current_state)), f"working_memory current_state from {rng_key_name} contains NaN values")

            logging.debug(f"Validation passed for {rng_key_name}")
            return consciousness_state, new_working_memory, working_memory_dict

        try:
            logging.debug("Calling simulate_consciousness with rng_keys1")
            result1 = self.consciousness_sim.apply(
                params, x, rngs=rng_keys1,
                method=self.consciousness_sim.simulate_consciousness, mutable=['working_memory', 'model_state']
            )
            consciousness_state1, new_working_memory1, working_memory_dict1 = validate_result(result1, "rng_keys1")

            logging.debug("Calling simulate_consciousness with rng_keys2")
            result2 = self.consciousness_sim.apply(
                params, x, rngs=rng_keys2,
                method=self.consciousness_sim.simulate_consciousness, mutable=['working_memory', 'model_state']
            )
            consciousness_state2, new_working_memory2, working_memory_dict2 = validate_result(result2, "rng_keys2")

            # Check that new_working_memory is different from the current working memory state
            self.assertFalse(jnp.allclose(new_working_memory1, working_memory_dict1['working_memory']['current_state']),
                             "New working memory should be different from current working memory state")

            # Check that the outputs are different due to different RNG keys
            self.assertFalse(jnp.allclose(consciousness_state1, consciousness_state2),
                             "Consciousness states should be different due to different RNG keys")
            self.assertFalse(jnp.allclose(new_working_memory1, new_working_memory2),
                             "New working memories should be different due to different RNG keys")
            self.assertFalse(jnp.allclose(working_memory_dict1['working_memory']['current_state'], working_memory_dict2['working_memory']['current_state']),
                             "Working memory states should be different due to different RNG keys")

            # Check for NaN and infinite values
            for name, array in [("Consciousness state", consciousness_state1),
                                ("New working memory", new_working_memory1),
                                ("Working memory", working_memory_dict1['working_memory']['current_state'])]:
                self.assertFalse(jnp.any(jnp.isnan(array)), f"{name} contains NaN values")
                self.assertFalse(jnp.any(jnp.isinf(array)), f"{name} contains infinite values")

            # Additional check for non-zero values in normal case
            self.assertFalse(jnp.all(consciousness_state1 == 0), "Consciousness state should not be all zeros")
            self.assertFalse(jnp.all(new_working_memory1 == 0), "New working memory should not be all zeros")
            self.assertFalse(jnp.all(working_memory_dict1['working_memory']['current_state'] == 0), "Working memory state should not be all zeros")

            # Check for error case with non-finite input
            error_x = jnp.array([[float('nan')]])  # Input that should cause an error
            logging.debug("Testing error case with non-finite input")
            error_result = self.consciousness_sim.apply(
                params, error_x, rngs=rng_keys1,
                method=self.consciousness_sim.simulate_consciousness, mutable=['working_memory', 'model_state']
            )
            self.assertEqual(len(error_result), 3, f"Error case should still return 3 values, got {len(error_result)}")
            error_consciousness, error_working_memory, error_dict = error_result

            # Validate error case outputs
            self.assertEqual(error_consciousness.shape, consciousness_state1.shape, "Error case consciousness shape mismatch")
            self.assertEqual(error_working_memory.shape, new_working_memory1.shape, "Error case working memory shape mismatch")
            self.assertIn('error', error_dict, "Error case should return an error message in the dictionary")
            self.assertIsInstance(error_dict['error'], str, "Error message should be a string")
            self.assertIn('working_memory', error_dict, "Error dictionary should contain 'working_memory' key")
            self.assertIn('current_state', error_dict['working_memory'], "Error dictionary's working_memory should contain 'current_state' key")
            self.assertIsInstance(error_dict['working_memory']['current_state'], jnp.ndarray, "Error case working memory should be a jax array")
            self.assertEqual(error_dict['working_memory']['current_state'].shape, new_working_memory1.shape, "Error case working memory shape mismatch")

            # Check that error case returns zero arrays
            self.assertTrue(jnp.all(error_consciousness == 0), "Error case consciousness should be all zeros")
            self.assertTrue(jnp.all(error_working_memory == 0), "Error case working memory should be all zeros")
            self.assertTrue(jnp.all(error_dict['working_memory']['current_state'] == 0), "Error case working memory state should be all zeros")

        except Exception as e:
            logging.error(f"simulate_consciousness method failed: {str(e)}")
            self.fail(f"simulate_consciousness method failed: {str(e)}")

        logging.debug("Finished test_simulate_consciousness")

    def test_generate_thought(self):
        logging.debug("Starting test_generate_thought")
        x = jax.random.normal(self.rng, (1, self.output_dim))
        logging.debug(f"Generated input x with shape: {x.shape}")
        params = self.consciousness_sim.init(self.rng, x)
        logging.debug(f"Initialized params: {jax.tree_util.tree_map(lambda x: x.shape, params)}")

        # Create PRNG keys for the simulate_consciousness method
        rng_key1 = jax.random.PRNGKey(0)
        rng_keys1 = {'dropout': jax.random.fold_in(rng_key1, 0), 'perturbation': jax.random.fold_in(rng_key1, 1)}

        try:
            # simulate_consciousness returns (consciousness_state, new_working_memory, working_memory_dict)
            result = self.consciousness_sim.apply(
                params, x, rngs=rng_keys1, method=self.consciousness_sim.simulate_consciousness, mutable=['working_memory', 'model_state']
            )
            self.assertEqual(len(result), 3, "simulate_consciousness should return 3 values")
            consciousness_state, new_working_memory, working_memory_dict = result
            logging.debug(f"Generated consciousness_state with shape: {consciousness_state.shape}")
            logging.debug(f"New working memory shape: {new_working_memory.shape}")
            logging.debug(f"Working memory dict keys: {working_memory_dict.keys()}")

            # Create PRNG keys for generate_thought
            rng_key2 = jax.random.PRNGKey(1)
            rng_keys2 = {'dropout': jax.random.fold_in(rng_key2, 0)}
            thought = self.consciousness_sim.apply(
                params, consciousness_state, rngs=rng_keys2, method=self.consciousness_sim.generate_thought
            )
            logging.debug(f"Generated thought with shape: {thought.shape}")

            self.assertIsInstance(thought, jnp.ndarray)
            self.assertEqual(thought.shape, (1, self.output_dim))
            self.assertAlmostEqual(jnp.sum(thought), 1.0, places=6)  # Check if it's a valid probability distribution
            self.assertTrue(jnp.all(thought >= 0) and jnp.all(thought <= 1))  # Check if values are between 0 and 1

            # Test with different RNG key
            rng_key3 = jax.random.PRNGKey(2)
            rng_keys3 = {'dropout': jax.random.fold_in(rng_key3, 0)}
            thought2 = self.consciousness_sim.apply(
                params, consciousness_state, rngs=rng_keys3, method=self.consciousness_sim.generate_thought
            )
            self.assertFalse(jnp.allclose(thought, thought2), "Thoughts should be different with different RNG keys")

            logging.debug("Finished test_generate_thought")
        except Exception as e:
            logging.error(f"Error in test_generate_thought: {str(e)}")
            raise

    def test_forward_pass_with_attention(self):
        logging.info("Starting test_forward_pass_with_attention")
        x = jax.random.normal(self.rng, (1, self.output_dim))
        logging.debug(f"Generated input x with shape: {x.shape}")

        params = self.consciousness_sim.init(self.rng, x)
        logging.debug(f"Initialized params: {jax.tree_util.tree_map(lambda x: x.shape, params)}")

        rng_key = jax.random.PRNGKey(0)
        rng_keys = {'dropout': jax.random.fold_in(rng_key, 0), 'perturbation': jax.random.fold_in(rng_key, 1)}
        try:
            result = self.consciousness_sim.apply(
                params, x, rngs=rng_keys, method=self.consciousness_sim.simulate_consciousness, mutable=['working_memory', 'model_state']
            )
            self.assertIsInstance(result, tuple, "Result should be a tuple")
            self.assertEqual(len(result), 3, "simulate_consciousness should return exactly 3 values")
            consciousness, new_working_memory, working_memory = result
            logging.debug(f"Consciousness shape: {consciousness.shape}")
            logging.debug(f"New working memory shape: {new_working_memory.shape}")
            logging.debug(f"Working memory shape: {working_memory['working_memory']['current_state'].shape}")
        except Exception as e:
            logging.error(f"Error in forward pass: {str(e)}")
            self.fail(f"Forward pass failed: {str(e)}")

        # Check the output shapes
        expected_consciousness_shape = (1, self.output_dim + 2*self.working_memory_size + 4 + self.working_memory_size)
        self.assertEqual(consciousness.shape, expected_consciousness_shape,
                         f"Expected consciousness shape {expected_consciousness_shape}, got {consciousness.shape}")
        self.assertEqual(new_working_memory.shape, (1, self.working_memory_size),
                         f"Expected working memory shape (1, {self.working_memory_size}), got {new_working_memory.shape}")
        self.assertIsInstance(working_memory, dict, "Expected working_memory to be a dictionary")
        self.assertIn('working_memory', working_memory, "Expected 'working_memory' key in working_memory dict")
        self.assertIn('current_state', working_memory['working_memory'], "Expected 'current_state' key in working_memory['working_memory']")
        self.assertEqual(working_memory['working_memory']['current_state'].shape, (1, self.working_memory_size),
                         f"Expected working memory current state shape (1, {self.working_memory_size}), got {working_memory['working_memory']['current_state'].shape}")
        logging.info("Output shape checks passed")

        # Check if the output is different from the input (processing occurred)
        self.assertFalse(jnp.allclose(consciousness[:, :self.output_dim], x),
                         "Output should be different from input")
        logging.info("Processing effect check passed")

        # Check if the output values are within a reasonable range
        self.assertTrue(jnp.all(jnp.isfinite(consciousness)),
                        "Consciousness output contains non-finite values")
        self.assertTrue(jnp.all(jnp.isfinite(new_working_memory)),
                        "New working memory output contains non-finite values")
        self.assertTrue(jnp.all(jnp.isfinite(working_memory['working_memory']['current_state'])),
                        "Working memory output contains non-finite values")
        logging.info("Output range check passed")

        # Check for NaN values
        self.assertFalse(jnp.any(jnp.isnan(consciousness)),
                         "Consciousness output contains NaN values")
        self.assertFalse(jnp.any(jnp.isnan(new_working_memory)),
                         "New working memory output contains NaN values")
        self.assertFalse(jnp.any(jnp.isnan(working_memory['working_memory']['current_state'])),
                         "Working memory output contains NaN values")
        logging.info("NaN check passed")

    def test_working_memory_update(self):
        logging.info("Starting test_working_memory_update")
        x1 = jax.random.normal(self.rng, (1, self.output_dim))
        x2 = jax.random.normal(self.rng, (1, self.output_dim))

        params = self.consciousness_sim.init(self.rng, x1)

        # First forward pass
        rng_key1 = jax.random.PRNGKey(0)
        rng_keys1 = {'dropout': jax.random.fold_in(rng_key1, 0), 'perturbation': jax.random.fold_in(rng_key1, 1)}
        try:
            result1 = self.consciousness_sim.apply(
                params, x1, rngs=rng_keys1,
                method=self.consciousness_sim.simulate_consciousness, mutable=['working_memory', 'model_state']
            )
            self.assertEqual(len(result1), 3, "simulate_consciousness should return 3 values")
            consciousness1, new_working_memory1, working_memory1 = result1
        except Exception as e:
            self.fail(f"First forward pass failed: {str(e)}")

        # Second forward pass with different input and RNG key
        rng_key2 = jax.random.PRNGKey(1)
        rng_keys2 = {'dropout': jax.random.fold_in(rng_key2, 0), 'perturbation': jax.random.fold_in(rng_key2, 1)}
        try:
            result2 = self.consciousness_sim.apply(
                params, x2, rngs=rng_keys2,
                method=self.consciousness_sim.simulate_consciousness, mutable=['working_memory', 'model_state']
            )
            self.assertEqual(len(result2), 3, "simulate_consciousness should return 3 values")
            consciousness2, new_working_memory2, working_memory2 = result2
        except Exception as e:
            self.fail(f"Second forward pass failed: {str(e)}")

        # Check if working memory has been updated
        self.assertFalse(jnp.allclose(working_memory1['working_memory']['current_state'], working_memory2['working_memory']['current_state']),
                         "Working memory should be updated between forward passes")
        self.assertFalse(jnp.allclose(new_working_memory1, new_working_memory2),
                         "New working memory should be different between forward passes")
        logging.info("Working memory update check passed")

    def test_invalid_rngs(self):
        logging.info("Starting test_invalid_rngs")
        x = jax.random.normal(self.rng, (1, self.output_dim))
        params = self.consciousness_sim.init(self.rng, x)

        # Test with None rngs
        try:
            result = self.consciousness_sim.apply(
                params, x, rngs=None, method=self.consciousness_sim.simulate_consciousness, mutable=['working_memory', 'model_state']
            )
            self.assertEqual(len(result), 3, "simulate_consciousness should return 3 values")
            logging.info("Test with None rngs passed")
        except Exception as e:
            self.fail(f"Test with None rngs failed: {str(e)}")

        # Test with missing 'dropout' key
        invalid_rngs1 = {'perturbation': jax.random.PRNGKey(0)}
        try:
            result = self.consciousness_sim.apply(
                params, x, rngs=invalid_rngs1, method=self.consciousness_sim.simulate_consciousness, mutable=['working_memory', 'model_state']
            )
            self.assertEqual(len(result), 3, "simulate_consciousness should return 3 values")
            logging.info("Test with missing 'dropout' key passed")
        except Exception as e:
            self.fail(f"Test with missing 'dropout' key failed: {str(e)}")

        # Test with missing 'perturbation' key
        invalid_rngs2 = {'dropout': jax.random.PRNGKey(0)}
        try:
            result = self.consciousness_sim.apply(
                params, x, rngs=invalid_rngs2, method=self.consciousness_sim.simulate_consciousness, mutable=['working_memory', 'model_state']
            )
            self.assertEqual(len(result), 3, "simulate_consciousness should return 3 values")
            logging.info("Test with missing 'perturbation' key passed")
        except Exception as e:
            self.fail(f"Test with missing 'perturbation' key failed: {str(e)}")

        # Test with non-PRNGKey values
        invalid_rngs3 = {'dropout': 0, 'perturbation': 1}
        try:
            result = self.consciousness_sim.apply(
                params, x, rngs=invalid_rngs3, method=self.consciousness_sim.simulate_consciousness, mutable=['working_memory', 'model_state']
            )
            self.assertEqual(len(result), 3, "simulate_consciousness should return 3 values")
            logging.info("Test with non-PRNGKey values passed")
        except Exception as e:
            self.fail(f"Test with non-PRNGKey values failed: {str(e)}")

        logging.info("All invalid rngs tests passed successfully")

    def test_neurolib_integration(self):
        logging.info("Starting test_neurolib_integration")
        x = jax.random.normal(self.rng, (1, self.output_dim))
        params = self.consciousness_sim.init(self.rng, x)

        rng_key = jax.random.PRNGKey(0)
        rng_keys = {'dropout': jax.random.fold_in(rng_key, 0), 'perturbation': jax.random.fold_in(rng_key, 1)}

        try:
            result = self.consciousness_sim.apply(
                params, x, rngs=rng_keys, method=self.consciousness_sim.simulate_consciousness, mutable=['working_memory', 'model_state']
            )
            self.assertEqual(len(result), 3, "simulate_consciousness should return 3 values")
            consciousness, new_working_memory, working_memory = result

            # Check if ALNModel simulation results are incorporated into the consciousness state
            self.assertGreater(jnp.sum(jnp.abs(consciousness)), 0, "ALNModel simulation should affect consciousness state")

            # Verify that the ALNModel parameters are updated
            self.assertNotEqual(self.consciousness_sim.aln_model.params['duration'], 0, "ALNModel duration should be set")

            logging.info("Neurolib integration test passed successfully")
        except Exception as e:
            self.fail(f"Neurolib integration test failed: {str(e)}")

    logging.info("All tests completed successfully")

if __name__ == '__main__':
    unittest.main()

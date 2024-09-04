import unittest
import jax
import jax.numpy as jnp
import logging
from NeuroFlex.consciousness_simulation import ConsciousnessSimulation, create_consciousness_simulation

# Configure basic logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TestConsciousnessModule(unittest.TestCase):
    def setUp(self):
        logging.info("Setting up TestConsciousnessModule")
        self.features = [64, 32]
        self.output_dim = 16
        self.working_memory_size = 64
        self.attention_heads = 4
        self.qkv_features = 64
        self.dropout_rate = 0.1
        logging.debug(f"Creating ConsciousnessSimulation with features={self.features}, output_dim={self.output_dim}")
        self.consciousness_sim = create_consciousness_simulation(
            features=self.features,
            output_dim=self.output_dim,
            working_memory_size=self.working_memory_size,
            attention_heads=self.attention_heads,
            qkv_features=self.qkv_features,
            dropout_rate=self.dropout_rate
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
            consciousness1, new_working_memory1, working_memory1 = self.consciousness_sim.apply(
                params, x, rngs=rng_keys1, method=self.consciousness_sim.simulate_consciousness, mutable=['working_memory', 'model_state']
            )

            # Second forward pass with the same input but different RNG key
            rng_key2 = jax.random.PRNGKey(1)
            rng_keys2 = {'dropout': jax.random.fold_in(rng_key2, 0), 'perturbation': jax.random.fold_in(rng_key2, 1)}
            consciousness2, new_working_memory2, working_memory2 = self.consciousness_sim.apply(
                params, x, rngs=rng_keys2, method=self.consciousness_sim.simulate_consciousness, mutable=['working_memory', 'model_state']
            )

            logging.debug(f"Consciousness shape: {consciousness1.shape}")
            logging.debug(f"Working memory shape: {new_working_memory1.shape}")
            expected_consciousness_shape = (1, self.output_dim * 2 + self.working_memory_size + 2)
            self.assertEqual(consciousness1.shape, expected_consciousness_shape)
            self.assertEqual(new_working_memory1.shape, (1, self.working_memory_size))
            self.assertIsInstance(working_memory1, dict)
            self.assertIn('working_memory', working_memory1)

            # Check that working memory is actually being perturbed
            self.assertFalse(jnp.allclose(working_memory1, working_memory2),
                             "Working memory should be different due to perturbation")

            # Check that consciousness states are different due to different RNG keys
            self.assertFalse(jnp.allclose(consciousness1, consciousness2),
                             "Consciousness states should be different due to different RNG keys")

            # Check for NaN values
            self.assertFalse(jnp.any(jnp.isnan(consciousness1)), "Consciousness output contains NaN values")
            self.assertFalse(jnp.any(jnp.isnan(working_memory1)), "Working memory output contains NaN values")

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

        try:
            logging.debug("Calling simulate_consciousness with rng_keys1")
            result1 = self.consciousness_sim.apply(
                params, x, rngs=rng_keys1,
                method=self.consciousness_sim.simulate_consciousness, mutable=['working_memory', 'model_state']
            )
            logging.debug(f"Result1 type: {type(result1)}, length: {len(result1) if isinstance(result1, tuple) else 'N/A'}")
            consciousness_state1, new_working_memory1, working_memory1 = result1

            logging.debug("Calling simulate_consciousness with rng_keys2")
            result2 = self.consciousness_sim.apply(
                params, x, rngs=rng_keys2,
                method=self.consciousness_sim.simulate_consciousness, mutable=['working_memory', 'model_state']
            )
            logging.debug(f"Result2 type: {type(result2)}, length: {len(result2) if isinstance(result2, tuple) else 'N/A'}")
            consciousness_state2, new_working_memory2, working_memory2 = result2
        except Exception as e:
            logging.error(f"simulate_consciousness method failed: {str(e)}")
            self.fail(f"simulate_consciousness method failed: {str(e)}")

        logging.debug(f"Consciousness state shape: {consciousness_state1.shape}")
        logging.debug(f"New working memory shape: {new_working_memory1.shape}")
        logging.debug(f"Working memory type: {type(working_memory1)}")
        if isinstance(working_memory1, dict) and 'working_memory' in working_memory1:
            logging.debug(f"Working memory shape: {working_memory1['working_memory']['current_state'].shape}")
        else:
            logging.warning(f"Unexpected working_memory1 structure: {working_memory1}")

        # Check types and shapes
        self.assertIsInstance(consciousness_state1, jnp.ndarray)
        self.assertIsInstance(new_working_memory1, jnp.ndarray)
        self.assertIsInstance(working_memory1, dict)
        expected_consciousness_shape = (1, self.output_dim * 2 + self.working_memory_size + 2)
        self.assertEqual(consciousness_state1.shape, expected_consciousness_shape,
                         f"Expected shape {expected_consciousness_shape}, got {consciousness_state1.shape}")
        self.assertEqual(new_working_memory1.shape, (1, self.working_memory_size))
        self.assertIn('working_memory', working_memory1, "working_memory1 should contain 'working_memory' key")
        self.assertIn('current_state', working_memory1['working_memory'], "working_memory1['working_memory'] should contain 'current_state' key")
        self.assertEqual(working_memory1['working_memory']['current_state'].shape, (1, self.working_memory_size))

        # Check that new_working_memory is different from the initial working memory
        self.assertFalse(jnp.allclose(new_working_memory1, working_memory1['working_memory']['current_state']),
                         "New working memory should be different due to perturbation")

        # Check that the outputs are different due to different RNG keys
        self.assertFalse(jnp.allclose(consciousness_state1, consciousness_state2),
                         "Consciousness states should be different due to different RNG keys")
        self.assertFalse(jnp.allclose(new_working_memory1, new_working_memory2),
                         "New working memories should be different due to different RNG keys")
        self.assertFalse(jnp.allclose(working_memory1['working_memory']['current_state'], working_memory2['working_memory']['current_state']),
                         "Working memory states should be different due to different RNG keys")

        # Check for NaN values
        self.assertFalse(jnp.any(jnp.isnan(consciousness_state1)), "Consciousness state contains NaN values")
        self.assertFalse(jnp.any(jnp.isnan(new_working_memory1)), "New working memory contains NaN values")

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
            consciousness_state, new_working_memory, working_memory = self.consciousness_sim.apply(
                params, x, rngs=rng_keys1, method=self.consciousness_sim.simulate_consciousness, mutable=['working_memory', 'model_state']
            )
            logging.debug(f"Generated consciousness_state with shape: {consciousness_state.shape}")

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
            consciousness, new_working_memory, working_memory = self.consciousness_sim.apply(
                params, x, rngs=rng_keys, method=self.consciousness_sim.simulate_consciousness, mutable=['working_memory', 'model_state']
            )
            logging.debug(f"Consciousness shape: {consciousness.shape}")
            logging.debug(f"New working memory shape: {new_working_memory.shape}")
            logging.debug(f"Working memory shape: {working_memory['working_memory']['current_state'].shape}")
        except Exception as e:
            logging.error(f"Error in forward pass: {str(e)}")
            self.fail(f"Forward pass failed: {str(e)}")

        # Check the output shapes
        expected_consciousness_shape = (1, self.output_dim * 2 + self.working_memory_size + 2)
        self.assertEqual(consciousness.shape, expected_consciousness_shape,
                         f"Expected consciousness shape {expected_consciousness_shape}, got {consciousness.shape}")
        self.assertEqual(working_memory.shape, (1, self.working_memory_size),
                         f"Expected working memory shape (1, {self.working_memory_size}), got {working_memory.shape}")
        logging.info("Output shape checks passed")

        # Check if the output is different from the input (processing occurred)
        self.assertFalse(jnp.allclose(consciousness[:, :self.output_dim], x),
                         "Output should be different from input")
        logging.info("Processing effect check passed")

        # Check if the output values are within a reasonable range
        self.assertTrue(jnp.all(jnp.isfinite(consciousness)),
                        "Consciousness output contains non-finite values")
        self.assertTrue(jnp.all(jnp.isfinite(working_memory)),
                        "Working memory output contains non-finite values")
        logging.info("Output range check passed")

        # Check for NaN values
        self.assertFalse(jnp.any(jnp.isnan(consciousness)),
                         "Consciousness output contains NaN values")
        self.assertFalse(jnp.any(jnp.isnan(working_memory)),
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
            consciousness1, new_working_memory1, working_memory1 = self.consciousness_sim.apply(
                params, x1, rngs=rng_keys1,
                method=self.consciousness_sim.simulate_consciousness, mutable=['working_memory', 'model_state']
            )
        except Exception as e:
            self.fail(f"First forward pass failed: {str(e)}")

        # Second forward pass with different input and RNG key
        rng_key2 = jax.random.PRNGKey(1)
        rng_keys2 = {'dropout': jax.random.fold_in(rng_key2, 0), 'perturbation': jax.random.fold_in(rng_key2, 1)}
        try:
            consciousness2, new_working_memory2, working_memory2 = self.consciousness_sim.apply(
                params, x2, rngs=rng_keys2,
                method=self.consciousness_sim.simulate_consciousness, mutable=['working_memory', 'model_state']
            )
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
            self.consciousness_sim.apply(
                params, x, rngs=None, method=self.consciousness_sim.simulate_consciousness, mutable=['working_memory', 'model_state']
            )
            logging.info("Test with None rngs passed")
        except Exception as e:
            self.fail(f"Test with None rngs failed: {str(e)}")

        # Test with missing 'dropout' key
        invalid_rngs1 = {'perturbation': jax.random.PRNGKey(0)}
        try:
            self.consciousness_sim.apply(
                params, x, rngs=invalid_rngs1, method=self.consciousness_sim.simulate_consciousness, mutable=['working_memory', 'model_state']
            )
            logging.info("Test with missing 'dropout' key passed")
        except Exception as e:
            self.fail(f"Test with missing 'dropout' key failed: {str(e)}")

        # Test with missing 'perturbation' key
        invalid_rngs2 = {'dropout': jax.random.PRNGKey(0)}
        try:
            self.consciousness_sim.apply(
                params, x, rngs=invalid_rngs2, method=self.consciousness_sim.simulate_consciousness, mutable=['working_memory', 'model_state']
            )
            logging.info("Test with missing 'perturbation' key passed")
        except Exception as e:
            self.fail(f"Test with missing 'perturbation' key failed: {str(e)}")

        # Test with non-PRNGKey values
        invalid_rngs3 = {'dropout': 0, 'perturbation': 1}
        try:
            self.consciousness_sim.apply(
                params, x, rngs=invalid_rngs3, method=self.consciousness_sim.simulate_consciousness, mutable=['working_memory', 'model_state']
            )
            logging.info("Test with non-PRNGKey values passed")
        except Exception as e:
            self.fail(f"Test with non-PRNGKey values failed: {str(e)}")

        logging.info("All invalid rngs tests passed successfully")

    logging.info("All tests completed successfully")

if __name__ == '__main__':
    unittest.main()

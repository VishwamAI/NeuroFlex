# Unit tests for the JAX module

import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import optax
from NeuroFlex.core_neural_networks import JAXModule, jax_train, jax_predict
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TestJAXModule(unittest.TestCase):
    def setUp(self):
        self.features = (10, 5)  # Tuple of features for JAXModule
        self.model = JAXModule(features=self.features)
        self.X = jnp.ones((20, self.features[0]))
        self.y = jax.random.randint(jax.random.PRNGKey(0), (20,), 0, self.features[-1])
        self.key = jax.random.PRNGKey(0)

    def test_jax_model_initialization(self):
        params = self.model.init(self.key, self.X)
        self.assertIsNotNone(params)
        self.assertIn('params', params)
        # Check if the model has the correct number of layers
        self.assertEqual(len(params['params']), len(self.features))

    def test_jax_train(self):
        try:
            jax_train(self.model, self.X, self.y, num_epochs=10, batch_size=32, learning_rate=0.01)
        except Exception as e:
            self.fail(f"Training failed with error: {str(e)}")

        # Note: Since jax_train is a placeholder, we can't check trained parameters
        # TODO: Add more assertions when jax_train is implemented

    def test_jax_predict(self):
        try:
            predictions = jax_predict(self.model, self.X)
            self.assertIsNotNone(predictions, "Predictions should not be None")
            # TODO: Add more assertions when jax_predict is implemented
        except Exception as e:
            self.fail(f"Prediction failed with error: {str(e)}")

    def test_model_forward_pass(self):
        params = self.model.init(self.key, self.X)
        output = self.model.apply(params, self.X)
        self.assertEqual(output.shape, (self.X.shape[0], self.features[-1]), "Output shape mismatch")
        self.assertTrue(jnp.all(jnp.isfinite(output)), "Output contains non-finite values")

    # Removed test_alignment_with_pytorch as it's no longer applicable

if __name__ == '__main__':
    unittest.main()

# Unit tests for the JAX module

import unittest
import jax
import jax.numpy as jnp
from src.NeuroFlex.modules.jax import JAXModel, train_jax_model

class TestJAXModule(unittest.TestCase):
    def setUp(self):
        self.model = JAXModel(features=10)
        self.X = jnp.ones((10, 256))  # Dummy input
        self.y = jnp.zeros(10)  # Dummy labels

    def test_jax_model_initialization(self):
        params = self.model.init(jax.random.PRNGKey(0), self.X)
        self.assertIsNotNone(params)

    def test_train_jax_model(self):
        trained_params = train_jax_model(self.model, self.X, self.y)
        self.assertIsNotNone(trained_params)

    def test_jax_vmap(self):
        def f(x):
            return x * 2

        batched_f = jax.vmap(f)
        result = batched_f(self.X)
        self.assertEqual(result.shape, self.X.shape)

if __name__ == '__main__':
    unittest.main()

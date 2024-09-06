# Unit tests for the JAX module

import unittest
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from NeuroFlex.core_neural_networks import JaxModel, create_jax_model, train_jax_model, jax_predict
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TestJaxModel(unittest.TestCase):
    def setUp(self):
        self.input_shape = (10,)
        self.output_dim = 5
        self.hidden_layers = [32, 16]
        self.model = create_jax_model(self.input_shape, self.output_dim, self.hidden_layers)
        self.X = jnp.ones((20,) + self.input_shape)
        self.y = jax.random.randint(jax.random.PRNGKey(0), (20, self.output_dim), 0, 2)
        self.key = jax.random.PRNGKey(0)

    def test_jax_model_initialization(self):
        params = self.model.init(self.key, self.X[0])
        self.assertIsNotNone(params)
        self.assertIn('params', params)
        # Check if the model has the correct number of layers
        self.assertEqual(len(params['params']), len(self.hidden_layers) + 1)  # +1 for output layer

    def test_jax_train(self):
        try:
            state, history = train_jax_model(
                self.model,
                self.X,
                self.y,
                self.input_shape,
                epochs=10,
                batch_size=32,
                learning_rate=0.01
            )
            self.assertIsInstance(state, train_state.TrainState)
            self.assertIsNotNone(history)
            self.assertIn('train_loss', history)
            self.assertEqual(len(history['train_loss']), 10)  # 10 epochs
        except Exception as e:
            self.fail(f"Training failed with error: {str(e)}")

    def test_jax_predict(self):
        try:
            params = self.model.init(self.key, self.X[0])['params']
            predictions = jax_predict(self.model, params, self.X)
            self.assertIsNotNone(predictions, "Predictions should not be None")
            self.assertEqual(predictions.shape, (self.X.shape[0], self.output_dim), "Predictions shape mismatch")
        except Exception as e:
            self.fail(f"Prediction failed with error: {str(e)}")

    def test_model_forward_pass(self):
        params = self.model.init(self.key, self.X[0])['params']
        output = self.model.apply({'params': params}, self.X[0])
        self.assertEqual(output.shape, (self.output_dim,), "Output shape mismatch")
        self.assertTrue(jnp.all(jnp.isfinite(output)), "Output contains non-finite values")

    def test_model_gradients(self):
        params = self.model.init(self.key, self.X[0])['params']

        @jax.jit
        def loss_fn(params, x, y):
            pred = self.model.apply({'params': params}, x)
            return jnp.mean((pred - y) ** 2)

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params, self.X[0], self.y[0])

        self.assertIsNotNone(grads)
        self.assertEqual(jax.tree_util.tree_structure(grads),
                         jax.tree_util.tree_structure(params))

if __name__ == '__main__':
    unittest.main()

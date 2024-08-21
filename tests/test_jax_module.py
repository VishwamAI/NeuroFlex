# Unit tests for the JAX module

import unittest
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from NeuroFlex.jax_module import JAXModel, train_jax_model, batch_predict

class TestJAXModule(unittest.TestCase):
    def setUp(self):
        self.model = JAXModel(features=[256, 128, 10])
        self.X_small = jnp.ones((5, 10))  # Small input
        self.X_medium = jnp.ones((10, 20))  # Medium input
        self.X_large = jnp.ones((20, 30))  # Large input
        self.y_small = jnp.zeros(5)
        self.y_medium = jnp.zeros(10)
        self.y_large = jnp.zeros(20)
        self.key = jax.random.PRNGKey(0)

    def test_jax_model_initialization(self):
        params_small = self.model.init(self.key, self.X_small)
        params_large = self.model.init(self.key, self.X_large)
        self.assertIsNotNone(params_small)
        self.assertIsNotNone(params_large)

    def test_train_jax_model(self):
        initial_params = self.model.init(self.key, self.X_medium)['params']

        # Define a more appropriate loss function (e.g., MSE for regression)
        def loss_fn(pred, y):
            return jnp.mean((pred - y) ** 2)

        # Calculate initial loss
        initial_loss = loss_fn(self.model.apply({'params': initial_params}, self.X_medium), self.y_medium)

        try:
            trained_params, final_loss, training_history = train_jax_model(
                self.model, initial_params, self.X_medium, self.y_medium,
                loss_fn=loss_fn, epochs=100, patience=20, min_delta=1e-6, batch_size=32
            )
        except Exception as e:
            self.fail(f"Training failed with error: {str(e)}")

        self.assertIsNotNone(trained_params, "Trained parameters should not be None")
        self.assertIsNotNone(final_loss, "Final loss should not be None")
        self.assertIsInstance(training_history, list, "Training history should be a list")
        self.assertGreater(len(training_history), 0, "Training history should not be empty")

        # Check if trained parameters are different from initial parameters
        param_diff = jax.tree_map(lambda x, y: jnp.sum(jnp.abs(x - y)), initial_params, trained_params)
        total_diff = sum(jax.tree_leaves(param_diff))
        self.assertGreater(total_diff, 0, "Trained parameters should be different from initial parameters")

        # Check if final loss is lower than initial loss
        self.assertLess(final_loss, initial_loss, "Final loss should be lower than initial loss")

        # Check if loss decreased during training (allow for small fluctuations and plateaus)
        self.assertTrue(all(training_history[i] >= training_history[i+1] * 0.99 or
                            abs(training_history[i] - training_history[i+1]) < 1e-6
                            for i in range(len(training_history)-1)),
                        "Loss should generally decrease or plateau during training")

        # Check if there's significant improvement in loss
        self.assertLess(training_history[-1], 0.9 * training_history[0], "Training should show improvement")

        # Test model application with trained parameters for different input sizes
        for X in [self.X_small, self.X_medium, self.X_large]:
            try:
                output = self.model.apply({'params': trained_params}, X)
                self.assertEqual(output.shape, (X.shape[0], self.model.features[-1]), f"Output shape mismatch for input shape {X.shape}")
                self.assertTrue(jnp.all(jnp.isfinite(output)), f"Output contains non-finite values for input shape {X.shape}")
                self.assertTrue(jnp.all(jnp.abs(output) < 1e5), f"Output values are not within a reasonable range for input shape {X.shape}")
            except Exception as e:
                self.fail(f"Model application failed for input shape {X.shape}: {str(e)}")

        # Verify the structure of trained parameters
        expected_layers = ['dense_layers_0', 'dense_layers_1', 'final_dense']
        for layer in expected_layers:
            self.assertIn(layer, trained_params, f"Trained params should contain '{layer}'")
            self.assertIn('kernel', trained_params[layer], f"'{layer}' should have a 'kernel'")
            self.assertIn('bias', trained_params[layer], f"'{layer}' should have a 'bias'")

        # Verify the shapes of the trained parameters
        expected_shapes = [(self.X_medium.shape[1], 256), (256, 128), (128, self.model.features[-1])]
        for layer, shape in zip(expected_layers, expected_shapes):
            self.assertEqual(trained_params[layer]['kernel'].shape, shape, f"Shape mismatch for {layer} kernel")
            self.assertEqual(trained_params[layer]['bias'].shape, (shape[1],), f"Shape mismatch for {layer} bias")

        # Test model prediction
        test_input = jax.random.normal(self.key, (5, self.X_medium.shape[1]))
        try:
            predictions = self.model.apply({'params': trained_params}, test_input)
            self.assertEqual(predictions.shape, (5, self.model.features[-1]), "Prediction shape mismatch")
            self.assertTrue(jnp.all(jnp.isfinite(predictions)), "Predictions contain non-finite values")
            self.assertTrue(jnp.all(jnp.abs(predictions) < 1e5), "Predictions are not within a reasonable range")
        except Exception as e:
            self.fail(f"Model prediction failed: {str(e)}")

        # Test for overfitting (use a more lenient threshold)
        train_loss = loss_fn(self.model.apply({'params': trained_params}, self.X_medium), self.y_medium)
        test_loss = loss_fn(self.model.apply({'params': trained_params}, self.X_large), self.y_large)
        self.assertLess(test_loss / train_loss, 2.0, "Model may be overfitting")

        # Additional checks for numerical stability
        self.assertFalse(jnp.any(jnp.isnan(final_loss)), "Final loss contains NaN values")
        self.assertFalse(jnp.any(jnp.isinf(final_loss)), "Final loss contains infinite values")

        # Check for consistent output across multiple runs
        predictions1 = self.model.apply({'params': trained_params}, self.X_medium)
        predictions2 = self.model.apply({'params': trained_params}, self.X_medium)
        self.assertTrue(jnp.allclose(predictions1, predictions2), "Model output is not deterministic")

        # Verify gradients
        gradients = jax.grad(lambda p: loss_fn(self.model.apply({'params': p}, self.X_medium), self.y_medium))(trained_params)
        for grad in jax.tree_leaves(gradients):
            self.assertFalse(jnp.any(jnp.isnan(grad)), "Gradients contain NaN values")
            self.assertFalse(jnp.any(jnp.isinf(grad)), "Gradients contain infinite values")

    def test_jax_vmap(self):
        def f(x):
            return x * 2

        batched_f = jax.vmap(f)
        for X in [self.X_small, self.X_medium, self.X_large]:
            result = batched_f(X)
            self.assertEqual(result.shape, X.shape)

    def test_batch_predict(self):
        params = self.model.init(self.key, self.X_medium)['params']
        for X in [self.X_small, self.X_medium, self.X_large]:
            try:
                predictions = batch_predict(params, X)
                self.assertEqual(predictions.shape, (X.shape[0], self.model.features[-1]),
                                 f"Prediction shape mismatch for input shape {X.shape}")
                self.assertTrue(jnp.all(jnp.isfinite(predictions)),
                                f"Predictions contain non-finite values for input shape {X.shape}")
                self.assertTrue(jnp.all(jnp.abs(predictions) < 1e5),
                                f"Predictions are not within a reasonable range for input shape {X.shape}")
            except ValueError as e:
                if X.shape[1] != self.X_medium.shape[1]:
                    self.assertIn("Input dimension mismatch", str(e),
                                  f"Unexpected error for input shape {X.shape}: {str(e)}")
                else:
                    self.fail(f"Unexpected ValueError for input shape {X.shape}: {str(e)}")
            except Exception as e:
                self.fail(f"Unexpected error for input shape {X.shape}: {str(e)}")

        # Test with incorrect parameter structure
        incorrect_params = {'wrong_key': params['dense_layers_0']}
        with self.assertRaises(ValueError):
            batch_predict(incorrect_params, self.X_medium)

        # Test with empty input
        with self.assertRaises(ValueError):
            batch_predict(params, jnp.array([]))

        # Test with 1D input
        X_1d = jnp.ones(self.X_medium.shape[1])
        predictions_1d = batch_predict(params, X_1d)
        self.assertEqual(predictions_1d.shape, (1, self.model.features[-1]),
                         "Prediction shape mismatch for 1D input")

        # Test with 3D input
        X_3d = jnp.ones((5, 4, self.X_medium.shape[1]))
        predictions_3d = batch_predict(params, X_3d)
        self.assertEqual(predictions_3d.shape, (5 * 4, self.model.features[-1]),
                         "Prediction shape mismatch for 3D input")

        # Test with very large input
        X_large = jnp.ones((1000, self.X_medium.shape[1]))
        predictions_large = batch_predict(params, X_large)
        self.assertEqual(predictions_large.shape, (1000, self.model.features[-1]),
                         "Prediction shape mismatch for large input")

if __name__ == '__main__':
    unittest.main()

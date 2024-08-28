# Unit tests for the JAX module

import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import optax
from NeuroFlex.jax_module import JAXModel, train_jax_model, batch_predict
from NeuroFlex.pytorch_integration import PyTorchModel, train_pytorch_model
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TestJAXModule(unittest.TestCase):
    def setUp(self):
        self.features = 10
        self.model = JAXModel(features=self.features)
        self.X = jnp.ones((20, self.features))
        self.y = jax.random.randint(jax.random.PRNGKey(0), (20,), 0, self.features)
        self.key = jax.random.PRNGKey(0)

    def test_jax_model_initialization(self):
        params = self.model.init(self.key, self.X)
        self.assertIsNotNone(params)
        self.assertIn('params', params)
        self.assertIn('layer', params['params'])
        self.assertIn('kernel', params['params']['layer'])
        self.assertIn('bias', params['params']['layer'])

    def test_train_jax_model(self):
        initial_params = self.model.init(self.key, self.X)['params']

        try:
            trained_params = train_jax_model(
                self.model, initial_params, self.X, self.y,
                epochs=10, learning_rate=0.01
            )
        except Exception as e:
            self.fail(f"Training failed with error: {str(e)}")

        self.assertIsNotNone(trained_params, "Trained parameters should not be None")

        # Check if trained parameters are different from initial parameters
        param_diff = jax.tree_map(lambda x, y: jnp.sum(jnp.abs(x - y)), initial_params, trained_params)
        total_diff = sum(jax.tree_leaves(param_diff))
        self.assertGreater(total_diff, 0, "Trained parameters should be different from initial parameters")

        # Test model prediction
        test_input = jax.random.normal(self.key, (5, self.features))
        try:
            predictions = self.model.apply({'params': trained_params}, test_input)
            self.assertEqual(predictions.shape, (5, self.features), "Prediction shape mismatch")
            self.assertTrue(jnp.all(jnp.isfinite(predictions)), "Predictions contain non-finite values")
        except Exception as e:
            self.fail(f"Model prediction failed: {str(e)}")

    def test_batch_predict(self):
        params = self.model.init(self.key, self.X)['params']
        try:
            predictions = batch_predict(params, self.X)
            self.assertEqual(predictions.shape, (self.X.shape[0], self.features),
                             f"Prediction shape mismatch for input shape {self.X.shape}")
            self.assertTrue(jnp.all(jnp.isfinite(predictions)),
                            f"Predictions contain non-finite values for input shape {self.X.shape}")
        except Exception as e:
            self.fail(f"Unexpected error for input shape {self.X.shape}: {str(e)}")

        # Test with 1D input
        X_1d = jnp.ones(self.features)
        predictions_1d = batch_predict(params, X_1d)
        self.assertEqual(predictions_1d.shape, (1, self.features),
                         "Prediction shape mismatch for 1D input")

        # Test with incorrect input dimension
        X_incorrect = jnp.ones((20, self.features + 1))
        with self.assertRaises(ValueError):
            batch_predict(params, X_incorrect)

    def test_alignment_with_pytorch(self):
        # Set random seeds for reproducibility
        jax_key = jax.random.PRNGKey(0)
        torch.manual_seed(0)
        np.random.seed(0)

        # Initialize JAX model with specific initialization
        jax_model = JAXModel(features=self.features)
        jax_params = jax_model.init(jax_key, self.X)['params']

        # Initialize PyTorch model with matching initialization
        torch_model = PyTorchModel(features=self.features)
        with torch.no_grad():
            torch_model.layer.weight.copy_(torch.tensor(np.array(jax_params['layer']['kernel'].T)))
            torch_model.layer.bias.copy_(torch.tensor(np.array(jax_params['layer']['bias']).flatten()))

        # Train JAX model
        jax_losses = []
        def jax_callback(loss):
            jax_losses.append(loss)

        jax_trained_params = train_jax_model(jax_model, jax_params, self.X, self.y, epochs=10, learning_rate=0.01,
                                             callback=jax_callback)

        # Train PyTorch model
        torch_losses = []
        torch_params_history = []
        def torch_callback(loss, model):
            torch_losses.append(loss)
            torch_params_history.append({name: param.clone().detach().numpy() for name, param in model.named_parameters()})

        torch_trained_params = train_pytorch_model(torch_model, np.array(self.X), np.array(self.y), epochs=10, learning_rate=0.01,
                                                   callback=torch_callback)

        # Log training losses and parameter changes
        logging.info("JAX training losses: %s", jax_losses)
        logging.info("PyTorch training losses: %s", torch_losses)

        # Log final parameter differences
        logging.debug("Final parameter differences:")
        for jax_key in jax_trained_params['layer']:
            jax_param = np.array(jax_trained_params['layer'][jax_key])
            torch_key = 'weight' if jax_key == 'kernel' else jax_key
            torch_param = torch_trained_params[f'layer.{torch_key}'].detach().numpy()
            if jax_key == 'kernel':
                torch_param = torch_param.T  # Transpose for proper comparison
            param_diff = np.mean(np.abs(jax_param - torch_param))
            logging.debug(f"  {jax_key} (JAX) vs {torch_key} (PyTorch): {param_diff}")

        # Compare predictions
        test_input_key = jax.random.PRNGKey(42)  # Create a new PRNG key for test input
        test_input = jax.random.normal(test_input_key, (5, self.features))
        jax_predictions = jax_model.apply({'params': jax_trained_params}, test_input)
        torch_predictions = torch_model(torch.FloatTensor(np.array(test_input)))

        # Log predictions and final model parameters
        logging.debug("JAX predictions:\n%s", jax_predictions)
        logging.debug("PyTorch predictions:\n%s", torch_predictions.detach().numpy())
        logging.debug("JAX trained parameters:\n%s", jax.tree_map(lambda x: x, jax_trained_params))
        logging.debug("PyTorch trained parameters:\n%s", {k: v.detach().numpy() for k, v in torch_trained_params.items()})

        # Calculate and log the mean absolute difference
        mean_abs_diff = jnp.mean(jnp.abs(jax_predictions - torch_predictions.detach().numpy()))
        logging.info("Mean absolute difference between JAX and PyTorch predictions: %f", mean_abs_diff)

        # Check if predictions are similar (using a smaller tolerance)
        self.assertTrue(jnp.allclose(jax_predictions, torch_predictions.detach().numpy(), atol=1e-4),
                        f"JAX and PyTorch model predictions should be very similar. Mean absolute difference: {mean_abs_diff}")

        # Compare individual prediction differences
        max_diff = jnp.max(jnp.abs(jax_predictions - torch_predictions.detach().numpy()))
        logging.info("Maximum difference between JAX and PyTorch predictions: %f", max_diff)

        # Verify that both models produce valid probability distributions
        self.assertTrue(jnp.allclose(jnp.sum(jnp.exp(jax_predictions), axis=1), 1.0, atol=1e-6),
                        "JAX predictions do not sum to 1 after exponentiation")
        self.assertTrue(jnp.allclose(torch.sum(torch.exp(torch_predictions), dim=1).detach().numpy(), 1.0, atol=1e-6),
                        "PyTorch predictions do not sum to 1 after exponentiation")

if __name__ == '__main__':
    unittest.main()

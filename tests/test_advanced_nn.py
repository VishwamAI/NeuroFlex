import unittest
import unittest.mock
import logging
import jax
import jax.numpy as jnp
from flax import linen as nn
from NeuroFlex.advanced_nn import NeuroFlexNN, ResidualBlock, AdvancedNNComponents, create_rl_train_state, adversarial_training
from NeuroFlex.rl_module import select_action, PrioritizedReplayBuffer

logging.basicConfig(level=logging.DEBUG)

class TestReinforcementLearning(unittest.TestCase):
    def setUp(self):
        self.rng = jax.random.PRNGKey(0)
        self.input_shape = (1, 4)  # Assuming a simple environment with 4 state dimensions
        self.output_shape = (1, 2)  # Assuming 2 possible actions
        self.action_dim = 2
        self.model = NeuroFlexNN(
            features=(32, 16, self.action_dim),
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            use_rl=True,
            action_dim=self.action_dim
        )
        self.advanced_components = AdvancedNNComponents()
        self.advanced_components.initialize_rl_components(buffer_size=1000, learning_rate=0.001, epsilon_start=1.0)

    def test_model_initialization(self):
        self.assertIsInstance(self.model, NeuroFlexNN)
        self.assertEqual(self.model.action_dim, self.action_dim)
        self.assertTrue(self.model.use_rl)

    # Removed test_replay_buffer method as it's no longer applicable with PrioritizedReplayBuffer

    def test_optimizer(self):
        self.assertIsNotNone(self.advanced_components.optimizer)

    def test_epsilon(self):
        self.assertEqual(self.advanced_components.epsilon, 1.0)

    def test_select_action(self):
        observation = jnp.ones(self.input_shape)
        state = create_rl_train_state(self.rng, self.model, observation, self.advanced_components.optimizer)
        action = self.advanced_components.select_action(state, observation, epsilon=0.5)
        self.assertIsInstance(action, jnp.ndarray)
        self.assertEqual(action.shape, ())
        self.assertTrue(0 <= action < self.action_dim)

    def test_update_rl_model(self):
        observation = jnp.ones(self.input_shape)
        state = create_rl_train_state(self.rng, self.model, observation, self.advanced_components.optimizer)
        target_state = create_rl_train_state(self.rng, self.model, observation, self.advanced_components.optimizer)

        batch = {
            'observations': jnp.ones((32,) + self.input_shape),
            'actions': jnp.zeros((32,), dtype=jnp.int32),
            'rewards': jnp.ones((32,)),
            'next_observations': jnp.ones((32,) + self.input_shape),
            'dones': jnp.zeros((32,), dtype=jnp.bool_)
        }

        updated_state, loss = self.advanced_components.update_rl_model(state, target_state, batch)

        self.assertIsInstance(updated_state, type(state))
        self.assertIsInstance(loss, jnp.ndarray)
        self.assertEqual(loss.shape, ())

    def test_q_value_computation(self):
        observation = jnp.ones(self.input_shape)
        state = create_rl_train_state(self.rng, self.model, observation, self.advanced_components.optimizer)
        q_values = self.model.apply({'params': state.params}, observation)
        self.assertEqual(q_values.shape, (1, self.action_dim))

    def test_epsilon_greedy_policy(self):
        observation = jnp.ones(self.input_shape)
        state = create_rl_train_state(self.rng, self.model, observation, self.advanced_components.optimizer)

        with unittest.mock.patch('jax.random.uniform', return_value=jnp.array([0.05])):
            action = self.advanced_components.select_action(state, observation, epsilon=0.1)
        self.assertIsInstance(action, jnp.ndarray)
        self.assertEqual(action.shape, ())

        with unittest.mock.patch('jax.random.uniform', return_value=jnp.array([0.95])):
            action = self.advanced_components.select_action(state, observation, epsilon=0.1)
        self.assertIsInstance(action, jnp.ndarray)
        self.assertEqual(action.shape, ())

    def test_gradients(self):
        # Create a dummy input
        dummy_input = jnp.ones(self.input_shape)

        # Initialize the model
        params = self.model.init(self.rng, dummy_input)['params']

        # Define a simple loss function
        def loss_fn(params):
            output = self.model.apply({'params': params}, dummy_input)
            return jnp.sum(output)

        # Compute gradients
        grads = jax.grad(loss_fn)(params)

        # Check if gradients are non-zero
        self.assertTrue(any(jnp.any(g != 0) for g in jax.tree_leaves(grads)),
                        "Gradients should not be all zero")

        # Check gradient magnitudes
        grad_magnitudes = [jnp.max(jnp.abs(g)) for g in jax.tree_leaves(grads)]
        self.assertTrue(all(1e-8 < m < 1e5 for m in grad_magnitudes),
                        "Gradient magnitudes should be within reasonable range")

        logging.info("Gradient test completed successfully")

    def test_input_shape_mismatch(self):
        input_shape = (1, 28, 28, 1)
        output_shape = (1, 10)
        model = NeuroFlexNN(features=(32, 64, 10), use_cnn=True, input_shape=input_shape, output_shape=output_shape)
        params = model.init(jax.random.PRNGKey(0), jnp.ones(input_shape))['params']

        incorrect_shape = input_shape[:-1] + (input_shape[-1] + 1,)
        with self.assertRaises(ValueError) as cm:
            model.apply({'params': params}, jnp.ones(incorrect_shape))
        self.assertIn("Input shape mismatch", str(cm.exception))
        logging.info(f"Input shape mismatch test passed for {input_shape}")

    def test_batch_size_mismatch(self):
        input_shape = (1, 28, 28, 1)
        output_shape = (1, 10)
        model = NeuroFlexNN(features=(32, 64, 10), use_cnn=True, input_shape=input_shape, output_shape=output_shape)
        params = model.init(jax.random.PRNGKey(0), jnp.ones(input_shape))['params']

        incorrect_batch_shape = (2,) + input_shape[1:]
        with self.assertRaises(ValueError) as cm:
            model.apply({'params': params}, jnp.ones(incorrect_batch_shape))
        self.assertIn("Batch size mismatch", str(cm.exception))
        logging.info(f"Batch size mismatch test passed for {input_shape}")

    def test_nan_inf_handling(self):
        input_shape = (1, 28, 28, 1)
        output_shape = (1, 10)
        model = NeuroFlexNN(features=(32, 64, 10), use_cnn=True, input_shape=input_shape, output_shape=output_shape)
        params = model.init(jax.random.PRNGKey(0), jnp.ones(input_shape))['params']

        nan_input = jnp.ones(input_shape).at[0, 0, 0, 0].set(jnp.nan)
        with self.assertRaises(ValueError) as cm:
            model.apply({'params': params}, nan_input)
        self.assertIn("NaN", str(cm.exception), "Expected error for NaN input")

        inf_input = jnp.ones(input_shape).at[0, 0, 0, 0].set(jnp.inf)
        with self.assertRaises(ValueError) as cm:
            model.apply({'params': params}, inf_input)
        self.assertIn("Inf", str(cm.exception), "Expected error for Inf input")

        logging.info("NaN and Inf handling tests passed")

class TestConvolutionLayers(unittest.TestCase):
    def setUp(self):
        self.rng = jax.random.PRNGKey(0)
        self.input_shapes_2d = [(1, 28, 28, 1), (2, 32, 32, 3), (4, 64, 64, 1)]
        self.input_shapes_3d = [(1, 16, 16, 16, 1), (2, 32, 32, 32, 3), (4, 8, 8, 8, 1)]

    def test_2d_convolution(self):
        for input_shape in self.input_shapes_2d:
            with self.subTest(input_shape=input_shape):
                output_shape = (input_shape[0], 10)
                model = NeuroFlexNN(features=(32, 64, 10), use_cnn=True, conv_dim=2, input_shape=input_shape, output_shape=output_shape)
                variables = model.init(self.rng, jnp.ones(input_shape))
                params = variables['params']
                output = model.apply(variables, jnp.ones(input_shape), deterministic=True)
                self.assertEqual(output.shape, output_shape)
                self.assertIn('conv_layers', params)
                self.assertIn('bn_layers', params)
                cnn_output = model.apply(variables, jnp.ones(input_shape), method=model.cnn_block, deterministic=True)
                self.assertIsInstance(cnn_output, jnp.ndarray)

                # Calculate expected flattened output size
                input_height, input_width = input_shape[1:3]
                expected_height = input_height // 4  # Two 2x2 max pooling operations
                expected_width = input_width // 4
                expected_channels = 64  # Number of filters in the last conv layer
                expected_flat_size = expected_height * expected_width * expected_channels
                expected_shape = (input_shape[0], expected_flat_size)

                self.assertEqual(cnn_output.shape, expected_shape, f"Expected shape {expected_shape}, got {cnn_output.shape}")
                self.assertTrue(jnp.all(jnp.isfinite(cnn_output)), "CNN output should contain only finite values")
                self.assertTrue(jnp.any(cnn_output != 0), "CNN output should not be all zeros")
                self.assertTrue(jnp.all(cnn_output >= 0), "CNN output should be non-negative after ReLU")
                self.assertLess(jnp.max(cnn_output), 1e5, "CNN output values should be reasonably bounded")
                self.assertEqual(params['conv_layers'][0]['kernel'].shape, (3, 3, input_shape[-1], 32))
                self.assertEqual(params['conv_layers'][1]['kernel'].shape, (3, 3, 32, 64))
                self.assertEqual(len(cnn_output.shape), 2, "CNN output should be 2D (flattened)")

                # Check if the output is different for different inputs
                random_input = jax.random.normal(self.rng, input_shape)
                random_output = model.apply(variables, random_input, method=model.cnn_block, deterministic=True)
                self.assertFalse(jnp.allclose(cnn_output, random_output), "Output should be different for different inputs")

                # Check if gradients can be computed
                def loss_fn(params):
                    output = model.apply({'params': params}, jnp.ones(input_shape), deterministic=True)
                    return jnp.sum(output)
                grads = jax.grad(loss_fn)(params)
                self.assertIsNotNone(grads, "Gradients should be computable")
                self.assertTrue(any(jnp.any(g != 0) for g in jax.tree_leaves(grads)), "Some gradients should be non-zero")

    def test_3d_convolution(self):
        for input_shape in self.input_shapes_3d:
            with self.subTest(input_shape=input_shape):
                output_shape = (input_shape[0], 10)
                model = NeuroFlexNN(features=(32, 64, 10), use_cnn=True, conv_dim=3, input_shape=input_shape, output_shape=output_shape)
                variables = model.init(self.rng, jnp.ones(input_shape))
                params = variables['params']
                output = model.apply(variables, jnp.ones(input_shape), deterministic=True)
                self.assertEqual(output.shape, output_shape)
                self.assertIn('conv_layers', params)
                self.assertIn('bn_layers', params)
                cnn_output = model.apply(variables, jnp.ones(input_shape), method=model.cnn_block, deterministic=True)
                self.assertIsInstance(cnn_output, jnp.ndarray)

                # Calculate expected output shape
                input_depth, input_height, input_width = input_shape[1:4]
                expected_depth = input_depth // 4  # Two 2x2x2 max pooling operations
                expected_height = input_height // 4
                expected_width = input_width // 4
                expected_channels = 64  # Number of filters in the last conv layer
                expected_flat_size = expected_depth * expected_height * expected_width * expected_channels
                expected_shape = (input_shape[0], expected_flat_size)

                self.assertEqual(cnn_output.shape, expected_shape, f"Expected shape {expected_shape}, got {cnn_output.shape}")
                self.assertTrue(jnp.all(jnp.isfinite(cnn_output)), "CNN output should contain only finite values")
                self.assertTrue(jnp.any(cnn_output != 0), "CNN output should not be all zeros")
                self.assertTrue(jnp.all(cnn_output >= 0), "CNN output should be non-negative after ReLU")
                self.assertLess(jnp.max(cnn_output), 1e5, "CNN output values should be reasonably bounded")
                self.assertEqual(params['conv_layers'][0]['kernel'].shape, (3, 3, 3, input_shape[-1], 32))
                self.assertEqual(params['conv_layers'][1]['kernel'].shape, (3, 3, 3, 32, 64))
                self.assertEqual(len(cnn_output.shape), 2, "CNN output should be 2D (flattened)")
                self.assertEqual(cnn_output.size, expected_flat_size,
                                 f"Expected {expected_flat_size} elements, got {cnn_output.size}")

                # Test with different input values
                random_input = jax.random.normal(self.rng, input_shape)
                random_output = model.apply(variables, random_input, method=model.cnn_block, deterministic=True)
                self.assertEqual(random_output.shape, expected_shape, "Shape mismatch with random input")
                self.assertFalse(jnp.allclose(cnn_output, random_output), "Output should differ for different inputs")

                # Test for gradient flow
                def loss_fn(params):
                    output = model.apply({'params': params}, jnp.ones(input_shape), method=model.cnn_block, deterministic=True)
                    return jnp.sum(output)
                grads = jax.grad(loss_fn)(params)
                self.assertTrue(all(jnp.any(jnp.abs(g) > 0) for g in jax.tree_leaves(grads)),
                                "Gradients should flow through all layers")

    def test_cnn_block_accessibility(self):
        model_2d = NeuroFlexNN(features=(32, 64, 10), use_cnn=True, conv_dim=2, input_shape=(1, 28, 28, 1), output_shape=(1, 10))
        model_3d = NeuroFlexNN(features=(32, 64, 10), use_cnn=True, conv_dim=3, input_shape=(1, 16, 16, 16, 1), output_shape=(1, 10))

        self.assertTrue(hasattr(model_2d, 'cnn_block'), "cnn_block should be accessible in 2D model")
        self.assertTrue(hasattr(model_3d, 'cnn_block'), "cnn_block should be accessible in 3D model")

    def test_residual_connections(self):
        input_shape = (1, 28, 28, 1)
        output_shape = (1, 10)
        model = NeuroFlexNN(features=(32, 64, 10), use_cnn=True, conv_dim=2, input_shape=input_shape, output_shape=output_shape, use_residual=True)
        variables = model.init(self.rng, jnp.ones(input_shape))
        params = variables['params']
        output = model.apply(variables, jnp.ones(input_shape), deterministic=True)
        self.assertEqual(output.shape, output_shape)
        self.assertIn('dense_layers', params)
        self.assertTrue(any('ResidualBlock' in layer_name for layer_name in params['dense_layers'].keys()),
                        "ResidualBlock should be present in dense layers")

        # Test residual connection functionality
        x = jnp.ones(input_shape)
        for i, layer in enumerate(model.dense_layers[:-1]):  # Exclude dropout layer
            y = model.apply({'params': params}, x, method=lambda m, x: m.dense_layers[i](x))
            self.assertFalse(jnp.allclose(x, y), f"Residual connection in layer {i} should modify the input")
            x = y

    def test_mixed_cnn_dnn(self):
        input_shape = (1, 28, 28, 1)
        output_shape = (1, 10)
        model = NeuroFlexNN(features=[32, 64, 128, 10], use_cnn=True, conv_dim=2, input_shape=input_shape, output_shape=output_shape)
        variables = model.init(self.rng, jnp.ones(input_shape))
        params = variables['params']
        output = model.apply(variables, jnp.ones(input_shape), deterministic=True)
        self.assertEqual(output.shape, output_shape)
        self.assertIn('conv_layers', params)
        self.assertIn('bn_layers', params)
        self.assertIn('dense_layers', params)
        self.assertIn('final_dense', params)

        # Test CNN block output
        cnn_output = model.apply(variables, jnp.ones(input_shape), method=model.cnn_block, deterministic=True)
        self.assertIsInstance(cnn_output, jnp.ndarray)
        self.assertEqual(len(cnn_output.shape), 2, "CNN output should be 2D (flattened)")

        # Test DNN block output
        dnn_input = jnp.ones((1, 128))  # Assuming 128 is the flattened size after CNN
        dnn_output = model.apply(variables, dnn_input, method=model.dnn_block, deterministic=True)
        self.assertEqual(dnn_output.shape, output_shape)

        # Test end-to-end forward pass
        full_output = model.apply(variables, jnp.ones(input_shape), deterministic=True)
        self.assertEqual(full_output.shape, output_shape)

    def test_error_handling(self):
        with self.assertRaises(ValueError):
            NeuroFlexNN(features=[32, 64, 10], use_cnn=True, conv_dim=4, input_shape=(1, 28, 28, 1), output_shape=(1, 10))

        model = NeuroFlexNN(features=[32, 64, 10], use_cnn=True, conv_dim=2, input_shape=(1, 28, 28, 1), output_shape=(1, 10))
        variables = model.init(self.rng, jnp.ones((1, 28, 28, 1)))
        params = variables['params']
        del params['conv_layers'][0]
        with self.assertRaises(KeyError):
            model.apply({'params': params}, jnp.ones((1, 28, 28, 1)))

        with self.assertRaises(ValueError):
            model.apply(variables, jnp.ones((1, 32, 32, 1)))

    def test_input_dimension_mismatch(self):
        model = NeuroFlexNN(features=[32, 64, 10], use_cnn=True, conv_dim=2, input_shape=(1, 28, 28, 1), output_shape=(1, 10))
        variables = model.init(self.rng, jnp.ones((1, 28, 28, 1)))

        incorrect_shape = (1, 32, 32, 1)
        with self.assertRaises(ValueError):
            model.apply(variables, jnp.ones(incorrect_shape))

    def test_gradients(self):
        input_shape = (1, 28, 28, 1)
        output_shape = (1, 10)
        model = NeuroFlexNN(features=[32, 64, 10], use_cnn=True, conv_dim=2, input_shape=input_shape, output_shape=output_shape)
        variables = model.init(self.rng, jnp.ones(input_shape))

        def loss_fn(params):
            output = model.apply({'params': params}, jnp.ones(input_shape), deterministic=True)
            return jnp.sum(output)

        grads = jax.grad(loss_fn)(variables['params'])

        self.assertIn('conv_layers', grads)
        self.assertIn('bn_layers', grads)
        self.assertIn('final_dense', grads)

        for layer_grad in jax.tree_leaves(grads):
            self.assertTrue(jnp.any(layer_grad != 0), "Gradients should be non-zero")

    def test_model_consistency(self):
        input_shape = (1, 28, 28, 1)
        output_shape = (1, 10)
        model = NeuroFlexNN(features=[32, 64, 10], use_cnn=True, conv_dim=2, input_shape=input_shape, output_shape=output_shape)
        variables = model.init(self.rng, jnp.ones(input_shape))

        output1 = model.apply(variables, jnp.ones(input_shape), deterministic=True)
        output2 = model.apply(variables, jnp.ones(input_shape), deterministic=True)
        self.assertTrue(jnp.allclose(output1, output2), "Model should produce consistent output for the same input")

    def test_activation_function(self):
        def custom_activation(x):
            return jnp.tanh(x)

        input_shape = (1, 28, 28, 1)
        output_shape = (1, 10)
        model = NeuroFlexNN(features=[32, 64, 10], use_cnn=True, conv_dim=2, input_shape=input_shape, output_shape=output_shape)
        variables = model.init(self.rng, jnp.ones(input_shape))
        output = model.apply(variables, jnp.ones(input_shape), deterministic=True)

        self.assertTrue(jnp.all(jnp.abs(output) <= 1), "Output should be bounded between -1 and 1")

from unittest import mock

# TestReinforcementLearning class for testing RL components
class TestReinforcementLearning(unittest.TestCase):
    def setUp(self):
        self.rng = jax.random.PRNGKey(0)
        self.input_shape = (1, 4)  # Assuming a simple environment with 4 state dimensions
        self.output_shape = (1, 2)  # Assuming 2 possible actions
        self.action_dim = 2
        self.model = NeuroFlexNN(
            features=(32, 16, self.action_dim),
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            use_rl=True,
            action_dim=self.action_dim
        )

        # Mock the PrioritizedReplayBuffer
        self.mock_replay_buffer = mock.Mock(spec=PrioritizedReplayBuffer)
        self.model.replay_buffer = self.mock_replay_buffer

        # Mock the epsilon-greedy policy
        self.mock_epsilon = mock.Mock(return_value=0.1)
        self.model.rl_epsilon = self.mock_epsilon

    def test_action_selection(self):
        state = jnp.ones(self.input_shape)
        with mock.patch('jax.random.uniform', return_value=jnp.array([0.5])):
            action = select_action(state, self.model, self.model.params)
        self.assertIsInstance(action, jnp.ndarray)
        self.assertEqual(action.shape, (1,))

    def test_q_value_computation(self):
        state = jnp.ones(self.input_shape)
        q_values = self.model.apply({'params': self.model.params}, state)
        self.assertEqual(q_values.shape, (1, self.action_dim))

    def test_epsilon_greedy_policy(self):
        state = jnp.ones(self.input_shape)
        with mock.patch('jax.random.uniform', return_value=jnp.array([0.05])):
            action = select_action(state, self.model, self.model.params)
        self.assertIsInstance(action, jnp.ndarray)
        self.assertEqual(action.shape, (1,))

class TestDNNBlock(unittest.TestCase):
    def setUp(self):
        self.rng = jax.random.PRNGKey(0)
        self.input_shape = (1, 100)
        self.output_shape = (1, 16)
        self.model = NeuroFlexNN(features=(64, 32, 16), input_shape=self.input_shape, output_shape=self.output_shape)
        self.model_with_residual = NeuroFlexNN(features=(64, 32, 16), input_shape=self.input_shape, output_shape=self.output_shape, use_residual=True)

    def test_dnn_block(self):
        variables = self.model.init(self.rng, jnp.ones(self.input_shape))
        params = variables['params']
        output = self.model.apply({'params': params}, jnp.ones(self.input_shape))

        # Check output shape
        self.assertEqual(output.shape, (1, 16))

        # Verify the presence and correct sizes of dense layers
        self.assertIn('dense_layers_0', params)
        self.assertEqual(params['dense_layers_0']['kernel'].shape, (100, 64))
        self.assertIn('dense_layers_1', params)
        self.assertEqual(params['dense_layers_1']['kernel'].shape, (64, 32))
        self.assertIn('final_dense', params)
        self.assertEqual(params['final_dense']['kernel'].shape, (32, 16))

        # Check for ReLU activation in each layer
        for i, layer in enumerate(self.model.dense_layers[:-1]):  # Exclude dropout layer
            layer_output = self.model.apply({'params': params}, jnp.ones(self.input_shape), method=lambda m, x: m.dense_layers[i](x))
            self.assertTrue(jnp.any(layer_output > 0), f"ReLU activation in layer {i} should produce some positive values")
            self.assertTrue(jnp.all(layer_output >= 0), f"ReLU activation in layer {i} should produce non-negative values")

        # Verify that the output is different from the input, indicating processing
        self.assertFalse(jnp.allclose(output, jnp.ones(output.shape)))

        # Check if output values are within a reasonable range
        self.assertTrue(jnp.all(jnp.isfinite(output)), "Output should contain only finite values")
        self.assertLess(jnp.max(jnp.abs(output)), 1e5, "Output values should be reasonably bounded")

        # Test the dnn_block method specifically
        dnn_output = self.model.apply({'params': params}, jnp.ones(self.input_shape), method=self.model.dnn_block)
        self.assertEqual(dnn_output.shape, (1, 16), "DNN block output shape should match the final layer")
        self.assertTrue(jnp.all(jnp.isfinite(dnn_output)), "DNN block output should contain only finite values")

        # Check if at least some activations are non-zero
        self.assertTrue(jnp.any(dnn_output != 0), "DNN block output should have some non-zero values")

        # Check gradients flow through the network
        def loss_fn(params):
            output = self.model.apply({'params': params}, jnp.ones(self.input_shape))
            return jnp.sum(output)

        grads = jax.grad(loss_fn)(params)
        for layer_grad in jax.tree_leaves(grads):
            self.assertTrue(jnp.any(layer_grad != 0), "Gradients should flow through all layers")

        # Check if the final output has some non-zero values and is different from intermediate layers
        final_output = self.model.apply({'params': params}, jnp.ones(self.input_shape), method=self.model.dnn_block)
        self.assertTrue(jnp.any(final_output != 0), "Final output should have some non-zero values")
        for i, layer in enumerate(self.model.dense_layers[:-1]):  # Exclude dropout layer
            layer_output = self.model.apply({'params': params}, jnp.ones(self.input_shape), method=lambda m, x: m.dense_layers[i](x))
            self.assertFalse(jnp.allclose(final_output, layer_output), f"Final output should be different from layer {i} output")

    def test_residual_connections(self):
        variables = self.model_with_residual.init(self.rng, jnp.ones(self.input_shape))
        params = variables['params']
        output = self.model_with_residual.apply({'params': params}, jnp.ones(self.input_shape))

        # Check output shape
        self.assertEqual(output.shape, (1, 16))

        # Verify the presence of residual blocks
        self.assertIn('ResidualBlock_0', params)
        self.assertIn('ResidualBlock_1', params)

        # Test the dnn_block method with residual connections
        dnn_output = self.model_with_residual.apply({'params': params}, jnp.ones(self.input_shape), method=self.model_with_residual.dnn_block)
        self.assertEqual(dnn_output.shape, (1, 16), "DNN block output shape should match the final layer")
        self.assertTrue(jnp.all(jnp.isfinite(dnn_output)), "DNN block output should contain only finite values")

        # Check if residual connections are working
        for i, layer in enumerate(self.model_with_residual.dense_layers[:-1]):  # Exclude dropout layer
            layer_input = jnp.ones(self.input_shape)
            layer_output = self.model_with_residual.apply({'params': params}, layer_input, method=lambda m, x: m.dense_layers[i](x))
            self.assertFalse(jnp.allclose(layer_input, layer_output), f"Residual connection in layer {i} should modify the input")

# The TestSHAPInterpretability class has been removed as it relies on the removed shap dependency.

class TestNeuroFlexNNShapeValidation(unittest.TestCase):
    def setUp(self):
        self.rng = jax.random.PRNGKey(0)
        self.input_shape = (1, 28, 28, 1)
        self.output_shape = (1, 10)
        self.features = (32, 64, 10)

    def test_valid_shapes(self):
        model = NeuroFlexNN(features=self.features, input_shape=self.input_shape, output_shape=self.output_shape, use_cnn=True)
        self.assertIsInstance(model, NeuroFlexNN)

    def test_invalid_input_shape_type(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=self.features, input_shape=[1, 28, 28, 1], output_shape=self.output_shape, use_cnn=True)
        self.assertIn("Input and output shapes must be tuples", str(cm.exception))

    def test_invalid_output_shape_type(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=self.features, input_shape=self.input_shape, output_shape=[1, 10], use_cnn=True)
        self.assertIn("Input and output shapes must be tuples", str(cm.exception))

    def test_invalid_input_shape_dimensions(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=self.features, input_shape=(28, 28, 1), output_shape=self.output_shape, use_cnn=True)
        self.assertIn("Input shape must have at least 2 dimensions", str(cm.exception))

    def test_invalid_output_shape_dimensions(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=self.features, input_shape=self.input_shape, output_shape=(10,), use_cnn=True)
        self.assertIn("Output shape must have at least 2 dimensions", str(cm.exception))

    def test_mismatched_features_output(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=(32, 64, 20), input_shape=self.input_shape, output_shape=self.output_shape, use_cnn=True)
        self.assertIn("Last feature dimension", str(cm.exception))

    def test_invalid_cnn_input_shape(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=self.features, input_shape=(1, 28, 28), output_shape=self.output_shape, use_cnn=True)
        self.assertIn("For CNN with conv_dim=2, input shape must have 4 dimensions", str(cm.exception))

    def test_batch_size_mismatch(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=self.features, input_shape=(2, 28, 28, 1), output_shape=(1, 10), use_cnn=True)
        self.assertIn("Batch size mismatch", str(cm.exception))

    def test_missing_action_dim(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=self.features, input_shape=self.input_shape, output_shape=self.output_shape, use_rl=True)
        self.assertIn("action_dim must be provided when use_rl is True", str(cm.exception))

    def test_negative_dimensions(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=self.features, input_shape=(1, -28, 28, 1), output_shape=self.output_shape, use_cnn=True)
        self.assertIn("All dimensions must be positive", str(cm.exception))

class TestAdversarialTraining(unittest.TestCase):
    def setUp(self):
        self.rng = jax.random.PRNGKey(0)
        self.input_shape = (1, 28, 28, 1)
        self.output_shape = (1, 10)
        self.model = NeuroFlexNN(features=(32, 64, 10), use_cnn=True, input_shape=self.input_shape, output_shape=self.output_shape)

    def test_adversarial_training(self):
        params = self.model.init(self.rng, jnp.ones(self.input_shape))['params']
        key, subkey = jax.random.split(self.rng)
        input_data = {
            'image': jax.random.uniform(subkey, self.input_shape, minval=0, maxval=1),
            'label': jax.nn.one_hot(jnp.array([0]), 10)
        }
        epsilon = 0.1
        step_size = 0.01
        perturbed_input = adversarial_training(self.model, params, input_data, epsilon, step_size)

        # Check if perturbed input is not None
        self.assertIsNotNone(perturbed_input)

        # Check if perturbed input has the same shape as original input
        self.assertEqual(perturbed_input['image'].shape, self.input_shape)

        # Check if perturbed input is different from original input
        self.assertFalse(jnp.allclose(perturbed_input['image'], input_data['image']))

        # Check if the magnitude of perturbation is within epsilon
        perturbation = perturbed_input['image'] - input_data['image']
        self.assertTrue(jnp.all(jnp.abs(perturbation) <= epsilon + 1e-6))

        # Check if perturbed input is clipped to [0, 1] range
        self.assertTrue(jnp.all(perturbed_input['image'] >= 0) and jnp.all(perturbed_input['image'] <= 1))

        # Check if the perturbation changes the model's output
        original_output = self.model.apply({'params': params}, input_data['image'])
        perturbed_output = self.model.apply({'params': params}, perturbed_input['image'])
        self.assertFalse(jnp.allclose(original_output, perturbed_output, atol=1e-4))

        # Quantify the change in model output
        output_diff = jnp.abs(original_output - perturbed_output)
        self.assertTrue(jnp.any(output_diff > 1e-4))

        # Check if the label remains unchanged
        self.assertTrue(jnp.array_equal(input_data['label'], perturbed_input['label']))

        # Test with different epsilon and step_size
        epsilon_small = 0.01
        step_size_small = 0.001
        perturbed_input_small = adversarial_training(self.model, params, input_data, epsilon_small, step_size_small)
        perturbation_small = perturbed_input_small['image'] - input_data['image']
        self.assertTrue(jnp.all(jnp.abs(perturbation_small) <= epsilon_small + 1e-6))

        # Test with multi-class output
        multi_class_output = self.model.apply({'params': params}, input_data['image'])
        self.assertEqual(multi_class_output.shape[-1], 10)

        # Check if convolution layers are correctly named and applied
        self.assertIn('conv_layers_0', params)
        self.assertIn('conv_layers_1', params)
        conv_output = self.model.apply({'params': params}, input_data['image'], method=self.model.cnn_block)
        self.assertIsNotNone(conv_output)
        self.assertEqual(conv_output.ndim, 2)  # Flattened output from CNN block

        # Additional checks for CNN block
        cnn_params = self.model.init(self.rng, jnp.ones(self.input_shape))['params']
        self.assertIn('conv_layers_0', cnn_params)
        self.assertIn('conv_layers_1', cnn_params)
        self.assertEqual(cnn_params['conv_layers_0']['kernel'].shape, (3, 3, 1, 32))
        self.assertEqual(cnn_params['conv_layers_1']['kernel'].shape, (3, 3, 32, 64))

        # Test CNN block output
        cnn_output = self.model.apply({'params': cnn_params}, input_data['image'], method=self.model.cnn_block)
        self.assertIsNotNone(cnn_output)
        self.assertEqual(cnn_output.ndim, 2)
        self.assertTrue(cnn_output.shape[-1] % 64 == 0)  # Flattened output should be multiple of 64

if __name__ == '__main__':
    unittest.main()

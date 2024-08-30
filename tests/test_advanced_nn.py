import sys
sys.path.insert(0, 'src')
import unittest
import unittest.mock
import logging
import jax
import jax.numpy as jnp
import tensorflow as tf
from flax import linen as nn
from NeuroFlex.advanced_nn import NeuroFlexNN
from NeuroFlex.rl_module import (
    select_action, PrioritizedReplayBuffer, create_train_state, RLAgent,
    RLEnvironment, train_rl_agent, ExtendedTrainState, run_validation
)
from NeuroFlex.tensorflow_convolutions import TensorFlowConvolutions
from NeuroFlex.utils import convert_array, create_backend
from NeuroFlex.advanced_nn import adversarial_training

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
        self.replay_buffer = PrioritizedReplayBuffer(capacity=1000)

    def test_model_initialization(self):
        self.assertIsInstance(self.model, NeuroFlexNN)
        self.assertEqual(self.model.action_dim, self.action_dim)
        self.assertTrue(self.model.use_rl)

    def test_replay_buffer(self):
        self.assertIsInstance(self.replay_buffer, PrioritizedReplayBuffer)
        self.assertEqual(self.replay_buffer.capacity, 1000)

    def test_select_action(self):
        observation = jnp.ones(self.input_shape)
        state = create_train_state(self.rng, self.model, observation)
        action = select_action(state, observation)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < self.action_dim)

    def test_q_value_computation(self):
        observation = jnp.ones(self.input_shape)
        state = create_train_state(self.rng, self.model, observation)
        q_values = self.model.apply({'params': state.train_state.params}, observation)
        self.assertEqual(q_values.shape, (1, self.action_dim))

    def test_train_rl_agent(self):
        env = RLEnvironment("CartPole-v1")
        agent = RLAgent(features=[32, 16], action_dim=env.action_space.n)
        trained_state, rewards, info = train_rl_agent(
            agent, env, num_episodes=10, max_steps=100
        )
        self.assertIsInstance(trained_state, ExtendedTrainState)
        self.assertIsInstance(rewards, list)
        self.assertIsInstance(info, dict)
        self.assertGreater(len(rewards), 0)

    def test_run_validation(self):
        env = RLEnvironment("CartPole-v1")
        agent = RLAgent(features=[32, 16], action_dim=env.action_space.n)
        state = create_train_state(self.rng, agent, jnp.ones(self.input_shape))
        validation_rewards = run_validation(state, env, num_episodes=5, max_steps=100)
        self.assertIsInstance(validation_rewards, list)
        self.assertEqual(len(validation_rewards), 5)

    def test_select_action(self):
        observation = jnp.ones(self.input_shape)
        model = NeuroFlexNN(
            features=(32, 16, self.action_dim),
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            use_rl=True,
            action_dim=self.action_dim
        )
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, observation)['params']
        state = create_train_state(rng, model, observation)

        with unittest.mock.patch('jax.random.uniform', return_value=jnp.array([0.05])):
            action = select_action(state, observation)
        self.assertIsInstance(int(action.item()), int)
        self.assertTrue(0 <= int(action.item()) < self.action_dim)

        with unittest.mock.patch('jax.random.uniform', return_value=jnp.array([0.95])):
            action = select_action(state, observation)
        self.assertIsInstance(int(action.item()), int)
        self.assertTrue(0 <= int(action.item()) < self.action_dim)

        # Test with different epsilon values
        state = state.replace(epsilon=0.0)  # Greedy action
        action = select_action(state, observation)
        self.assertIsInstance(int(action.item()), int)
        self.assertTrue(0 <= int(action.item()) < self.action_dim)

        state = state.replace(epsilon=1.0)  # Random action
        action = select_action(state, observation)
        self.assertIsInstance(int(action.item()), int)
        self.assertTrue(0 <= int(action.item()) < self.action_dim)

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

    # Residual connections are no longer part of the current implementation
    # This test has been removed as it's no longer applicable

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
        observation = jnp.ones(self.input_shape)
        rng = jax.random.PRNGKey(0)
        state = create_train_state(rng, self.model, observation)

        with mock.patch('jax.random.uniform', return_value=jnp.array([0.5])):
            action = select_action(state, observation)

        action_int = int(action.item())
        self.assertIsInstance(action_int, int)
        self.assertTrue(0 <= action_int < self.action_dim)

    def test_q_value_computation(self):
        observation = jnp.ones(self.input_shape)
        rng = jax.random.PRNGKey(0)
        state = create_train_state(rng, self.model, observation)

        q_values, _ = state.apply_fn(
            {'params': state.train_state.params, 'batch_stats': state.batch_stats},
            observation[None, ...],
            mutable=['batch_stats']
        )
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
        self.assertEqual(model.input_shape, self.input_shape)
        self.assertEqual(model.output_shape, self.output_shape)
        self.assertEqual(model.features, self.features)
        self.assertTrue(model.use_cnn)
        self.assertFalse(model.use_rl)

    def test_invalid_input_shape_type(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=self.features, input_shape=[1, 28, 28, 1], output_shape=self.output_shape, use_cnn=True)
        self.assertEqual(str(cm.exception), "Input shape must be a tuple, got <class 'list'>")

    def test_invalid_output_shape_type(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=self.features, input_shape=self.input_shape, output_shape=[1, 10], use_cnn=True)
        self.assertEqual(str(cm.exception), "Output shape must be a tuple, got <class 'list'>")

    def test_invalid_input_shape_dimensions(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=self.features, input_shape=(28, 28, 1), output_shape=self.output_shape, use_cnn=True)
        self.assertEqual(str(cm.exception), "Input shape must have at least 2 dimensions, got (28, 28, 1)")

    def test_invalid_output_shape_dimensions(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=self.features, input_shape=self.input_shape, output_shape=(10,), use_cnn=True)
        self.assertEqual(str(cm.exception), "Output shape must have at least 2 dimensions, got (10,)")

    def test_mismatched_features_output(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=(32, 64, 20), input_shape=self.input_shape, output_shape=self.output_shape, use_cnn=True)
        self.assertEqual(str(cm.exception), "Last feature dimension 20 must match output shape 10")

    def test_invalid_cnn_input_shape(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=self.features, input_shape=(1, 28, 28), output_shape=self.output_shape, use_cnn=True)
        self.assertEqual(str(cm.exception), "For CNN with conv_dim=2, input shape must have 4 dimensions, got 3")

    def test_batch_size_mismatch(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=self.features, input_shape=(2, 28, 28, 1), output_shape=(1, 10), use_cnn=True)
        self.assertEqual(str(cm.exception), "Batch size mismatch. Input shape: 2, Output shape: 1")

    def test_missing_action_dim(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=self.features, input_shape=self.input_shape, output_shape=self.output_shape, use_rl=True)
        self.assertEqual(str(cm.exception), "action_dim must be provided when use_rl is True")

    def test_negative_dimensions(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=self.features, input_shape=(1, -28, 28, 1), output_shape=self.output_shape, use_cnn=True)
        self.assertEqual(str(cm.exception), "All dimensions in Input shape must be positive integers, got (1, -28, 28, 1)")

    def test_zero_dimensions(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=self.features, input_shape=(1, 0, 28, 1), output_shape=self.output_shape, use_cnn=True)
        self.assertEqual(str(cm.exception), "All dimensions in Input shape must be positive integers, got (1, 0, 28, 1)")

    def test_invalid_feature_dimensions(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=(32, 0, 10), input_shape=self.input_shape, output_shape=self.output_shape, use_cnn=True)
        self.assertEqual(str(cm.exception), "All features must be positive integers, got (32, 0, 10)")

    def test_invalid_conv_dim(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=self.features, input_shape=self.input_shape, output_shape=self.output_shape, use_cnn=True, conv_dim=4)
        self.assertEqual(str(cm.exception), "conv_dim must be 2 or 3, got 4")

    def test_cnn_input_channel_mismatch(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=(16, 32, 10), input_shape=(1, 28, 28, 3), output_shape=self.output_shape, use_cnn=True)
        self.assertEqual(str(cm.exception), "For CNN, input channels 3 must match first feature dimension 16")

    def test_dnn_input_feature_mismatch(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=(64, 32, 10), input_shape=(1, 100), output_shape=self.output_shape, use_cnn=False)
        self.assertEqual(str(cm.exception), "For DNN, input features 100 must match first feature dimension 64")

    def test_rl_action_dim_mismatch(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=self.features, input_shape=self.input_shape, output_shape=self.output_shape, use_rl=True, action_dim=5)
        self.assertEqual(str(cm.exception), "action_dim 5 must match last dimension of output_shape 10")

    def test_valid_rl_configuration(self):
        model = NeuroFlexNN(features=self.features, input_shape=self.input_shape, output_shape=self.output_shape, use_rl=True, action_dim=10)
        self.assertTrue(model.use_rl)
        self.assertEqual(model.action_dim, 10)

    def test_invalid_features_type(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=[32, 64, 10], input_shape=self.input_shape, output_shape=self.output_shape, use_cnn=True)
        self.assertEqual(str(cm.exception), "Features must be a tuple, got <class 'list'>")

    def test_cnn_3d_input_shape(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=self.features, input_shape=(1, 16, 16, 16, 1), output_shape=self.output_shape, use_cnn=True, conv_dim=2)
        self.assertEqual(str(cm.exception), "For CNN with conv_dim=2, input shape must have 4 dimensions, got 5")

    def test_valid_3d_cnn_configuration(self):
        model = NeuroFlexNN(features=self.features, input_shape=(1, 16, 16, 16, 1), output_shape=self.output_shape, use_cnn=True, conv_dim=3)
        self.assertTrue(model.use_cnn)
        self.assertEqual(model.conv_dim, 3)

    def test_invalid_action_dim_type(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=self.features, input_shape=self.input_shape, output_shape=self.output_shape, use_rl=True, action_dim=5.5)
        self.assertEqual(str(cm.exception), "action_dim must be a positive integer, got 5.5")

    def test_negative_action_dim(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=self.features, input_shape=self.input_shape, output_shape=self.output_shape, use_rl=True, action_dim=-5)
        self.assertEqual(str(cm.exception), "action_dim must be a positive integer, got -5")

    def test_valid_dnn_configuration(self):
        model = NeuroFlexNN(features=(64, 32, 10), input_shape=(1, 64), output_shape=(1, 10), use_cnn=False)
        self.assertFalse(model.use_cnn)
        self.assertEqual(model.features, (64, 32, 10))

    def test_invalid_feature_count(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=(32,), input_shape=self.input_shape, output_shape=self.output_shape, use_cnn=True)
        self.assertEqual(str(cm.exception), "Features must have at least two elements, got (32,)")

    def test_non_integer_feature(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=(32, 64.5, 10), input_shape=self.input_shape, output_shape=self.output_shape, use_cnn=True)
        self.assertEqual(str(cm.exception), "All elements in features must be integers, got (32, 64.5, 10)")

    def test_valid_cnn_dnn_mixed_configuration(self):
        model = NeuroFlexNN(features=(32, 64, 128, 10), input_shape=(1, 28, 28, 1), output_shape=(1, 10), use_cnn=True)
        self.assertTrue(model.use_cnn)
        self.assertEqual(model.features, (32, 64, 128, 10))

    def test_valid_rl_dueling_configuration(self):
        model = NeuroFlexNN(features=self.features, input_shape=self.input_shape, output_shape=self.output_shape, use_rl=True, action_dim=10, use_dueling=True)
        self.assertTrue(model.use_rl)
        self.assertTrue(model.use_dueling)
        self.assertEqual(model.action_dim, 10)

    def test_valid_rl_double_configuration(self):
        model = NeuroFlexNN(features=self.features, input_shape=self.input_shape, output_shape=self.output_shape, use_rl=True, action_dim=10, use_double=True)
        self.assertTrue(model.use_rl)
        self.assertTrue(model.use_double)
        self.assertEqual(model.action_dim, 10)

    def test_invalid_input_shape_ndim(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=self.features, input_shape=(1,), output_shape=self.output_shape, use_cnn=True)
        self.assertEqual(str(cm.exception), "Input shape must have at least 2 dimensions, got (1,)")

    def test_invalid_output_shape_ndim(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=self.features, input_shape=self.input_shape, output_shape=(10,), use_cnn=True)
        self.assertEqual(str(cm.exception), "Output shape must have at least 2 dimensions, got (10,)")

    def test_invalid_features_length(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=(32,), input_shape=self.input_shape, output_shape=self.output_shape, use_cnn=True)
        self.assertEqual(str(cm.exception), "Features must have at least two elements, got (32,)")

    def test_invalid_conv_dim_type(self):
        with self.assertRaises(ValueError) as cm:
            NeuroFlexNN(features=self.features, input_shape=self.input_shape, output_shape=self.output_shape, use_cnn=True, conv_dim=2.5)
        self.assertEqual(str(cm.exception), "conv_dim must be 2 or 3, got 2.5")

class TestAdversarialTraining(unittest.TestCase):
    def setUp(self):
        self.rng = jax.random.PRNGKey(0)
        self.input_shape = (1, 28, 28, 1)
        self.output_shape = (1, 10)

    def test_adversarial_training_jax(self):
        model = NeuroFlexNN(features=(32, 64, 10), use_cnn=True, input_shape=self.input_shape, output_shape=self.output_shape, backend='jax')
        self._run_adversarial_training_test(model, jnp.ones, jnp.array_equal, jnp.allclose, jnp.abs, jnp.all, jax.random)

    def test_adversarial_training_tensorflow(self):
        model = NeuroFlexNN(features=(32, 64, 10), use_cnn=True, input_shape=self.input_shape, output_shape=self.output_shape, backend='tensorflow')
        self._run_adversarial_training_test(model, tf.ones, tf.math.equal, tf.math.reduce_all, tf.abs, tf.reduce_all, tf.random)

    def _run_adversarial_training_test(self, model, ones_func, array_equal_func, allclose_func, abs_func, all_func, random_module):
        params = model.init(self.rng, ones_func(self.input_shape))['params']
        key, subkey = jax.random.split(self.rng)
        input_data = {
            'image': random_module.uniform(subkey, self.input_shape, minval=0, maxval=1),
            'label': tf.one_hot(tf.constant([0]), 10) if model.backend == 'tensorflow' else jax.nn.one_hot(jnp.array([0]), 10)
        }
        epsilon = 0.1
        step_size = 0.01
        perturbed_input = adversarial_training(model, params, input_data, epsilon, step_size)

        self.assertIsNotNone(perturbed_input)
        self.assertEqual(perturbed_input['image'].shape, self.input_shape)
        self.assertFalse(allclose_func(perturbed_input['image'], input_data['image']))

        perturbation = perturbed_input['image'] - input_data['image']
        self.assertTrue(all_func(abs_func(perturbation) <= epsilon + 1e-6))

        self.assertTrue(all_func(perturbed_input['image'] >= 0) and all_func(perturbed_input['image'] <= 1))

        original_output = model.apply({'params': params}, input_data['image'])
        perturbed_output = model.apply({'params': params}, perturbed_input['image'])
        self.assertFalse(allclose_func(original_output, perturbed_output, atol=1e-4))

        output_diff = abs_func(original_output - perturbed_output)
        self.assertTrue(tf.reduce_any(output_diff > 1e-4) if model.backend == 'tensorflow' else jnp.any(output_diff > 1e-4))

        self.assertTrue(array_equal_func(input_data['label'], perturbed_input['label']))

        epsilon_small = 0.01
        step_size_small = 0.001
        perturbed_input_small = adversarial_training(model, params, input_data, epsilon_small, step_size_small)
        perturbation_small = perturbed_input_small['image'] - input_data['image']
        self.assertTrue(all_func(abs_func(perturbation_small) <= epsilon_small + 1e-6))

        multi_class_output = model.apply({'params': params}, input_data['image'])
        self.assertEqual(multi_class_output.shape[-1], 10)

        self.assertIn('conv_layers_0', params)
        self.assertIn('conv_layers_1', params)
        conv_output = model.apply({'params': params}, input_data['image'], method=model.cnn_block)
        self.assertIsNotNone(conv_output)
        self.assertEqual(conv_output.ndim, 2)

        cnn_params = model.init(self.rng, ones_func(self.input_shape))['params']
        self.assertIn('conv_layers_0', cnn_params)
        self.assertIn('conv_layers_1', cnn_params)
        self.assertEqual(cnn_params['conv_layers_0']['kernel'].shape, (3, 3, 1, 32))
        self.assertEqual(cnn_params['conv_layers_1']['kernel'].shape, (3, 3, 32, 64))

        cnn_output = model.apply({'params': cnn_params}, input_data['image'], method=model.cnn_block)
        self.assertIsNotNone(cnn_output)
        self.assertEqual(cnn_output.ndim, 2)
        self.assertTrue(cnn_output.shape[-1] % 64 == 0)

if __name__ == '__main__':
    unittest.main()

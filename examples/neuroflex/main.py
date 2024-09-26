# MIT License
# 
# Copyright (c) 2024 VishwamAI
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import unittest
import jax
import jax.numpy as jnp
from jax import jit
import flax.linen as nn
import gym
from NeuroFlex.core_neural_networks.advanced_nn import data_augmentation, NeuroFlexNN, create_train_state, select_action
from NeuroFlex.utils import validate_util_config
from NeuroFlex.config import get_neuroflex_version


class TestDataAugmentation(unittest.TestCase):
    def setUp(self):
        self.rng = jax.random.PRNGKey(0)
        self.image_shape = (32, 32, 3)
        self.batch_size = 4
        self.images = jax.random.uniform(
            self.rng,
            (self.batch_size,) + self.image_shape
        )

    def test_horizontal_flip(self):
        augmented, _ = data_augmentation(self.images, self.rng)
        self.assertEqual(augmented.shape, self.images.shape)
        # Check if at least one image is flipped horizontally
        flipped = jnp.any(jnp.not_equal(augmented[:, :, ::-1, :], self.images))
        self.assertTrue(flipped)

    def test_vertical_flip(self):
        augmented, _ = data_augmentation(self.images, self.rng)
        self.assertEqual(augmented.shape, self.images.shape)
        # Check if at least one image is flipped vertically
        flipped = jnp.any(jnp.not_equal(augmented[:, ::-1, :, :], self.images))
        self.assertTrue(flipped)

    def test_rotation(self):
        augmented, _ = data_augmentation(self.images, self.rng)
        self.assertEqual(augmented.shape, self.images.shape)
        # Check if at least one image is rotated
        rotated = jnp.any(jnp.not_equal(augmented, self.images))
        self.assertTrue(rotated)

    def test_brightness_adjustment(self):
        augmented, _ = data_augmentation(self.images, self.rng)
        self.assertEqual(augmented.shape, self.images.shape)
        # Check if brightness is adjusted while maintaining the valid range
        self.assertTrue(jnp.all(augmented >= 0) and jnp.all(augmented <= 1))
        brightness_changed = jnp.any(jnp.not_equal(augmented, self.images))
        self.assertTrue(brightness_changed)

    def test_contrast_adjustment(self):
        augmented, _ = data_augmentation(self.images, self.rng)
        self.assertEqual(augmented.shape, self.images.shape)
        # Check if contrast is adjusted while maintaining the valid range
        self.assertTrue(jnp.all(augmented >= 0) and jnp.all(augmented <= 1))
        contrast_changed = jnp.any(jnp.not_equal(augmented, self.images))
        self.assertTrue(contrast_changed)

    def test_key_update(self):
        _, key1 = data_augmentation(self.images, self.rng)
        _, key2 = data_augmentation(self.images, key1)
        self.assertFalse(jnp.array_equal(key1, key2))


class TestXLAOptimizations(unittest.TestCase):
    def setUp(self):
        self.rng = jax.random.PRNGKey(0)
        self.input_shape = (1, 28, 28, 1)
        self.model = NeuroFlexNN(features=[32, 10], use_cnn=True)

    def test_jit_compilation(self):
        params = self.model.init(self.rng, jnp.ones(self.input_shape))['params']

        @jit
        def forward(params, x):
            return self.model.apply({'params': params}, x)

        x = jnp.ones(self.input_shape)
        output = forward(params, x)
        self.assertEqual(output.shape, (1, 10))


class TestConvolutionLayers(unittest.TestCase):
    def setUp(self):
        self.rng = jax.random.PRNGKey(0)
        self.input_shape_2d = (1, 28, 28, 1)
        self.input_shape_3d = (1, 16, 16, 16, 1)

    def test_2d_convolution(self):
        model = NeuroFlexNN(features=[32, 10], use_cnn=True, conv_dim=2)
        params = model.init(self.rng, jnp.ones(self.input_shape_2d))['params']
        output = model.apply({'params': params}, jnp.ones(self.input_shape_2d))
        self.assertEqual(output.shape, (1, 10))
        self.assertIsInstance(model.cnn_block.layers[0], nn.Conv)

    def test_3d_convolution(self):
        model = NeuroFlexNN(features=[32, 10], use_cnn=True, conv_dim=3)
        params = model.init(self.rng, jnp.ones(self.input_shape_3d))['params']
        output = model.apply({'params': params}, jnp.ones(self.input_shape_3d))
        self.assertEqual(output.shape, (1, 10))
        self.assertIsInstance(model.cnn_block.layers[0], nn.Conv)


class TestReinforcementLearning(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('CartPole-v1')
        self.input_shape = self.env.observation_space.shape
        self.action_space = self.env.action_space.n
        self.model = NeuroFlexNN(features=[64, 32, self.action_space], use_rl=True, output_dim=self.action_space)

    def test_rl_model_initialization(self):
        rng = jax.random.PRNGKey(0)
        state, _, _ = create_train_state(rng, self.model, self.input_shape, 1e-3)
        self.assertIsNotNone(state)

    def test_action_selection(self):
        rng = jax.random.PRNGKey(0)
        state, _, _ = create_train_state(rng, self.model, self.input_shape, 1e-3)
        observation = self.env.reset()
        action = select_action(observation, self.model, state.params)
        self.assertIsInstance(action, jax.numpy.ndarray)
        self.assertEqual(action.shape, ())
        self.assertTrue(0 <= int(action) < self.action_space)


class TestUtilsAndConfig(unittest.TestCase):
    def test_validate_util_config(self):
        valid_config = {
            'util_name': 'load_data',
            'parameters': {'file_path': 'data.csv'}
        }
        self.assertTrue(validate_util_config(valid_config))

        invalid_config = {
            'util_name': 'invalid_util',
            'parameters': {}
        }
        with self.assertRaises(ValueError):
            validate_util_config(invalid_config)

    def test_neuroflex_version(self):
        version = get_neuroflex_version()
        self.assertEqual(version, "0.1.3")  # Update this if the version changes


if __name__ == '__main__':
    print(f"Running tests for NeuroFlex version: {get_neuroflex_version()}")
    unittest.main()

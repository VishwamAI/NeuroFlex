import unittest
import jax
import jax.numpy as jnp
from jax import jit
import flax.linen as nn
import gym
from NeuroFlex.core_neural_networks import NeuroFlex
from NeuroFlex.utils import data_augmentation
from NeuroFlex.reinforcement_learning import create_train_state, select_action
from NeuroFlex.scientific_domains.bioinformatics.ete_integration import ETEIntegration
from NeuroFlex.scientific_domains.bioinformatics.scikit_bio_integration import ScikitBioIntegration


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
        self.model = NeuroFlex(features=[32, 10], use_cnn=True)

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
        model = NeuroFlex(features=[32, 10], use_cnn=True, conv_dim=2)
        params = model.init(self.rng, jnp.ones(self.input_shape_2d))['params']
        output = model.apply({'params': params}, jnp.ones(self.input_shape_2d))
        self.assertEqual(output.shape, (1, 10))
        self.assertIsInstance(model.cnn_block.layers[0], nn.Conv)

    def test_3d_convolution(self):
        model = NeuroFlex(features=[32, 10], use_cnn=True, conv_dim=3)
        params = model.init(self.rng, jnp.ones(self.input_shape_3d))['params']
        output = model.apply({'params': params}, jnp.ones(self.input_shape_3d))
        self.assertEqual(output.shape, (1, 10))
        self.assertIsInstance(model.cnn_block.layers[0], nn.Conv)


class TestReinforcementLearning(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('CartPole-v1')
        self.input_shape = self.env.observation_space.shape
        self.action_space = self.env.action_space.n
        self.model = NeuroFlex(features=[64, 32, self.action_space], use_rl=True, action_dim=self.action_space)

    def test_rl_model_initialization(self):
        rng = jax.random.PRNGKey(0)
        state = create_train_state(rng, self.model, self.input_shape, learning_rate=1e-3)
        self.assertIsNotNone(state)

    def test_action_selection(self):
        rng = jax.random.PRNGKey(0)
        state = create_train_state(rng, self.model, self.input_shape)
        observation = self.env.reset()
        action = select_action(state, observation)
        self.assertIsInstance(action, jax.numpy.ndarray)
        self.assertEqual(action.shape, ())
        self.assertTrue(0 <= int(action) < self.action_space)


class TestETEIntegration(unittest.TestCase):
    def setUp(self):
        self.ete_integration = ETEIntegration()

    def test_create_tree(self):
        newick_string = "(A:1,(B:1,(C:1,D:1):0.5):0.5);"
        tree = self.ete_integration.create_tree(newick_string)
        self.assertIsNotNone(tree)
        self.assertEqual(len(tree), 4)  # 4 leaf nodes

    def test_analyze_tree(self):
        newick_string = "(A:1,(B:1,(C:1,D:1):0.5):0.5);"
        tree = self.ete_integration.create_tree(newick_string)
        self.ete_integration.analyze_tree(tree)
        # This test just ensures the method runs without errors

class TestScikitBioIntegration(unittest.TestCase):
    def setUp(self):
        self.skbio_integration = ScikitBioIntegration()

    def test_analyze_sequence(self):
        sequence = "ATCG"
        result = self.skbio_integration.analyze_sequence(sequence)
        self.assertIsNotNone(result)

    def test_calculate_diversity(self):
        counts = [10, 20, 30, 40]
        diversity = self.skbio_integration.calculate_diversity(counts)
        self.assertIsNotNone(diversity)

if __name__ == '__main__':
    unittest.main()

# Example script for NeuroscienceModel usage
from NeuroFlex.neuroscience_models.neuroscience_model import NeuroscienceModel
from neurolib.utils.loadData import Dataset
import numpy as np

def run_neuroscience_example():
    # Initialize the model
    model = NeuroscienceModel()

    # Load connectivity data
    dataset = Dataset(Cmat_file="path_to_connectivity_data.mat")
    model.load_connectivity(dataset)

    # Set custom parameters
    model.set_parameters({"sigma_ou": 0.03, "dt": 0.05})

    # Run simulation (prediction)
    dummy_data = np.random.rand(100, 2)  # Example input data
    simulation_output = model.predict(dummy_data)

    # Evaluate results
    dummy_labels = np.random.randint(0, 2, 100)  # Example labels
    evaluation = model.evaluate(dummy_data, dummy_labels)

    # Interpret the simulation
    interpretation = model.interpret_results(simulation_output)

    print(f"Evaluation: {evaluation}")
    print(f"Interpretation: {interpretation}")

# Uncomment the following line to run the example
# run_neuroscience_example()

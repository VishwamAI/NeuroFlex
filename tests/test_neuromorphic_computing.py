import unittest
import jax
import jax.numpy as jnp
import optax
from NeuroFlex.neuromorphic_computing import spiking_neuron, SpikingNeuralNetwork, create_neuromorphic_model

class TestNeuromorphicComputing(unittest.TestCase):
    def setUp(self):
        self.input_shape = (1, 8)
        self.num_neurons = [8, 4, 2]
        self.rng = jax.random.PRNGKey(0)

    def test_spiking_neuron(self):
        threshold, reset_potential, leak_factor = 1.0, 0.0, 0.9
        inputs = jnp.array([0.5, 0.8, 1.2])
        membrane_potential = jnp.zeros_like(inputs)

        spike, new_membrane_potential = spiking_neuron(inputs, membrane_potential, threshold, reset_potential, leak_factor)

        self.assertIsInstance(spike, jnp.ndarray)
        self.assertTrue(jnp.all((spike == 0.0) | (spike == 1.0)))
        self.assertFalse(jnp.array_equal(membrane_potential, new_membrane_potential))
        self.assertTrue(jnp.all(new_membrane_potential >= reset_potential))
        self.assertTrue(jnp.all(new_membrane_potential <= threshold))

    def test_spiking_neural_network(self):
        snn = SpikingNeuralNetwork(num_neurons=self.num_neurons)

        # Test with 2D input (batch, features)
        inputs_2d = jnp.ones(self.input_shape)
        membrane_potentials_2d = [jnp.zeros(self.input_shape[:1] + (n,)) for n in self.num_neurons]

        params = snn.init(self.rng, inputs_2d, membrane_potentials_2d)
        outputs_2d, new_membrane_potentials_2d = snn.apply(params, inputs_2d, membrane_potentials_2d)

        self.assertEqual(outputs_2d.shape, self.input_shape[:1] + (self.num_neurons[-1],))
        self.assertEqual(len(new_membrane_potentials_2d), len(self.num_neurons))
        for i, n in enumerate(self.num_neurons):
            self.assertEqual(new_membrane_potentials_2d[i].shape, self.input_shape[:1] + (n,))

        # Test with 1D input (features)
        inputs_1d = jnp.ones((8,))
        membrane_potentials_1d = [jnp.zeros((n,)) for n in self.num_neurons]

        outputs_1d, new_membrane_potentials_1d = snn.apply(params, inputs_1d, membrane_potentials_1d)

        self.assertEqual(outputs_1d.shape, (self.num_neurons[-1],))
        for i, n in enumerate(self.num_neurons):
            self.assertEqual(new_membrane_potentials_1d[i].shape, (n,))

        # Test with 3D input (batch, time, features)
        inputs_3d = jnp.ones((2, 3, 8))
        membrane_potentials_3d = [jnp.zeros((2, 3, n)) for n in self.num_neurons]

        outputs_3d, new_membrane_potentials_3d = snn.apply(params, inputs_3d, membrane_potentials_3d)

        self.assertEqual(outputs_3d.shape, (2, 3, self.num_neurons[-1]))
        for i, n in enumerate(self.num_neurons):
            self.assertEqual(new_membrane_potentials_3d[i].shape, (2, 3, n))

        # Test error handling for mismatched shapes
        with self.assertRaises(ValueError):
            mismatched_inputs = jnp.ones((3, 7))  # Mismatched input shape
            snn.apply(params, mismatched_inputs, membrane_potentials_2d)

    def test_neuromorphic_computing(self):
        nc = create_neuromorphic_model(self.num_neurons)
        params = nc.init_model(self.rng, self.input_shape)

        # Test with 2D input
        inputs_2d = jnp.ones(self.input_shape)
        membrane_potentials_2d = [jnp.zeros(self.input_shape[:1] + (n,)) for n in self.num_neurons]
        outputs_2d, new_membrane_potentials_2d = nc.forward(params, inputs_2d, membrane_potentials_2d)

        self.assertEqual(outputs_2d.shape, self.input_shape[:1] + (self.num_neurons[-1],))
        self.assertIsInstance(outputs_2d, jnp.ndarray)
        self.assertTrue(jnp.all((outputs_2d >= 0.0) & (outputs_2d <= 1.0)))

        # Test with 1D input
        inputs_1d = jnp.ones((8,))
        membrane_potentials_1d = [jnp.zeros((n,)) for n in self.num_neurons]
        outputs_1d, new_membrane_potentials_1d = nc.forward(params, inputs_1d, membrane_potentials_1d)

        self.assertEqual(outputs_1d.shape, (self.num_neurons[-1],))
        for i, n in enumerate(self.num_neurons):
            self.assertEqual(new_membrane_potentials_1d[i].shape, (n,))

        # Test with 3D input
        inputs_3d = jnp.ones((2, 3, 8))
        membrane_potentials_3d = [jnp.zeros((2, 3, n)) for n in self.num_neurons]
        outputs_3d, new_membrane_potentials_3d = nc.forward(params, inputs_3d, membrane_potentials_3d)

        self.assertEqual(outputs_3d.shape, (2, 3, self.num_neurons[-1]))

        # Test multiple forward passes
        outputs2, _ = nc.forward(params, inputs_2d, new_membrane_potentials_2d)
        self.assertFalse(jnp.array_equal(outputs_2d, outputs2), "Outputs should differ due to changing membrane potentials")

        # Test with different input values
        varied_inputs = jnp.array([[0.1, 0.5, 1.0, 0.8, 0.3, 0.7, 0.2, 0.9]])
        varied_outputs, _ = nc.forward(params, varied_inputs, membrane_potentials_2d)
        self.assertFalse(jnp.array_equal(outputs_2d, varied_outputs), "Outputs should differ for different input values")

    def test_train_step(self):
        nc = create_neuromorphic_model(self.num_neurons)
        params = nc.init_model(self.rng, self.input_shape)

        inputs = jnp.ones(self.input_shape)
        targets = jnp.zeros((1, self.num_neurons[-1]))
        membrane_potentials = [jnp.zeros(self.input_shape[:1] + (n,)) for n in self.num_neurons]
        optimizer = optax.adam(learning_rate=0.01)

        new_params, loss, new_membrane_potentials, new_optimizer = nc.train_step(params, inputs, targets, membrane_potentials, optimizer)

        self.assertIsInstance(loss, jnp.ndarray)
        self.assertLess(loss, 1.0)  # Assuming loss is reasonably small after one step
        self.assertIsInstance(new_optimizer, optax.GradientTransformation)
        self.assertNotEqual(id(optimizer), id(new_optimizer))  # Ensure a new optimizer instance is returned

        # Test multiple training steps
        for _ in range(5):
            params, loss, membrane_potentials, optimizer = nc.train_step(params, inputs, targets, membrane_potentials, optimizer)

        self.assertLess(loss, 0.5)  # Assuming loss decreases over multiple steps

    def test_edge_cases(self):
        snn = SpikingNeuralNetwork(num_neurons=self.num_neurons)

        # Test with empty input
        empty_input = jnp.array([])
        empty_potentials = [jnp.array([]) for _ in self.num_neurons]

        with self.assertRaises(ValueError):
            snn.init(self.rng, empty_input, empty_potentials)

        # Test with very large input
        large_input = jnp.ones((1000, 8))
        large_potentials = [jnp.zeros((1000, n)) for n in self.num_neurons]

        params = snn.init(self.rng, large_input, large_potentials)
        outputs, _ = snn.apply(params, large_input, large_potentials)

        self.assertEqual(outputs.shape, (1000, self.num_neurons[-1]))

        # Test with very small values
        small_input = jnp.full(self.input_shape, 1e-10)
        small_potentials = [jnp.full(self.input_shape[:1] + (n,), 1e-10) for n in self.num_neurons]

        outputs, _ = snn.apply(params, small_input, small_potentials)

        self.assertTrue(jnp.all(jnp.isfinite(outputs)))

if __name__ == '__main__':
    unittest.main()

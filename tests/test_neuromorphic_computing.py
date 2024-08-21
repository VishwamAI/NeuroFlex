import unittest
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from NeuroFlex.neuromorphic_computing import spiking_neuron, SpikingNeuralNetwork, NeuromorphicComputing, create_neuromorphic_model

class TestNeuromorphicComputing(unittest.TestCase):
    def setUp(self):
        self.input_shape = (1, 10)
        self.num_neurons = [8, 4, 2]
        self.rng = jax.random.PRNGKey(0)

    def test_spiking_neuron(self):
        threshold, reset_potential, leak_factor = 1.0, 0.0, 0.9
        inputs = jnp.array([0.5, 0.8, 1.2])
        membrane_potential = jnp.zeros_like(inputs)

        spike, new_membrane_potential = spiking_neuron(inputs, membrane_potential, threshold, reset_potential, leak_factor)

        self.assertIsInstance(spike, jnp.ndarray)
        self.assertTrue(jnp.all((spike == 0.0) | (spike == 1.0)))

        # Test that the membrane potential is being updated
        spike2, new_membrane_potential2 = spiking_neuron(inputs, new_membrane_potential, threshold, reset_potential, leak_factor)
        self.assertFalse(jnp.array_equal(spike, spike2))
        self.assertFalse(jnp.array_equal(membrane_potential, new_membrane_potential))

        # Test with different threshold
        spike_high, _ = spiking_neuron(inputs, membrane_potential, 2.0, reset_potential, leak_factor)
        self.assertTrue(jnp.all(spike_high == 0.0))

    def test_spiking_neural_network(self):
        snn = SpikingNeuralNetwork(num_neurons=self.num_neurons)
        inputs = jnp.ones(self.input_shape)
        membrane_potentials = [jnp.zeros(self.input_shape[:1] + (n,)) for n in self.num_neurons]

        params = snn.init(self.rng, inputs, membrane_potentials)
        outputs, new_membrane_potentials = snn.apply(params, inputs, membrane_potentials)

        print(f"Input shape: {inputs.shape}")
        print(f"Output shape: {outputs.shape}")
        print(f"New membrane potential shapes: {[mp.shape for mp in new_membrane_potentials]}")

        self.assertEqual(outputs.shape, self.input_shape[:1] + (self.num_neurons[-1],))
        self.assertEqual(len(new_membrane_potentials), len(self.num_neurons))
        for i, n in enumerate(self.num_neurons):
            self.assertEqual(new_membrane_potentials[i].shape, self.input_shape[:1] + (n,))

        # Test with different input shape
        different_input_shape = (2, 5)
        different_inputs = jnp.ones(different_input_shape)
        different_membrane_potentials = [jnp.zeros(different_input_shape[:1] + (n,)) for n in self.num_neurons]

        different_params = snn.init(self.rng, different_inputs, different_membrane_potentials)
        different_outputs, different_new_membrane_potentials = snn.apply(different_params, different_inputs, different_membrane_potentials)

        print(f"Different input shape: {different_inputs.shape}")
        print(f"Different output shape: {different_outputs.shape}")
        print(f"Different new membrane potential shapes: {[mp.shape for mp in different_new_membrane_potentials]}")

        self.assertEqual(different_outputs.shape, different_input_shape[:1] + (self.num_neurons[-1],))
        self.assertEqual(len(different_new_membrane_potentials), len(self.num_neurons))
        for i, n in enumerate(self.num_neurons):
            self.assertEqual(different_new_membrane_potentials[i].shape, different_input_shape[:1] + (n,))

        # Test error handling for mismatched shapes
        with self.assertRaises(ValueError):
            mismatched_inputs = jnp.ones((3, 7))  # Mismatched input shape
            snn.apply(params, mismatched_inputs, membrane_potentials)

        with self.assertRaises(ValueError):
            mismatched_potentials = [jnp.zeros((1, n+1)) for n in self.num_neurons]  # Mismatched potential shapes
            snn.apply(params, inputs, mismatched_potentials)

        # Test with 1D input
        one_d_input = jnp.ones((10,))
        one_d_potentials = [jnp.zeros((n,)) for n in self.num_neurons]
        one_d_params = snn.init(self.rng, one_d_input, one_d_potentials)
        one_d_outputs, one_d_new_potentials = snn.apply(one_d_params, one_d_input, one_d_potentials)

        self.assertEqual(one_d_outputs.shape, (self.num_neurons[-1],))
        for i, n in enumerate(self.num_neurons):
            self.assertEqual(one_d_new_potentials[i].shape, (n,))

    def test_neuromorphic_computing(self):
        nc = create_neuromorphic_model(self.num_neurons)
        params = nc.init_model(self.rng, self.input_shape)

        inputs = jnp.ones(self.input_shape)
        membrane_potentials = [jnp.zeros(self.input_shape[:1] + (n,)) for n in self.num_neurons]
        outputs, new_membrane_potentials = nc.forward(params, inputs, membrane_potentials)

        self.assertEqual(outputs.shape, self.input_shape[:1] + (self.num_neurons[-1],))
        self.assertIsInstance(outputs, jnp.ndarray)
        self.assertTrue(jnp.all((outputs >= 0.0) & (outputs <= 1.0)))

        # Test multiple forward passes
        outputs2, new_membrane_potentials2 = nc.forward(params, inputs, new_membrane_potentials)
        self.assertFalse(jnp.array_equal(outputs, outputs2), "Outputs should differ due to changing membrane potentials")

        # Test with different inputs
        different_inputs = jnp.array([[0.5, 1.0, 0.2, 0.8, 1.5, 0.3, 0.9, 0.1, 0.7, 0.4]])
        different_outputs, _ = nc.forward(params, different_inputs, membrane_potentials)
        self.assertFalse(jnp.array_equal(outputs, different_outputs), "Outputs should differ for different inputs")

    def test_train_step(self):
        nc = create_neuromorphic_model(self.num_neurons)
        params = nc.init_model(self.rng, self.input_shape)

        inputs = jnp.ones(self.input_shape)
        targets = jnp.zeros((1, self.num_neurons[-1]))
        membrane_potentials = [jnp.zeros(self.input_shape[:1] + (n,)) for n in self.num_neurons]
        optimizer = optax.adam(learning_rate=0.01)
        optimizer_state = optimizer.init(params)

        new_params, loss, new_membrane_potentials, new_optimizer = nc.train_step(params, inputs, targets, membrane_potentials, optimizer.replace(state=optimizer_state))

        self.assertIsInstance(loss, jnp.ndarray)
        self.assertLess(loss, 1.0)  # Assuming loss is reasonably small after one step
        self.assertIsInstance(new_optimizer, optax.GradientTransformation)
        self.assertNotEqual(id(optimizer), id(new_optimizer))  # Ensure a new optimizer instance is returned

if __name__ == '__main__':
    unittest.main()

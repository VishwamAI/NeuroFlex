import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import List, Tuple, Callable, Optional
import logging

def spiking_neuron(x, membrane_potential, threshold=1.0, reset_potential=0.0, leak_factor=0.9):
    new_membrane_potential = jnp.add(leak_factor * membrane_potential, x)
    spike = jnp.where(new_membrane_potential >= threshold, 1.0, 0.0)
    new_membrane_potential = jnp.where(spike == 1.0, reset_potential, new_membrane_potential)
    return spike, new_membrane_potential

class SpikingNeuralNetwork(nn.Module):
    num_neurons: List[int]
    activation: Callable = nn.relu
    spike_function: Callable = lambda x: jnp.where(x > 0, 1.0, 0.0)
    threshold: float = 1.0
    reset_potential: float = 0.0
    leak_factor: float = 0.9

    @nn.compact
    def __call__(self, inputs, membrane_potentials=None):
        logging.debug(f"Input shape: {inputs.shape}")
        x = inputs

        # Input validation and reshaping
        if len(inputs.shape) == 1:
            x = jnp.expand_dims(x, axis=0)
        elif len(inputs.shape) > 2:
            x = jnp.reshape(x, (-1, x.shape[-1]))

        if x.shape[1] != self.num_neurons[0]:
            raise ValueError(f"Input shape {x.shape} does not match first layer neurons {self.num_neurons[0]}")

        if membrane_potentials is None:
            membrane_potentials = [jnp.zeros((x.shape[0], num_neuron)) for num_neuron in self.num_neurons]
        else:
            if len(membrane_potentials) != len(self.num_neurons):
                raise ValueError(f"Expected {len(self.num_neurons)} membrane potentials, got {len(membrane_potentials)}")
            membrane_potentials = [jnp.broadcast_to(mp, (x.shape[0], mp.shape[-1])) for mp in membrane_potentials]

        logging.debug(f"Adjusted input shape: {x.shape}")
        logging.debug(f"Adjusted membrane potentials shapes: {[mp.shape for mp in membrane_potentials]}")

        new_membrane_potentials = []
        for i, (num_neuron, membrane_potential) in enumerate(zip(self.num_neurons, membrane_potentials)):
            logging.debug(f"Layer {i} - Input shape: {x.shape}, Membrane potential shape: {membrane_potential.shape}")

            spiking_layer = jax.vmap(lambda x, mp: spiking_neuron(x, mp, self.threshold, self.reset_potential, self.leak_factor),
                                     in_axes=(0, 0), out_axes=0)
            spikes, new_membrane_potential = spiking_layer(x, membrane_potential)

            logging.debug(f"Layer {i} - Spikes shape: {spikes.shape}, New membrane potential shape: {new_membrane_potential.shape}")

            x = self.activation(spikes)
            new_membrane_potentials.append(new_membrane_potential)

            # Adjust x for the next layer
            if i < len(self.num_neurons) - 1:
                x = nn.Dense(self.num_neurons[i+1])(x)

        logging.debug(f"Final output shape: {x.shape}")
        return self.spike_function(x), new_membrane_potentials

class NeuromorphicComputing(nn.Module):
    num_neurons: List[int]
    threshold: float = 1.0
    reset_potential: float = 0.0
    leak_factor: float = 0.9

    def setup(self):
        self.model = SpikingNeuralNetwork(num_neurons=self.num_neurons,
                                          threshold=self.threshold,
                                          reset_potential=self.reset_potential,
                                          leak_factor=self.leak_factor)
        logging.info(f"Initialized NeuromorphicComputing with {len(self.num_neurons)} layers")

    def __call__(self, inputs, membrane_potentials=None):
        return self.model(inputs, membrane_potentials)

    def init_model(self, rng, input_shape):
        dummy_input = jnp.zeros(input_shape)
        membrane_potentials = [jnp.zeros(input_shape[:-1] + (n,)) for n in self.num_neurons]
        # Ensure consistent shapes between inputs and membrane potentials
        if dummy_input.shape[1] != membrane_potentials[0].shape[1]:
            dummy_input = jnp.reshape(dummy_input, (-1, membrane_potentials[0].shape[1]))
        return self.init(rng, dummy_input, membrane_potentials)

    @jax.jit
    def forward(self, params, inputs, membrane_potentials):
        return self.apply(params, inputs, membrane_potentials)

    def train_step(self, params, inputs, targets, membrane_potentials, optimizer):
        def loss_fn(params):
            outputs, new_membrane_potentials = self.forward(params, inputs, membrane_potentials)
            return jnp.mean((outputs - targets) ** 2), new_membrane_potentials

        (loss, new_membrane_potentials), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, optimizer_state = optimizer.update(grads, optimizer.state)
        params = optax.apply_updates(params, updates)
        optimizer = optimizer.replace(state=optimizer_state)
        return params, loss, new_membrane_potentials, optimizer

    @staticmethod
    def handle_error(e: Exception) -> None:
        logging.error(f"Error in NeuromorphicComputing: {str(e)}")
        if isinstance(e, jax.errors.JAXException):
            logging.error("JAX-specific error occurred. Check JAX configuration and input shapes.")
        elif isinstance(e, ValueError):
            logging.error("Value error occurred. Check input data and model parameters.")
        else:
            logging.error("Unexpected error occurred. Please review the stack trace for more information.")
        raise

def create_neuromorphic_model(num_neurons: List[int]) -> NeuromorphicComputing:
    return NeuromorphicComputing(num_neurons=num_neurons)

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
    dtype: jnp.dtype = jnp.float32
    learning_rate: float = 0.01

    def setup(self):
        self.model = SpikingNeuralNetwork(
            num_neurons=self.num_neurons,
            threshold=self.threshold,
            reset_potential=self.reset_potential,
            leak_factor=self.leak_factor
        )
        self.optimizer = optax.adam(learning_rate=self.learning_rate)
        self.initialized = False  # Flag to track initialization status
        self.variables = None  # Store model variables
        logging.info(f"NeuromorphicComputing setup complete with {len(self.num_neurons)} layers")

    def __call__(self, inputs: jnp.ndarray, rng_key: jnp.ndarray, membrane_potentials: Optional[List[jnp.ndarray]] = None) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
        if not self.initialized:
            raise RuntimeError("NeuromorphicComputing model is not initialized. Call init_model() first.")

        try:
            inputs = self._preprocess_inputs(inputs)
            membrane_potentials = self._initialize_or_preprocess_membrane_potentials(inputs, membrane_potentials)

            noise_key, threshold_key, model_key = jax.random.split(rng_key, 3)
            membrane_potentials = self._add_noise_to_membrane_potentials(membrane_potentials, noise_key)
            dynamic_threshold = self._get_dynamic_threshold(threshold_key)

            outputs, new_membrane_potentials = self.model.apply(
                self.variables,
                inputs,
                membrane_potentials,
                rngs={'dropout': model_key},
                threshold=dynamic_threshold
            )
            outputs = self._add_noise_to_outputs(outputs, model_key)
            outputs = self._ensure_2d_output(outputs)

            return outputs, new_membrane_potentials
        except ValueError as e:
            logging.error(f"ValueError in NeuromorphicComputing.__call__: {str(e)}")
            self.handle_error(e)
        except Exception as e:
            logging.error(f"Unexpected error in NeuromorphicComputing.__call__: {str(e)}")
            self.handle_error(e)

    def _preprocess_inputs(self, inputs: jnp.ndarray) -> jnp.ndarray:
        if len(inputs.shape) == 1:
            return jnp.expand_dims(inputs, axis=0)
        elif len(inputs.shape) > 2:
            return jnp.reshape(inputs, (-1, inputs.shape[-1]))
        return inputs

    def _initialize_or_preprocess_membrane_potentials(self, inputs: jnp.ndarray, membrane_potentials: Optional[List[jnp.ndarray]]) -> List[jnp.ndarray]:
        if membrane_potentials is None:
            return [jnp.zeros((inputs.shape[0], n), dtype=self.dtype) for n in self.num_neurons]
        return [jnp.broadcast_to(mp, (inputs.shape[0], mp.shape[-1])) for mp in membrane_potentials]

    def _add_noise_to_membrane_potentials(self, membrane_potentials: List[jnp.ndarray], noise_key: jnp.ndarray) -> List[jnp.ndarray]:
        noise_factor = 1e-2
        return [
            mp + jax.random.normal(noise_key, mp.shape, dtype=self.dtype) * noise_factor
            for mp in membrane_potentials
        ]

    def _get_dynamic_threshold(self, threshold_key: jnp.ndarray) -> float:
        threshold_noise = jax.random.uniform(threshold_key, (), minval=-0.1, maxval=0.1)
        return self.threshold + threshold_noise

    def _add_noise_to_outputs(self, outputs: jnp.ndarray, model_key: jnp.ndarray) -> jnp.ndarray:
        output_noise_key, _ = jax.random.split(model_key)
        noise_factor = 1e-2
        return outputs + jax.random.normal(output_noise_key, outputs.shape, dtype=self.dtype) * noise_factor

    def _ensure_2d_output(self, outputs: jnp.ndarray) -> jnp.ndarray:
        if len(outputs.shape) == 1:
            return jnp.expand_dims(outputs, axis=0)
        elif len(outputs.shape) > 2:
            return jnp.reshape(outputs, (-1, outputs.shape[-1]))
        return outputs

    def init_model(self, rng: jnp.ndarray, input_shape: Tuple[int, ...]) -> None:
        try:
            dummy_input = jnp.zeros(input_shape, dtype=self.dtype)
            dummy_input = self._preprocess_inputs(dummy_input)
            dummy_membrane_potentials = [jnp.zeros((dummy_input.shape[0], n), dtype=self.dtype) for n in self.num_neurons]

            self.init_variables = self.init(rng, dummy_input, rng, dummy_membrane_potentials)
            self.opt_state = self.optimizer.init(self.init_variables['params'])
            self.variables = self.init_variables  # Set variables attribute
            logging.info(f"Model initialized with parameters. Params shape: {jax.tree_map(lambda x: x.shape, self.init_variables['params'])}")
            logging.debug(f"Full params structure: {jax.tree_map(lambda x: x.shape, self.init_variables['params'])}")
            self.initialized = True  # Set initialization flag
        except Exception as e:
            logging.error(f"Error during model initialization: {str(e)}")
            self.initialized = False
            raise

    def train_step(self, inputs: jnp.ndarray, targets: jnp.ndarray, rng_key: jnp.ndarray) -> Tuple[dict, float]:
        def loss_fn(params):
            rng_key_step, rng_key_noise = jax.random.split(rng_key)
            noisy_inputs = inputs + jax.random.normal(rng_key_noise, inputs.shape, dtype=self.dtype) * 0.01
            outputs, _ = self.apply({'params': params}, noisy_inputs, rng_key_step)
            return jnp.mean((outputs - targets) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(self.variables['params'])
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.variables['params'] = optax.apply_updates(self.variables['params'], updates)
        return self.variables['params'], loss

    @staticmethod
    def handle_error(e: Exception) -> None:
        logging.error(f"Error in NeuromorphicComputing: {str(e)}")
        if isinstance(e, ValueError):
            logging.error("Value error occurred. Check input data and model parameters.")
        elif isinstance(e, TypeError):
            logging.error("Type error occurred. Ensure all inputs have correct types and shapes.")
        else:
            logging.error("Unexpected error occurred. Please review the stack trace for more information.")
        logging.debug(f"Error details: {e}", exc_info=True)
        raise

def create_neuromorphic_model(num_neurons: List[int]) -> NeuromorphicComputing:
    return NeuromorphicComputing(num_neurons=num_neurons)

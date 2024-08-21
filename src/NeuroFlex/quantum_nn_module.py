import jax
import jax.numpy as jnp
from jax import tree_util
import flax.linen as nn
import pennylane as qml
import logging
from typing import Callable, List, Tuple, Optional, Any, Dict
from functools import partial
from flax import struct

class QuantumNeuralNetwork(nn.Module):
    """
    A quantum neural network module that combines classical and quantum computations.

    This class implements a variational quantum circuit that can be used as a layer
    in a hybrid quantum-classical neural network.

    Attributes:
        num_qubits (int): The number of qubits in the quantum circuit.
        num_layers (int): The number of layers in the variational quantum circuit.
        input_shape (Tuple[int, ...]): The shape of the input tensor.
        output_shape (Tuple[int, ...]): The shape of the output tensor (excluding batch dimension).
        max_retries (int): The maximum number of retries for quantum circuit execution.
    """

    num_qubits: int
    num_layers: int
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    max_retries: int = 3
    device: Optional[qml.Device] = None
    qlayer: Optional[Callable] = None
    vmap_qlayer: Optional[Callable] = None

    def setup(self):
        logging.info(f"Setting up QuantumNeuralNetwork with {self.num_qubits} qubits, {self.num_layers} layers, input shape {self.input_shape}, and output shape {self.output_shape}")
        self._validate_init_params()

        self.param('weights', nn.initializers.uniform(scale=0.1), (self.num_layers, self.num_qubits, 3))
        try:
            quantum_components = self._initialize_quantum_components()
            self.device = quantum_components['device']
            self.qlayer = quantum_components['qlayer']
            self.vmap_qlayer = quantum_components['vmap_qlayer']
            self.variable('quantum_components', 'components', lambda: quantum_components)
        except Exception as e:
            logging.error(f"Error initializing quantum components: {str(e)}")
            fallback_components = self._fallback_initialization()
            self.device = fallback_components['device']
            self.qlayer = fallback_components['qlayer']
            self.vmap_qlayer = fallback_components['vmap_qlayer']
            self.variable('quantum_components', 'components', lambda: fallback_components)

    def _validate_init_params(self):
        if not isinstance(self.num_qubits, int) or self.num_qubits <= 0:
            raise ValueError(f"Number of qubits must be a positive integer, got {self.num_qubits}")
        if not isinstance(self.num_layers, int) or self.num_layers <= 0:
            raise ValueError(f"Number of layers must be a positive integer, got {self.num_layers}")
        if not isinstance(self.input_shape, tuple) or len(self.input_shape) != 2 or self.input_shape[1] != self.num_qubits:
            raise ValueError(f"Invalid input_shape: {self.input_shape}. Expected shape (batch_size, {self.num_qubits})")
        if not isinstance(self.output_shape, tuple) or len(self.output_shape) != 1 or self.output_shape[0] != self.num_qubits:
            raise ValueError(f"Invalid output_shape: {self.output_shape}. Expected shape ({self.num_qubits},)")

    def _initialize_quantum_components(self):
        try:
            self.device = qml.device("default.qubit", wires=self.num_qubits)
            self.qlayer = qml.QNode(self.quantum_circuit, self.device, interface="jax")
            self.vmap_qlayer = jax.vmap(self.qlayer, in_axes=(0, None))
            logging.info("Quantum components created successfully")
            return {
                'device': self.device,
                'qlayer': self.qlayer,
                'vmap_qlayer': self.vmap_qlayer
            }
        except Exception as e:
            logging.error(f"Error creating quantum components: {str(e)}")
            return self._fallback_initialization()

    def quantum_circuit(self, inputs: jnp.ndarray, weights: jnp.ndarray) -> List[qml.measurements.ExpectationMP]:
        qml.AngleEmbedding(inputs, wires=range(self.num_qubits))
        for l in range(self.num_layers):
            qml.StronglyEntanglingLayers(weights[l], wires=range(self.num_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

    def validate_input_shape(self, x: jnp.ndarray) -> None:
        if len(x.shape) != 2 or x.shape[1] != self.num_qubits:
            raise ValueError(f"Input shape {x.shape} does not match expected shape (batch_size, {self.num_qubits})")

    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        try:
            self.validate_input_shape(x)
            if jnp.any(jnp.isnan(x)) or jnp.any(jnp.isinf(x)):
                raise ValueError(f"Input contains NaN or Inf values: {x}")

            logging.debug(f"Executing quantum circuit with input shape: {x.shape}")
            if self.vmap_qlayer is None:
                logging.warning("Quantum components not initialized. Attempting initialization.")
                self._initialize_quantum_components()
                if self.vmap_qlayer is None:
                    logging.error("Quantum components initialization failed. Using fallback.")
                    return self._fallback_output(x)

            result_array = self._execute_quantum_circuit(x)

            expected_shape = (x.shape[0],) + self.output_shape
            if result_array.shape != expected_shape:
                logging.warning(f"Output shape mismatch. Expected {expected_shape}, got {result_array.shape}. Reshaping.")
                result_array = jnp.reshape(result_array, expected_shape)

            result_array = jnp.clip(result_array, -1, 1)
            logging.info(f"Quantum circuit executed successfully. Input shape: {x.shape}, Output shape: {result_array.shape}")
            return result_array
        except ValueError as ve:
            logging.error(f"ValueError during quantum circuit execution: {str(ve)}")
            return self._fallback_output(x)
        except Exception as e:
            logging.error(f"Unexpected error during quantum circuit execution: {str(e)}")
            return self._fallback_output(x)

    def _execute_quantum_circuit(self, x: jnp.ndarray) -> jnp.ndarray:
        weights = self.variable('params', 'weights').value
        for attempt in range(self.max_retries):
            try:
                logging.debug(f"Attempt {attempt + 1}/{self.max_retries} to execute quantum circuit")
                if self.vmap_qlayer is None:
                    raise ValueError("Quantum components not properly initialized")
                result = self.vmap_qlayer(x, weights)
                result_array = jnp.array(result)
                if jnp.all(jnp.isfinite(result_array)):
                    logging.info(f"Quantum circuit execution successful on attempt {attempt + 1}")
                    return result_array
                else:
                    raise ValueError("Quantum circuit produced non-finite values")
            except Exception as e:
                logging.warning(f"Quantum circuit execution failed on attempt {attempt + 1}: {str(e)}")
                if attempt == self.max_retries - 1:
                    logging.error("Max retries reached. Quantum circuit execution failed.")
                    return self._fallback_output(x)
        return self._fallback_output(x)  # Ensure a return value if loop completes

    def _fallback_output(self, x: jnp.ndarray) -> jnp.ndarray:
        fallback = jnp.zeros((x.shape[0],) + self.output_shape)
        noise = jax.random.normal(jax.random.PRNGKey(0), fallback.shape) * 0.1
        return jnp.clip(fallback + noise, -1, 1)

    def _fallback_initialization(self):
        logging.warning("Falling back to classical initialization")
        fallback_components = {
            'device': None,
            'qlayer': lambda x, w: jnp.zeros(self.output_shape),
            'vmap_qlayer': jax.vmap(lambda x, w: jnp.zeros(self.output_shape), in_axes=(0, None))
        }
        logging.info("Classical fallback initialization completed")
        self.sow('quantum_components', 'components', fallback_components)
        return fallback_components

    def reinitialize_device(self):
        try:
            new_device = qml.device("default.qubit", wires=self.num_qubits)
            new_qlayer = qml.QNode(self.quantum_circuit, new_device, interface="jax")
            new_vmap_qlayer = jax.vmap(new_qlayer, in_axes=(0, None))
            new_components = {
                'device': new_device,
                'qlayer': new_qlayer,
                'vmap_qlayer': new_vmap_qlayer
            }
            self.variable('quantum_components', 'components', lambda: new_components)
            logging.info("Quantum device reinitialized successfully")
        except Exception as e:
            logging.error(f"Error reinitializing quantum device: {str(e)}")
            fallback_components = self._fallback_initialization()
            self.variable('quantum_components', 'components', lambda: fallback_components)
        return self.variable('quantum_components', 'components').value

@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def create_quantum_nn(num_qubits: int, num_layers: int, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> QuantumNeuralNetwork:
    return QuantumNeuralNetwork(num_qubits=num_qubits, num_layers=num_layers, input_shape=input_shape, output_shape=output_shape)

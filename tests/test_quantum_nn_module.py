import pytest
import jax
from jax import numpy as jnp
from jax import random
import pennylane as qml
import logging
import flax.linen as nn
from NeuroFlex.quantum_nn_module import QuantumNeuralNetwork

@pytest.fixture(params=[
    (3, 2, (1, 3), (3,)),
    (5, 3, (1, 5), (5,)),
    (2, 1, (1, 2), (2,)),
])
def qnn(request):
    num_qubits, num_layers, input_shape, output_shape = request.param
    return QuantumNeuralNetwork(num_qubits=num_qubits, num_layers=num_layers, input_shape=input_shape, output_shape=output_shape)

# def test_initialization(qnn):
#     assert qnn.num_qubits == qnn.input_shape[1]
#     assert qnn.num_layers == 2
#     assert qnn.output_shape == (qnn.num_qubits,)
#     assert qnn.max_retries == 3
#     assert isinstance(qnn.device, qml.Device)
#     assert callable(qnn.quantum_circuit)
#     assert callable(qnn.qlayer)
#     assert callable(qnn.vmap_qlayer)
#     assert isinstance(qnn.weights, jnp.ndarray)
#     assert qnn.weights.shape == (qnn.num_layers, qnn.num_qubits, 3)
#     assert isinstance(qnn, nn.Module)

#     # Test initialization with invalid parameters
#     with pytest.raises(ValueError):
#         QuantumNeuralNetwork(num_qubits=0, num_layers=2, input_shape=(1, 0), output_shape=(3,))
#     with pytest.raises(ValueError):
#         QuantumNeuralNetwork(num_qubits=3, num_layers=0, input_shape=(1, 3), output_shape=(3,))
#     with pytest.raises(ValueError):
#         QuantumNeuralNetwork(num_qubits=3, num_layers=2, input_shape=(1, 3), output_shape=(4,))

#     # Test fallback initialization
#     qnn._fallback_initialization()
#     assert qnn.device is None
#     assert qnn.qlayer is None
#     assert qnn.vmap_qlayer is None
#     assert qnn.weights is None

# def test_quantum_circuit(qnn):
#     inputs = jnp.array([0.1, 0.2, 0.3])
#     result = qnn.quantum_circuit(inputs, qnn.weights)
#     assert len(result) == qnn.num_qubits
#     assert all(isinstance(r, qml.measurements.ExpectationMP) for r in result)
#     assert all(-1 <= r.evaluate() <= 1 for r in result), "Expectation values should be in [-1, 1]"
#     assert jnp.allclose(jnp.array([r.evaluate() for r in result]), qnn.qlayer(inputs, qnn.weights), atol=1e-5)

# def test_forward(qnn):
#     inputs = jnp.ones(qnn.input_shape)
#     variables = qnn.init(jax.random.PRNGKey(0), inputs)
#     result = qnn.apply(variables, inputs)
#     assert isinstance(result, jnp.ndarray)
#     assert result.shape == qnn.input_shape[0:1] + qnn.output_shape
#     assert jnp.all(jnp.isfinite(result)), "Output should contain only finite values"
#     assert jnp.all((result >= -1) & (result <= 1)), "Output values should be in range [-1, 1]"

#     # Test with batch input
#     batch_size = 2
#     batch_inputs = jnp.ones((batch_size,) + qnn.input_shape[1:])
#     batch_result = qnn.apply(variables, batch_inputs)
#     assert batch_result.shape == (batch_size,) + qnn.output_shape

# def test_end_to_end(qnn):
#     inputs = jnp.ones(qnn.input_shape)
#     variables = qnn.init(jax.random.PRNGKey(0), inputs)
#     result = qnn.apply(variables, inputs)
#     assert isinstance(result, jnp.ndarray)
#     assert result.shape == qnn.input_shape[0:1] + qnn.output_shape
#     assert jnp.all((result >= -1) & (result <= 1))

# def test_input_shape_validation(qnn):
#     with pytest.raises(ValueError, match="Input shape .* does not match expected shape"):
#         qnn.validate_input_shape(jnp.ones((1, qnn.num_qubits - 1)))
#     with pytest.raises(ValueError, match="Input shape .* does not match expected shape"):
#         qnn.validate_input_shape(jnp.ones((1, qnn.num_qubits + 1)))

# def test_quantum_device_initialization_error():
#     with pytest.raises(ValueError, match="Number of qubits must be positive"):
#         QuantumNeuralNetwork(num_qubits=-1, num_layers=2, input_shape=(1, 3), output_shape=(3,))

# def test_batch_processing(qnn):
#     batch_size = 2
#     inputs = jnp.ones((batch_size,) + qnn.input_shape[1:])
#     variables = qnn.init(jax.random.PRNGKey(0), inputs)
#     result = qnn.apply(variables, inputs)
#     assert result.shape == (batch_size,) + qnn.output_shape

# def test_quantum_circuit_execution_error(qnn, monkeypatch, caplog):
#     retry_count = 0
#     def mock_quantum_circuit(*args):
#         nonlocal retry_count
#         retry_count += 1
#         raise RuntimeError(f"Quantum circuit execution failed (attempt {retry_count})")

#     monkeypatch.setattr(qnn, "quantum_circuit", mock_quantum_circuit)

#     with caplog.at_level(logging.WARNING):
#         inputs = jnp.ones(qnn.input_shape)
#         variables = qnn.init(jax.random.PRNGKey(0), inputs)
#         result = qnn.apply(variables, inputs)

#         assert retry_count == qnn.max_retries, f"Expected {qnn.max_retries} retries, got {retry_count}"
#         assert any("Quantum circuit execution failed" in record.message for record in caplog.records)
#         assert any("Max retries reached" in record.message for record in caplog.records)

#         expected_shape = qnn.input_shape[0:1] + qnn.output_shape
#         assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"
#         assert jnp.all(jnp.isfinite(result)), "Result should contain only finite values"
#         assert jnp.all((result >= -1) & (result <= 1)), "Result should be in range [-1, 1]"

# def test_device_accessibility(qnn):
#     assert hasattr(qnn, 'device')
#     assert qnn.device is not None
#     assert isinstance(qnn.device, qml.Device)

# def test_device_reinitialization(qnn):
#     original_device = qnn.device
#     qnn.reinitialize_device()
#     assert qnn.device is not original_device
#     assert isinstance(qnn.device, qml.Device)

# def test_gradients(qnn):
#     inputs = jnp.ones(qnn.input_shape)
#     variables = qnn.init(jax.random.PRNGKey(0), inputs)
#     def loss_fn(params):
#         return jnp.sum(qnn.apply({'params': params}, inputs))
#     grad_fn = jax.grad(loss_fn)
#     grads = grad_fn(variables['params'])
#     assert jax.tree_util.tree_all(jax.tree_map(lambda x: jnp.any(x != 0), grads))

# def test_large_batch_processing(qnn):
#     batch_size = 100
#     inputs = jnp.ones((batch_size,) + qnn.input_shape[1:])
#     variables = qnn.init(jax.random.PRNGKey(0), inputs)
#     result = qnn.apply(variables, inputs)
#     assert result.shape == (batch_size,) + qnn.output_shape

# def test_input_range(qnn):
#     inputs = jnp.array([
#         [0.0] * qnn.num_qubits,  # Normal range
#         [-1.0] * qnn.num_qubits,  # Out of normal range
#         [jnp.pi] * qnn.num_qubits,  # Angular values
#         [1e-5] * qnn.num_qubits,  # Very small values
#     ])
#     variables = qnn.init(jax.random.PRNGKey(0), inputs)
#     result = qnn.apply(variables, inputs)

#     assert result.shape == (4,) + qnn.output_shape
#     assert jnp.all((result >= -1) & (result <= 1))

#     # Check if outputs are different for different inputs
#     assert not jnp.allclose(result[0], result[1])
#     assert not jnp.allclose(result[0], result[2])

#     # Check if the model handles extreme values
#     assert jnp.all(jnp.isfinite(result))

#     # Test with zero input
#     zero_input = jnp.zeros(qnn.input_shape)
#     zero_result = qnn.apply(variables, zero_input)
#     assert zero_result.shape == qnn.input_shape[0:1] + qnn.output_shape
#     assert jnp.all((zero_result >= -1) & (zero_result <= 1))

#     # Test with NaN input
#     nan_input = jnp.full(qnn.input_shape, jnp.nan)
#     with pytest.raises(ValueError, match="Input contains NaN values"):
#         qnn.apply(variables, nan_input)

#     # Test with infinity input
#     inf_input = jnp.full(qnn.input_shape, jnp.inf)
#     with pytest.raises(ValueError, match="Input contains infinite values"):
#         qnn.apply(variables, inf_input)

#     # Test with mixed valid and invalid inputs
#     mixed_input = jnp.array([[0.1] + [jnp.nan] * (qnn.num_qubits - 2) + [jnp.inf]])
#     with pytest.raises(ValueError, match="Input contains NaN or infinite values"):
#         qnn.apply(variables, mixed_input)

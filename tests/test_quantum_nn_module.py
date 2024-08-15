import pytest
import jax
import jax.numpy as jnp
import pennylane as qml
from quantum_nn_module import QuantumNeuralNetwork

@pytest.fixture
def qnn():
    return QuantumNeuralNetwork(n_qubits=3, n_layers=2)

def test_initialization(qnn):
    assert qnn.n_qubits == 3
    assert qnn.n_layers == 2
    assert isinstance(qnn.dev, qml.devices.default_qubit.DefaultQubit)
    assert callable(qnn.quantum_circuit)

def test_circuit(qnn):
    inputs = jnp.array([0.1, 0.2, 0.3])
    weights = jnp.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
                         [[1.0, 1.1, 1.2], [1.3, 1.4, 1.5], [1.6, 1.7, 1.8]]])
    result = qnn.circuit(inputs, weights)
    assert len(result) == qnn.n_qubits
    assert all(isinstance(r, qml.measurements.ExpectationMP) for r in result)

def test_encode_input(qnn):
    inputs = jnp.array([0.1, 0.2, 0.3])
    with qml.tape.QuantumTape() as tape:
        qnn.encode_input(inputs)
    assert len(tape.operations) == qnn.n_qubits
    assert all(isinstance(op, qml.RY) for op in tape.operations)

def test_variational_layer(qnn):
    weights = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    with qml.tape.QuantumTape() as tape:
        qnn.variational_layer(weights)
    assert len(tape.operations) == qnn.n_qubits + (qnn.n_qubits - 1)
    assert all(isinstance(op, (qml.Rot, qml.CNOT)) for op in tape.operations)

def test_forward(qnn):
    inputs = jnp.array([0.1, 0.2, 0.3])
    weights = qnn.initialize_weights()
    result = qnn.forward(inputs, weights)
    assert isinstance(result, jnp.ndarray)
    assert result.shape == (qnn.n_qubits,)

def test_initialize_weights(qnn):
    weights = qnn.initialize_weights()
    assert isinstance(weights, jnp.ndarray)
    assert weights.shape == (qnn.n_layers, qnn.n_qubits, 3)
    assert jnp.all((weights >= 0) & (weights <= 2*jnp.pi))

def test_end_to_end(qnn):
    inputs = jnp.array([0.1, 0.2, 0.3])
    weights = qnn.initialize_weights()
    result = qnn.forward(inputs, weights)
    assert isinstance(result, jnp.ndarray)
    assert result.shape == (qnn.n_qubits,)
    assert jnp.all((result >= -1) & (result <= 1))

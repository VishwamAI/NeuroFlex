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
    assert callable(qnn.encoding_method)

def test_circuit(qnn):
    inputs = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    inputs = inputs / jnp.linalg.norm(inputs)  # Normalize the input
    weights = jnp.array([[[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0], [1.1, 1.2, 1.3, 1.4, 1.5]],
                         [[1.6, 1.7, 1.8, 1.9, 2.0], [2.1, 2.2, 2.3, 2.4, 2.5], [2.6, 2.7, 2.8, 2.9, 3.0]]])
    result = qnn.circuit(inputs, weights)
    assert len(result) == 2 * qnn.n_qubits  # PauliZ and PauliX measurements for each qubit
    assert all(isinstance(r, qml.measurements.ExpectationMP) for r in result)

def test_amplitude_encoding(qnn):
    inputs = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    inputs = inputs / jnp.linalg.norm(inputs)  # Normalize the input
    with qml.tape.QuantumTape() as tape:
        qnn.amplitude_encoding(inputs)
    assert len(tape.operations) == 1
    assert isinstance(tape.operations[0], qml.QubitStateVector)

def test_angle_encoding(qnn):
    inputs = jnp.array([0.1, 0.2, 0.3])
    with qml.tape.QuantumTape() as tape:
        qnn.angle_encoding(inputs)
    assert len(tape.operations) == qnn.n_qubits
    assert all(isinstance(op, qml.RY) for op in tape.operations)

def test_variational_layer(qnn):
    weights = jnp.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0], [1.1, 1.2, 1.3, 1.4, 1.5]])
    with qml.tape.QuantumTape() as tape:
        qnn.variational_layer(weights)
    assert len(tape.operations) == qnn.n_qubits * 2 + (qnn.n_qubits - 1) + 1  # Rot, RZ, CNOT, and CRZ
    assert all(isinstance(op, (qml.Rot, qml.RZ, qml.CNOT, qml.CRZ)) for op in tape.operations)

def test_entangling_layer(qnn):
    with qml.tape.QuantumTape() as tape:
        qnn.entangling_layer()
    assert len(tape.operations) == 2 * qnn.n_qubits  # Hadamard and CZ for each qubit
    assert all(isinstance(op, (qml.Hadamard, qml.CZ)) for op in tape.operations)

def test_forward(qnn):
    inputs = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    inputs = inputs / jnp.linalg.norm(inputs)  # Normalize the input
    weights = qnn.initialize_weights()
    result = qnn.forward(inputs, weights)
    assert isinstance(result, jnp.ndarray)
    assert result.shape == (2 * qnn.n_qubits,)  # PauliZ and PauliX measurements for each qubit

def test_initialize_weights(qnn):
    weights = qnn.initialize_weights()
    assert isinstance(weights, jnp.ndarray)
    assert weights.shape == (qnn.n_layers, qnn.n_qubits, 5)  # 5 parameters per qubit per layer
    assert jnp.all((weights >= 0) & (weights <= 2*jnp.pi))

def test_quantum_classical_hybrid(qnn):
    inputs = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    inputs = inputs / jnp.linalg.norm(inputs)  # Normalize the input
    weights = qnn.initialize_weights()
    def classical_layer(x):
        return jnp.sum(x)
    result = qnn.quantum_classical_hybrid(inputs, weights, classical_layer)
    assert isinstance(result, jnp.ndarray)
    assert result.shape == ()  # Scalar output from the classical layer

def test_end_to_end(qnn):
    inputs = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    inputs = inputs / jnp.linalg.norm(inputs)  # Normalize the input
    weights = qnn.initialize_weights()
    result = qnn.forward(inputs, weights)
    assert isinstance(result, jnp.ndarray)
    assert result.shape == (2 * qnn.n_qubits,)  # PauliZ and PauliX measurements for each qubit
    assert jnp.all((result >= -1) & (result <= 1))

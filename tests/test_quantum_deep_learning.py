import pytest
import jax
import jax.numpy as jnp
from NeuroFlex.quantum_deep_learning.quantum_boltzmann_machine import QuantumBoltzmannMachine
from NeuroFlex.quantum_deep_learning.quantum_cnn import QuantumCNN
from NeuroFlex.quantum_deep_learning.quantum_rnn import QuantumRNN

@pytest.fixture
def qbm():
    return QuantumBoltzmannMachine(num_visible=4, num_hidden=2, num_qubits=6)

@pytest.fixture
def qcnn():
    return QuantumCNN(num_qubits=4, num_layers=2)

@pytest.fixture
def qrnn():
    return QuantumRNN(num_qubits=4, num_layers=2)

def test_qbm_initialization(qbm):
    assert isinstance(qbm, QuantumBoltzmannMachine)
    assert qbm.num_visible == 4
    assert qbm.num_hidden == 2
    assert qbm.num_qubits == 6

def test_qcnn_initialization(qcnn):
    assert isinstance(qcnn, QuantumCNN)
    assert qcnn.num_qubits == 4
    assert qcnn.num_layers == 2

def test_qrnn_initialization(qrnn):
    assert isinstance(qrnn, QuantumRNN)
    assert qrnn.num_qubits == 4
    assert qrnn.num_layers == 2

def test_qbm_execution(qbm):
    visible_data = jnp.array([1, 0, 1, 1])
    hidden_state = qbm.sample_hidden(visible_data)
    assert hidden_state.shape == (2,)
    assert jnp.all((hidden_state == 0) | (hidden_state == 1))

    # Test energy calculation
    energy = qbm.energy(visible_data, hidden_state)
    assert isinstance(energy, float)
    assert energy < 0  # Energy should typically be negative

def test_qbm_training():
    qbm = QuantumBoltzmannMachine(num_visible=4, num_hidden=2, num_qubits=6)
    data = jnp.array([[1, 0, 1, 1], [0, 1, 1, 0], [1, 1, 0, 0]])
    initial_energy = qbm.energy(data[0], qbm.sample_hidden(data[0]))
    qbm.train(data, num_epochs=10, learning_rate=0.1)
    final_energy = qbm.energy(data[0], qbm.sample_hidden(data[0]))
    assert final_energy <= initial_energy

    # Test parameter updates
    assert jnp.any(jnp.not_equal(qbm.weights, jnp.zeros_like(qbm.weights)))

def test_qbm_error_handling():
    with pytest.raises(ValueError):
        QuantumBoltzmannMachine(num_visible=-1, num_hidden=2, num_qubits=6)

def test_qcnn_execution(qcnn):
    key = jax.random.PRNGKey(0)
    params = qcnn.init(key, jnp.zeros((1, 4)))
    x = jax.random.normal(key, (10, 4))
    output = qcnn.apply(params, x)
    assert output.shape == (10, 16)  # Assuming 4 qubits, output should be 2^4

    # Test different input sizes
    x_small = jax.random.normal(key, (5, 4))
    output_small = qcnn.apply(params, x_small)
    assert output_small.shape == (5, 16)

def test_qcnn_gradient():
    qcnn = QuantumCNN(num_qubits=2, num_layers=1)
    key = jax.random.PRNGKey(0)
    params = qcnn.init(key, jnp.zeros((1, 2)))
    x = jax.random.normal(key, (1, 2))

    def loss(params, x):
        return jnp.sum(qcnn.apply(params, x))

    grad_fn = jax.grad(loss)
    grads = grad_fn(params, x)
    assert jax.tree_util.tree_all(jax.tree_map(lambda x: jnp.all(jnp.isfinite(x)), grads))

    # Test gradient descent step
    learning_rate = 0.01
    new_params = jax.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    assert jax.tree_util.tree_all(jax.tree_map(lambda x, y: jnp.any(x != y), params, new_params))

def test_qrnn_execution(qrnn):
    key = jax.random.PRNGKey(0)
    params = qrnn.init(key, jnp.zeros((1, 1, 4)))
    x = jax.random.normal(key, (10, 1, 4))  # 3D array: (batch_size, time_steps, features)
    output = qrnn.apply(params, x)
    assert output.shape == (10, 1, 4)  # Output shape should match input shape

    # Test sequence processing
    x_seq = jax.random.normal(key, (5, 3, 4))  # 5 sequences, 3 time steps, 4 features
    output_seq = qrnn.apply(params, x_seq)
    assert output_seq.shape == (5, 3, 4)  # Output shape should match input shape

    # Test individual qubit_layer call
    single_input = jax.random.normal(key, (1, 4))
    single_output = qrnn.qubit_layer(params[0], single_input)
    assert single_output.shape == (1, 4)  # Single qubit output should match input shape

def test_qrnn_gradient():
    qrnn = QuantumRNN(num_qubits=2, num_layers=1)
    key = jax.random.PRNGKey(0)
    params = qrnn.init(key, jnp.zeros((1, 2)))
    x = jax.random.normal(key, (1, 2))

    def loss(params, x):
        return jnp.sum(qrnn.apply(params, x))

    grad_fn = jax.grad(loss)
    grads = grad_fn(params, x)
    assert jax.tree_util.tree_all(jax.tree_map(lambda x: jnp.all(jnp.isfinite(x)), grads))

    # Test gradient clipping
    max_norm = 1.0
    clipped_grads = jax.tree_map(lambda g: jnp.clip(g, -max_norm, max_norm), grads)
    assert jax.tree_util.tree_all(jax.tree_map(lambda x: jnp.all(jnp.abs(x) <= max_norm), clipped_grads))

if __name__ == "__main__":
    pytest.main([__file__])

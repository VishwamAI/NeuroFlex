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
    assert energy <= 0  # Energy should be non-positive

    # Test entangle_qubits function
    params1 = jnp.array([0.1, 0.2, 0.3])
    params2 = jnp.array([0.4, 0.5, 0.6])
    entangled_state = qbm.entangle_qubits(params1, params2)
    assert entangled_state.shape == (4,)  # Probabilities for 2 qubits: |00>, |01>, |10>, |11>
    assert jnp.allclose(jnp.sum(entangled_state), 1.0)  # Probabilities should sum to 1

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
    params = qcnn.init(key, jnp.zeros((1, qcnn.num_qubits)))
    x = jax.random.normal(key, (10, qcnn.num_qubits))
    output = qcnn.apply(params, x)
    assert output.shape == (8, qcnn.num_qubits)  # Output shape should be (input_size - 2, num_qubits)

    # Test qubit_layer function
    from NeuroFlex.quantum_deep_learning.quantum_cnn import qubit_layer
    input_val = 0.5  # Pass input_val as a scalar
    qubit_params = jnp.array([0.1, 0.2])  # Ensure 2 parameters for RY, RZ
    qubit_output = qubit_layer(qubit_params, input_val)
    assert isinstance(qubit_output, jnp.ndarray)  # Output should be a JAX array

    # Test different input sizes
    x_small = jax.random.normal(key, (5, qcnn.num_qubits))
    output_small = qcnn.apply(params, x_small)
    assert output_small.shape == (3, qcnn.num_qubits)

    # Verify parameter shapes
    assert params.shape == (qcnn.num_layers, qcnn.num_qubits, 2)  # (num_layers, num_qubits, 2)

def test_qcnn_gradient():
    qcnn = QuantumCNN(num_qubits=2, num_layers=1)
    key = jax.random.PRNGKey(0)
    params = qcnn.init(key, jnp.zeros((1, 2)))
    x = jax.random.normal(key, (1, 2))
    print(f"Input x shape: {x.shape}, values: {x}")
    print(f"Initial params shape: {params.shape}, values: {params}")

    def loss(params, x):
        print(f"Loss function input x shape: {x.shape}, values: {x}")
        print(f"Loss function params shape: {params.shape}, values: {params}")
        output = qcnn.apply(params, x)
        print(f"QCNN output shape: {output.shape}, values: {output}")
        loss_value = jnp.sum(output)
        print(f"Loss value: {loss_value}")
        return loss_value

    initial_loss = loss(params, x)
    print(f"Initial loss: {initial_loss}")

    grad_fn = jax.grad(loss)
    grads = grad_fn(params, x)
    print(f"Gradients shape: {grads.shape}, values: {grads}")
    print(f"Gradients sum: {jax.tree_map(lambda x: jnp.sum(jnp.abs(x)), grads)}")
    assert jax.tree_util.tree_all(jax.tree_map(lambda x: jnp.all(jnp.isfinite(x)), grads))

    # Test gradient descent step
    learning_rate = 0.01
    new_params = jax.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    param_diff = jax.tree_map(lambda x, y: jnp.abs(x - y), params, new_params)
    print(f"New params shape: {new_params.shape}, values: {new_params}")
    print(f"Parameter difference shape: {param_diff.shape}, values: {param_diff}")
    print(f"Parameter difference sum: {jax.tree_map(lambda x: jnp.sum(jnp.abs(x)), param_diff)}")
    assert jax.tree_util.tree_reduce(lambda x, y: x or y, jax.tree_map(lambda x: jnp.any(x > 1e-6), param_diff))

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

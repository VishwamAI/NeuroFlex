import pytest
import jax
import jax.numpy as jnp
from NeuroFlex.quantum_neural_networks.quantum_module import VQEModel, QAOAModel

@pytest.fixture
def vqe_model():
    return VQEModel(num_qubits=4, num_layers=2)

@pytest.fixture
def qaoa_model():
    return QAOAModel(num_qubits=4, num_layers=2)

def test_vqe_model_initialization(vqe_model):
    assert isinstance(vqe_model, VQEModel)
    assert vqe_model.num_qubits == 4
    assert vqe_model.num_layers == 2

def test_qaoa_model_initialization(qaoa_model):
    assert isinstance(qaoa_model, QAOAModel)
    assert qaoa_model.num_qubits == 4
    assert qaoa_model.num_layers == 2

def test_vqe_model_execution(vqe_model):
    key = jax.random.PRNGKey(0)
    params = vqe_model.init(key, jnp.zeros((1, 4)))
    x = jax.random.normal(key, (10, 4))
    output = vqe_model.apply(params, x)
    assert output.shape == (10, 2)  # Updated to match the new output shape (2 probabilities for 1 qubit)

def test_qaoa_model_execution(qaoa_model):
    key = jax.random.PRNGKey(0)
    params = qaoa_model.init(key, jnp.zeros((1, 4)))
    x = jax.random.normal(key, (10, 4))
    output = qaoa_model.apply(params, x)
    assert output.shape == (10, 16)  # Updated to match the new output shape (2^4 probabilities for 4 qubits)

def test_vqe_model_gradient():
    model = VQEModel(num_qubits=2, num_layers=1)
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.zeros((1, 2)))
    x = jax.random.normal(key, (1, 2))

    def loss(params, x):
        return jnp.sum(model.apply(params, x))

    grad_fn = jax.grad(loss)
    grads = grad_fn(params, x)
    assert jax.tree_util.tree_all(jax.tree_map(lambda x: jnp.all(jnp.isfinite(x)), grads))

def test_qaoa_model_gradient():
    model = QAOAModel(num_qubits=2, num_layers=1)
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.zeros((1, 2)))
    x = jax.random.normal(key, (1, 2))

    def loss(params, x):
        return jnp.sum(model.apply(params, x))

    grad_fn = jax.grad(loss)
    grads = grad_fn(params, x)
    assert jax.tree_util.tree_all(jax.tree_map(lambda x: jnp.all(jnp.isfinite(x)), grads))

if __name__ == "__main__":
    pytest.main([__file__])

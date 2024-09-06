import pytest
import jax
import jax.numpy as jnp
import torch
from jax import random
from NeuroFlex.core_neural_networks import CNNBlock, LRNN, LSTMModule, JaxModel, CDSTDP, NeuroFlex, MachineLearning
from NeuroFlex.utils.utils import get_activation_function

@pytest.fixture(scope="module")
def rng_key():
    return random.PRNGKey(0)  # Use a fixed seed for reproducibility

@pytest.fixture
def cnn_block(rng_key):
    model = CNNBlock(features=(32, 64), conv_dim=2, dtype=jnp.float32, activation=jax.nn.relu)
    variables = model.init(rng_key, jnp.ones((1, 28, 28, 1)))
    return model.bind(variables)

@pytest.fixture
def lrnn(rng_key):
    model = LRNN(features=(64, 32), activation=jax.nn.tanh)
    variables = model.init(rng_key, jnp.ones((1, 10, 5)))
    return model.bind(variables)

@pytest.fixture
def lstm_module(rng_key):
    model = LSTMModule(hidden_size=64, num_layers=2, dropout=0.1)
    variables = model.init(rng_key, jnp.ones((1, 10, 5)))
    return model.bind(variables)

@pytest.fixture
def jax_model(rng_key):
    model = JaxModel(hidden_layers=(64, 32), output_dim=10)
    variables = model.init(rng_key, jnp.ones((1, 100)))
    return model.bind(variables)

@pytest.fixture
def cdstdp():
    return CDSTDP()

@pytest.fixture
def neuroflex():
    return NeuroFlex(features=[64, 32, 10])

@pytest.fixture
def machine_learning():
    return MachineLearning(features=[100, 64, 32, 10])

def test_cnn_block_creation(cnn_block):
    assert isinstance(cnn_block, CNNBlock)
    assert cnn_block.features == (32, 64)
    assert cnn_block.conv_dim == 2
    assert cnn_block.dtype == jnp.float32
    assert cnn_block.activation == jax.nn.relu

def test_cnn_block_forward_pass(cnn_block):
    input_data = jnp.ones((1, 28, 28, 1))
    output = cnn_block(input_data)
    assert output.shape == (1, 7, 7, 64)

def test_lrnn_creation(lrnn):
    assert isinstance(lrnn, LRNN)
    assert lrnn.features == (64, 32)
    assert lrnn.activation == jax.nn.tanh

def test_lrnn_forward_pass(lrnn):
    input_data = jnp.ones((1, 10, 5))
    output, final_state = lrnn(input_data)
    assert output.shape == (1, 10, 32)
    assert final_state.shape == (1, 32)

def test_lstm_module_creation(lstm_module):
    assert isinstance(lstm_module, LSTMModule)
    assert lstm_module.hidden_size == 64
    assert lstm_module.num_layers == 2
    assert lstm_module.dropout == 0.1

def test_lstm_module_forward_pass(lstm_module, rng_key):
    input_data = jnp.ones((1, 10, 5))
    rng_key, dropout_key = jax.random.split(rng_key)
    outputs, (final_h, final_c) = lstm_module(input_data, rngs={'dropout': dropout_key})
    assert outputs.shape == (1, 10, 64)
    assert final_h.shape == (2, 1, 64)
    assert final_c.shape == (2, 1, 64)

def test_jax_model_creation(jax_model):
    assert isinstance(jax_model, JaxModel)
    assert jax_model.hidden_layers == (64, 32)
    assert jax_model.output_dim == 10

def test_jax_model_forward_pass(jax_model):
    input_data = jnp.ones((1, 100))
    output = jax_model(input_data)
    assert output.shape == (1, 10)

def test_cdstdp_creation(cdstdp):
    assert isinstance(cdstdp, CDSTDP)
    assert hasattr(cdstdp, 'update_weights')

def test_neuroflex_creation(neuroflex):
    assert isinstance(neuroflex, NeuroFlex)
    assert neuroflex.features == [64, 32, 10]

def test_machine_learning_creation(machine_learning):
    assert isinstance(machine_learning, MachineLearning)
    assert machine_learning.features == [100, 64, 32, 10]

def test_machine_learning_forward_pass(machine_learning):
    input_data = torch.ones((1, 100))
    output = machine_learning(input_data)
    assert output.shape == (1, 10)
    assert isinstance(output, torch.Tensor)

if __name__ == "__main__":
    pytest.main([__file__])

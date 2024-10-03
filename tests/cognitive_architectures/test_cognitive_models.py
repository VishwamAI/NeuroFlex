import pytest
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training import train_state
from flax.training.train_state import TrainState
import optax
from flax.core.frozen_dict import FrozenDict
from flax.core import freeze, unfreeze
from NeuroFlex.cognitive_architectures.attention_schema_theory import ASTModel
from NeuroFlex.cognitive_architectures.global_workspace_theory import GWTModel
from NeuroFlex.cognitive_architectures.higher_order_thoughts import HOTModel
from NeuroFlex.cognitive_architectures.integrated_information_theory import IITModel

@pytest.fixture
def ast_model():
    return ASTModel(attention_dim=10, hidden_dim=20)

@pytest.fixture
def gwt_model():
    return GWTModel(num_processes=5, workspace_size=100)

@pytest.fixture
def hot_model():
    return HOTModel(num_layers=3, hidden_dim=10)

@pytest.fixture
def iit_model():
    return IITModel(num_components=5)

def test_ast_model(ast_model):
    key = random.PRNGKey(0)
    x = random.normal(key, (1, 10))
    params = ast_model.init(key, x)

    output = ast_model.apply(params, x)
    assert output.shape == (1, 10)  # Updated to match the input shape
    assert jnp.isfinite(output).all()

def test_gwt_model(gwt_model):
    key = random.PRNGKey(0)
    input_stimulus = random.normal(key, (1, 100))

    variables = gwt_model.init({"params": key}, input_stimulus)
    # Ensure the correct structure is used for the apply function
    assert isinstance(variables['params'], FrozenDict), "variables['params'] should be a FrozenDict"

    state = train_state.TrainState.create(
        apply_fn=gwt_model.apply,
        params=variables['params'],
        tx=optax.adam(learning_rate=1e-3)
    )

    # Ensure the model is bound before accessing variables
    bound_gwt_model = gwt_model.bind(variables)

    # Pass the PRNG key correctly during model application
    broadcasted_workspace, integrated_workspace = bound_gwt_model.apply(variables, input_stimulus, rngs={"params": key})

    assert len(broadcasted_workspace) == gwt_model.num_processes
    assert all(bw.shape == (1, gwt_model.workspace_size) for bw in broadcasted_workspace)
    assert integrated_workspace.shape == (1, gwt_model.workspace_size)
    assert all(jnp.isfinite(bw).all() for bw in broadcasted_workspace)
    assert jnp.isfinite(integrated_workspace).all()

    # Verify weights initialization
    assert 'weights' in variables['params']
    assert variables['params']['weights'].shape == (gwt_model.num_processes,)

    # Investigate the GWT model's weight update issue
    new_weights = jnp.array([0.1, 0.2, 0.3, 0.2, 0.2])
    print(f"Initial weights: {variables['params']['weights']}")
    updated_variables = bound_gwt_model.apply({'params': variables['params']}, new_weights, method=bound_gwt_model.update_weights, rngs={'params': key})
    updated_weights = updated_variables['params']['weights']
    expected_weights = new_weights / jnp.sum(new_weights)
    print(f"Updated weights: {updated_weights}")
    print(f"Expected weights: {expected_weights}")
    print(f"Difference: {jnp.abs(updated_weights - expected_weights)}")
    assert jnp.allclose(updated_weights, expected_weights, atol=1e-5)

def test_ast_model(ast_model):
    key = random.PRNGKey(0)
    x = random.normal(key, (1, 10))  # Input shape (1, 10)
    variables = ast_model.init(key, x)
    output = ast_model.apply(variables, x)
    assert output.shape == (1, 10)  # Ensure output shape matches input shape

def test_hot_model(hot_model):
    key = random.PRNGKey(0)
    x = random.normal(key, (1, hot_model.input_dim))  # Use hot_model.input_dim for input shape
    params = hot_model.init(key, x)

    output = hot_model.apply(params, x)
    assert output.shape == (1, hot_model.output_dim), f"Expected output shape (1, {hot_model.output_dim}), but got {output.shape}"
    assert jnp.isfinite(output).all(), "Output contains non-finite values"

    # Verify the model's dimensions
    assert hot_model.input_dim == hot_model.output_dim, f"Expected input_dim to match output_dim, but got input_dim={hot_model.input_dim} and output_dim={hot_model.output_dim}"
    assert hot_model.hidden_dim == 10, f"Expected hidden_dim 10, but got {hot_model.hidden_dim}"
    print(f"HOT model dimensions: input_dim={hot_model.input_dim}, hidden_dim={hot_model.hidden_dim}, output_dim={hot_model.output_dim}")

def test_iit_model(iit_model):
    key = random.PRNGKey(0)
    state = random.normal(key, (5,))

    # Initialize the model
    params = iit_model.init(key, None)

    # Apply the model to calculate integrated information
    phi = iit_model.apply(params, method=iit_model.calculate_integrated_information)

    assert jnp.isscalar(phi), f"Expected phi to be a scalar, but got {phi}"
    assert jnp.isfinite(phi)
    assert phi >= 0

def test_ast_model_training(ast_model):
    key = random.PRNGKey(0)
    x = random.normal(key, (10, 10))  # Batch size of 10, input dimension of 10
    y = random.normal(key, (10, 10))  # Output dimension should match input dimension

    params = ast_model.init(key, x[0])
    output = ast_model.apply(params, x)

    assert output.shape == (10, 10)
    assert jnp.isfinite(output).all()

    # Check if the model can be trained
    loss = jnp.mean((output - y) ** 2)
    assert isinstance(loss, jnp.ndarray) and loss.ndim == 0, "Loss should be a scalar"
    assert jnp.isfinite(loss)
    assert loss >= 0

def test_gwt_model_update_weights(gwt_model):
    key = random.PRNGKey(0)
    input_stimulus = random.normal(key, (1, gwt_model.workspace_size))
    variables = gwt_model.init(key, input_stimulus)
    bound_gwt_model = gwt_model.bind(variables)

    new_weights = jnp.array([0.1, 0.2, 0.3, 0.2, 0.2])
    assert new_weights.shape == (gwt_model.num_processes,), "New weights shape mismatch"

    updated_variables = bound_gwt_model.apply({'params': variables['params']}, new_weights, method=bound_gwt_model.update_weights, rngs={'params': key})
    updated_weights = updated_variables['params']['weights']
    expected_weights = new_weights / jnp.sum(new_weights)
    assert jnp.allclose(updated_weights, expected_weights, atol=1e-5), f"Expected {expected_weights}, but got {updated_weights}"

def test_hot_model_higher_order_thought(hot_model):
    key = random.PRNGKey(0)
    x = random.normal(key, (1, hot_model.input_dim))  # Changed from hidden_dim to input_dim
    params = hot_model.init(key, x)

    first_order_thought = hot_model.apply(params, x)
    higher_order_thought = hot_model.generate_higher_order_thought(params, first_order_thought)

    assert higher_order_thought.shape == (1, hot_model.output_dim)
    assert jnp.isfinite(higher_order_thought).all()
    assert first_order_thought.shape == (1, hot_model.output_dim)

def test_iit_model_cause_effect_structure(iit_model):
    key = random.PRNGKey(0)
    state = random.normal(key, (5,))

    # Initialize the model
    params = iit_model.init(key, state)
    initialized_iit_model = iit_model.bind(params)

    ces = initialized_iit_model.compute_cause_effect_structure(state)
    assert isinstance(ces, dict)
    assert len(ces) > 0
    for mechanism, purview in ces.items():
        assert isinstance(mechanism, tuple)
        assert isinstance(purview, dict)
        assert 'cause' in purview and 'effect' in purview
        assert isinstance(purview['cause'], tuple)
        assert isinstance(purview['effect'], tuple)

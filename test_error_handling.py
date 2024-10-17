import pytest
import logging
import re
from NeuroFlex.cognitive_architectures.error_handling import enhanced_error_handling
from NeuroFlex.cognitive_architectures.consciousness_simulation import ConsciousnessSimulation
import jax
import jax.numpy as jnp

@pytest.fixture
def setup_logging():
    logging.basicConfig(level=logging.DEBUG)
    yield
    logging.getLogger().handlers = []

def test_enhanced_error_handling(setup_logging, caplog):
    @enhanced_error_handling
    def faulty_function():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        faulty_function()

    assert "Error in faulty_function: Test error" in caplog.text

def test_consciousness_simulation_error_handling():
    rng = jax.random.PRNGKey(0)
    model = ConsciousnessSimulation(features=[64, 32], output_dim=16)

    # Test with invalid input shape
    invalid_input = jnp.ones((2, 5))  # Assuming the model expects (batch_size, 64)
    expected_error_pattern = re.escape(f"Invalid input shape. Expected (batch_size, 64), but got ") + r"\(2, 5\)"

    with pytest.raises(ValueError, match=expected_error_pattern):
        params = model.init(rng, invalid_input)

    # Test with valid input shape
    valid_input = jnp.ones((1, 64))
    try:
        params = model.init(rng, valid_input)
    except ValueError:
        pytest.fail("Unexpected ValueError raised with valid input")

if __name__ == "__main__":
    pytest.main([__file__])

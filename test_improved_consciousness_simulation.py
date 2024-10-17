import pytest
import jax
import jax.numpy as jnp
from NeuroFlex.cognitive_architectures.consciousness_simulation import ImprovedConsciousnessSimulation, create_improved_consciousness_simulation

@pytest.fixture
def improved_model():
    return create_improved_consciousness_simulation(features=[64, 32], output_dim=16)

def test_improved_model_initialization(improved_model):
    assert isinstance(improved_model, ImprovedConsciousnessSimulation)
    assert improved_model.features == [64, 32]
    assert improved_model.output_dim == 16

def test_improved_model_call(improved_model):
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (1, 10))
    external_stimuli = jax.random.normal(rng, (1, 5))

    params = improved_model.init(rng, x, external_stimuli)
    consciousness_state, working_memory_state, long_term_memory = improved_model.apply(params, x, external_stimuli)

    assert consciousness_state.shape == (1, improved_model.output_dim * 3)  # Thought + Metacognition + Memory Output
    assert working_memory_state[0].shape == (1, improved_model.working_memory_size)
    assert working_memory_state[1].shape == (1, improved_model.working_memory_size)
    assert long_term_memory.shape == (1, improved_model.long_term_memory_size)

def test_components_presence(improved_model):
    assert hasattr(improved_model, 'enhanced_attention')
    assert hasattr(improved_model, 'advanced_working_memory')
    assert hasattr(improved_model, 'advanced_metacognition')
    assert hasattr(improved_model, 'thought_generator')
    assert hasattr(improved_model, 'environmental_interaction')
    assert hasattr(improved_model, 'long_term_memory')
    assert hasattr(improved_model, 'lr_scheduler')
    assert hasattr(improved_model, 'self_healing')

def test_environmental_interaction(improved_model):
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (1, 10))
    external_stimuli = jax.random.normal(rng, (1, 5))

    params = improved_model.init(rng, x, external_stimuli)
    consciousness_state, _, _ = improved_model.apply(params, x, external_stimuli)

    consciousness_state_no_stimuli, _, _ = improved_model.apply(params, x, None)

    assert not jnp.allclose(consciousness_state, consciousness_state_no_stimuli)

def test_adaptive_learning_rate(improved_model):
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (1, 10))
    external_stimuli = jax.random.normal(rng, (1, 5))

    params = improved_model.init(rng, x, external_stimuli)
    initial_lr = improved_model.apply(params, x, external_stimuli, method=lambda self, *args: self.variable('model_state', 'learning_rate').value)

    # Run the model multiple times to trigger learning rate updates
    for _ in range(10):
        params, _ = improved_model.apply(params, x, external_stimuli, mutable=['model_state', 'working_memory', 'long_term_memory'])

    final_lr = improved_model.apply(params, x, external_stimuli, method=lambda self, *args: self.variable('model_state', 'learning_rate').value)

    assert initial_lr != final_lr, f"Learning rate did not change. Initial: {initial_lr}, Final: {final_lr}"

if __name__ == "__main__":
    pytest.main([__file__])

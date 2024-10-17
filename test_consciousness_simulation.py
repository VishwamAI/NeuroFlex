import jax
import jax.numpy as jnp
from NeuroFlex.cognitive_architectures.consciousness_simulation import ConsciousnessSimulation, create_consciousness_simulation

def test_enhanced_attention(model, params, x):
    print("Testing enhanced attention...")
    attention_output = model.apply(params, x, method=model.enhanced_attention)
    assert attention_output.shape == (1, model.working_memory_size), f"Expected shape (1, {model.working_memory_size}), got {attention_output.shape}"
    print("Enhanced attention test passed.")

def test_advanced_working_memory(model, params, x):
    print("Testing advanced working memory...")
    current_memory = jnp.zeros((1, model.working_memory_size))
    new_memory, _ = model.apply(params, x, current_memory, method=model.advanced_working_memory)
    assert new_memory.shape == (1, model.working_memory_size), f"Expected shape (1, {model.working_memory_size}), got {new_memory.shape}"
    print("Advanced working memory test passed.")

def test_detailed_brain_simulation(model):
    print("Testing detailed brain simulation...")
    simulation_result = model.aln_model.run()
    assert len(simulation_result) > 0, "Brain simulation returned empty result"
    print("Detailed brain simulation test passed.")

def test_metacognition(model, params, x):
    print("Testing metacognition...")
    metacognition_output = model.apply(params, x, method=model.advanced_metacognition)
    assert metacognition_output.shape[0] == 1, f"Expected batch size 1, got {metacognition_output.shape[0]}"
    print("Metacognition test passed.")

def test_error_handling(model, params, x):
    print("Testing error handling...")
    try:
        model.apply(params, jnp.ones((1, 100)), method=model.__call__)  # Intentionally wrong input shape
    except ValueError as e:
        print(f"Caught expected error: {str(e)}")
    else:
        raise AssertionError("Error handling test failed: expected ValueError")
    print("Error handling test passed.")

def test_adaptive_learning_rate(model):
    print("Testing adaptive learning rate scheduling...")
    initial_lr = model.lr_scheduler.lr
    model.lr_scheduler.step(0.5)  # Simulate performance decrease
    assert model.lr_scheduler.lr < initial_lr, "Learning rate should decrease after performance drop"
    print("Adaptive learning rate test passed.")

def test_self_healing(model, params):
    print("Testing self-healing mechanisms...")
    issues = model.self_healing.diagnose(model)
    model.self_healing.heal(model, issues)
    assert model.variable('model_state', 'healing_attempts').value > 0, "Healing attempts should be recorded"
    print("Self-healing test passed.")

def test_thought_generation(model, params, x):
    print("Testing thought generation...")
    thought = model.apply(params, x, method=model.thought_generator)
    assert thought.shape == (1, model.output_dim), f"Expected shape (1, {model.output_dim}), got {thought.shape}"
    print("Thought generation test passed.")

def test_environmental_interaction(model, params, x, external_stimuli):
    print("Testing environmental interaction...")
    interaction_result = model.apply(params, x, external_stimuli, method=model.environmental_interaction)
    assert interaction_result.shape == x.shape, f"Expected shape {x.shape}, got {interaction_result.shape}"
    print("Environmental interaction test passed.")

def test_long_term_memory(model, params, x):
    print("Testing long-term memory...")
    current_memory = jnp.zeros((1, model.long_term_memory_size))
    new_memory, _ = model.apply(params, x, current_memory, method=model.long_term_memory)
    assert new_memory.shape == (1, model.long_term_memory_size), f"Expected shape (1, {model.long_term_memory_size}), got {new_memory.shape}"
    print("Long-term memory test passed.")

def test_consciousness_simulation():
    print("Starting consciousness simulation tests...")

    # Initialize the model
    rng = jax.random.PRNGKey(0)
    features = [64, 32]
    output_dim = 16
    model = create_consciousness_simulation(features, output_dim)

    # Generate random input and external stimuli
    x = jax.random.normal(rng, (1, 10))
    external_stimuli = jax.random.normal(rng, (1, 5))

    # Initialize parameters
    params = model.init(rng, x, external_stimuli)

    # Test the __call__ method
    consciousness_state, new_working_memory, updated_long_term_memory = model.apply(params, x, external_stimuli)

    print("Consciousness state shape:", consciousness_state.shape)
    print("New working memory shape:", new_working_memory.shape)
    print("Updated long-term memory shape:", updated_long_term_memory.shape)

    # Test the simulate_consciousness method
    simulated_state, simulated_working_memory, simulated_long_term_memory = model.apply(params, x, external_stimuli, method=model.simulate_consciousness)

    print("Simulated consciousness state shape:", simulated_state.shape)
    print("Simulated working memory shape:", simulated_working_memory.shape)
    print("Simulated long-term memory shape:", simulated_long_term_memory.shape)

    # Run specific tests for each component
    test_enhanced_attention(model, params, x)
    test_advanced_working_memory(model, params, x)
    test_detailed_brain_simulation(model)
    test_metacognition(model, params, x)
    test_error_handling(model, params, x)
    test_adaptive_learning_rate(model)
    test_self_healing(model, params)
    test_thought_generation(model, params, x)
    test_environmental_interaction(model, params, x, external_stimuli)
    test_long_term_memory(model, params, x)

    print("All tests completed successfully.")

if __name__ == "__main__":
    test_consciousness_simulation()

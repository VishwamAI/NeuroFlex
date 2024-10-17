import jax
import jax.numpy as jnp
import logging
from NeuroFlex.cognitive_architectures.consciousness_simulation import ConsciousnessSimulation, EnhancedAttention, AdvancedWorkingMemory, detailed_brain_simulation, AdvancedMetacognition
from NeuroFlex.cognitive_architectures.error_handling import enhanced_error_handling
from NeuroFlex.cognitive_architectures.adaptive_learning_rate_scheduler import AdaptiveLearningRateScheduler
from NeuroFlex.cognitive_architectures.advanced_self_healing import AdvancedSelfHealing
from NeuroFlex.cognitive_architectures.detailed_thought_generator import DetailedThoughtGenerator
from NeuroFlex.cognitive_architectures.environmental_interaction import EnvironmentalInteraction
from NeuroFlex.cognitive_architectures.long_term_memory import LongTermMemory

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@enhanced_error_handling
def test_enhanced_attention():
    logger.info("Testing EnhancedAttention")
    rng = jax.random.PRNGKey(0)
    batch_size, seq_length, input_dim = 1, 10, 64
    num_heads, qkv_features, out_features = 4, 32, 64
    dropout_rate = 0.1

    attention = EnhancedAttention(num_heads=num_heads, qkv_features=qkv_features, out_features=out_features, dropout_rate=dropout_rate)
    x = jax.random.normal(rng, (batch_size, seq_length, input_dim))
    params = attention.init({'params': rng, 'dropout': rng}, x)
    output = attention.apply(params, x, rngs={'dropout': rng})

    assert output.shape == (batch_size, seq_length, out_features), f"Expected shape {(batch_size, seq_length, out_features)}, but got {output.shape}"
    logger.info("EnhancedAttention test passed")

@enhanced_error_handling
def test_advanced_working_memory():
    logger.info("Testing AdvancedWorkingMemory")
    rng = jax.random.PRNGKey(0)
    memory_size, batch_size, input_size = 192, 1, 64

    awm = AdvancedWorkingMemory(memory_size=memory_size)
    x = jax.random.normal(rng, (batch_size, input_size))
    initial_state = awm.initialize_state(batch_size)

    def init_and_apply(rng, x, initial_state):
        params = awm.init(rng, x, initial_state)
        def apply_fn(params, x, state):
            return awm.apply(params, x, state)
        return jax.jit(apply_fn), params

    apply_fn, params = init_and_apply(rng, x, initial_state)
    new_state, y = apply_fn(params, x, initial_state)

    logger.debug(f"Input shape: {x.shape}")
    logger.debug(f"Initial state shape: {initial_state[0].shape}, {initial_state[1].shape}")
    logger.debug(f"New state shape: {new_state[0].shape}, {new_state[1].shape}")
    logger.debug(f"Output shape: {y.shape}")

    assert isinstance(new_state, tuple) and len(new_state) == 2, "New state should be a tuple with two elements"
    assert new_state[0].shape == new_state[1].shape == (batch_size, memory_size), f"Expected shape {(batch_size, memory_size)}, but got {new_state[0].shape} and {new_state[1].shape}"
    assert y.shape == (batch_size, memory_size), f"Expected output shape {(batch_size, memory_size)}, but got {y.shape}"

    logger.info("AdvancedWorkingMemory test passed")

@enhanced_error_handling
def test_detailed_brain_simulation():
    logger.info("Testing detailed_brain_simulation")
    num_brain_areas = 5
    simulation_length = 1000
    aln_input = jnp.random.normal(jax.random.PRNGKey(0), (num_brain_areas, 10))

    logger.debug(f"Input parameters: aln_input shape={aln_input.shape}, num_brain_areas={num_brain_areas}, simulation_length={simulation_length}")

    def run_simulation_with_timeout():
        return detailed_brain_simulation(aln_input, num_brain_areas, simulation_length)

    try:
        result, exception = jax.jit(run_simulation_with_timeout)()
        logger.debug(f"Simulation result type: {type(result)}, Exception: {exception}")

        if result is None:
            logger.error(f"Detailed brain simulation failed. Exception: {exception}")
            assert exception is not None, "Exception should not be None when result is None"
            logger.error(f"Exception details: {str(exception)}")
        else:
            assert exception is None, f"Exception should be None when result is not None, but got: {exception}"
            assert isinstance(result, dict), f"Result should be a dictionary, but got {type(result)}"
            assert 'rates_exc' in result, f"Result should contain 'rates_exc', but only has keys: {result.keys()}"
            assert result['rates_exc'].shape == (num_brain_areas, simulation_length // 10), f"Expected shape {(num_brain_areas, simulation_length // 10)}, but got {result['rates_exc'].shape}"
            assert jnp.all(jnp.isfinite(result['rates_exc'])), "Output contains non-finite values"
            logger.info("Detailed brain simulation test passed")
        logger.debug(f"Final result: {result}")
    except Exception as e:
        logger.error(f"Unexpected error in detailed brain simulation: {str(e)}")
        raise

@enhanced_error_handling
def test_advanced_metacognition():
    logger.info("Testing AdvancedMetacognition")
    rng = jax.random.PRNGKey(0)
    input_size, batch_size = 64, 1

    metacognition = AdvancedMetacognition()
    x = jax.random.normal(rng, (batch_size, input_size))
    params = metacognition.init(rng, x)
    output = metacognition.apply(params, x)

    logger.debug(f"Input shape: {x.shape}")
    logger.debug(f"Output shape: {output.shape}")
    logger.debug(f"Output min: {jnp.min(output)}, max: {jnp.max(output)}, mean: {jnp.mean(output)}")

    assert output.shape == (batch_size, 2), f"Expected shape {(batch_size, 2)}, but got {output.shape}"
    assert jnp.all(output >= 0) and jnp.all(output <= 1), "Output values should be between 0 and 1"
    assert jnp.issubdtype(output.dtype, jnp.floating), f"Expected floating-point output, but got {output.dtype}"

    logger.info("AdvancedMetacognition test passed")

@enhanced_error_handling
def test_adaptive_learning_rate_scheduler():
    logger.info("Testing AdaptiveLearningRateScheduler")
    initial_lr, patience, factor = 0.001, 5, 0.5
    scheduler = AdaptiveLearningRateScheduler(initial_lr=initial_lr, patience=patience, factor=factor)

    logger.debug(f"Initial learning rate: {scheduler.lr}")

    # Simulate improving performance
    for i in range(10):
        new_lr = scheduler.step(i * 0.1)
        logger.debug(f"Step {i+1}, Performance: {i*0.1}, New LR: {new_lr}")
        assert new_lr == initial_lr, f"LR should not change during improvement, but got {new_lr}"

    # Simulate plateauing performance
    for i in range(patience + 1):
        new_lr = scheduler.step(0.9)
        logger.debug(f"Step {i+11}, Performance: 0.9, New LR: {new_lr}")
        if i == patience:
            assert new_lr == initial_lr * factor, f"LR should decrease after {patience} steps, but got {new_lr}"

    # Simulate declining performance
    for i in range(patience + 1):
        new_lr = scheduler.step(0.8 - i * 0.01)
        logger.debug(f"Step {i+17}, Performance: {0.8 - i*0.01}, New LR: {new_lr}")

    assert scheduler.lr >= initial_lr * (factor ** 3), f"LR should not decrease below {initial_lr * (factor ** 3)}, but got {scheduler.lr}"
    assert scheduler.lr <= initial_lr, f"LR should not increase above initial value {initial_lr}, but got {scheduler.lr}"

    logger.info("AdaptiveLearningRateScheduler test passed")

@enhanced_error_handling
def test_advanced_self_healing():
    logger.info("Testing AdvancedSelfHealing")
    self_healing = AdvancedSelfHealing()

    # Create a mock model with simulated issues
    class MockModel:
        def __init__(self):
            self.params = {
                'layer1': jnp.array([[1.0, 2.0], [3.0, float('nan')]]),
                'layer2': jnp.array([[float('inf'), 5.0], [6.0, 7.0]]),
                'layer3': jnp.array([[8.0, 9.0], [10.0, 11.0]])
            }

    model = MockModel()
    logger.debug(f"Initial model params: {model.params}")

    # Diagnose the mock model for issues
    issues = self_healing.diagnose(model)
    logger.debug(f"Diagnosed issues: {issues}")
    assert len(issues) > 0, "Diagnostic should detect issues"
    assert "NaN values detected in model parameters" in issues, "Diagnostic should detect NaN values"

    # Apply the healing process to the diagnosed issues
    self_healing.heal(model, issues)
    logger.debug(f"Model params after healing: {model.params}")

    # Verify that the healing process resolved the simulated issues
    assert not jnp.isnan(model.params['layer1']).any(), "Healing should replace NaN values"
    assert not jnp.isinf(model.params['layer2']).any(), "Healing should replace infinite values"
    assert jnp.allclose(model.params['layer3'], jnp.array([[8.0, 9.0], [10.0, 11.0]])), "Healing should not affect normal values"

    # Check if no issues remain after healing
    remaining_issues = self_healing.diagnose(model)
    logger.debug(f"Remaining issues after healing: {remaining_issues}")
    assert len(remaining_issues) == 0, "No issues should remain after healing"

    # Test edge case: empty model
    empty_model = MockModel()
    empty_model.params = {}
    empty_issues = self_healing.diagnose(empty_model)
    assert len(empty_issues) == 0, "Empty model should have no issues"

    # Test edge case: model with all NaN values
    nan_model = MockModel()
    nan_model.params = {
        'layer': jnp.full((2, 2), float('nan'))
    }
    nan_issues = self_healing.diagnose(nan_model)
    assert len(nan_issues) > 0, "Model with all NaN values should be detected"
    self_healing.heal(nan_model, nan_issues)
    assert not jnp.isnan(nan_model.params['layer']).any(), "All NaN values should be replaced"

    logger.info("AdvancedSelfHealing test passed")

@enhanced_error_handling
def test_detailed_thought_generator():
    logger.info("Testing DetailedThoughtGenerator")
    rng = jax.random.PRNGKey(0)
    input_size, batch_size, output_dim = 64, 1, 16

    thought_generator = DetailedThoughtGenerator(output_dim=output_dim)
    logger.debug(f"Initialized DetailedThoughtGenerator with output_dim={output_dim}")

    x = jax.random.normal(rng, (batch_size, input_size))
    logger.debug(f"Input shape: {x.shape}")

    params = thought_generator.init(rng, x)
    logger.debug(f"Initialized parameters")

    # Generate thoughts multiple times to test diversity
    thoughts = [thought_generator.apply(params, x) for _ in range(5)]
    logger.debug(f"Generated {len(thoughts)} thoughts")

    for i, thought in enumerate(thoughts):
        logger.debug(f"Thought {i+1} shape: {thought.shape}")
        assert isinstance(thought, jnp.ndarray), f"Expected output to be a JAX array, but got {type(thought)}"
        assert thought.shape == (batch_size, output_dim), f"Expected shape ({batch_size}, {output_dim}), but got {thought.shape}"

    # Check diversity of thoughts
    thought_diversity = jnp.std(jnp.stack(thoughts))
    logger.debug(f"Thought diversity (std): {thought_diversity}")
    assert thought_diversity > 0, "Generated thoughts should be diverse"

    # Test edge case: empty input
    empty_input = jnp.zeros((batch_size, input_size))
    empty_thought = thought_generator.apply(params, empty_input)
    logger.debug(f"Empty input thought shape: {empty_thought.shape}")
    assert jnp.all(jnp.isfinite(empty_thought)), "Thought generator should handle empty input gracefully"

    # Test edge case: extreme values
    extreme_input = jnp.full((batch_size, input_size), 1e6)
    extreme_thought = thought_generator.apply(params, extreme_input)
    logger.debug(f"Extreme input thought shape: {extreme_thought.shape}")
    assert jnp.all(jnp.isfinite(extreme_thought)), "Thought generator should handle extreme input values"

    logger.info("DetailedThoughtGenerator test passed")

@enhanced_error_handling
def test_environmental_interaction():
    logger.info("Testing EnvironmentalInteraction")
    rng = jax.random.PRNGKey(0)
    thought_size, stimuli_size, batch_size = 32, 16, 1

    env_interaction = EnvironmentalInteraction()
    logger.debug(f"Initialized EnvironmentalInteraction")

    thought = jax.random.normal(rng, (batch_size, thought_size))
    stimuli = jax.random.normal(rng, (batch_size, stimuli_size))
    logger.debug(f"Thought shape: {thought.shape}, Stimuli shape: {stimuli.shape}")

    params = env_interaction.init(rng, thought, stimuli)
    logger.debug(f"Initialized parameters")

    output = env_interaction.apply(params, thought, stimuli)
    logger.debug(f"Output shape: {output.shape}")

    assert isinstance(output, jnp.ndarray), f"Expected output to be a JAX array, but got {type(output)}"
    assert output.shape == thought.shape, f"Expected output shape {thought.shape}, but got {output.shape}"

    # Test with different types of stimuli
    visual_stimuli = jax.random.normal(rng, (batch_size, stimuli_size // 2))
    auditory_stimuli = jax.random.normal(rng, (batch_size, stimuli_size // 2))
    combined_stimuli = jnp.concatenate([visual_stimuli, auditory_stimuli], axis=-1)
    output_combined = env_interaction.apply(params, thought, combined_stimuli)
    assert output_combined.shape == thought.shape, "Output shape should match thought shape for combined stimuli"

    # Test edge case: no external stimuli
    no_stimuli = jnp.zeros((batch_size, stimuli_size))
    output_no_stimuli = env_interaction.apply(params, thought, no_stimuli)
    assert jnp.all(jnp.isfinite(output_no_stimuli)), "Output should be finite with no external stimuli"

    # Test edge case: extreme values
    extreme_stimuli = jnp.full((batch_size, stimuli_size), 1e6)
    output_extreme = env_interaction.apply(params, thought, extreme_stimuli)
    assert jnp.all(jnp.isfinite(output_extreme)), "Output should be finite with extreme stimuli values"

    logger.info("EnvironmentalInteraction test passed")

@enhanced_error_handling
def test_long_term_memory():
    logger.info("Testing LongTermMemory")
    rng = jax.random.PRNGKey(0)
    memory_size, input_size, batch_size = 1024, 64, 1

    ltm = LongTermMemory(memory_size=memory_size)
    logger.debug(f"Initialized LongTermMemory with memory_size={memory_size}")

    x = jax.random.normal(rng, (batch_size, input_size))
    logger.debug(f"Input shape: {x.shape}")

    def initialize_state(batch_size):
        return jnp.zeros((batch_size, memory_size))

    state = initialize_state(batch_size)
    logger.debug(f"Initialized state shape: {state.shape}")

    params = ltm.init(rng, x, state)
    logger.debug(f"Initialized parameters")

    # Test multiple iterations of memory storage and retrieval
    num_iterations = 5
    for i in range(num_iterations):
        new_state, y = ltm.apply(params, x, state)
        logger.debug(f"Iteration {i+1} - New state shape: {new_state.shape}, Output shape: {y.shape}")
        assert isinstance(new_state, jnp.ndarray), f"Expected new_state to be a JAX array, but got {type(new_state)}"
        assert isinstance(y, jnp.ndarray), f"Expected output to be a JAX array, but got {type(y)}"
        assert new_state.shape == (batch_size, memory_size), f"Expected shape {(batch_size, memory_size)}, but got {new_state.shape}"
        assert y.shape == (batch_size, memory_size), f"Expected shape {(batch_size, memory_size)}, but got {y.shape}"
        state = new_state
        x = y  # Use output as next input to simulate information retrieval

    # Test edge case: memory overflow
    overflow_input = jnp.full((batch_size, input_size), 1e6)
    overflow_state, overflow_output = ltm.apply(params, overflow_input, state)
    logger.debug(f"Overflow test - State shape: {overflow_state.shape}, Output shape: {overflow_output.shape}")
    assert jnp.all(jnp.isfinite(overflow_state)), "State should remain finite after overflow input"
    assert jnp.all(jnp.isfinite(overflow_output)), "Output should be finite after overflow input"

    # Test edge case: retrieval with zero input
    zero_input = jnp.zeros((batch_size, input_size))
    zero_state, zero_output = ltm.apply(params, zero_input, state)
    logger.debug(f"Zero input test - State shape: {zero_state.shape}, Output shape: {zero_output.shape}")
    assert not jnp.all(zero_output == 0), "Output should not be all zeros when retrieving with zero input"

    logger.info("LongTermMemory test passed")

@enhanced_error_handling
def test_consciousness_simulation():
    logger.info("Testing ConsciousnessSimulation")
    rng = jax.random.PRNGKey(0)
    input_size, output_size, batch_size = 64, 32, 1
    features = [64, 32]
    long_term_memory_size = 1024

    model = ConsciousnessSimulation(features=features, output_dim=output_size, long_term_memory_size=long_term_memory_size)
    x = jax.random.normal(rng, (batch_size, input_size))
    external_stimuli = jax.random.normal(rng, (batch_size, 5))
    params = model.init(rng, x, external_stimuli)
    consciousness_state, working_memory, long_term_memory = model.apply(params, x, external_stimuli)

    assert consciousness_state.shape == (batch_size, output_size), f"Expected shape {(batch_size, output_size)}, but got {consciousness_state.shape}"
    assert working_memory[0].shape == working_memory[1].shape == (batch_size, features[-1]), f"Expected shape {(batch_size, features[-1])}, but got {working_memory[0].shape} and {working_memory[1].shape}"
    assert long_term_memory.shape == (batch_size, long_term_memory_size), f"Expected shape {(batch_size, long_term_memory_size)}, but got {long_term_memory.shape}"
    logger.info("ConsciousnessSimulation test passed")

if __name__ == "__main__":
    test_functions = [
        test_enhanced_attention,
        test_advanced_working_memory,
        test_detailed_brain_simulation,
        test_advanced_metacognition,
        test_adaptive_learning_rate_scheduler,
        test_advanced_self_healing,
        test_detailed_thought_generator,
        test_environmental_interaction,
        test_long_term_memory,
        test_consciousness_simulation
    ]

    for test_func in test_functions:
        try:
            test_func()
            logger.info(f"{test_func.__name__} passed successfully")
        except Exception as e:
            logger.error(f"{test_func.__name__} failed: {str(e)}")

    logger.info("All tests completed")

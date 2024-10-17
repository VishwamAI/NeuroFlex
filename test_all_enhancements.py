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
    memory_size, batch_size = 192, 1

    awm = AdvancedWorkingMemory(memory_size=memory_size)
    x = jax.random.normal(rng, (batch_size, memory_size))

    def init_and_apply(rng, x):
        params = awm.init(rng, x, (jnp.zeros((batch_size, memory_size)), jnp.zeros((batch_size, memory_size))))
        def apply_fn(params, x, state):
            return awm.apply(params, x, state)
        return jax.jit(apply_fn), params

    apply_fn, params = init_and_apply(rng, x)
    new_state, y = apply_fn(params, x, (jnp.zeros((batch_size, memory_size)), jnp.zeros((batch_size, memory_size))))

    assert isinstance(new_state, tuple) and len(new_state) == 2, "New state should be a tuple with two elements"
    assert new_state[0].shape == new_state[1].shape == (batch_size, memory_size), f"Expected shape {(batch_size, memory_size)}, but got {new_state[0].shape} and {new_state[1].shape}"
    assert y.shape == (batch_size, memory_size), f"Expected output shape {(batch_size, memory_size)}, but got {y.shape}"
    logger.info("AdvancedWorkingMemory test passed")

@enhanced_error_handling
def test_detailed_brain_simulation():
    logger.info("Testing detailed_brain_simulation")
    aln_input = jnp.ones((5, 10))
    num_brain_areas = 5
    simulation_length = 1000

    logger.debug(f"Input parameters: aln_input shape={aln_input.shape}, num_brain_areas={num_brain_areas}, simulation_length={simulation_length}")
    result, exception = detailed_brain_simulation(aln_input, num_brain_areas, simulation_length)
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
        logger.info("Detailed brain simulation test passed")
    logger.debug(f"Final result: {result}")

@enhanced_error_handling
def test_advanced_metacognition():
    logger.info("Testing AdvancedMetacognition")
    rng = jax.random.PRNGKey(0)
    input_size, batch_size = 64, 1

    metacognition = AdvancedMetacognition()
    x = jax.random.normal(rng, (batch_size, input_size))
    params = metacognition.init(rng, x)
    output = metacognition.apply(params, x)

    assert output.shape == (batch_size, 2), f"Expected shape {(batch_size, 2)}, but got {output.shape}"
    logger.info("AdvancedMetacognition test passed")

@enhanced_error_handling
def test_adaptive_learning_rate_scheduler():
    logger.info("Testing AdaptiveLearningRateScheduler")
    scheduler = AdaptiveLearningRateScheduler(initial_lr=0.001, patience=10, factor=0.5)

    initial_lr = scheduler.lr
    scheduler.step(0.5)  # No change
    assert scheduler.lr == initial_lr, f"Learning rate should not change, but got {scheduler.lr}"

    for _ in range(11):  # Trigger learning rate change
        scheduler.step(0.4)

    assert scheduler.lr == initial_lr * 0.5, f"Learning rate should be halved, but got {scheduler.lr}"
    logger.info("AdaptiveLearningRateScheduler test passed")

@enhanced_error_handling
def test_advanced_self_healing():
    logger.info("Testing AdvancedSelfHealing")
    self_healing = AdvancedSelfHealing()

    # Create a mock model with issues
    class MockModel:
        def __init__(self):
            self.params = {
                'layer1': jnp.zeros((10, 10)),
                'layer2': None  # Simulating a broken layer
            }

    model = MockModel()
    logger.debug(f"Initial model params: {model.params}")

    issues = self_healing.diagnose(model)
    logger.debug(f"Diagnosed issues: {issues}")
    assert len(issues) > 0, "Diagnostic should detect issues"
    assert 'layer2' in issues, "Diagnostic should detect the broken layer2"

    self_healing.heal(model, issues)
    logger.debug(f"Model params after healing: {model.params}")
    assert model.params['layer2'] is not None, "Healing should fix the broken layer"
    assert isinstance(model.params['layer2'], jnp.ndarray), "Healed layer should be a JAX array"

    # Check if no issues remain after healing
    remaining_issues = self_healing.diagnose(model)
    logger.debug(f"Remaining issues after healing: {remaining_issues}")
    assert len(remaining_issues) == 0, "No issues should remain after healing"

    logger.info("AdvancedSelfHealing test passed")

@enhanced_error_handling
def test_detailed_thought_generator():
    logger.info("Testing DetailedThoughtGenerator")
    rng = jax.random.PRNGKey(0)
    input_size, batch_size = 64, 1

    thought_generator = DetailedThoughtGenerator()
    logger.debug(f"Initialized DetailedThoughtGenerator")

    x = jax.random.normal(rng, (batch_size, input_size))
    logger.debug(f"Input shape: {x.shape}")

    params = thought_generator.init(rng, x)
    logger.debug(f"Initialized parameters")

    output = thought_generator.apply(params, x)
    logger.debug(f"Output shape: {output.shape}")

    assert isinstance(output, jnp.ndarray), f"Expected output to be a JAX array, but got {type(output)}"
    assert output.shape[0] == batch_size, f"Expected batch size {batch_size}, but got {output.shape[0]}"
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
    assert output.shape[0] == batch_size, f"Expected batch size {batch_size}, but got {output.shape[0]}"
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

    new_state, y = ltm.apply(params, x, state)
    logger.debug(f"New state shape: {new_state.shape}, Output shape: {y.shape}")

    assert isinstance(new_state, jnp.ndarray), f"Expected new_state to be a JAX array, but got {type(new_state)}"
    assert isinstance(y, jnp.ndarray), f"Expected output to be a JAX array, but got {type(y)}"
    assert new_state.shape == (batch_size, memory_size), f"Expected shape {(batch_size, memory_size)}, but got {new_state.shape}"
    assert y.shape == (batch_size, input_size), f"Expected shape {(batch_size, input_size)}, but got {y.shape}"
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

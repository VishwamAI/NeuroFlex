import jax
import jax.numpy as jnp
from NeuroFlex.cognitive_architectures.advanced_working_memory import AdvancedWorkingMemory
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_advanced_working_memory():
    logger.info("Starting advanced working memory test")

    try:
        rng = jax.random.PRNGKey(0)
        memory_size = 192
        batch_size = 1

        # Initialize the AdvancedWorkingMemory
        awm = AdvancedWorkingMemory(memory_size=memory_size)

        # Create a random input
        x = jax.random.normal(rng, (batch_size, memory_size))
        logger.debug(f"Input shape: {x.shape}")

        # Initialize the state
        state = awm.initialize_state(batch_size)
        logger.debug(f"Initial state: {state}")

        # Initialize parameters
        params = awm.init(rng, x, state)

        # Apply the AdvancedWorkingMemory
        new_state, y = awm.apply(params, x, state)

        logger.debug(f"New state type: {type(new_state)}")
        logger.debug(f"New state shapes: {new_state[0].shape}, {new_state[1].shape}")
        logger.debug(f"Output shape: {y.shape}")

        # Assertions
        assert isinstance(new_state, tuple), "New state should be a tuple"
        assert len(new_state) == 2, "New state should have two elements"
        assert new_state[0].shape == (batch_size, memory_size), f"Expected shape {(batch_size, memory_size)}, but got {new_state[0].shape}"
        assert new_state[1].shape == (batch_size, memory_size), f"Expected shape {(batch_size, memory_size)}, but got {new_state[1].shape}"
        assert y.shape == (batch_size, memory_size), f"Expected output shape {(batch_size, memory_size)}, but got {y.shape}"

        logger.info("Advanced working memory test passed successfully")
    except Exception as e:
        logger.error(f"Advanced working memory test failed with error: {str(e)}")
        logger.exception("Traceback for the error:")
        raise

if __name__ == "__main__":
    test_advanced_working_memory()

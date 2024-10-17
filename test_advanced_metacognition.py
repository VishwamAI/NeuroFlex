import jax
import jax.numpy as jnp
from NeuroFlex.cognitive_architectures.advanced_metacognition import AdvancedMetacognition
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_advanced_metacognition():
    logger.info("Starting advanced metacognition test")

    try:
        rng = jax.random.PRNGKey(0)
        batch_size = 1
        input_dim = 64

        # Initialize the AdvancedMetacognition
        metacognition = AdvancedMetacognition()

        # Create a random input
        x = jax.random.normal(rng, (batch_size, input_dim))
        logger.debug(f"Input shape: {x.shape}")

        # Initialize parameters
        params = metacognition.init(rng, x)

        # Apply the AdvancedMetacognition
        output = metacognition.apply(params, x)

        logger.debug(f"Output shape: {output.shape}")

        # Assertions
        assert output.shape == (batch_size, 2), f"Expected shape {(batch_size, 2)}, but got {output.shape}"
        assert jnp.all(output >= 0) and jnp.all(output <= 1), "Output values should be between 0 and 1"

        logger.info("Advanced metacognition test passed successfully")
    except Exception as e:
        logger.error(f"Advanced metacognition test failed with error: {str(e)}")
        logger.exception("Traceback for the error:")
        raise

if __name__ == "__main__":
    test_advanced_metacognition()

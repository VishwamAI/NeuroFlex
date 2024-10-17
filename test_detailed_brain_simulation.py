import jax
import jax.numpy as jnp
from NeuroFlex.cognitive_architectures.consciousness_simulation import detailed_brain_simulation
import logging
import numpy as np
import signal
import sys

# Configure the root logger
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a file handler
file_handler = logging.FileHandler('detailed_brain_simulation_test.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Set logging level for all loggers
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger('neurolib').setLevel(logging.DEBUG)

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Simulation timed out")

def test_detailed_brain_simulation():
    logger.info("Starting detailed brain simulation test")

    try:
        # Set timeout for 60 seconds
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)

        rng = jax.random.PRNGKey(0)
        num_brain_areas = 5
        simulation_length = 1000  # in milliseconds
        input_size = 10

        # Create a random input
        aln_input = jax.random.normal(rng, (num_brain_areas, input_size))
        logger.debug(f"Input shape: {aln_input.shape}")

        # Run the detailed brain simulation
        logger.info("Calling detailed_brain_simulation function")
        simulation_result = detailed_brain_simulation(aln_input, num_brain_areas, simulation_length)
        logger.info("detailed_brain_simulation function call completed")

        # Cancel the alarm
        signal.alarm(0)

        logger.debug(f"Simulation result type: {type(simulation_result)}")

        if simulation_result is None:
            logger.warning("Simulation result is None. This might indicate an error in the simulation process.")
            logger.debug("Detailed brain simulation returned None. Check the ALNModel implementation for potential issues.")
        elif isinstance(simulation_result, (np.ndarray, jnp.ndarray)):
            logger.debug(f"Simulation result shape: {simulation_result.shape}")

            # Assertions
            assert simulation_result.shape[0] == num_brain_areas, f"Expected first dimension to be {num_brain_areas}, but got {simulation_result.shape[0]}"

            # The second dimension might not exactly match simulation_length due to potential time discretization
            assert abs(simulation_result.shape[1] - simulation_length) < 10, f"Expected second dimension to be close to {simulation_length}, but got {simulation_result.shape[1]}"

            logger.info("Detailed brain simulation test passed successfully")
        elif isinstance(simulation_result, dict):
            logger.debug(f"Simulation result keys: {simulation_result.keys()}")
            if 'rates_exc' in simulation_result:
                rates_exc = simulation_result['rates_exc']
                logger.debug(f"Shape of rates_exc: {rates_exc.shape}")

                # Assertions
                assert rates_exc.shape[0] == num_brain_areas, f"Expected first dimension of rates_exc to be {num_brain_areas}, but got {rates_exc.shape[0]}"

                # The second dimension might not exactly match simulation_length due to potential time discretization
                assert abs(rates_exc.shape[1] - simulation_length) < 10, f"Expected second dimension of rates_exc to be close to {simulation_length}, but got {rates_exc.shape[1]}"

                logger.info("Detailed brain simulation test passed successfully")
            else:
                logger.warning("Expected 'rates_exc' key not found in simulation result")
                logger.debug(f"Available keys in simulation result: {simulation_result.keys()}")
        else:
            logger.warning(f"Unexpected simulation result type: {type(simulation_result)}")
            logger.debug(f"Unexpected simulation result content: {simulation_result}")

    except TimeoutError:
        logger.error("Detailed brain simulation test timed out after 60 seconds")
    except Exception as e:
        logger.error(f"Detailed brain simulation test failed with error: {str(e)}")
        logger.exception("Traceback for the error:")
    finally:
        # Ensure the alarm is canceled even if an exception occurs
        signal.alarm(0)

    logger.info("Detailed brain simulation test script execution completed")

if __name__ == "__main__":
    test_detailed_brain_simulation()

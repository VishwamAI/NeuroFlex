import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from neurolib.utils.loadData import Dataset
from NeuroFlex.neuroscience_models.neuroscience_model import NeuroscienceModel
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_neuroscience_model():
    try:
        # Initialize the model
        model = NeuroscienceModel()
        logger.info("NeuroscienceModel initialized")

        # Set up basic parameters
        model.set_parameters({"exc_ext_rate": 0.1, "dt": 0.1})
        logger.info("Model parameters set")

        # Create and load connectivity
        connectivity = np.random.rand(2, 2)
        dataset = Dataset()
        dataset.Cmat = connectivity
        model.load_connectivity(dataset)
        logger.info("Connectivity loaded")

        # Run a simple prediction
        dummy_data = np.random.rand(100, 2)
        logger.debug(f"Generated dummy data with shape: {dummy_data.shape}")
        logger.debug(f"Dummy data sample: {dummy_data[:5]}")  # Log first 5 rows
        try:
            predictions = model.predict(dummy_data)
            logger.info(f"Prediction shape: {predictions.shape}")
            logger.debug(f"Prediction sample: {predictions[:5]}")  # Log first 5 rows
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

        # Run a simulation
        try:
            simulation_output = model.run_simulation(duration=1.0, dt=0.1)
            logger.info(f"Simulation output shape: {simulation_output.shape}")
            logger.debug(f"Simulation output sample: {simulation_output[:5]}")  # Log first 5 rows
        except Exception as e:
            logger.error(f"Error during simulation: {str(e)}")
            raise

        logger.info("NeuroscienceModel run successfully")
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    run_neuroscience_model()

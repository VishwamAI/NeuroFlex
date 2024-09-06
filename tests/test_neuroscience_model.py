import unittest
import numpy as np
from neurolib.utils.loadData import Dataset
from NeuroFlex.neuroscience_models.neuroscience_model import NeuroscienceModel

import logging
from io import StringIO

class TestNeuroscienceModel(unittest.TestCase):
    def setUp(self):
        self.model = NeuroscienceModel()
        self.dummy_data = np.random.rand(100, 2)  # ALNModel expects 2D input
        self.dummy_labels = np.random.randint(0, 2, 100)
        self.log_capture = StringIO()
        self.log_handler = logging.StreamHandler(self.log_capture)
        logging.getLogger().addHandler(self.log_handler)
        logging.getLogger().setLevel(logging.DEBUG)

    def tearDown(self):
        logging.getLogger().removeHandler(self.log_handler)

    def test_basic_simulation(self):
        self.model.set_parameters({"exc_ext_rate": 0.1, "dt": 0.1})
        connectivity = np.random.rand(2, 2)
        dataset = Dataset()
        dataset.Cmat = connectivity
        self.model.load_connectivity(dataset)
        dummy_data = np.random.rand(100, 2)
        predictions = self.model.predict(dummy_data)
        expected_time_steps = int(self.model.model.params['duration'] / self.model.model.params['dt'])
        self.assertEqual(predictions.shape, (expected_time_steps, 2))
        self.assertIn("Predict method called with data shape", self.log_capture.getvalue())
        self.assertIn("Model run completed successfully", self.log_capture.getvalue())

    def test_connectivity_matrix(self):
        connectivity = np.random.rand(32, 32)
        dataset = Dataset()
        dataset.Cmat = connectivity
        self.model.load_connectivity(dataset)
        self.assertIsNotNone(self.model.connectivity)
        self.assertEqual(self.model.connectivity.shape, (32, 32))

    def test_parameter_setting(self):
        new_params = {"sigma_ou": 0.05, "dt": 0.2}
        self.model.set_parameters(new_params)
        for key, value in new_params.items():
            self.assertEqual(self.model.model.params[key], value)

    def test_prediction(self):
        connectivity = np.random.rand(2, 2)
        dataset = Dataset()
        dataset.Cmat = connectivity
        self.model.load_connectivity(dataset)
        self.model.set_parameters({"exc_ext_rate": 0.1, "dt": 0.1})
        dummy_data = np.random.rand(100, 2)
        predictions = self.model.predict(dummy_data)
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions.shape), 2)  # Should be a 2D array
        expected_time_steps = int(self.model.model.params['duration'] / self.model.model.params['dt'])
        self.assertEqual(predictions.shape, (expected_time_steps, 2))
        log_output = self.log_capture.getvalue()
        self.assertIn("Predict method called with data shape", log_output)
        self.assertIn("Current model parameters", log_output)
        self.assertIn("Connectivity matrix shape", log_output)
        self.assertIn("Model run completed successfully", log_output)
        self.assertIn("Model output shape", log_output)

    def test_integration(self):
        self.model.set_parameters({"exc_ext_rate": 0.1, "dt": 0.1})
        connectivity = np.random.rand(2, 2)
        dataset = Dataset()
        dataset.Cmat = connectivity
        self.model.load_connectivity(dataset)

        # Test prediction
        dummy_data = np.random.rand(100, 2)
        predictions = self.model.predict(dummy_data)
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions.shape), 2)

        # Test simulation
        simulation_output = self.model.run_simulation(duration=1.0, dt=0.1)
        self.assertIsInstance(simulation_output, np.ndarray)
        self.assertEqual(len(simulation_output.shape), 2)
        expected_time_steps = int(1.0 / 0.1)  # duration / dt
        self.assertEqual(simulation_output.shape, (expected_time_steps, 2))
        self.assertIn(f"Simulation output shape: {simulation_output.shape}", self.log_capture.getvalue())

        log_output = self.log_capture.getvalue()
        self.assertIn("Predict method called with data shape", log_output)
        self.assertIn("Model run completed successfully", log_output)

    def test_error_handling(self):
        with self.assertRaises(ValueError):
            self.model.predict(self.dummy_data)  # Should raise error due to missing connectivity
        log_output = self.log_capture.getvalue()
        self.assertIn("Connectivity matrix not loaded", log_output)

    def test_train(self):
        self.model.set_parameters({"exc_ext_rate": 0.1, "dt": 0.1})
        connectivity = np.random.rand(2, 2)
        dataset = Dataset()
        dataset.Cmat = connectivity
        self.model.load_connectivity(dataset)
        dummy_data = np.random.rand(100, 2)
        dummy_labels = np.random.randint(0, 2, 100)
        self.model.train(dummy_data, dummy_labels)
        # Since train is a placeholder, we just check if it runs without error
        self.assertTrue(True)

    def test_evaluate(self):
        self.model.set_parameters({"exc_ext_rate": 0.1, "dt": 0.1})
        connectivity = np.random.rand(2, 2)
        dataset = Dataset()
        dataset.Cmat = connectivity
        self.model.load_connectivity(dataset)
        dummy_data = np.random.rand(100, 2)
        dummy_labels = np.random.randint(0, 2, 100)
        evaluation = self.model.evaluate(dummy_data, dummy_labels)
        self.assertIsInstance(evaluation, dict)
        self.assertIn('accuracy', evaluation)
        self.assertIsInstance(evaluation['accuracy'], float)

    def test_interpret_results(self):
        dummy_results = np.random.rand(100, 2)
        interpretation = self.model.interpret_results(dummy_results)
        self.assertIsInstance(interpretation, dict)
        self.assertIn('mean_activity', interpretation)
        self.assertIn('max_activity', interpretation)
        self.assertIn('min_activity', interpretation)
        self.assertIn('interpretation', interpretation)

    def test_get_model_info(self):
        self.model.set_parameters({"exc_ext_rate": 0.1, "dt": 0.1})
        connectivity = np.random.rand(2, 2)
        dataset = Dataset()
        dataset.Cmat = connectivity
        self.model.load_connectivity(dataset)
        info = self.model.get_model_info()
        self.assertIsInstance(info, dict)
        self.assertIn('num_regions', info)
        self.assertIn('current_parameters', info)
        self.assertIn('has_connectivity', info)
        self.assertTrue(info['has_connectivity'])

if __name__ == '__main__':
    unittest.main()

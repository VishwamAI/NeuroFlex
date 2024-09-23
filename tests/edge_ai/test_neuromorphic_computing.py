import unittest
import torch
import numpy as np
import time
from unittest.mock import patch
from NeuroFlex.edge_ai.neuromorphic_computing import NeuromorphicComputing
from NeuroFlex.constants import (
    PERFORMANCE_THRESHOLD,
    UPDATE_INTERVAL,
)


class TestNeuromorphicComputing(unittest.TestCase):
    def setUp(self):
        self.nc = NeuromorphicComputing()

    def test_create_spiking_neural_network(self):
        lif_network = self.nc.create_spiking_neural_network("LIF", num_neurons=100)
        self.assertIsInstance(lif_network, torch.nn.Module)

        izhikevich_network = self.nc.create_spiking_neural_network(
            "Izhikevich", num_neurons=50
        )
        self.assertIsInstance(izhikevich_network, torch.nn.Module)

        with self.assertRaises(ValueError):
            self.nc.create_spiking_neural_network("InvalidModel", num_neurons=10)

    def test_simulate_network(self):
        network = self.nc.create_spiking_neural_network("LIF", num_neurons=100)
        input_data = torch.randn(1, 100)
        output = self.nc.simulate_network(network, input_data, simulation_time=1000)
        self.assertEqual(output.shape, (1, 100))

    def test_update_performance(self):
        initial_performance = self.nc.performance
        output = torch.tensor([0.5])
        self.nc._update_performance(output)
        self.assertNotEqual(self.nc.performance, initial_performance)
        self.assertEqual(len(self.nc.performance_history), 1)

        # Test performance history limit
        for _ in range(110):
            self.nc._update_performance(torch.tensor([0.5]))
        self.assertEqual(len(self.nc.performance_history), 100)

    @patch("NeuroFlex.edge_ai.neuromorphic_computing.NeuromorphicComputing._self_heal")
    def test_update_performance_triggers_self_heal(self, mock_self_heal):
        self.nc.performance = PERFORMANCE_THRESHOLD - 0.1
        output = torch.tensor([PERFORMANCE_THRESHOLD - 0.1])
        self.nc._update_performance(output)
        mock_self_heal.assert_called_once()

    def test_self_heal(self):
        self.nc.performance = PERFORMANCE_THRESHOLD - 0.1
        initial_performance = self.nc.performance
        self.nc._self_heal()
        self.assertGreaterEqual(self.nc.performance, initial_performance)
        self.assertLessEqual(self.nc.performance, PERFORMANCE_THRESHOLD)

    def test_adjust_learning_rate(self):
        initial_lr = self.nc.learning_rate
        self.nc.performance_history = [0.5, 0.6]
        self.nc._adjust_learning_rate()
        self.assertGreater(self.nc.learning_rate, initial_lr)

        self.nc.performance_history = [0.6, 0.5]
        self.nc._adjust_learning_rate()
        self.assertLess(self.nc.learning_rate, initial_lr)

        # Test learning rate bounds
        self.nc.learning_rate = 0.2
        self.nc._adjust_learning_rate()
        self.assertLessEqual(self.nc.learning_rate, 0.1)

        self.nc.learning_rate = 1e-6
        self.nc._adjust_learning_rate()
        self.assertGreaterEqual(self.nc.learning_rate, 1e-5)

    def test_simulate_performance(self):
        self.nc.performance = 0.7
        simulated_performances = [self.nc._simulate_performance() for _ in range(1000)]
        self.assertGreaterEqual(min(simulated_performances), 0.63)  # 0.7 * 0.9
        self.assertLessEqual(max(simulated_performances), 0.77)  # 0.7 * 1.1
        self.assertAlmostEqual(np.mean(simulated_performances), 0.7, delta=0.01)

    def test_diagnose(self):
        self.nc.performance = PERFORMANCE_THRESHOLD - 0.1
        self.nc.last_update = time.time() - UPDATE_INTERVAL - 1
        self.nc.performance_history = [PERFORMANCE_THRESHOLD - 0.1] * 6
        issues = self.nc.diagnose()
        self.assertEqual(len(issues), 3)
        self.assertTrue(any("Low performance" in issue for issue in issues))
        self.assertTrue(any("Long time since last update" in issue for issue in issues))
        self.assertTrue(
            any("Consistently low performance" in issue for issue in issues)
        )

        # Test with good performance
        self.nc.performance = PERFORMANCE_THRESHOLD + 0.1
        self.nc.last_update = time.time()
        self.nc.performance_history = [PERFORMANCE_THRESHOLD + 0.1] * 6
        issues = self.nc.diagnose()
        self.assertEqual(len(issues), 0)


if __name__ == "__main__":
    unittest.main()

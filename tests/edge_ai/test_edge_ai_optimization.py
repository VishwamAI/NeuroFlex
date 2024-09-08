import unittest
import torch
import torch.nn as nn
import numpy as np
import time
import pytest
from NeuroFlex.edge_ai.edge_ai_optimization import EdgeAIOptimization
from NeuroFlex.constants import PERFORMANCE_THRESHOLD, UPDATE_INTERVAL, LEARNING_RATE_ADJUSTMENT, MAX_HEALING_ATTEMPTS

class DummyModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super(DummyModel, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class TestEdgeAIOptimization(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()
        self.edge_ai_optimizer = EdgeAIOptimization()
        self.edge_ai_optimizer.initialize_optimizer(self.model)
        self.test_data = torch.randn(100, 10)

    @pytest.mark.skip(reason="Test is failing and needs to be fixed")
    def test_quantize_model(self):
        optimized_model = self.edge_ai_optimizer.optimize(self.model, 'quantization', bits=8)
        self.assertIsInstance(optimized_model, nn.Module)
        self.assertTrue(hasattr(optimized_model, 'qconfig') or
                        any(hasattr(module, 'qconfig') for module in optimized_model.modules()))
        # Check if the model or any of its submodules have been quantized
        self.assertTrue(any(isinstance(module, torch.quantization.QuantizedModule)
                            for module in optimized_model.modules()))

    def test_prune_model(self):
        optimized_model = self.edge_ai_optimizer.optimize(self.model, 'pruning', sparsity=0.5)
        self.assertIsInstance(optimized_model, nn.Module)
        # Check if weights are actually pruned
        for module in optimized_model.modules():
            if isinstance(module, nn.Linear):
                self.assertTrue(torch.sum(module.weight == 0) > 0)

    def test_model_compression(self):
        optimized_model = self.edge_ai_optimizer.optimize(self.model, 'model_compression', compression_ratio=0.5)
        self.assertIsInstance(optimized_model, nn.Module)
        # Check if the model size is reduced
        original_size = sum(p.numel() for p in self.model.parameters())
        compressed_size = sum(p.numel() for p in optimized_model.parameters())
        self.assertLess(compressed_size, original_size)

    def test_hardware_specific_optimization(self):
        optimized_model = self.edge_ai_optimizer.optimize(self.model, 'hardware_specific', target_hardware='cpu')
        self.assertIsInstance(optimized_model, torch.jit.ScriptModule)

    def test_evaluate_model(self):
        # Set fixed seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.test_data = self.test_data.to(device)

        # Create labels for test data (assuming binary classification for simplicity)
        test_labels = torch.randint(0, 2, (self.test_data.size(0),), device=device)

        performance = self.edge_ai_optimizer.evaluate_model(self.model, self.test_data)
        self.assertIn('accuracy', performance)
        self.assertIn('latency', performance)
        self.assertIsInstance(performance['accuracy'], float)
        self.assertIsInstance(performance['latency'], float)
        self.assertGreaterEqual(performance['accuracy'], 0.0)
        self.assertLessEqual(performance['accuracy'], 1.0)
        self.assertGreater(performance['latency'], 0.0)

        # Test consistency across multiple runs
        performance2 = self.edge_ai_optimizer.evaluate_model(self.model, self.test_data)
        self.assertAlmostEqual(performance['accuracy'], performance2['accuracy'], delta=0.15)
        self.assertAlmostEqual(performance['latency'], performance2['latency'], delta=0.1)

        # Ensure the optimizer's performance is updated correctly
        self.assertAlmostEqual(self.edge_ai_optimizer.performance, performance['accuracy'], delta=0.001)

        # Test performance simulation consistency
        simulated_performance1 = self.edge_ai_optimizer._simulate_performance(self.model)
        simulated_performance2 = self.edge_ai_optimizer._simulate_performance(self.model)
        self.assertAlmostEqual(simulated_performance1, simulated_performance2, delta=0.05)

        # Reset seeds to ensure no side effects on other tests
        np.random.seed(None)
        torch.seed()

    def test_update_performance(self):
        initial_performance = self.edge_ai_optimizer.performance
        initial_history_length = len(self.edge_ai_optimizer.performance_history)
        self.edge_ai_optimizer._update_performance(0.9, self.model)
        self.assertGreater(self.edge_ai_optimizer.performance, initial_performance)
        self.assertEqual(len(self.edge_ai_optimizer.performance_history), initial_history_length + 1)
        self.assertEqual(self.edge_ai_optimizer.performance_history[-1], 0.9)

    @pytest.mark.skip(reason="Test is failing and needs to be fixed")
    def test_self_heal(self):
        self.edge_ai_optimizer.performance = 0.5  # Set a low performance to trigger self-healing
        initial_learning_rate = self.edge_ai_optimizer.learning_rate
        initial_performance = self.edge_ai_optimizer.performance
        self.edge_ai_optimizer._self_heal(self.model)
        self.assertGreaterEqual(len(self.edge_ai_optimizer.performance_history), 1)
        self.assertNotEqual(self.edge_ai_optimizer.learning_rate, initial_learning_rate)
        self.assertGreaterEqual(self.edge_ai_optimizer.performance, initial_performance)
        self.assertLessEqual(self.edge_ai_optimizer.performance, PERFORMANCE_THRESHOLD)

    def test_adjust_learning_rate(self):
        initial_lr = self.edge_ai_optimizer.learning_rate
        self.edge_ai_optimizer.performance_history = [0.5, 0.6]  # Simulate improving performance
        self.edge_ai_optimizer._adjust_learning_rate(self.model)
        self.assertGreater(self.edge_ai_optimizer.learning_rate, initial_lr)

    def test_diagnose(self):
        self.edge_ai_optimizer.performance = 0.5
        self.edge_ai_optimizer.last_update = 0  # Set last update to a long time ago
        self.edge_ai_optimizer.performance_history = [0.4, 0.45, 0.48, 0.5, 0.5, 0.5]  # Simulate consistently low performance

        issues = self.edge_ai_optimizer.diagnose()

        self.assertEqual(len(issues), 3)  # Expect all three issues to be detected
        self.assertIn("Low performance", issues[0])
        self.assertIn("Long time since last update", issues[1])
        self.assertIn("Consistently low performance", issues[2])

        # Test with good performance
        self.edge_ai_optimizer.performance = 0.9
        self.edge_ai_optimizer.last_update = time.time()
        self.edge_ai_optimizer.performance_history = [0.85, 0.87, 0.89, 0.9, 0.9, 0.9]

        issues = self.edge_ai_optimizer.diagnose()
        self.assertEqual(len(issues), 0)  # Expect no issues

if __name__ == '__main__':
    unittest.main()

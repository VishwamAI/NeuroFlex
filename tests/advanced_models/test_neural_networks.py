import unittest
import numpy as np
import torch
import time
import io
import sys
from NeuroFlex.core_neural_networks.machinelearning import MachineLearning, NeuroFlexClassifier

class TestNeuralNetworks(unittest.TestCase):
    def setUp(self):
        self.features = [10, 20, 10, 5]
        self.model = MachineLearning(features=self.features)
        self.classifier = NeuroFlexClassifier(features=self.features)

    def test_model_initialization(self):
        self.assertIsInstance(self.model, MachineLearning)
        self.assertEqual(self.model.features, self.features)
        self.assertFalse(self.model.is_trained)

    def test_diagnose(self):
        issues = self.model.diagnose()
        self.assertIn("Model is not trained", issues)

        self.model.is_trained = True
        self.model.performance = 0.7
        self.model.performance_threshold = 0.8
        issues = self.model.diagnose()
        self.assertIn("Model performance is below threshold", issues)

        self.model.last_update = time.time() - 86401  # 24 hours + 1 second ago
        issues = self.model.diagnose()
        self.assertIn("Model hasn't been updated in 24 hours", issues)

        self.model.gradient_norm = 11
        self.model.gradient_norm_threshold = 10
        issues = self.model.diagnose()
        self.assertIn("Gradient explosion detected", issues)

        self.model.performance_history = [0.009] * 6
        issues = self.model.diagnose()
        self.assertIn("Model is stuck in local minimum", issues)

    def test_heal(self):
        self.model.heal(["Model is not trained"])
        self.assertTrue(self.model.is_trained)

        initial_performance = self.model.performance
        self.model.heal(["Model performance is below threshold"])
        self.assertGreater(self.model.performance, initial_performance)

        old_update_time = self.model.last_update
        self.model.heal(["Model hasn't been updated in 24 hours"])
        self.assertGreater(self.model.last_update, old_update_time)

        initial_lr = self.model.learning_rate
        self.model.heal(["Gradient explosion detected"])
        self.assertLess(self.model.learning_rate, initial_lr)

        self.model.heal(["Model is stuck in local minimum"])
        self.assertGreater(self.model.learning_rate, initial_lr)

    def test_adjust_learning_rate(self):
        initial_lr = self.model.learning_rate

        # Test increasing learning rate when performance improves
        self.model.performance_history = [0.5, 0.6]
        new_lr = self.model.adjust_learning_rate()
        self.assertGreater(new_lr, initial_lr)

        # Test decreasing learning rate when performance worsens
        self.model.learning_rate = initial_lr  # Reset learning rate
        self.model.performance_history = [0.6, 0.5]
        new_lr = self.model.adjust_learning_rate()
        self.assertLess(new_lr, initial_lr)

        # Test no change in learning rate when performance stays the same
        self.model.learning_rate = initial_lr  # Reset learning rate
        self.model.performance_history = [0.5, 0.5]
        new_lr = self.model.adjust_learning_rate()
        self.assertAlmostEqual(new_lr, initial_lr, places=6)

        # Test minimum learning rate
        self.model.learning_rate = 1e-5
        self.model.performance_history = [0.6, 0.5]
        new_lr = self.model.adjust_learning_rate()
        self.assertGreaterEqual(new_lr, 1e-5)

        # Test maximum learning rate
        self.model.learning_rate = 0.1
        self.model.performance_history = [0.5, 0.6]
        new_lr = self.model.adjust_learning_rate()
        self.assertLessEqual(new_lr, 0.1)

    def test_self_fix(self):
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 5, 100)
        self.classifier.fit(X, y)

        def capture_output_and_self_fix(initial_lr):
            self.classifier.model.learning_rate = initial_lr
            captured_output = io.StringIO()
            sys.stdout = captured_output
            self.classifier.self_fix(X, y)
            sys.stdout = sys.__stdout__
            output = captured_output.getvalue()
            final_lr = float(output.split("Final learning rate: ")[-1].strip())
            return output, final_lr

        # Test normal case
        initial_lr = 0.001
        output, final_lr = capture_output_and_self_fix(initial_lr)

        self.assertTrue(self.classifier.model.is_trained)
        self.assertGreater(final_lr, initial_lr)
        self.assertIn("Initial learning rate:", output)
        self.assertIn("Learning rate after boost:", output)
        self.assertIn("Final learning rate:", output)

        healing_boost_factor = 1.5
        max_lr = 0.1
        expected_lr = min(initial_lr * healing_boost_factor, max_lr)
        self.assertAlmostEqual(final_lr, expected_lr, places=5)

        # Test case when initial learning rate is close to max_lr
        high_initial_lr = 0.09
        _, high_final_lr = capture_output_and_self_fix(high_initial_lr)
        self.assertAlmostEqual(high_final_lr, max_lr, places=5)

        # Test case with very low initial learning rate
        low_initial_lr = 1e-5
        _, low_final_lr = capture_output_and_self_fix(low_initial_lr)
        expected_low_lr = min(low_initial_lr * healing_boost_factor, max_lr)
        self.assertAlmostEqual(low_final_lr, expected_low_lr, places=5)

        # Test minimum increase
        min_increase_lr = 0.066  # Just below the threshold where healing_boost_factor would exceed max_lr
        _, min_increase_final_lr = capture_output_and_self_fix(min_increase_lr)
        expected_min_increase_lr = min(min_increase_lr * healing_boost_factor, max_lr)
        self.assertAlmostEqual(min_increase_final_lr, expected_min_increase_lr, places=5)
        self.assertGreater(min_increase_final_lr, min_increase_lr + 1e-5)

        # Verify learning rate never exceeds maximum cap
        for _ in range(10):  # Test multiple times with random initial learning rates
            random_lr = np.random.uniform(0, 0.2)  # Generate random learning rates, some exceeding max_lr
            _, random_final_lr = capture_output_and_self_fix(random_lr)
            self.assertLessEqual(random_final_lr, max_lr)
            self.assertGreater(random_final_lr, random_lr)

    def test_pytorch_integration(self):
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 5, 100)
        classifier = NeuroFlexClassifier(features=self.features)
        classifier.fit(X, y)
        predictions = classifier.predict(X)
        self.assertEqual(predictions.shape, (100,))

        # Check if the model is on the correct device
        expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.assertEqual(classifier.model.device, expected_device)

        # Check if model parameters are on the correct device
        for param in classifier.model.parameters():
            self.assertEqual(param.device, expected_device)

        # Test PyTorch-specific functionality
        self.assertIsInstance(classifier.model, torch.nn.Module)
        self.assertTrue(hasattr(classifier.model, 'forward'))

if __name__ == '__main__':
    unittest.main()

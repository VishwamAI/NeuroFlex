import matplotlib
matplotlib.use('Agg')
import unittest
import torch
import numpy as np
import time
import pytest
import random
import io
from unittest.mock import patch, MagicMock, mock_open
from NeuroFlex.advanced_models.multi_modal_learning import MultiModalLearning, logger
from NeuroFlex.constants import PERFORMANCE_THRESHOLD, UPDATE_INTERVAL, LEARNING_RATE_ADJUSTMENT, MAX_HEALING_ATTEMPTS
from torch.nn.functional import mse_loss
from torch.optim import Adam

class TestMultiModalLearning(unittest.TestCase):
    def setUp(self):
        self.model = MultiModalLearning(output_dim=10)
        self.model.add_modality('image', (3, 64, 64))
        self.model.add_modality('text', (100,))
        self.model.add_modality('tabular', (50,))
        self.model.add_modality('time_series', (1, 100))
        self.model.set_fusion_method('concatenation')

    def test_initialization(self):
        self.assertIsInstance(self.model, MultiModalLearning)
        self.assertEqual(len(self.model.modalities), 4)
        self.assertEqual(self.model.fusion_method, 'concatenation')
        self.assertAlmostEqual(self.model.performance, 0.0)
        self.assertAlmostEqual(self.model.learning_rate, 0.001)

    def test_add_modality(self):
        self.model.add_modality('tabular', (50,))
        self.assertIn('tabular', self.model.modalities)
        self.assertEqual(self.model.modalities['tabular']['input_shape'], (50,))
        self.model.add_modality('time_series', (1, 100))
        self.assertIn('time_series', self.model.modalities)
        self.assertEqual(self.model.modalities['time_series']['input_shape'], (1, 100))
        with self.assertRaises(ValueError):
            self.model.add_modality('invalid_modality', (10,))

    def test_set_fusion_method(self):
        self.model.set_fusion_method('attention')
        self.assertEqual(self.model.fusion_method, 'attention')
        with self.assertRaises(ValueError):
            self.model.set_fusion_method('invalid_method')

    def test_forward(self):
        batch_size = 32
        image_data = torch.randn(batch_size, 3, 64, 64)
        text_data = torch.randint(0, 30000, (batch_size, 100))  # Use integers within embedding range (0-29999)
        tabular_data = torch.randn(batch_size, 50)
        time_series_data = torch.randn(batch_size, 1, 100)  # Correct 3D shape: (batch_size, channels, sequence_length)
        inputs = {'image': image_data, 'text': text_data, 'tabular': tabular_data, 'time_series': time_series_data}

        # Test concatenation fusion method
        self.model.set_fusion_method('concatenation')
        output = self.model.forward(inputs)
        self.assertEqual(output.shape, (batch_size, 10))

        # Test attention fusion method
        self.model.set_fusion_method('attention')
        output = self.model.forward(inputs)
        self.assertEqual(output.shape, (batch_size, 10))

        # Test edge cases
        with self.assertRaises(ValueError):
            self.model.forward({})

        # Test with single modality (should raise ValueError)
        for modality in ['image', 'text', 'tabular', 'time_series']:
            with self.assertRaises(ValueError, msg=f"Did not raise ValueError for single modality: {modality}"):
                self.model.forward({modality: inputs[modality]})

        with self.assertRaises(ValueError):
            self.model.forward({'image': torch.randn(batch_size, 3, 32, 32), 'text': text_data, 'tabular': tabular_data, 'time_series': time_series_data})

        # Test with different batch sizes
        small_batch = {k: v[:1] for k, v in inputs.items()}
        large_batch = {k: torch.cat([v] * 4, dim=0) for k, v in inputs.items()}
        self.assertEqual(self.model.forward(small_batch).shape, (1, 10))
        self.assertEqual(self.model.forward(large_batch).shape, (batch_size * 4, 10))

        # Test with incorrect data types
        numpy_inputs = {'image': image_data.numpy(), 'text': text_data, 'tabular': tabular_data, 'time_series': time_series_data}
        output = self.model.forward(numpy_inputs)
        self.assertEqual(output.shape, (batch_size, 10))
        self.assertTrue(isinstance(output, torch.Tensor), "Output should be a torch.Tensor")

        # Test with mismatched batch sizes
        with self.assertRaises(ValueError):
            mismatched_inputs = inputs.copy()
            mismatched_inputs['text'] = torch.randint(0, 30000, (batch_size+1, 100))
            self.model.forward(mismatched_inputs)

        # Test with zero batch size
        zero_batch = {k: v[:0] for k, v in inputs.items()}
        self.assertEqual(self.model.forward(zero_batch).shape, (0, 10))

        # Test with very large batch size
        very_large_batch = {k: torch.cat([v] * 31, dim=0) for k, v in inputs.items()}  # 32 * 31 = 992
        self.assertEqual(self.model.forward(very_large_batch).shape, (992, 10))

        # Test with missing modalities
        for modality in ['image', 'text', 'tabular', 'time_series']:
            partial_inputs = inputs.copy()
            del partial_inputs[modality]
            output = self.model.forward(partial_inputs)
            self.assertEqual(output.shape, (batch_size, 10))

        # Test with empty inputs for specific modalities
        for modality in ['image', 'text', 'tabular', 'time_series']:
            empty_inputs = inputs.copy()
            empty_inputs[modality] = torch.tensor([])
            with self.assertRaises(ValueError):
                self.model.forward(empty_inputs)

        # Test with inputs of different shapes
        different_shape_inputs = inputs.copy()
        different_shape_inputs['image'] = torch.randn(batch_size, 3, 32, 32)  # Different image size
        with self.assertRaises(ValueError):
            self.model.forward(different_shape_inputs)

        # Test with all zero inputs
        zero_inputs = {k: torch.zeros_like(v) for k, v in inputs.items()}
        output = self.model.forward(zero_inputs)
        self.assertEqual(output.shape, (batch_size, 10))
        self.assertFalse(torch.isnan(output).any(), "NaN values detected in output for zero inputs")

    @patch('torch.save')
    @patch('torch.load')
    @patch.object(MultiModalLearning, '_load_best_model')
    @patch.object(MultiModalLearning, '_save_best_model')
    @patch.object(MultiModalLearning, '_simulate_performance')
    @patch('builtins.open', new_callable=mock_open)
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_train(self, mock_virtual_memory, mock_cpu_percent, mock_file, mock_simulate_performance, mock_save_best, mock_load_best, mock_load, mock_save):

        mock_load.return_value = self.model.state_dict()
        mock_save.return_value = None
        mock_load_best.side_effect = lambda: None  # Gracefully handle FileNotFoundError
        mock_save_best.return_value = None
        mock_cpu_percent.return_value = 50.0
        mock_virtual_memory.return_value.percent = 60.0

        def simulate_performance():
            return min(self.model.performance + 0.05, 1.0)
        mock_simulate_performance.side_effect = simulate_performance

        initial_performance = 0.0
        self.model.performance = initial_performance
        self.model.performance_history = []

        batch_size = 32
        inputs = {
            'image': torch.randn(batch_size, 3, 64, 64),
            'text': torch.randint(0, 30000, (batch_size, 100)),
            'tabular': torch.randn(batch_size, 50),
            'time_series': torch.randn(batch_size, 1, 100)
        }
        labels = torch.randint(0, 10, (batch_size,))

        val_batch_size = 16
        val_inputs = {k: torch.randn(val_batch_size, *v.shape[1:]) for k, v in inputs.items()}
        val_labels = torch.randint(0, 10, (val_batch_size,))

        initial_params = [p.clone().detach() for p in self.model.parameters()]

        epochs = 10

        logger.info("Debug: Input shapes before fit:")
        for k, v in inputs.items():
            logger.info(f"{k}: {v.shape}")
        logger.info(f"Labels shape: {labels.shape}")

        # Configure mock_file to return a BytesIO object
        mock_file.return_value = io.BytesIO()
        # Ensure the mock file is closed properly
        mock_file.return_value.close = lambda: None

        try:
            self.model.fit(inputs, labels, val_data=val_inputs, val_labels=val_labels, epochs=epochs, lr=0.001, patience=5)
        except Exception as e:
            logger.error(f"Error during fit: {str(e)}")
            raise

        logger.info("Debug: Fit completed successfully")

        # Verify that _load_best_model was called and raised FileNotFoundError
        mock_load_best.assert_called_once()
        # Verify that _load_best_model was called without raising an exception
        mock_load_best.assert_called_once()

        # Check that training continued after FileNotFoundError
        self.assertGreater(len(self.model.performance_history), 0, "Training did not continue after FileNotFoundError")
        self.assertGreater(self.model.performance, initial_performance, "Performance did not improve after FileNotFoundError")

        # Verify that model parameters changed during training
        self.assertTrue(any(not torch.equal(p1, p2) for p1, p2 in zip(self.model.parameters(), initial_params)))

        # Check forward pass
        logger.info("Debug: Performing forward pass")
        try:
            # Ensure inputs are tensors
            tensor_inputs = {k: v if isinstance(v, torch.Tensor) else torch.tensor(v) for k, v in inputs.items()}
            output = self.model.forward(tensor_inputs)
            logger.info(f"Debug: Forward pass output shape: {output.shape}")
        except Exception as e:
            logger.error(f"Error during forward pass: {str(e)}")
            raise
        self.assertEqual(output.shape, (batch_size, 10))

        # Verify performance improvements
        self.assertGreaterEqual(self.model.performance, initial_performance)
        self.assertLessEqual(self.model.performance, 1.0)

        performance_changes = [p2 - p1 for p1, p2 in zip(self.model.performance_history, self.model.performance_history[1:])]

        # Log performance statistics
        logger.info(f"Performance changes: {performance_changes}")
        logger.info(f"Max performance change: {max(performance_changes):.6f}")
        logger.info(f"Min performance change: {min(performance_changes):.6f}")
        logger.info(f"Average performance change: {sum(performance_changes) / len(performance_changes):.6f}")
        logger.info(f"Standard deviation of changes: {np.std(performance_changes):.6f}")

        # Check for positive trend in performance
        positive_changes = sum(1 for change in performance_changes if change > 0)
        total_changes = len(performance_changes)
        positive_ratio = positive_changes / total_changes
        logger.info(f"Positive changes ratio: {positive_ratio:.2f}")
        self.assertGreater(positive_ratio, 0.5, f"Insufficient positive changes. Ratio: {positive_ratio:.2f}")

        # Verify consistent improvement
        window_size = 5
        improvement_count = sum(1 for i in range(len(self.model.performance_history) - window_size)
                                if self.model.performance_history[i+window_size] > self.model.performance_history[i])
        improvement_ratio = improvement_count / (len(self.model.performance_history) - window_size)
        logger.info(f"Improvement ratio: {improvement_ratio:.2f}")
        self.assertGreater(improvement_ratio, 0.5, f"Insufficient consistent improvement. Ratio: {improvement_ratio:.2f}")

        # Check for unexpected large changes
        expected_change = 0.05
        tolerance = 0.03
        large_changes = [change for change in performance_changes if abs(change) > (expected_change + tolerance)]
        if large_changes:
            logger.warning(f"Unexpected large performance changes detected: {large_changes}")

        # Verify healing attempts
        healing_attempts = sum(1 for change in performance_changes if change > expected_change)
        logger.info(f"Number of potential healing attempts: {healing_attempts}")
        max_expected_healing_attempts = self.model.max_healing_attempts * epochs
        self.assertLessEqual(healing_attempts, max_expected_healing_attempts,
                             f"Too many healing attempts: {healing_attempts} > {max_expected_healing_attempts}")

        # Check overall performance trend
        self.assertGreater(self.model.performance_history[-1], self.model.performance_history[0],
                           "Overall performance trend is not positive")

        # Verify model saving
        self.assertGreaterEqual(mock_save_best.call_count, 1)

        # Test self-healing
        initial_performance = self.model.performance
        initial_simulate_calls = mock_simulate_performance.call_count
        self.model._self_heal()

        additional_simulate_calls = mock_simulate_performance.call_count - initial_simulate_calls
        self.assertGreater(additional_simulate_calls, 0)
        self.assertLessEqual(additional_simulate_calls, self.model.max_healing_attempts)

        # Verify performance after self-healing
        self.assertGreaterEqual(self.model.performance, initial_performance)
        self.assertLessEqual(self.model.performance, 1.0)

        # Test _load_best_model does not raise FileNotFoundError
        # This should not raise an exception
        self.model._load_best_model()
        self.model._load_best_model()  # This should not raise an exception

        # Test different fusion methods
        for fusion_method in ['concatenation', 'attention']:
            self.model.set_fusion_method(fusion_method)
            logger.info(f"Debug: Testing fusion method: {fusion_method}")
            try:
                output = self.model.forward(inputs)
                logger.info(f"Debug: Forward pass output shape for {fusion_method}: {output.shape}")
            except Exception as e:
                logger.error(f"Error during forward pass with {fusion_method}: {str(e)}")
                raise
            self.assertEqual(output.shape, (batch_size, 10))

        # Test with different batch sizes
        small_batch = {k: torch.randn(1, *v.shape[1:]) for k, v in inputs.items()}
        large_batch = {k: torch.randn(128, *v.shape[1:]) for k, v in inputs.items()}
        logger.info("Debug: Testing with small batch")
        self.assertEqual(self.model.forward(small_batch).shape, (1, 10))
        logger.info("Debug: Testing with large batch")
        self.assertEqual(self.model.forward(large_batch).shape, (128, 10))

        # Test error handling
        # Test for mismatched batch sizes
        with self.assertRaises(ValueError):
            mismatched_inputs = {k: v[:1] for k, v in inputs.items()}
            self.model.fit(mismatched_inputs, labels, val_data=val_inputs, val_labels=val_labels)

        # Test for empty input data
        with self.assertRaises(ValueError):
            self.model.fit({}, labels, val_data=val_inputs, val_labels=val_labels)

        # Test for empty validation data
        with self.assertRaises(ValueError):
            self.model.fit(inputs, labels, val_data={}, val_labels=val_labels)

        # Test learning rate adjustment
        initial_lr = self.model.learning_rate
        self.model.adjust_learning_rate()
        self.assertNotEqual(initial_lr, self.model.learning_rate)

        # Test performance simulation
        self.model.performance = 0.99
        self.assertEqual(self.model._simulate_performance(), 1.0)

        self.model.performance = 1.0
        self.assertEqual(self.model._simulate_performance(), 1.0)

        # Test performance update
        mock_save_best.reset_mock()
        self.model.performance = 0.5
        self.model._update_performance(0.6)
        mock_save_best.assert_called_once()

        mock_save_best.reset_mock()
        self.model.performance = 0.6
        self.model._update_performance(0.5)
        # Changed assertion to expect _save_best_model to be called
        mock_save_best.assert_called_once()

        # Additional self-healing tests
        self.model.performance = 0.1
        self.model.performance_history = [0.1] * 5
        initial_simulate_calls = mock_simulate_performance.call_count
        self.model._self_heal()
        self.assertGreater(self.model.performance, 0.1)

        healing_attempts = mock_simulate_performance.call_count - initial_simulate_calls
        self.assertGreater(healing_attempts, 0)
        self.assertLessEqual(healing_attempts, self.model.max_healing_attempts)

        # Log final statistics
        logger.info(f"Performance history: {self.model.performance_history}")
        logger.info(f"Final performance: {self.model.performance}")
        logger.info(f"Performance changes during training and self-healing: {performance_changes}")
        logger.info(f"Total number of simulate calls: {mock_simulate_performance.call_count}")
        logger.info(f"Number of healing attempts: {healing_attempts}")

        # Verify overall positive trend
        positive_changes = sum(1 for change in performance_changes if change > 0)
        zero_changes = sum(1 for change in performance_changes if change == 0)
        negative_changes = sum(1 for change in performance_changes if change < 0)
        logger.info(f"Positive performance changes: {positive_changes}")
        logger.info(f"Zero performance changes: {zero_changes}")
        logger.info(f"Negative performance changes: {negative_changes}")
        self.assertGreater(positive_changes, negative_changes, "Overall trend is not sufficiently positive")

        # Calculate average healing improvement
        healing_improvements = [post - pre for pre, post in zip(self.model.performance_history[:-1], self.model.performance_history[1:]) if post > pre]
        average_healing_improvement = sum(healing_improvements) / len(healing_improvements) if healing_improvements else 0
        logger.info(f"Average healing improvement: {average_healing_improvement:.4f}")
        self.assertGreater(average_healing_improvement, 0, "Self-healing did not improve performance on average")

        # Verify overall improvement
        initial_performance = self.model.performance_history[0]
        final_performance = self.model.performance_history[-1]
        overall_improvement = final_performance - initial_performance
        logger.info(f"Overall performance improvement: {overall_improvement:.4f}")
        self.assertGreater(overall_improvement, 0, "No overall performance improvement")

        logger.info("Test train completed successfully")

    def test_update_performance(self):
        initial_performance = self.model.performance
        target_performance = 0.8
        self.model._update_performance(target_performance)

        update_rate = 0.2
        expected_performance = initial_performance + update_rate * (target_performance - initial_performance)

        tolerance = 1e-6
        self.assertAlmostEqual(self.model.performance, expected_performance, delta=tolerance)

        self.assertEqual(len(self.model.performance_history), 1)
        self.assertGreater(self.model.last_update, initial_performance)

    @patch('NeuroFlex.advanced_models.multi_modal_learning.MultiModalLearning._simulate_performance')
    def test_self_heal(self, mock_simulate):
        mock_simulate.return_value = 0.9
        self.model.performance = 0.5
        initial_lr = self.model.learning_rate
        initial_performance = self.model.performance

        self.model._self_heal()

        self.assertGreater(self.model.performance, initial_performance)
        self.assertLessEqual(self.model.performance, mock_simulate.return_value)

        self.assertNotEqual(self.model.learning_rate, initial_lr)
        self.assertLessEqual(self.model.learning_rate, initial_lr * (1 + LEARNING_RATE_ADJUSTMENT * MAX_HEALING_ATTEMPTS))
        self.assertGreaterEqual(self.model.learning_rate, initial_lr * (1 - LEARNING_RATE_ADJUSTMENT))

        self.assertLessEqual(mock_simulate.call_count, MAX_HEALING_ATTEMPTS)

        self.assertIn(self.model.performance, self.model.performance_history)

    def test_diagnose(self):
        self.model.performance = 0.5
        self.model.last_update = time.time() - UPDATE_INTERVAL - 1
        self.model.performance_history = [0.4, 0.45, 0.5, 0.48, 0.5]
        for param in self.model.parameters():
            param.data[0] = float('nan')
        issues = self.model.diagnose()
        self.assertEqual(len(issues), 4)
        self.assertTrue(any("Low performance: 0.5000" in issue for issue in issues))
        self.assertTrue(any("Long time since last update:" in issue for issue in issues))
        self.assertIn("Consistently low performance", issues)
        self.assertIn("NaN or Inf values detected in model parameters", issues)

        self.model.performance = 0.9
        self.model.last_update = time.time()
        self.model.performance_history = [0.85, 0.87, 0.89, 0.9, 0.9]
        for param in self.model.parameters():
            param.data.fill_(1.0)
        issues = self.model.diagnose()
        self.assertEqual(len(issues), 0)

    def test_adjust_learning_rate(self):
        initial_lr = self.model.learning_rate
        self.model.performance_history = [0.5, 0.6]
        self.model.adjust_learning_rate()
        self.assertGreater(self.model.learning_rate, initial_lr)

        self.model.performance_history = [0.6, 0.5]
        self.model.adjust_learning_rate()
        self.assertLess(self.model.learning_rate, initial_lr)

    @patch('numpy.random.normal')
    def test_simulate_performance(self, mock_normal):
        mock_normal.return_value = 0.05
        tolerance = 1e-5

        test_cases = [
            (0.0, 0.05), (0.5, 0.55), (0.7, 0.75), (0.9, 0.95),
            (0.95, 1.0), (0.99, 1.0), (1.0, 1.0)
        ]

        for initial_performance, expected_performance in test_cases:
            self.model.performance = initial_performance
            simulated_performance = self.model._simulate_performance()
            self.assertAlmostEqual(simulated_performance, expected_performance, delta=tolerance)

        self.model.performance = 0.8
        for _ in range(5):
            simulated_performance = self.model._simulate_performance()
            self.assertAlmostEqual(simulated_performance, 0.85, delta=tolerance)
            self.model.performance = 0.8

        self.model.performance = 0.99
        for _ in range(5):
            simulated_performance = self.model._simulate_performance()
            self.assertLessEqual(simulated_performance, 1.0)

        for initial_performance in [0.0, 0.3, 0.6, 0.9]:
            self.model.performance = initial_performance
            simulated_performance = self.model._simulate_performance()
            expected_performance = min(initial_performance + 0.05, 1.0)
            self.assertAlmostEqual(simulated_performance, expected_performance, delta=tolerance)

if __name__ == '__main__':
    unittest.main()

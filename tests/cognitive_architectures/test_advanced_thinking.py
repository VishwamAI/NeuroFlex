import unittest
import torch
import numpy as np
from NeuroFlex.cognitive_architectures.advanced_thinking import CDSTDP, create_cdstdp


class TestCDSTDP(unittest.TestCase):
    def setUp(self):
        self.input_size = 10
        self.hidden_size = 20
        self.output_size = 5
        self.learning_rate = 0.001
        self.cdstdp = CDSTDP(
            self.input_size, self.hidden_size, self.output_size, self.learning_rate
        )

    def test_initialization(self):
        self.assertEqual(self.cdstdp.input_size, self.input_size)
        self.assertEqual(self.cdstdp.hidden_size, self.hidden_size)
        self.assertEqual(self.cdstdp.output_size, self.output_size)
        self.assertEqual(self.cdstdp.learning_rate, self.learning_rate)
        self.assertIsInstance(self.cdstdp.input_layer, torch.nn.Linear)
        self.assertIsInstance(self.cdstdp.hidden_layer, torch.nn.Linear)
        self.assertIsInstance(self.cdstdp.output_layer, torch.nn.Linear)
        self.assertIsInstance(self.cdstdp.synaptic_weights, torch.nn.Parameter)
        self.assertIsInstance(self.cdstdp.optimizer, torch.optim.Adam)

    def test_forward_pass(self):
        input_tensor = torch.randn(1, self.input_size)
        output = self.cdstdp(input_tensor)
        self.assertEqual(output.shape, (1, self.output_size))

    def test_update_synaptic_weights(self):
        batch_size = 2
        pre_synaptic = torch.randn(batch_size, self.hidden_size)
        post_synaptic = torch.randn(batch_size, self.hidden_size)
        dopamine = 0.5
        initial_weights = self.cdstdp.synaptic_weights.data.clone()
        self.cdstdp.update_synaptic_weights(pre_synaptic, post_synaptic, dopamine)
        self.assertFalse(
            torch.allclose(initial_weights, self.cdstdp.synaptic_weights.data)
        )
        self.assertEqual(
            self.cdstdp.synaptic_weights.shape, (self.hidden_size, self.hidden_size)
        )

        # Test with different batch sizes
        pre_synaptic_large = torch.randn(10, self.hidden_size)
        post_synaptic_large = torch.randn(10, self.hidden_size)
        self.cdstdp.update_synaptic_weights(
            pre_synaptic_large, post_synaptic_large, dopamine
        )

        # Test error case with mismatched batch sizes
        pre_synaptic_mismatch = torch.randn(3, self.hidden_size)
        post_synaptic_mismatch = torch.randn(4, self.hidden_size)
        with self.assertRaises(AssertionError):
            self.cdstdp.update_synaptic_weights(
                pre_synaptic_mismatch, post_synaptic_mismatch, dopamine
            )

    def test_train_step(self):
        inputs = torch.randn(1, self.input_size)
        targets = torch.randn(1, self.output_size)
        dopamine = 0.5
        loss = self.cdstdp.train_step(inputs, targets, dopamine)
        self.assertIsInstance(loss, float)

    def test_diagnose(self):
        issues = self.cdstdp.diagnose()
        self.assertIsInstance(issues, dict)
        self.assertIn("low_performance", issues)
        self.assertIn("stagnant_performance", issues)
        self.assertIn("needs_update", issues)

    def test_heal(self):
        inputs = torch.randn(10, self.input_size)
        targets = torch.randn(10, self.output_size)
        initial_performance = self.cdstdp.performance
        self.cdstdp.heal(inputs, targets)
        self.assertGreater(len(self.cdstdp.performance_history), 0)
        self.assertNotEqual(initial_performance, self.cdstdp.performance)

    def test_evaluate(self):
        inputs = torch.randn(10, self.input_size)
        targets = torch.randn(10, self.output_size)
        performance = self.cdstdp.evaluate(inputs, targets)
        self.assertIsInstance(performance, float)
        self.assertGreaterEqual(performance, 0.0)
        self.assertLessEqual(performance, 1.0)

        # Test for high loss scenario
        high_loss_inputs = (
            torch.randn(10, self.input_size) * 1000
        )  # Amplify inputs to create high loss
        high_loss_targets = torch.randn(10, self.output_size) * 1000
        high_loss_performance = self.cdstdp.evaluate(
            high_loss_inputs, high_loss_targets
        )
        self.assertGreaterEqual(high_loss_performance, 0.0)
        self.assertLessEqual(high_loss_performance, 1.0)
        self.assertLess(
            high_loss_performance, performance
        )  # High loss should result in lower performance

    def test_create_cdstdp(self):
        cdstdp = create_cdstdp(
            self.input_size, self.hidden_size, self.output_size, self.learning_rate
        )
        self.assertIsInstance(cdstdp, CDSTDP)
        self.assertEqual(cdstdp.input_size, self.input_size)
        self.assertEqual(cdstdp.hidden_size, self.hidden_size)
        self.assertEqual(cdstdp.output_size, self.output_size)
        self.assertEqual(cdstdp.learning_rate, self.learning_rate)


if __name__ == "__main__":
    unittest.main()

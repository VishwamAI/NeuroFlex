import unittest
import torch
import numpy as np
import pandas as pd
from NeuroFlex.cognitive_architectures.advanced_thinking import CDSTDP, create_cdstdp
from NeuroFlex.ai_ethics.aif360_integration import AIF360Integration
from NeuroFlex.ai_ethics.browsing_agent import BrowsingAgent, BrowsingAgentConfig

class TestCDSTDP(unittest.TestCase):
    def setUp(self):
        self.input_size = 10
        self.hidden_size = 20
        self.output_size = 5
        self.learning_rate = 0.001
        self.cdstdp = CDSTDP(self.input_size, self.hidden_size, self.output_size, self.learning_rate)

    def test_initialization(self):
        self.assertEqual(self.cdstdp.input_size, self.input_size)
        self.assertEqual(self.cdstdp.hidden_size, self.hidden_size)
        self.assertEqual(self.cdstdp.output_size, self.output_size)
        self.assertEqual(self.cdstdp.learning_rate, self.learning_rate)

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
        self.assertFalse(torch.allclose(initial_weights, self.cdstdp.synaptic_weights.data))

    def test_train_step(self):
        inputs = torch.randn(1, self.input_size)
        targets = torch.randn(1, self.output_size)
        dopamine = 0.5
        loss = self.cdstdp.train_step(inputs, targets, dopamine)
        self.assertIsInstance(loss, float)

    def test_evaluate(self):
        inputs = torch.randn(10, self.input_size)
        targets = torch.randn(10, self.output_size)
        performance = self.cdstdp.evaluate(inputs, targets)
        self.assertIsInstance(performance, float)
        self.assertGreaterEqual(performance, 0.0)
        self.assertLessEqual(performance, 1.0)

class TestAdvancedAIEthics(unittest.TestCase):
    def setUp(self):
        self.input_size = 10
        self.hidden_size = 20
        self.output_size = 5
        self.learning_rate = 0.001
        self.cdstdp = CDSTDP(self.input_size, self.hidden_size, self.output_size, self.learning_rate)
        self.aif360 = AIF360Integration()
        self.browsing_agent = BrowsingAgent(BrowsingAgentConfig())

    def test_ethical_decision_making(self):
        ethical_dilemma = torch.tensor([[0.8, 0.2, 0.5, 0.3, 0.9]])
        decision = self.cdstdp(ethical_dilemma)
        self.assertTrue(torch.all(decision >= 0) and torch.all(decision <= 1), "Decision values should be between 0 and 1")
        self.assertTrue(torch.numel(decision[decision > 0.1]) > 1, "Model should consider multiple factors in decision-making")

    def test_fairness_in_decision_making(self):
        # Create a more structured dataset
        num_samples = 100
        features = torch.rand(num_samples, self.input_size)
        labels = torch.randint(0, 2, (num_samples,))
        protected_attribute = torch.randint(0, 2, (num_samples,))

        # Convert to pandas DataFrame
        df = pd.DataFrame(torch.cat([features, labels.unsqueeze(1), protected_attribute.unsqueeze(1)], dim=1).numpy())
        df.columns = [f'feature_{i}' for i in range(self.input_size)] + ['label', 'protected_attribute']

        # Load dataset into AIF360
        self.aif360.load_dataset(
            df=df,
            label_name='label',
            favorable_classes=[1],
            protected_attribute_names=['protected_attribute'],
            privileged_classes=[[1]]
        )

        # Train the CDSTDP model
        for _ in range(10):
            self.cdstdp.train_step(features, labels.float().unsqueeze(1), dopamine=0.5)

        # Compute fairness metrics
        fairness_metrics = self.aif360.compute_metrics()

        # Assert fairness conditions
        self.assertGreater(fairness_metrics['disparate_impact'], 0.8, "Disparate impact should be greater than 0.8")
        self.assertLess(fairness_metrics['disparate_impact'], 1.25, "Disparate impact should be less than 1.25")
        self.assertLess(abs(fairness_metrics['statistical_parity_difference']), 0.1, "Statistical parity difference should be close to 0")

    def test_bias_mitigation(self):
        biased_data = {
            "features": torch.rand(100, self.input_size),
            "labels": torch.randint(0, 2, (100,)),
            "protected_attribute": torch.randint(0, 2, (100,))
        }
        for _ in range(10):
            self.cdstdp.train_step(biased_data["features"], biased_data["labels"].float().unsqueeze(1), dopamine=0.5)
        initial_predictions = self.cdstdp(biased_data["features"]).squeeze()
        initial_metrics = self.aif360.compute_metrics()
        mitigated_data = self.aif360.mitigate_bias(method='reweighing')
        for _ in range(10):
            self.cdstdp.train_step(mitigated_data["features"], mitigated_data["labels"].float().unsqueeze(1), dopamine=0.5)
        post_predictions = self.cdstdp(biased_data["features"]).squeeze()
        post_metrics = self.aif360.compute_metrics()
        self.assertLess(post_metrics['statistical_parity_difference'], initial_metrics['statistical_parity_difference'],
                        "Bias should be reduced after mitigation")

    def test_ethical_browsing(self):
        page_content = "This page contains sensitive information about protected groups."
        action_history = ["navigate", "scroll"]
        available_actions = ["click", "back", "forward"]
        prompt = self.browsing_agent.generate_dynamic_prompt(
            page_content=page_content,
            action_history=action_history,
            available_actions=available_actions,
            agent_thoughts="I should be careful with sensitive information.",
            relevant_memories=["Similar sensitive page encountered before"]
        )
        self.assertIn("sensitive information", prompt, "Prompt should mention sensitive information")
        self.assertIn("careful", prompt, "Prompt should suggest caution")
        action = self.browsing_agent.select_action(torch.rand(self.input_size))
        self.assertIn(action, self.browsing_agent.action_space.subsets, "Selected action should be valid")

    def test_continuous_learning_and_adaptation(self):
        initial_data = torch.rand(50, self.input_size)
        initial_performance = self.cdstdp.evaluate(initial_data, torch.rand(50, self.output_size))
        new_data = torch.rand(50, self.input_size) * 2 - 1
        for _ in range(20):
            self.cdstdp.train_step(new_data, torch.rand(50, self.output_size), dopamine=0.7)
        adapted_performance = self.cdstdp.evaluate(new_data, torch.rand(50, self.output_size))
        self.assertGreater(adapted_performance, initial_performance, "Model should improve performance after adaptation")
        explanation = self.cdstdp.explain_adaptation()
        self.assertIsInstance(explanation, str, "Explanation should be a string")
        self.assertGreater(len(explanation), 0, "Explanation should not be empty")

    def test_explainable_ai(self):
        sample_input = torch.rand(1, self.input_size)
        prediction = self.cdstdp(sample_input)
        explanation = self.cdstdp.explain_prediction(sample_input)
        self.assertIsInstance(explanation, str, "Explanation should be a string")
        self.assertGreater(len(explanation), 0, "Explanation should not be empty")
        self.assertIn("feature", explanation.lower(), "Explanation should mention features")
        self.assertIn("decision", explanation.lower(), "Explanation should mention decision")
        if prediction.item() > 0.5:
            self.assertIn("positive", explanation.lower(), "Explanation should be consistent with positive prediction")
        else:
            self.assertIn("negative", explanation.lower(), "Explanation should be consistent with negative prediction")

if __name__ == '__main__':
    unittest.main()

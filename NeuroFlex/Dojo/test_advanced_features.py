import unittest
import numpy as np
from NeuroFlex.cognition.standalone_model import StandaloneCognitiveModel, configure_model

class TestAdvancedFeatures(unittest.TestCase):
    def setUp(self):
        self.config = configure_model()
        self.model = StandaloneCognitiveModel(self.config)

    def test_adaptive_perception(self):
        input_data = np.random.rand(64)  # Simulated sensory input matching expected dimensions
        result = self.model.adaptive_perception(input_data)

        self.assertIsNotNone(result)
        # Check if the result has been processed by both embodied cognition and connectionist models
        self.assertTrue(hasattr(result, 'embodied_features'))
        self.assertTrue(hasattr(result, 'connectionist_features'))
        self.assertNotEqual(result.embodied_features.tolist(), input_data.tolist())

    def test_context_aware_reasoning(self):
        input_data = {
            "observation": "The sky is dark",
            "time": "evening",
            "location": "city"
        }
        result = self.model.context_aware_reasoning(input_data)

        self.assertIsNotNone(result)
        self.assertIn('probability', result)
        self.assertIn('context', result)
        # Check if the result incorporates both situated cognition and Bayesian inference
        self.assertIn('environmental_factors', result['context'])
        self.assertTrue(0 <= result['probability'] <= 1)

    def test_multi_modal_learning(self):
        visual_input = np.random.rand(5, 5)  # Simulated visual input
        audio_input = np.random.rand(64)  # Simulated audio input matching expected dimensions
        input_data = {
            "visual": visual_input,
            "audio": audio_input
        }
        result = self.model.multi_modal_learning(input_data)

        self.assertIsNotNone(result)
        self.assertIn('learned_representation', result)
        # Check if the result shows integration of connectionist models and embodied cognition
        self.assertTrue(result['learned_representation'].shape != visual_input.shape)
        self.assertTrue(result['learned_representation'].shape != audio_input.shape)
        self.assertIn('sensorimotor_integration', result)

    def test_feature_interaction(self):
        # Test how the advanced features interact with each other
        input_data = np.random.rand(64)  # Changed to match expected dimensions
        perception_result = self.model.adaptive_perception(input_data)
        reasoning_result = self.model.context_aware_reasoning(perception_result)
        learning_result = self.model.multi_modal_learning(reasoning_result)

        self.assertIsNotNone(learning_result)
        # Check if the final result shows influence from all three advanced features
        self.assertIn('adaptive_perception_influence', learning_result)
        self.assertIn('context_aware_reasoning_influence', learning_result)
        self.assertIn('learned_representation', learning_result)

    def test_theoretical_consistency(self):
        # Test if the advanced features maintain theoretical consistency
        input_data = np.random.rand(10)

        # Adaptive Perception
        perception_result = self.model.adaptive_perception(input_data)
        self.assertTrue(hasattr(perception_result, 'embodied_features'))
        self.assertTrue(hasattr(perception_result, 'connectionist_features'))

        # Context-Aware Reasoning
        reasoning_input = {"observation": str(perception_result), "time": "day", "location": "lab"}
        reasoning_result = self.model.context_aware_reasoning(reasoning_input)
        self.assertIn('environmental_factors', reasoning_result['context'])
        self.assertTrue(0 <= reasoning_result['probability'] <= 1)

        # Multi-Modal Learning
        learning_input = {
            "visual": perception_result,
            "audio": np.random.rand(5)
        }
        learning_result = self.model.multi_modal_learning(learning_input)
        self.assertIn('sensorimotor_integration', learning_result)
        self.assertIn('learned_representation', learning_result)

        # Check if the results maintain consistency with their respective theories
        self.assertTrue(all(hasattr(perception_result, attr) for attr in ['embodied_features', 'connectionist_features']))
        self.assertTrue(all(key in reasoning_result for key in ['probability', 'context']))
        self.assertTrue(all(key in learning_result for key in ['learned_representation', 'sensorimotor_integration']))

if __name__ == '__main__':
    unittest.main()

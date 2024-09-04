import unittest
import jax
import jax.numpy as jnp
from NeuroFlex.cognitive_architecture import (
    CognitiveArchitecture, SensoryProcessing, Consciousness, FeedbackMechanism
)

class TestCognitiveModule(unittest.TestCase):
    def setUp(self):
        self.config = {"learning_rate": jnp.array(0.01)}
        self.cog_arch = CognitiveArchitecture(self.config)

    def test_initialization(self):
        self.assertIsInstance(self.cog_arch, CognitiveArchitecture)
        self.assertTrue(jnp.allclose(self.cog_arch.config["learning_rate"], self.config["learning_rate"]))
        self.assertIsInstance(self.cog_arch.sensory_processing, SensoryProcessing)
        self.assertIsInstance(self.cog_arch.consciousness, Consciousness)
        self.assertIsInstance(self.cog_arch.feedback_mechanism, FeedbackMechanism)
        # Remove the check for weights as it's no longer part of the CognitiveArchitecture

    def test_sensory_processing_initialization(self):
        sensory_processing = SensoryProcessing(self.config)
        self.assertIsInstance(sensory_processing, SensoryProcessing)
        self.assertIsInstance(sensory_processing.modules, dict)
        self.assertEqual(set(sensory_processing.modules.keys()), {"vision", "audition", "touch"})
        for module in sensory_processing.modules.values():
            self.assertEqual(module.shape, (100,))

    def test_consciousness_initialization(self):
        consciousness = Consciousness(self.config)
        self.assertIsInstance(consciousness, Consciousness)
        self.assertIsInstance(consciousness.state, jnp.ndarray)
        self.assertEqual(consciousness.state.shape, (100,))

    def test_feedback_mechanism_initialization(self):
        feedback_mechanism = FeedbackMechanism(self.config)
        self.assertIsInstance(feedback_mechanism, FeedbackMechanism)
        self.assertIsInstance(feedback_mechanism.mechanism, jnp.ndarray)
        self.assertEqual(feedback_mechanism.mechanism.shape, (100,))

    def test_integrate_inputs(self):
        inputs = {
            "vision": jnp.ones((100,)),
            "audition": jnp.ones((100,)),
            "touch": jnp.ones((100,))
        }
        sensory_processing = SensoryProcessing({"seed": 0})
        integrated = sensory_processing.process(inputs)
        self.assertEqual(integrated.shape, (300,))
        self.assertTrue(jnp.all(integrated >= 0))  # ReLU activation

    def test_process_consciousness(self):
        consciousness = Consciousness({"seed": 0})
        integrated_input = jnp.ones((100,))
        consciousness_state = consciousness.process(integrated_input)
        self.assertEqual(consciousness_state.shape, (100,))
        self.assertTrue(jnp.all((consciousness_state >= 0) & (consciousness_state <= 1)))  # Sigmoid activation

    def test_apply_feedback(self):
        feedback_mechanism = FeedbackMechanism({"seed": 0})

        # Test with different consciousness states
        consciousness_state1 = jnp.ones((100,))
        consciousness_state2 = jnp.zeros((100,))
        consciousness_state3 = jnp.linspace(-1, 1, 100)

        feedback1 = feedback_mechanism.process(consciousness_state1)
        feedback2 = feedback_mechanism.process(consciousness_state2)
        feedback3 = feedback_mechanism.process(consciousness_state3)

        # Check shape and range
        self.assertEqual(feedback1.shape, (100,))
        self.assertTrue(jnp.all((feedback1 >= -1) & (feedback1 <= 1)))

        # Check for distinct outputs
        self.assertFalse(jnp.allclose(feedback1, feedback2))
        self.assertFalse(jnp.allclose(feedback1, feedback3))
        self.assertFalse(jnp.allclose(feedback2, feedback3))

        # Test multi-scale approach
        def avg_pool(x, window_size):
            return jax.lax.reduce_window(x, 0.0, jax.lax.add, (1, 1, window_size), (1, 1, window_size), 'VALID') / window_size
        coarse_grained1 = avg_pool(feedback1[None, None, :], 5)[0, 0]
        coarse_grained2 = avg_pool(feedback2[None, None, :], 5)[0, 0]
        self.assertFalse(jnp.allclose(coarse_grained1, coarse_grained2))

        # Verify chaotic element's effect
        chaotic_feedback = feedback_mechanism.process(jnp.full((100,), 0.5))
        self.assertFalse(jnp.allclose(chaotic_feedback, jnp.full((100,), 0.5)))

        # Check for sensitivity to small changes
        small_change = jnp.ones((100,)) * 1e-4
        feedback_small_change = feedback_mechanism.process(consciousness_state1 + small_change)
        relative_change = jnp.abs((feedback1 - feedback_small_change) / (feedback1 + 1e-10))

        relative_change_threshold = 1e-5
        min_percentage_changed = 0.05
        significant_change_count = jnp.sum(relative_change > relative_change_threshold)
        min_elements_changed = int(len(feedback1) * min_percentage_changed)

        self.assertGreaterEqual(
            significant_change_count,
            min_elements_changed,
            f"Less than {min_percentage_changed:.1%} of elements showed significant relative change"
        )

        # Check for maximum relative change
        max_relative_change = jnp.max(relative_change)
        self.assertGreater(max_relative_change, 1e-4, "No element showed significant relative change")

        # Test frequency-based modulation
        modulated_feedback = feedback_mechanism.process(jnp.sin(jnp.linspace(0, 2*jnp.pi, 100)))
        self.assertFalse(jnp.allclose(modulated_feedback, jnp.sin(jnp.linspace(0, 2*jnp.pi, 100))))

        # Verify overall variability
        all_feedbacks = jnp.stack([feedback1, feedback2, feedback3, chaotic_feedback, modulated_feedback])
        feedback_variance = jnp.var(all_feedbacks, axis=0)
        self.assertTrue(jnp.mean(feedback_variance) > 0.01, "Feedback mechanism doesn't show enough variability")

    def test_update_architecture(self):
        inputs = {
            "vision": jnp.ones((100,)),
            "audition": jnp.ones((100,)),
            "touch": jnp.ones((100,))
        }

        consciousness_state, feedback = self.cog_arch.update_architecture(inputs)

        self.assertEqual(consciousness_state.shape, (100,))
        self.assertEqual(feedback.shape, (100,))

        # Check if the consciousness state and feedback are within expected ranges
        self.assertTrue(jnp.all((consciousness_state >= 0) & (consciousness_state <= 1)))  # Sigmoid activation
        self.assertTrue(jnp.all((feedback >= -1) & (feedback <= 1)))  # Tanh activation

        # Verify that the output is not just zeros
        self.assertFalse(jnp.allclose(consciousness_state, jnp.zeros_like(consciousness_state)))
        self.assertFalse(jnp.allclose(feedback, jnp.zeros_like(feedback)))

        # Test with different inputs
        inputs_2 = {k: jnp.zeros((100,)) for k in inputs.keys()}
        consciousness_state_2, feedback_2 = self.cog_arch.update_architecture(inputs_2)

        # Verify that different inputs produce different outputs
        self.assertFalse(jnp.allclose(consciousness_state, consciousness_state_2))
        self.assertFalse(jnp.allclose(feedback, feedback_2))

    def test_agi_prototype_module(self):
        input_dim = 10
        hidden_dim = 12
        output_dim = 8
        weights1 = jnp.ones((input_dim, hidden_dim))
        weights2 = jnp.ones((hidden_dim, output_dim))

        input_data = jnp.ones((1, input_dim))
        output = self.cog_arch.agi_prototype_module(input_data, weights1, weights2)

        self.assertEqual(output.shape, (1, output_dim))
        self.assertTrue(jnp.allclose(jnp.sum(output), 1.0))  # Softmax output should sum to 1
        self.assertTrue(jnp.all(output >= 0) and jnp.all(output <= 1))  # Check output range

        # Test with different input dimensions
        input_data_2d = jnp.ones((5, input_dim))
        output_2d = self.cog_arch.agi_prototype_module(input_data_2d, weights1, weights2)
        self.assertEqual(output_2d.shape, (5, output_dim))

        # Test that different inputs produce different outputs
        input_data_3 = jnp.zeros((1, input_dim))
        output_3 = self.cog_arch.agi_prototype_module(input_data_3, weights1, weights2)
        self.assertFalse(jnp.allclose(output, output_3))

if __name__ == '__main__':
    unittest.main()

import unittest
import jax.numpy as jnp
from unittest.mock import patch, MagicMock
from NeuroFlex.scientific_domains import IBMIntegration

class TestIBMIntegration(unittest.TestCase):
    def setUp(self):
        self.ibm_integration = IBMIntegration()

    @patch('NeuroFlex.scientific_domains.IBMIntegration.quantum_circuit')
    def test_run_quantum_circuit(self, mock_quantum_circuit):
        mock_result = MagicMock()
        mock_result.get_counts.return_value = {'00': 500, '11': 500}
        mock_quantum_circuit.return_value = mock_result

        result = self.ibm_integration.run_quantum_circuit()
        self.assertEqual(result, {'00': 500, '11': 500})
        mock_quantum_circuit.assert_called_once()

    @patch('NeuroFlex.scientific_domains.IBMIntegration.watson_nlp')
    def test_analyze_text(self, mock_watson_nlp):
        mock_watson_nlp.return_value = {
            'sentiment': 'positive',
            'entities': ['NeuroFlex', 'AI']
        }

        result = self.ibm_integration.analyze_text("NeuroFlex is an amazing AI project!")
        self.assertEqual(result['sentiment'], 'positive')
        self.assertIn('NeuroFlex', result['entities'])
        self.assertIn('AI', result['entities'])
        mock_watson_nlp.assert_called_once_with("NeuroFlex is an amazing AI project!")

    @patch('NeuroFlex.scientific_domains.ibm_integration.ibm_quantum_inspired_optimization')
    def test_integrate_ibm_quantum(self, mock_optimization):
        input_data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        expected_output = jnp.array([0.5, 0.5])
        mock_optimization.return_value = expected_output

        result = self.ibm_integration.integrate_ibm_quantum(input_data)

        # Check if the result is close to the expected output
        self.assertTrue(jnp.allclose(result, jnp.dot(input_data, expected_output)))

        # Verify that the mock function was called with correct arguments
        problem_matrix = jnp.dot(input_data.T, input_data)
        mock_optimization.assert_called_once_with(problem_matrix, input_data.shape[-1])

        # Check the shape of the result
        self.assertEqual(result.shape, input_data.shape)

        # Ensure the result is a JAX array
        self.assertIsInstance(result, jnp.ndarray)

        # Verify that the result is within a valid range (assuming normalized data)
        self.assertTrue(jnp.all(result >= 0) and jnp.all(result <= 1))

if __name__ == '__main__':
    unittest.main()

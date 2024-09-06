import unittest
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

if __name__ == '__main__':
    unittest.main()

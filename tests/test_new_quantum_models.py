import unittest
import numpy as np
from quantum_algorithms_for_protein_prediction import HybridQuantumNeuralNetwork
from quantum_neurobiology_applications import QuantumSynapticTransmission

class TestNewQuantumModels(unittest.TestCase):
    def setUp(self):
        self.hqnn = HybridQuantumNeuralNetwork(num_qubits=4, num_layers=2)
        self.qst = QuantumSynapticTransmission(num_qubits=4)

    def test_hybrid_quantum_neural_network_initialization(self):
        self.assertIsInstance(self.hqnn, HybridQuantumNeuralNetwork)
        self.assertEqual(self.hqnn.num_qubits, 4)
        self.assertEqual(self.hqnn.num_layers, 2)

    def test_hybrid_quantum_neural_network_fit_predict(self):
        print("Starting test_hybrid_quantum_neural_network_fit_predict")
        X = np.random.rand(100, 4)
        y = np.random.randint(2, size=100)
        print(f"Data generated - X shape: {X.shape}, y shape: {y.shape}")
        print(f"X dtype: {X.dtype}, y dtype: {y.dtype}")

        print("Instantiating HybridQuantumNeuralNetwork")
        try:
            hqnn = HybridQuantumNeuralNetwork(num_qubits=4, num_layers=2)
            print("HybridQuantumNeuralNetwork instantiated successfully")
        except Exception as e:
            print(f"Error instantiating HybridQuantumNeuralNetwork: {str(e)}")
            raise

        print("Starting fit method")
        try:
            hqnn.fit(X, y)
            print("Fit method completed")
        except Exception as e:
            print(f"Error during fit method: {str(e)}")
            raise

        try:
            predictions = hqnn.predict(X)
            print(f"Predictions made - shape: {predictions.shape}, dtype: {predictions.dtype}")
        except Exception as e:
            print(f"Error during predict method: {str(e)}")
            raise

        self.assertEqual(len(predictions), 100)
        self.assertTrue(all(isinstance(pred, (np.integer, int, float)) for pred in predictions))

    def test_hybrid_quantum_neural_network_scalability(self):
        large_hqnn = HybridQuantumNeuralNetwork(num_qubits=10, num_layers=5)
        X = np.random.rand(1000, 10)
        y = np.random.randint(2, size=1000)

        large_hqnn.fit(X, y)
        predictions = large_hqnn.predict(X[:10])  # Predict for a subset to save time

        self.assertEqual(len(predictions), 10)

    def test_quantum_synaptic_transmission_initialization(self):
        self.assertIsInstance(self.qst, QuantumSynapticTransmission)
        self.assertEqual(self.qst.num_qubits, 4)

    def test_quantum_synaptic_transmission_simulate(self):
        params = np.random.rand(8)  # 8 parameters for 4 qubits (θ and φ for each)
        transmission_prob = self.qst.simulate_transmission(params)

        self.assertIsInstance(transmission_prob, float)
        self.assertGreaterEqual(transmission_prob, 0)
        self.assertLessEqual(transmission_prob, 1)

    def test_quantum_synaptic_transmission_plasticity(self):
        initial_params = np.random.rand(8) * 2 * np.pi
        plasticity_results = self.qst.simulate_plasticity(initial_params, num_iterations=10)

        self.assertEqual(len(plasticity_results), 10)
        self.assertTrue(all(isinstance(prob, float) for prob in plasticity_results))
        self.assertTrue(all(0 <= prob <= 1 for prob in plasticity_results))

if __name__ == '__main__':
    unittest.main()

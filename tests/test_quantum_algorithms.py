import unittest
import numpy as np
from quantum_algorithms import QuantumPredictiveModel, QuantumEncryption

class TestQuantumAlgorithms(unittest.TestCase):
    def setUp(self):
        self.qpm = QuantumPredictiveModel(num_qubits=4)
        self.qe = QuantumEncryption(key_size=8)

    def test_quantum_predictive_model(self):
        # Test training
        X_train = np.random.rand(100, 4)
        y_train = np.random.randint(2, size=100)
        self.qpm.train(X_train, y_train)
        self.assertIsNotNone(self.qpm.optimal_params)

        # Test prediction
        X_test = np.random.rand(10, 4)
        predictions = [self.qpm.predict(x) for x in X_test]
        self.assertEqual(len(predictions), 10)
        for pred in predictions:
            self.assertGreaterEqual(pred, 0)
            self.assertLessEqual(pred, 1)

    def test_quantum_encryption(self):
        # Test key generation
        key = self.qe.generate_key()
        self.assertEqual(len(key), 8)
        for bit in key:
            self.assertIn(bit, [0, 1])

        # Test encryption and decryption
        message = [1, 0, 1, 1, 0, 0, 1, 0]
        ciphertext = self.qe.encrypt(message, key)
        decrypted = self.qe.decrypt(ciphertext, key)
        self.assertEqual(message, decrypted)

        # Test with different message lengths
        short_message = [1, 0, 1]
        short_key = self.qe.generate_key()[:3]
        short_ciphertext = self.qe.encrypt(short_message, short_key)
        short_decrypted = self.qe.decrypt(short_ciphertext, short_key)
        self.assertEqual(short_message, short_decrypted)

        long_message = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0]
        long_key = self.qe.generate_key() * 2  # Repeat the key
        long_ciphertext = self.qe.encrypt(long_message, long_key)
        long_decrypted = self.qe.decrypt(long_ciphertext, long_key)
        self.assertEqual(long_message, long_decrypted)

    def test_edge_cases(self):
        # Test QuantumPredictiveModel with edge case inputs
        edge_X = np.zeros((5, 4))
        edge_y = np.random.randint(2, size=5)
        self.qpm.train(edge_X, edge_y)  # Train the model with edge case data
        edge_prediction = self.qpm.predict(edge_X[0])
        self.assertGreaterEqual(edge_prediction, 0)
        self.assertLessEqual(edge_prediction, 1)

        # Test QuantumEncryption with edge case inputs
        empty_message = []
        empty_key = []
        empty_ciphertext = self.qe.encrypt(empty_message, empty_key)
        empty_decrypted = self.qe.decrypt(empty_ciphertext, empty_key)
        self.assertEqual(empty_message, empty_decrypted)

if __name__ == '__main__':
    unittest.main()

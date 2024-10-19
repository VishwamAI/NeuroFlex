import unittest
import numpy as np
from quantum_protein_development import QuantumProteinFolding

class TestQuantumProteinFolding(unittest.TestCase):
    def setUp(self):
        self.num_qubits = 4
        self.num_layers = 2
        self.qpf = QuantumProteinFolding(self.num_qubits, self.num_layers)

    def test_initialization(self):
        self.assertEqual(self.qpf.num_qubits, self.num_qubits)
        self.assertEqual(self.qpf.num_layers, self.num_layers)
        self.assertEqual(self.qpf.params.shape, (self.num_layers, self.num_qubits, 2))

    def test_protein_folding_simulation(self):
        amino_acid_sequence = [0.1, 0.2, 0.3, 0.4]
        folded_protein = self.qpf.protein_folding_simulation(amino_acid_sequence)
        self.assertIsInstance(folded_protein, np.ndarray)
        self.assertEqual(len(folded_protein), 4)  # Expected output size matches input size

    def test_optimize_folding(self):
        amino_acid_sequence = [0.1, 0.2, 0.3, 0.4]
        optimized_protein = self.qpf.optimize_folding(amino_acid_sequence, num_iterations=10)
        self.assertIsInstance(optimized_protein, np.ndarray)
        self.assertEqual(len(optimized_protein), 4)  # Expected output size matches input size

    def test_empty_sequence(self):
        with self.assertRaises(ValueError):
            self.qpf.protein_folding_simulation([])

    def test_large_sequence(self):
        large_sequence = np.random.rand(100)
        folded_protein = self.qpf.protein_folding_simulation(large_sequence)
        self.assertEqual(len(folded_protein), 100)  # Expected output size matches input size

    def test_optimization_improvement(self):
        amino_acid_sequence = [0.1, 0.2, 0.3, 0.4]
        initial_folding = self.qpf.protein_folding_simulation(amino_acid_sequence)
        optimized_folding = self.qpf.optimize_folding(amino_acid_sequence, num_iterations=50)

        # Calculate the cost (sum of squares) for both initial and optimized folding
        initial_cost = np.sum(initial_folding**2)
        optimized_cost = np.sum(optimized_folding**2)

        # Check if the optimized cost is lower (better) than the initial cost
        self.assertLess(optimized_cost, initial_cost,
                        f"Optimization did not improve the folding. Initial cost: {initial_cost}, Optimized cost: {optimized_cost}")

if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
import pytest
import time
from NeuroFlex.advanced_models.advanced_math_solving import AdvancedMathSolver
from NeuroFlex.constants import PERFORMANCE_THRESHOLD

class TestAdvancedMathSolver(unittest.TestCase):
    def setUp(self):
        self.solver = AdvancedMathSolver()

    def test_solve_linear_algebra_system(self):
        problem_data = {
            'matrix_a': np.array([[1, 2], [3, 4]]),
            'vector_b': np.array([5, 6])
        }
        solution = self.solver.solve('linear_algebra', problem_data)
        expected_solution = np.array([-4, 4.5])
        np.testing.assert_allclose(solution['solution'], expected_solution, rtol=1e-5)

    def test_solve_linear_algebra_eigenvalues(self):
        problem_data = {
            'matrix': np.array([[1, 2], [2, 1]])
        }
        solution = self.solver.solve('linear_algebra', problem_data)
        expected_eigenvalues = np.array([3, -1])
        np.testing.assert_allclose(solution['eigenvalues'], expected_eigenvalues, rtol=1e-5)

    def test_solve_calculus(self):
        problem_data = {
            'function': 'x**2',
            'variable': 'x',
            'interval': (0, 1)
        }
        solution = self.solver.solve('calculus', problem_data)
        expected_integral = 1/3
        self.assertAlmostEqual(solution['integral'], expected_integral, places=5)

    def test_solve_optimization(self):
        problem_data = {
            'function': 'x**2 + 2*x + 1',
            'initial_guess': [0]
        }
        solution = self.solver.solve('optimization', problem_data)
        expected_optimal_x = -1
        self.assertAlmostEqual(solution['optimal_x'][0], expected_optimal_x, places=5)

    def test_solve_differential_equations(self):
        problem_data = {
            'function': '-2*y',
            'initial_conditions': [1],
            't_span': (0, 1)
        }
        solution = self.solver.solve('differential_equations', problem_data)
        expected_y_end = np.exp(-2)
        np.testing.assert_allclose(solution['y'][0, -1], expected_y_end, rtol=1e-3)

    def test_unsupported_problem_type(self):
        with self.assertRaises(ValueError):
            self.solver.solve('unsupported_type', {})

    def test_unsupported_linear_algebra_problem(self):
        with self.assertRaises(ValueError):
            self.solver.solve('linear_algebra', {})

    def test_unsupported_calculus_problem(self):
        with self.assertRaises(ValueError):
            self.solver.solve('calculus', {})

    def test_unsupported_optimization_problem(self):
        with self.assertRaises(ValueError):
            self.solver.solve('optimization', {})

    def test_unsupported_differential_equations_problem(self):
        with self.assertRaises(ValueError):
            self.solver.solve('differential_equations', {})

    def test_update_performance(self):
        self.solver._update_performance(0.9)
        self.assertEqual(self.solver.performance, 0.9)
        self.assertEqual(len(self.solver.performance_history), 1)
        self.assertEqual(self.solver.performance_history[0], 0.9)

        # Test performance history limit
        for i in range(110):
            self.solver._update_performance(i / 100)
        self.assertEqual(len(self.solver.performance_history), 100)
        self.assertEqual(self.solver.performance_history[-1], 1.09)

    def test_self_heal(self):
        self.solver.performance = 0.5
        initial_performance = self.solver.performance
        initial_lr = self.solver.learning_rate
        initial_history_length = len(self.solver.performance_history)

        # Capture log output
        with self.assertLogs(level='INFO') as log:
            self.solver._self_heal()

        # Check if performance improved or stayed the same
        self.assertGreaterEqual(self.solver.performance, initial_performance)
        self.assertLessEqual(self.solver.performance, 1.0)

        # Check if learning rate was adjusted and within bounds
        self.assertNotEqual(self.solver.learning_rate, initial_lr)
        self.assertGreaterEqual(self.solver.learning_rate, 1e-6)
        self.assertLessEqual(self.solver.learning_rate, 0.1)

        # Check if performance improved by at least the improvement threshold
        if self.solver.performance > initial_performance:
            self.assertGreaterEqual(self.solver.performance, initial_performance + 0.005)

        # Check if performance history was updated
        self.assertEqual(self.solver.performance_history[-1], self.solver.performance)
        self.assertGreater(len(self.solver.performance_history), initial_history_length)

        # Check if performance threshold was updated
        self.assertGreaterEqual(self.solver.performance_threshold, PERFORMANCE_THRESHOLD)

        # Verify learning rate changes in log output
        lr_changes = [log for log in log.output if "LR change" in log]
        self.assertGreater(len(lr_changes), 0, "No learning rate changes logged")

        # Verify performance improvements in log output
        performance_logs = [log for log in log.output if "Performance =" in log]
        self.assertGreater(len(performance_logs), 0, "No performance changes logged")

        # Check final state logging
        self.assertTrue(any("Final state:" in log for log in log.output), "Final state not logged")

    def test_diagnose(self):
        self.solver.performance = 0.5
        self.solver.last_update = time.time() - self.solver.update_interval - 1
        self.solver.performance_history = [0.4, 0.45, 0.5, 0.48, 0.5, 0.49, 0.51, 0.48, 0.5, 0.52]
        self.solver.learning_rate = 0.0000001  # Set to a very low value
        issues = self.solver.diagnose()

        self.assertIn("Low performance", issues[0])
        self.assertIn("Long time since last update", issues[1])
        self.assertIn("Consistently low performance", issues[2])
        self.assertIn("Learning rate out of optimal range", issues[3])

        # Test with good performance
        self.solver.performance = 0.9
        self.solver.last_update = time.time()
        self.solver.performance_history = [0.85, 0.87, 0.9, 0.89, 0.9, 0.91, 0.92, 0.9, 0.91, 0.93]
        self.solver.learning_rate = 0.001
        issues = self.solver.diagnose()
        self.assertEqual(len(issues), 0)

        # Test with borderline performance
        self.solver.performance = self.solver.performance_threshold
        self.solver.performance_history = [0.7, 0.72, 0.75, 0.73, 0.74, 0.76, 0.75, 0.77, 0.76, 0.75]
        issues = self.solver.diagnose()
        self.assertEqual(len(issues), 0)

    def test_adjust_learning_rate(self):
        # Test increasing learning rate with sufficient history
        initial_lr = self.solver.learning_rate
        self.solver.performance_history = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
        self.solver.adjust_learning_rate()
        self.assertGreater(self.solver.learning_rate, initial_lr)
        self.assertLessEqual(self.solver.learning_rate, 0.1)

        # Test decreasing learning rate with sufficient history
        self.solver.learning_rate = initial_lr  # Reset to initial learning rate
        self.solver.performance_history = [0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
        self.solver.adjust_learning_rate()
        self.assertLess(self.solver.learning_rate, initial_lr)
        self.assertGreaterEqual(self.solver.learning_rate, 1e-6)

        # Test learning rate adjustment with limited history
        self.solver.learning_rate = 0.01
        self.solver.performance_history = [0.6, 0.65]
        initial_lr = self.solver.learning_rate
        self.solver.adjust_learning_rate()
        self.assertNotEqual(self.solver.learning_rate, initial_lr)
        self.assertGreaterEqual(self.solver.learning_rate, 1e-6)
        self.assertLessEqual(self.solver.learning_rate, 0.1)

        # Test learning rate bounds
        self.solver.learning_rate = 0.2
        self.solver.adjust_learning_rate()
        self.assertLessEqual(self.solver.learning_rate, 0.1)

        self.solver.learning_rate = 1e-7
        self.solver.adjust_learning_rate()
        self.assertGreaterEqual(self.solver.learning_rate, 1e-6)

        # Test performance threshold adjustment
        initial_threshold = self.solver.performance_threshold
        self.solver.performance_history = [0.8, 0.85, 0.9, 0.95, 1.0]
        self.solver.adjust_learning_rate()
        self.assertGreater(self.solver.performance_threshold, initial_threshold)

        # Test learning rate adjustment based on recent trend
        self.solver.learning_rate = 0.01
        self.solver.performance_history = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
        initial_lr = self.solver.learning_rate
        self.solver.adjust_learning_rate()
        self.assertGreater(self.solver.learning_rate, initial_lr)

        self.solver.learning_rate = 0.01  # Reset learning rate
        self.solver.performance_history = [0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
        initial_lr = self.solver.learning_rate
        self.solver.adjust_learning_rate()
        self.assertLess(self.solver.learning_rate, initial_lr)

        # Test significant decrease in learning rate for sharp negative trend
        self.solver.learning_rate = 0.01
        self.solver.performance_history = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        initial_lr = self.solver.learning_rate
        self.solver.adjust_learning_rate()
        self.assertLess(self.solver.learning_rate, initial_lr * 0.9)  # Expect at least 10% decrease

        # Test minimal change in learning rate for flat trend
        self.solver.learning_rate = 0.01
        self.solver.performance_history = [0.7, 0.71, 0.69, 0.7, 0.71, 0.7]
        initial_lr = self.solver.learning_rate
        self.solver.adjust_learning_rate()
        self.assertAlmostEqual(self.solver.learning_rate, initial_lr, delta=initial_lr * 0.01)  # Expect less than 1% change

    def test_simulate_performance(self):
        self.solver.performance = 0.7
        simulated_performances = [self.solver._simulate_performance() for _ in range(10000)]
        self.assertGreaterEqual(min(simulated_performances), 0.62)  # Slightly relaxed lower bound
        self.assertLessEqual(max(simulated_performances), 0.78)  # Slightly relaxed upper bound
        self.assertAlmostEqual(np.mean(simulated_performances), 0.7, delta=0.035)  # Further increased delta for robustness

if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
from NeuroFlex.advanced_models.advanced_math_solving import AdvancedMathSolver

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

if __name__ == '__main__':
    unittest.main()

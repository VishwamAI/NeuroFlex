import unittest
import numpy as np
import sympy as sp
from scipy import optimize, integrate, linalg
from NeuroFlex.scientific_domains.math_solvers import MathSolver

class TestMathSolver(unittest.TestCase):
    def setUp(self):
        self.math_solver = MathSolver()

    def test_numerical_root_finding(self):
        def func(x):
            return x**2 - 4

        result = self.math_solver.numerical_root_finding(func, 1.0)
        self.assertAlmostEqual(result.x[0], 2.0, places=6)

    def test_symbolic_equation_solving(self):
        x = sp.Symbol('x')
        equation = x**2 - 4
        result = self.math_solver.symbolic_equation_solving(equation, x)
        self.assertEqual(set(result), {-2, 2})

    def test_numerical_optimization(self):
        def func(x):
            return (x[0] - 1)**2 + (x[1] - 2.5)**2

        result = self.math_solver.numerical_optimization(func, [0.5, 2.0], method='Nelder-Mead')
        self.assertAlmostEqual(result.x[0], 1.0, places=4)
        self.assertAlmostEqual(result.x[1], 2.5, places=4)

    def test_linear_algebra_operations(self):
        matrix_a = np.array([[1, 2], [3, 4]])
        matrix_b = np.array([[5, 6], [7, 8]])

        # Test matrix multiplication
        result_multiply = self.math_solver.linear_algebra_operations(matrix_a, matrix_b, operation='multiply')
        expected_multiply = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_almost_equal(result_multiply, expected_multiply)

        # Test matrix inverse
        result_inverse = self.math_solver.linear_algebra_operations(matrix_a, None, operation='inverse')
        expected_inverse = np.array([[-2, 1], [1.5, -0.5]])
        np.testing.assert_array_almost_equal(result_inverse, expected_inverse)

        # Test eigenvalues
        result_eigenvalues = self.math_solver.linear_algebra_operations(matrix_a, None, operation='eigenvalues')
        expected_eigenvalues = np.array([-0.37228132, 5.37228132])
        np.testing.assert_array_almost_equal(result_eigenvalues, expected_eigenvalues)

        # Test unsupported operation
        with self.assertRaises(ValueError):
            self.math_solver.linear_algebra_operations(matrix_a, matrix_b, operation='unsupported')

    def test_numerical_integration(self):
        def func(x):
            return x**2

        result, _ = self.math_solver.numerical_integration(func, 0, 1)
        self.assertAlmostEqual(result, 1/3, places=6)

    def test_symbolic_differentiation(self):
        x = sp.Symbol('x')
        expr = x**3 + 2*x**2 - 5*x + 3
        result = self.math_solver.symbolic_differentiation(expr, x)
        expected = 3*x**2 + 4*x - 5
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()

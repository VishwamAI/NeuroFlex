import unittest
import numpy as np
import sympy as sp
from NeuroFlex.scientific_domains.math_solvers import MathSolver

class TestMathSolver(unittest.TestCase):
    def setUp(self):
        self.math_solver = MathSolver()

    def test_numerical_root_finding(self):
        def f(x):
            return x**2 - 5*x + 6
        result = self.math_solver.numerical_root_finding(f, 0)
        self.assertIsNotNone(result.x)
        self.assertAlmostEqual(f(result.x[0]), 0, places=6)

    def test_symbolic_equation_solving(self):
        x = sp.Symbol('x')
        equation = x**2 - 5*x + 6
        solution = self.math_solver.symbolic_equation_solving(equation, x)
        self.assertIsInstance(solution, list)
        self.assertEqual(len(solution), 2)

    def test_numerical_optimization(self):
        def g(x):
            return (x[0] - 1)**2 + (x[1] - 2.5)**2
        result = self.math_solver.numerical_optimization(g, [0, 0])
        self.assertIsNotNone(result.x)
        self.assertAlmostEqual(result.x[0], 1, places=4)
        self.assertAlmostEqual(result.x[1], 2.5, places=4)

    def test_linear_algebra_operations(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        result = self.math_solver.linear_algebra_operations(A, B)
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_numerical_integration(self):
        def h(x):
            return x**2
        result = self.math_solver.numerical_integration(h, 0, 1)
        self.assertAlmostEqual(result[0], 1/3, places=6)

    def test_symbolic_differentiation(self):
        x = sp.Symbol('x')
        expr = x**2 + 2*x + 1
        result = self.math_solver.symbolic_differentiation(expr, x)
        expected = 2*x + 2
        self.assertEqual(result, expected)

    def test_linear_algebra_operations_inverse(self):
        A = np.array([[1, 2], [3, 4]])
        result = self.math_solver.linear_algebra_operations(A, None, operation='inverse')
        expected = np.array([[-2, 1], [1.5, -0.5]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_linear_algebra_operations_eigenvalues(self):
        A = np.array([[1, 2], [2, 1]])
        result = self.math_solver.linear_algebra_operations(A, None, operation='eigenvalues')
        expected = np.array([3, -1])
        np.testing.assert_array_almost_equal(result, expected)

    def test_error_handling(self):
        with self.assertRaises(ValueError):
            self.math_solver.linear_algebra_operations(np.array([[1, 2], [3, 4]]), None, operation='invalid_operation')

if __name__ == '__main__':
    unittest.main()

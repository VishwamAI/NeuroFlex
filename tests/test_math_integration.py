import unittest
import sympy as sp
import random
from src.scientific_domains.math_integration import MathIntegration

class TestMathIntegration(unittest.TestCase):
    def setUp(self):
        self.math_integration = MathIntegration()

    def test_preprocess_problem(self):
        # Test LaTeX input
        latex_input = r"\frac{1}{2} \times \sqrt{4} + \sin(\pi)"
        expected_output = "(1/2) * math.sqrt(4) + math.sin(math.pi)"
        self.assertEqual(self.math_integration.preprocess_problem(latex_input), expected_output)

        # Test standard mathematical notation
        standard_input = "2^3 + 5 ÷ (3 - 1)"
        expected_output = "2**3 + 5 / (3 - 1)"
        self.assertEqual(self.math_integration.preprocess_problem(standard_input), expected_output)

        # Test additional mathematical functions
        function_input = "log(10) + exp(2) + abs(-5)"
        expected_output = "math.log(10) + math.exp(2) + math.abs(-5)"
        self.assertEqual(self.math_integration.preprocess_problem(function_input), expected_output)

    def test_identify_problem_type(self):
        self.assertEqual(self.math_integration.identify_problem_type("2x + 3 = 7"), "equation_or_inequality")
        self.assertEqual(self.math_integration.identify_problem_type("d/dx(x^2)"), "calculus")
        self.assertEqual(self.math_integration.identify_problem_type("[[1, 2], [3, 4]]"), "linear_algebra")
        self.assertEqual(self.math_integration.identify_problem_type("3 + 4i"), "complex_numbers")
        self.assertEqual(self.math_integration.identify_problem_type("17 mod 5"), "modular_arithmetic")

    def test_solve_problem(self):
        # Test equation solving
        equation = "2x + 3 = 7"
        solution = self.math_integration.solve_problem(equation)
        self.assertEqual(solution, [2])

        # Test inequality
        inequality = "x > 5"
        solution = self.math_integration.solve_problem(inequality)
        self.assertTrue(sp.sympify('x > 5').equals(solution))

        # Test calculus problem
        calculus_problem = "d/dx(x^2)"
        solution = self.math_integration.solve_problem(calculus_problem)
        self.assertEqual(solution, 2*sp.Symbol('x'))

        # Test integration
        integration_problem = "integrate(x^2)"
        solution = self.math_integration.solve_problem(integration_problem)
        self.assertEqual(solution, sp.Pow(sp.Symbol('x'), 3) / 3)

        # Test linear algebra problem
        linear_algebra_problem = "[[1, 2], [3, 4]]"
        solution = self.math_integration.solve_problem(linear_algebra_problem)
        self.assertIsInstance(solution, dict)
        self.assertIn('determinant', solution)
        self.assertIn('eigenvalues', solution)

        # Test system of equations
        system_problem = "x + y = 10, 2x - y = 5"
        solution = self.math_integration.solve_problem(system_problem)
        self.assertIsInstance(solution, dict)
        self.assertEqual(solution[sp.Symbol('x')], 5)
        self.assertEqual(solution[sp.Symbol('y')], 5)

        # Test complex numbers
        complex_problem = "3 + 4j"
        solution = self.math_integration.solve_problem(complex_problem)
        self.assertEqual(solution, 3 + 4j)

        # Test modular arithmetic
        modular_problem = "17 mod 5"
        solution = self.math_integration.solve_problem(modular_problem)
        self.assertEqual(solution, 2)

        # Test error handling
        invalid_problem = "invalid equation"
        solution = self.math_integration.solve_problem(invalid_problem)
        self.assertIsNone(solution)

        # Test LaTeX input
        latex_problem = r"\frac{1}{2} \times \sqrt{4} + \sin(\pi)"
        solution = self.math_integration.solve_problem(latex_problem)
        self.assertAlmostEqual(float(solution), 1.5)

    def test_evaluate_solution(self):
        problem = "2 + 2"
        correct_solution = 4
        incorrect_solution = 5
        self.assertTrue(self.math_integration.evaluate_solution(problem, correct_solution))
        self.assertFalse(self.math_integration.evaluate_solution(problem, incorrect_solution))

    def test_polynomial_coeffs_with_roots(self):
        roots = [1, 2]
        coeffs = self.math_integration.solve_problem(f"polynomial_coeffs_with_roots({roots})")
        self.assertEqual(coeffs, [2, -3, 1])  # x^2 - 3x + 2 has roots 1 and 2

    def test_polynomial_roots(self):
        coeffs = [1, -5, 6]  # x^2 - 5x + 6
        roots = self.math_integration.solve_problem(f"polynomial_roots({coeffs})")
        self.assertEqual(set(roots), {2, 3})

    def test_random_polynomial(self):
        for _ in range(5):  # Test with 5 random polynomials
            degree = random.randint(2, 5)
            roots = [random.randint(-10, 10) for _ in range(degree)]
            coeffs = self.math_integration.solve_problem(f"polynomial_coeffs_with_roots({roots})")
            calculated_roots = self.math_integration.solve_problem(f"polynomial_roots({coeffs})")
            self.assertEqual(set(map(float, calculated_roots)), set(map(float, roots)))

if __name__ == '__main__':
    unittest.main()

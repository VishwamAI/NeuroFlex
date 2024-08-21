import unittest
import jax
import jax.numpy as jnp
from NeuroFlex.advanced_math_solving import AdvancedMathSolving, create_advanced_math_solving
import sympy

class TestAdvancedMathSolving(unittest.TestCase):
    def setUp(self):
        self.math_solver = create_advanced_math_solving([64, 32], 10)

    def test_generate_problem(self):
        for difficulty in range(1, 5):
            problem, solution = self.math_solver.generate_problem(difficulty)
            self.assertIsInstance(problem, str)
            self.assertIsNotNone(solution)

    def test_solve_problem(self):
        problem = "2x + 3 = 7"
        result = self.math_solver.solve_problem(problem)
        self.assertIn('solution', result)
        self.assertIn('steps', result)
        self.assertIsInstance(result['solution'], list)
        self.assertIsInstance(result['steps'], list)

    def test_evaluate_solution(self):
        problem = "2x + 3 = 7"
        correct_solution = 2
        self.assertTrue(self.math_solver.evaluate_solution(problem, 2, correct_solution))
        self.assertFalse(self.math_solver.evaluate_solution(problem, 3, correct_solution))

    def test_consciousness_simulation(self):
        problem = "2x + 3 = 7"
        result = self.math_solver.solve_problem(problem)
        self.assertIn('consciousness_state', result)
        self.assertIsInstance(result['consciousness_state'], jnp.ndarray)

    def test_generate_step_by_step_solution(self):
        problem = "x^2 - 4 = 0"
        result = self.math_solver.solve_problem(problem)
        self.assertIn('steps', result)
        self.assertGreater(len(result['steps']), 0)

    def test_error_handling(self):
        with self.assertRaises(ValueError):
            self.math_solver.generate_problem(0)  # Invalid difficulty
        with self.assertRaises(Exception):
            self.math_solver.solve_problem("invalid problem")

if __name__ == '__main__':
    unittest.main()

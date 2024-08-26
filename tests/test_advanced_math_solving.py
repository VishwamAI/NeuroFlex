import unittest
import jax
import jax.numpy as jnp
from NeuroFlex.advanced_math_solving import AdvancedMathSolving, create_advanced_math_solving
import sympy
import logging
import sys
from io import StringIO

class TestAdvancedMathSolving(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up logging to capture output
        cls.log_capture = StringIO()
        logging.basicConfig(level=logging.DEBUG, stream=cls.log_capture, format='%(levelname)s: %(message)s')

        cls.math_solver = None
        initialization_attempts = [
            ([64, 32], 10),
            ([32, 16], 8),
            ([128, 64], 16)
        ]
        for features, output_dim in initialization_attempts:
            try:
                logging.info(f"Attempting to initialize AdvancedMathSolving with features={features}, output_dim={output_dim}")
                cls.math_solver = create_advanced_math_solving(features, output_dim)
                if cls.math_solver is None or not isinstance(cls.math_solver, AdvancedMathSolving):
                    raise ValueError(f"create_advanced_math_solving returned an invalid object with parameters: features={features}, output_dim={output_dim}")
                logging.info("AdvancedMathSolving initialized successfully")
                break
            except Exception as e:
                logging.error(f"Failed to initialize AdvancedMathSolving with parameters: features={features}, output_dim={output_dim}. Error: {str(e)}", exc_info=True)

        if cls.math_solver is None:
            logging.error("Failed to initialize AdvancedMathSolving after multiple attempts")
            cls.initialization_failed = True
        else:
            cls.initialization_failed = False

    def setUp(self):
        self.assertFalse(self.__class__.initialization_failed, "AdvancedMathSolving initialization failed. Check class initialization logs for details.")
        self.assertIsNotNone(self.math_solver, "AdvancedMathSolving not initialized in setUp. Class initialization might have failed.")

    def test_initialization(self):
        self.assertIsInstance(self.math_solver, AdvancedMathSolving)
        self.assertTrue(self.math_solver.is_initialized())

    def test_generate_problem(self):
        for difficulty in range(1, 5):
            problem, solution, thought = self.math_solver.generate_problem(difficulty)
            self.assertIsInstance(problem, str)
            self.assertIsNotNone(solution)
            self.assertIsNotNone(thought)

    def test_solve_problem(self):
        problem = "2x + 3 = 7"
        result = self.math_solver.solve_problem(problem)
        self.assertIn('solution', result)
        self.assertIn('steps', result)
        self.assertIsInstance(result['solution'], (list, str))
        self.assertIsInstance(result['steps'], list)

    def test_evaluate_solution(self):
        problem = "2x + 3 = 7"
        correct_solution = 2
        evaluation = self.math_solver.evaluate_solution(problem, 2, correct_solution)
        self.assertIn('is_correct', evaluation)
        self.assertTrue(evaluation['is_correct'])
        evaluation = self.math_solver.evaluate_solution(problem, 3, correct_solution)
        self.assertIn('is_correct', evaluation)
        self.assertFalse(evaluation['is_correct'])

    def test_consciousness_simulation(self):
        problem = "2x + 3 = 7"
        result = self.math_solver.solve_problem(problem)
        self.assertIn('consciousness_state', result)
        self.assertIsInstance(result['consciousness_state'], jnp.ndarray)
        self.assertIn('thought', result)
        self.assertIsInstance(result['thought'], str)

    def test_generate_step_by_step_solution(self):
        problem = "x^2 - 4 = 0"
        result = self.math_solver.solve_problem(problem)
        self.assertIn('steps', result)
        self.assertGreater(len(result['steps']), 0)

    def test_error_handling(self):
        with self.assertRaises(ValueError) as context:
            self.math_solver.generate_problem(0)  # Invalid difficulty
        self.assertIn("Difficulty level not supported", str(context.exception))

        result = self.math_solver.solve_problem("invalid problem")
        self.assertIn('error', result)
        self.assertIsInstance(result['error'], str)
        logging.info(f"Error message for invalid problem: {result['error']}")

        with self.assertRaises(Exception) as context:
            self.math_solver.evaluate_solution("2x + 3 = 7", "invalid", 2)
        logging.error(f"Exception raised during solution evaluation: {str(context.exception)}")

if __name__ == '__main__':
    unittest.main()

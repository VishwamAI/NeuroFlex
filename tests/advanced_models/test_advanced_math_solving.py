import unittest
import numpy as np
import pytest
import time
import itertools
from unittest.mock import Mock, patch
from NeuroFlex.advanced_models.advanced_math_solving import AdvancedMathSolver


class TestAdvancedMathSolver(unittest.TestCase):
    def setUp(self):
        self.solver = AdvancedMathSolver()

    def test_solve_linear_algebra_system(self):
        problem_data = {
            "matrix_a": np.array([[1, 2], [3, 4]]),
            "vector_b": np.array([5, 6]),
        }
        solution = self.solver.solve("linear_algebra", problem_data)
        expected_solution = np.array([-4, 4.5])
        np.testing.assert_allclose(solution["solution"], expected_solution, rtol=1e-5)

    def test_solve_linear_algebra_eigenvalues(self):
        problem_data = {"matrix": np.array([[1, 2], [2, 1]])}
        solution = self.solver.solve("linear_algebra", problem_data)
        expected_eigenvalues = np.array([3, -1])
        np.testing.assert_allclose(
            solution["eigenvalues"], expected_eigenvalues, rtol=1e-5
        )

    def test_solve_calculus(self):
        problem_data = {"function": "x**2", "variable": "x", "interval": (0, 1)}
        solution = self.solver.solve("calculus", problem_data)
        expected_integral = 1 / 3
        self.assertAlmostEqual(solution["integral"], expected_integral, places=5)

    def test_solve_optimization(self):
        problem_data = {"function": "x**2 + 2*x + 1", "initial_guess": [0]}
        solution = self.solver.solve("optimization", problem_data)
        expected_optimal_x = -1
        self.assertAlmostEqual(solution["optimal_x"][0], expected_optimal_x, places=5)

    def test_solve_differential_equations(self):
        problem_data = {"function": "-2*y", "initial_conditions": [1], "t_span": (0, 1)}
        solution = self.solver.solve("differential_equations", problem_data)
        expected_y_end = np.exp(-2)
        np.testing.assert_allclose(solution["y"][0, -1], expected_y_end, rtol=1e-3)

    def test_unsupported_problem_type(self):
        with self.assertRaises(ValueError):
            self.solver.solve("unsupported_type", {})

    def test_unsupported_linear_algebra_problem(self):
        with self.assertRaises(ValueError):
            self.solver.solve("linear_algebra", {})

    def test_unsupported_calculus_problem(self):
        with self.assertRaises(ValueError):
            self.solver.solve("calculus", {})

    def test_unsupported_optimization_problem(self):
        with self.assertRaises(ValueError):
            self.solver.solve("optimization", {})

    def test_unsupported_differential_equations_problem(self):
        with self.assertRaises(ValueError):
            self.solver.solve("differential_equations", {})

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

        # Mock the RL agent's methods
        self.solver.rl_agent.select_action = unittest.mock.Mock(return_value=0.1)
        self.solver.rl_agent.update = unittest.mock.Mock()

        # Mock the _simulate_performance method to return slightly improving values
        mock_performances = [0.51, 0.52, 0.53, 0.54, 0.55]
        mock_performance_iter = iter(mock_performances)

        def mock_simulate_performance():
            return next(mock_performance_iter)

        self.solver._simulate_performance = mock_simulate_performance

        self.solver._self_heal()

        # Check if performance improved
        self.assertGreater(self.solver.performance, initial_performance)
        self.assertLessEqual(self.solver.performance, 1.0)

        # Check if learning rate was adjusted
        self.assertNotEqual(self.solver.learning_rate, initial_lr)
        self.assertLessEqual(self.solver.learning_rate, 0.1)
        self.assertGreaterEqual(self.solver.learning_rate, 1e-5)

        # Check if performance improved but might not have reached the threshold
        self.assertGreater(self.solver.performance, initial_performance)
        self.assertLessEqual(self.solver.performance, self.solver.performance_threshold)

        # Verify RL agent method calls
        self.assertEqual(
            self.solver.rl_agent.select_action.call_count,
            self.solver.max_healing_attempts,
        )
        self.assertEqual(
            self.solver.rl_agent.update.call_count, self.solver.max_healing_attempts
        )

        # Reset mocks and solver state for the second test
        self.solver.rl_agent.select_action.reset_mock()
        self.solver.rl_agent.update.reset_mock()
        self.solver.performance = 0.5
        initial_performance = self.solver.performance

        # Mock _simulate_performance to return values with no improvement
        mock_no_improvement = [0.5, 0.5, 0.5, 0.5, 0.5]
        mock_no_improvement_iter = iter(mock_no_improvement)

        def mock_simulate_performance_no_improvement():
            return next(mock_no_improvement_iter)

        self.solver._simulate_performance = mock_simulate_performance_no_improvement

        self.solver._self_heal()

        # Check if performance remained unchanged
        self.assertEqual(self.solver.performance, initial_performance)

        # Verify RL agent method calls for the second test
        self.assertEqual(
            self.solver.rl_agent.select_action.call_count,
            self.solver.max_healing_attempts,
        )
        self.assertEqual(
            self.solver.rl_agent.update.call_count, self.solver.max_healing_attempts
        )

    def test_diagnose(self):
        # Test with low performance and outdated last_update
        self.solver.performance = 0.5
        self.solver.last_update = time.time() - (self.solver.update_interval + 1)
        self.solver.performance_history = [0.4, 0.45, 0.5, 0.48, 0.5]
        issues = self.solver.diagnose()
        self.assertEqual(len(issues), 3)
        self.assertTrue(any("Low performance" in issue for issue in issues))
        self.assertTrue(any("Long time since last update" in issue for issue in issues))
        self.assertTrue(
            any("Consistently low performance" in issue for issue in issues)
        )

        # Test with good performance
        self.solver.performance = 0.9
        self.solver.last_update = time.time()
        self.solver.performance_history = [0.85, 0.87, 0.9, 0.89, 0.9]
        issues = self.solver.diagnose()
        self.assertEqual(len(issues), 0)

        # Test with performance just below threshold
        self.solver.performance = self.solver.performance_threshold - 0.01
        self.solver.performance_history = [0.6, 0.65, 0.7, 0.75, 0.7]
        issues = self.solver.diagnose()
        self.assertEqual(len(issues), 2)
        self.assertTrue(any("Low performance" in issue for issue in issues))
        self.assertTrue(
            any("Consistently low performance" in issue for issue in issues)
        )

        # Test with consistently low performance but recent update
        self.solver.performance = 0.6
        self.solver.last_update = time.time()
        self.solver.performance_history = [0.55, 0.58, 0.6, 0.57, 0.6]
        issues = self.solver.diagnose()
        self.assertEqual(len(issues), 2)
        self.assertIn("Low performance", issues[0])
        self.assertIn("Consistently low performance", issues[1])

        # Test with all issues present
        self.solver.performance = 0.5
        self.solver.last_update = time.time() - (self.solver.update_interval + 1)
        self.solver.performance_history = [0.45, 0.48, 0.5, 0.47, 0.5]
        issues = self.solver.diagnose()
        self.assertEqual(len(issues), 3)
        self.assertTrue(any("Low performance" in issue for issue in issues))
        self.assertTrue(any("Long time since last update" in issue for issue in issues))
        self.assertTrue(
            any("Consistently low performance" in issue for issue in issues)
        )

    def test_simulate_performance(self):
        self.solver.performance = 0.7
        simulated_performances = [
            self.solver._simulate_performance() for _ in range(1000)
        ]
        self.assertGreaterEqual(min(simulated_performances), 0.63)  # 0.7 * 0.9
        self.assertLessEqual(max(simulated_performances), 0.77)  # 0.7 * 1.1
        self.assertAlmostEqual(np.mean(simulated_performances), 0.7, delta=0.01)

    def test_rl_agent_integration(self):
        # Test RLAgent initialization
        self.assertIsNotNone(self.solver.rl_agent)
        self.assertEqual(self.solver.rl_agent.state_dim, 3)
        self.assertEqual(self.solver.rl_agent.action_dim, 1)

        # Mock RL agent methods
        self.solver.rl_agent.select_action = Mock(return_value=0.1)
        self.solver.rl_agent.update = Mock()

        # Mock logging
        with self.assertLogs(level="INFO") as log_context:
            # Test learning process
            initial_performance = self.solver.performance
            initial_learning_rate = self.solver.learning_rate

            # Mock _simulate_performance to return gradually improving values
            mock_performances = [initial_performance + i * 0.05 for i in range(1, 11)]
            mock_performance_iter = itertools.cycle(mock_performances)
            self.solver._simulate_performance = lambda: next(mock_performance_iter)

            self.solver._self_heal()

            # Verify performance improvement
            self.assertGreater(self.solver.performance, initial_performance)
            self.assertLessEqual(self.solver.performance, 1.0)

            # Verify learning rate adjustment
            self.assertNotEqual(self.solver.learning_rate, initial_learning_rate)
            self.assertLessEqual(self.solver.learning_rate, 0.1)
            self.assertGreaterEqual(self.solver.learning_rate, 1e-5)

            # Verify logging of RL agent method calls
            log_messages = [record.getMessage() for record in log_context.records]
            self.assertTrue(any("RL agent method calls" in msg for msg in log_messages))

        # Reset mock call counts
        self.solver.rl_agent.select_action.reset_mock()
        self.solver.rl_agent.update.reset_mock()

        # Test multiple self-heal iterations
        with self.assertLogs(level="INFO") as log_context:
            for _ in range(4):
                self.solver._self_heal()

            # Verify overall improvement
            self.assertGreater(self.solver.performance, initial_performance)
            self.assertLessEqual(
                self.solver.performance, self.solver.performance_threshold
            )

            # Verify RL agent method calls
            log_messages = [record.getMessage() for record in log_context.records]
            rl_agent_calls = [
                msg for msg in log_messages if "RL agent method calls" in msg
            ]
            self.assertEqual(
                len(rl_agent_calls), 4
            )  # One log message per self-heal call

            # Extract the actual call counts from the log messages
            select_action_calls = sum(
                int(msg.split("select_action: ")[1].split(",")[0])
                for msg in rl_agent_calls
            )
            update_calls = sum(int(msg.split("update: ")[1]) for msg in rl_agent_calls)

            # Verify the total number of calls
            expected_calls = (
                self.solver.max_healing_attempts * 4
            )  # 4 total calls to _self_heal
            self.assertAlmostEqual(select_action_calls, expected_calls, delta=5)
            self.assertAlmostEqual(update_calls, expected_calls, delta=5)

            # Ensure the calls are within a reasonable range
            self.assertGreaterEqual(select_action_calls, expected_calls - 5)
            self.assertLessEqual(select_action_calls, expected_calls + 5)
            self.assertGreaterEqual(update_calls, expected_calls - 5)
            self.assertLessEqual(update_calls, expected_calls + 5)

            # Log the actual number of calls for debugging
            print(f"DEBUG: Expected calls: {expected_calls}")
            print(f"DEBUG: Actual select_action calls: {select_action_calls}")
            print(f"DEBUG: Actual update calls: {update_calls}")


if __name__ == "__main__":
    unittest.main()

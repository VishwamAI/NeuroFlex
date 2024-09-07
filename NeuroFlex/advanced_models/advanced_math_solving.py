import numpy as np
from scipy import optimize, integrate
from typing import List, Dict, Any, Union

class AdvancedMathSolver:
    def __init__(self):
        self.available_methods = {
            'linear_algebra': self.solve_linear_algebra,
            'calculus': self.solve_calculus,
            'optimization': self.solve_optimization,
            'differential_equations': self.solve_differential_equations
        }

    def solve(self, problem_type: str, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve an advanced math problem based on the problem type and data.

        Args:
            problem_type (str): Type of math problem (e.g., 'linear_algebra', 'calculus')
            problem_data (Dict[str, Any]): Dictionary containing problem-specific data

        Returns:
            Dict[str, Any]: Solution to the problem
        """
        if problem_type not in self.available_methods:
            raise ValueError(f"Unsupported problem type: {problem_type}")

        return self.available_methods[problem_type](problem_data)

    def solve_linear_algebra(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Solve linear algebra problems."""
        if 'matrix_a' in problem_data and 'vector_b' in problem_data:
            A = problem_data['matrix_a']
            b = problem_data['vector_b']
            x = np.linalg.solve(A, b)
            return {'solution': x}
        elif 'matrix' in problem_data:
            A = problem_data['matrix']
            eigenvalues, eigenvectors = np.linalg.eig(A)
            return {'eigenvalues': eigenvalues, 'eigenvectors': eigenvectors}
        else:
            raise ValueError("Unsupported linear algebra problem")

    def solve_calculus(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Solve calculus problems."""
        if 'function' in problem_data and 'variable' in problem_data:
            def f(x):
                return eval(problem_data['function'])

            a, b = problem_data.get('interval', (-np.inf, np.inf))
            integral, error = integrate.quad(f, a, b)
            return {'integral': integral, 'error': error}
        else:
            raise ValueError("Unsupported calculus problem")

    def solve_optimization(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Solve optimization problems."""
        if 'function' in problem_data and 'initial_guess' in problem_data:
            def f(x):
                return eval(problem_data['function'])

            x0 = problem_data['initial_guess']
            result = optimize.minimize(f, x0)
            return {'optimal_x': result.x, 'optimal_value': result.fun}
        else:
            raise ValueError("Unsupported optimization problem")

    def solve_differential_equations(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Solve differential equations."""
        if 'function' in problem_data and 'initial_conditions' in problem_data:
            def f(t, y):
                return eval(problem_data['function'])

            t_span = problem_data.get('t_span', (0, 10))
            y0 = problem_data['initial_conditions']

            # Use RK45 method with increased number of time points
            t_eval = np.linspace(t_span[0], t_span[1], 1000)
            solution = integrate.solve_ivp(f, t_span, y0, method='RK45', t_eval=t_eval, rtol=1e-8, atol=1e-8)
            return {'t': solution.t, 'y': solution.y}
        else:
            raise ValueError("Unsupported differential equation problem")

# Example usage
if __name__ == "__main__":
    solver = AdvancedMathSolver()

    # Example linear algebra problem
    linear_algebra_problem = {
        'matrix_a': np.array([[1, 2], [3, 4]]),
        'vector_b': np.array([5, 6])
    }

    solution = solver.solve('linear_algebra', linear_algebra_problem)
    print("Linear Algebra Solution:", solution)
